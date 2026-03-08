import express from "express";
import cors from "cors";
import multer from "multer";
import { v2 as cloudinary } from "cloudinary";
import { PrismaClient } from "./generated/prisma/client.js";
import { PrismaPg } from "@prisma/adapter-pg";
import { Queue } from "bullmq";
import { QdrantVectorStore } from "@langchain/qdrant";
import { CohereEmbeddings } from "@langchain/cohere";
import { CohereClientV2 } from "cohere-ai";
import dotenv from "dotenv";

dotenv.config();

cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME as string,
    api_key: process.env.CLOUDINARY_API_KEY as string,
    api_secret: process.env.CLOUDINARY_API_SECRET as string,
});

const prisma = new PrismaClient({ adapter: new PrismaPg({ connectionString: process.env.DATABASE_URL as string }) });
const client = new CohereClientV2({ token: process.env.COHERE_API_KEY });

const queue = new Queue('file-processing', {
    connection: {
        url: process.env.REDIS_URL as string,
        tls: {},
        maxRetriesPerRequest: null,
    }
});

const cloudinaryStorage: multer.StorageEngine = {
    _handleFile(req, file, cb) {
        const uploadStream = cloudinary.uploader.upload_stream(
            {
                resource_type: 'raw',
                folder: 'rag-documents',
                use_filename: true,
                unique_filename: true,
            },
            (error, result) => {
                if (error) return cb(error);
                cb(null, {
                    path: result!.secure_url,
                    filename: result!.public_id,
                    size: result!.bytes,
                });
            }
        );
        file.stream.pipe(uploadStream);
    },
    _removeFile(req, file, cb) {
        cloudinary.uploader.destroy(file.filename || '', { resource_type: 'raw' }, (error) => {
            cb(error || null);
        });
    }
};

const app = express();
const PORT = process.env.PORT || 5003;

app.use(cors());
const upload = multer({ storage: cloudinaryStorage });


app.get("/", (req, res) => {
    res.send("Hello from RAG Vector DB Microservice!");
});

app.post("/upload/", upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).send("No file uploaded.");
    }

    const userId = req.body.userId as string;
    if (!userId) {
        return res.status(400).json({ error: 'Missing userId in request body.' });
    }

    const pdfUrl = req.file.path;

    await prisma.pdf.create({
        data: { userId, pdfUrl, originalName: req.file.originalname },
    });

    await queue.add('file-processing', JSON.stringify({
        filePath: pdfUrl,
        originalName: req.file.originalname,
        userId,
    }));

    res.json({ message: `File ${req.file.originalname} uploaded successfully.`, pdfUrl });
});

app.get('/pdfs/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const pdfs = await prisma.pdf.findMany({
            where: { userId },
            orderBy: { id: 'desc' },
        });
        res.json({ pdfs });
    } catch (error: any) {
        console.error('Fetch PDFs error:', error);
        res.status(500).json({ error: error?.message || 'Internal server error.' });
    }
});

app.get('/chat', async (req, res) => {
    try {
        const userQuery = req.query.question;
        const userId = req.query.userId;
        if (!userQuery) {
            return res.status(400).json({ error: 'Missing question parameter.' });
        }
        if (!userId) {
            return res.status(400).json({ error: 'Missing userId parameter.' });
        }

        const query = userQuery as string;
        const collectionName = `notebook-${userId as string}`;

        // Use Cohere embeddings with inputType for query
        const embeddings = new CohereEmbeddings({
            apiKey: process.env.COHERE_API_KEY as string,
            model: 'embed-english-v3.0',
            inputType: 'search_query',
        });

        // Qdrant vector store
        const vectorStore = await QdrantVectorStore.fromExistingCollection(
            embeddings,
            {
                url: process.env.QDRANT_URL as string,
                apiKey: process.env.QDRANT_API_KEY as string,
                collectionName,
            }
        );
        const retriever = vectorStore.asRetriever({ k: 2 });
        const contextResults = await retriever.invoke(query);
        console.log("Retrieved context:", contextResults);

        // Prepare context for LLM
        const context = JSON.stringify(contextResults);
        const prompt = `You are a helpful AI assistant. Answer the user's question in Markdown format with proper and well formatted content like heading and paragraph should not be on same line and other such best practices of content and for heading write heading, sub-heading and paragraph in this way # heading, ## sub-heading, ### paragraph respectively.  Answer based only on the following context from PDF files. If context is not available then answer based on your knowledge.\nContext: ${context}.  `;

        // Use Cohere LLM (CohereClientV2 messages format)
        const response = await client.chat({
            model: 'command-a-03-2025',
            messages: [
                { role: 'system', content: prompt },
                { role: 'user', content: query }
            ]
        });

        const answer = response.message?.content?.[0]?.type === 'text'
            ? response.message.content[0].text
            : '';

        console.log("LLM response:", answer);

        res.json({
            message: answer,
            context: contextResults
        });
    } catch (error: any) {
        console.error('Chat error:', error);
        res.status(500).json({ error: error?.message || 'Internal server error.' });
    }
});


app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
