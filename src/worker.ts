import { config } from "dotenv";
config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Worker } from "bullmq";
import { CohereEmbeddings } from "@langchain/cohere";
import { QdrantVectorStore } from "@langchain/qdrant";


const worker = new Worker('file-processing', async job => {
    try {
        const data = typeof job.data === "string" ? JSON.parse(job.data) : job.data;
        const userId = data.userId as string | undefined;

        if (!userId) {
            throw new Error('Missing userId in file-processing job payload.');
        }

        const collectionName = `notebook-${userId}`;

        const response = await fetch(data.filePath);
        const blob = await response.blob();
        const loader = new PDFLoader(blob);
        const docs = await loader.load();
        console.log(`PDF ${data.originalName} loaded with ${docs.length} pages`);

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const chunks = await textSplitter.splitDocuments(docs);
        console.log(`PDF ${data.originalName} split into ${chunks.length} chunks`);

        const embeddings = new CohereEmbeddings({
            apiKey: process.env.COHERE_API_KEY as string,
            model: "embed-english-v3.0",
            inputType: "search_document",
        });

        console.log("Creating or connecting to Qdrant collection...");

        const vectorStore = await QdrantVectorStore.fromDocuments(chunks, embeddings, {
            url: process.env.QDRANT_URL as string,
            apiKey: process.env.QDRANT_API_KEY as string,
            collectionName,
        });

        console.log(`Successfully stored ${chunks.length} chunks to Qdrant collection ${collectionName}.`);
    } catch (error) {
        console.error("Error processing file:", error);
        throw error;
    }
}, {
    concurrency: 100, connection: {
        url: process.env.REDIS_URL as string,
        tls: {},
        maxRetriesPerRequest: null,
    }
});

worker.on("ready", () => {
    console.log("BullMQ worker is ready and listening for file-processing jobs.");
});

worker.on("completed", (job) => {
    console.log(`Job ${job.id} completed.`);
});

worker.on("failed", (job, err) => {
    console.error(`Job ${job?.id} failed:`, err);
});