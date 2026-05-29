import { Chroma } from "@langchain/community/vectorstores/chroma";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import type { ParsedPaper } from "./ingest.js";
import { cleanText, isValidText } from "../helpers.js";

export interface PaperChunk {
  content: string;
  metadata: {
    arxivId: string;
    title: string;
    section: string;
    chunkIndex: number;
    totalChunks: number;
  };
}

export interface VectorStore {
  store: Chroma;
  collectionName: string;
  chunkCount: number;
}

export async function chunkPaper(paper: ParsedPaper): Promise<PaperChunk[]> {
  paper.sections.unshift({ title: "Abstract", content: paper.abstract });

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", ". ", " "],
  });
  const chunks: PaperChunk[] = [];
  let globalChunkIndex = 0;

  for (const section of paper.sections) {
    const sectionChunks = await splitter.createDocuments([section.content]);
    const totalChunks = sectionChunks.length;

    for (let i = 0; i < totalChunks; i++) {
      const chunkContent = sectionChunks[i].pageContent;
      chunks.push({
        content: chunkContent,
        metadata: {
          arxivId: paper.arxivId,
          title: paper.title,
          section: section.title,
          chunkIndex: globalChunkIndex,
          totalChunks: totalChunks,
        },
      });
      globalChunkIndex++;
    }
  }
  return chunks;
}

function getVoyageEmbeddings(
  inputType: "document" | "query" = "document",
): VoyageEmbeddings {
  const apiKey = process.env.VOYAGE_API_KEY;
  if (!apiKey || apiKey.trim() === "") {
    throw new Error(
      "Missing VOYAGE_API_KEY. Add VOYAGE_API_KEY to your .env or environment and rerun the demo.",
    );
  }
  return new VoyageEmbeddings({
    apiKey,
    modelName: "voyage-4",
    batchSize: 32,
    inputType,
    truncation: true,
  });
}

export async function createVectorStore(
  paper: ParsedPaper,
  chunks: PaperChunk[],
): Promise<VectorStore> {
  const collectionName = `arxiv-${paper.arxivId.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
  const embeddings = getVoyageEmbeddings();
  const docs = chunks
    .map((chunk) => {
      const cleaned = cleanText(chunk.content);
      return {
        content: cleaned,
        metadata: chunk.metadata,
      };
    })
    .filter((c) => isValidText(c.content))
    .map(
      (chunk) =>
        new Document({
          pageContent: chunk.content,
          metadata: chunk.metadata,
        }),
    );
  const store = new Chroma(embeddings, {
    collectionName,
    url: process.env.CHROMA_URL ?? "http://localhost:8000",
  });

  await store.addDocuments(docs);
  const chunkCount = docs.length;
  return { store, collectionName, chunkCount };
}

export async function loadVectorStore(
  arxivId: string,
): Promise<VectorStore | null> {
  const collectionName = `arxiv-${arxivId.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
  const embeddings = getVoyageEmbeddings();
  try {
    const store = new Chroma(embeddings, {
      collectionName,
      url: process.env.CHROMA_URL ?? "http://localhost:8000",
    });
    const collection = await store.ensureCollection();
    const chunkCount = await collection.count();
    if (chunkCount === 0) {
      return null;
    }
    return { store, collectionName, chunkCount };
  } catch (error) {
    return null;
  }
}

export async function searchStore(
  store: VectorStore,
  query: string,
  k: number = 4,
): Promise<
  Array<{
    content: string;
    score: number;
    section: string;
    metadata: Record<string, unknown>;
  }>
> {
  const embeddings = getVoyageEmbeddings("query");
  const queryEmbedding = await embeddings.embedQuery(query);
  const results = await store.store.similaritySearchVectorWithScore(
    queryEmbedding,
    k,
  );
  return results.map(([doc, score]) => ({
    content: doc.pageContent,
    score,
    section: doc.metadata.section,
    metadata: doc.metadata,
  }));
}

export async function getOrCreateStore(
  paper: ParsedPaper,
): Promise<VectorStore> {
  let store = await loadVectorStore(paper.arxivId);
  if (store) {
    console.log(
      `Loaded existing vector store with ${store.chunkCount} chunks.`,
    );
    return store;
  }
  console.log("No existing vector store found. Creating a new one...");
  const chunks = await chunkPaper(paper);
  store = await createVectorStore(paper, chunks);
  console.log(`Created new vector store with ${store.chunkCount} chunks.`);
  return store;
}
