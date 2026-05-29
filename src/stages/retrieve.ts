import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { VectorStore } from "./embed.js";
import { searchStore } from "./embed.js";
import { VoyageAIClient } from "voyageai";
import { RunnableSequence } from "@langchain/core/runnables";

export interface RAGResult {
  question: string;
  answer: string;
  retrievedChunks: Array<{ content: string; section: string; score: number }>;
  queries: string[];
  model: string;
}

export function buildRAGPrompt(): ChatPromptTemplate {
  const template = `You are a helpful assistant that answers questions about a scientific paper. Use only the following retrieved chunks from the paper to answer the question. If you don't have enough information, say "I don't have enough information to answer that.Also, when you state a fact, cite the section it came from in parentheses, e.g. "(Methods section)".
{context}
Question: {question}
Answer:`;
  return ChatPromptTemplate.fromTemplate(template);
}

export async function expandQuery(question: string): Promise<string[]> {
  const systemPrompt = `You rewrite questions to maximize retrieval coverage. Given a question, produce 3 query variants that are different but still relevant to the original question. These will be used to retrieve different chunks from a paper, so they should be diverse. Only output the queries, one per line, without any explanation.`;
  const chat = new ChatAnthropic({ model: "claude-haiku-4-5-20251001" });
  const response = await chat.invoke([
    { role: "system", content: systemPrompt },
    { role: "user", content: question },
  ]);
  return response.content
    .toString()
    .split("\n")
    .map((q) => q.trim())
    .filter((q) => q.length > 0);
}

export async function retrieveAndRerank(
  question: string,
  store: VectorStore,
  k: number = 4,
): Promise<{
  chunks: Array<{ content: string; section: string; score: number }>;
  queries: string[];
}> {
  const questions: string[] = await expandQuery(question);
  const candidates: Array<{ content: string; section: string }> = [];
  for (const q of questions) {
    const results = await searchStore(store, q, 20);
    for (const result of results) {
      candidates.push({ content: result.content, section: result.section });
    }
  }
  const seen = new Set<string>();
  const uniqueCandidates = candidates.filter((c) => {
    if (seen.has(c.content)) return false;
    seen.add(c.content);
    return true;
  });
  const voyage = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY! });
  const rerankResults = await voyage.rerank({
    query: question,
    documents: uniqueCandidates.map((c) => c.content),
    model: "rerank-2.5",
    topK: k,
  });
  const topChunks = rerankResults.data!.map((r) => {
    const candidate = uniqueCandidates[r.index!];
    return {
      content: candidate.content,
      section: candidate.section,
      score: r.relevanceScore!,
    };
  });
  return { chunks: topChunks, queries: questions };
}

export function buildRAGChain(store: VectorStore) {
  return RunnableSequence.from([
    {
      context: async (inputs: { question: string }) => {
        const { chunks } = await retrieveAndRerank(inputs.question, store);
        const formattedChunks = chunks
          .map((c) => `Section: ${c.section}\nContent: ${c.content}`)
          .join("\n\n");
        return formattedChunks;
      },
      question: (inputs: { question: string }) => inputs.question,
    },
    buildRAGPrompt(),
    new ChatAnthropic({ model: "claude-haiku-4-5-20251001", apiKey: process.env.ANTHROPIC_API_KEY! }),
    new StringOutputParser(),
  ]);
}

export async function askPaper(
  question: string,
  store: VectorStore,
): Promise<RAGResult> {
  const { chunks, queries } = await retrieveAndRerank(question, store);
  const ragChain = buildRAGChain(store);
  const answer = await ragChain.invoke({ question });
  return {
    question,
    answer,
    retrievedChunks: chunks,
    queries,
    model: "claude-haiku-4-5-20251001",
  };
}

export async function interrogatePaper(
  questions: string[],
  store: VectorStore,
): Promise<RAGResult[]> {
  const results: RAGResult[] = [];
  for (const question of questions) {
    const result = await askPaper(question, store);
    results.push(result);
  }
  return results;
}
