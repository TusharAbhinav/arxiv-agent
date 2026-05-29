import { Chroma } from "@langchain/community/vectorstores/chroma";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
import { ChatAnthropic } from "@langchain/anthropic";
import { Document } from "@langchain/core/documents";
import { z } from "zod";
import type { ParsedPaper } from "./ingest.js";
import type { PaperChunk } from "./embed.js";

const KB_COLLECTION = "arxiv-knowledge-base";

export interface KnowledgeBase {
  store: Chroma;
  knownArxivIds: Set<string>;
}

export const SynthesisSchema = z.object({
  answer: z.string().describe("Direct answer to the question, citing each source by [arxivId]"),
  sources: z.array(z.object({
    arxivId: z.string(),
    title: z.string(),
    excerpt: z.string().describe("Verbatim quote from the paper supporting the answer"),
  })),
  synthesis: z.string().describe(
    "How do these papers relate? Do they agree? Disagree? What's the timeline of ideas?",
  ),
});

export type CrossPaperResult = z.infer<typeof SynthesisSchema>;

function getEmbeddings() {
  return new VoyageEmbeddings({
    apiKey: process.env.VOYAGE_API_KEY,
    modelName: "voyage-3",
  });
}

function getSonnet() {
  return new ChatAnthropic({
    model: "claude-sonnet-4-6",
    maxTokens: 4096,
    invocationKwargs: { top_p: undefined },
  });
}

export async function createKnowledgeBase(): Promise<KnowledgeBase> {
  const url = process.env.CHROMA_URL ?? "http://localhost:8000";
  const store = new Chroma(getEmbeddings(), {
    collectionName: KB_COLLECTION,
    url,
  });

  const knownArxivIds = new Set<string>();
  try {
    const existing = await store.similaritySearch("research retrieval generation", 1000);
    for (const doc of existing) {
      if (doc.metadata?.arxivId) {
        knownArxivIds.add(doc.metadata.arxivId as string);
      }
    }
  } catch {
  }

  return { store, knownArxivIds };
}

export async function addPaperToKnowledgeBase(
  kb: KnowledgeBase,
  paper: ParsedPaper,
  chunks: PaperChunk[],
): Promise<void> {
  if (kb.knownArxivIds.has(paper.arxivId)) return;

  const docs = chunks.map(
    (chunk) =>
      new Document({
        pageContent: chunk.content,
        metadata: {
          arxivId: paper.arxivId,
          title: paper.title,
          section: chunk.metadata.section,
        },
      }),
  );

  await kb.store.addDocuments(docs);
  kb.knownArxivIds.add(paper.arxivId);
}

export async function routeQueryToPapers(
  query: string,
  kb: KnowledgeBase,
): Promise<string[]> {
  const results = await kb.store.similaritySearch(query, 15);

  const counts = new Map<string, number>();
  for (const doc of results) {
    const id = doc.metadata?.arxivId as string | undefined;
    if (id) counts.set(id, (counts.get(id) ?? 0) + 1);
  }

  return [...counts.entries()]
    .filter(([, count]) => count >= 2)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([id]) => id);
}

export async function synthesizeAcrossPapers(
  query: string,
  arxivIds: string[],
  kb: KnowledgeBase,
): Promise<CrossPaperResult> {
  const paperBundles: string[] = [];

  for (let i = 0; i < arxivIds.length; i++) {
    const arxivId = arxivIds[i];
    const chunks = await kb.store.similaritySearch(query, 3, { arxivId });
    if (chunks.length === 0) continue;

    const title = (chunks[0].metadata?.title as string) ?? arxivId;
    const excerpts = chunks
      .map(
        (c, j) =>
          `  Excerpt ${j + 1} (section: ${c.metadata?.section ?? "unknown"}): "${c.pageContent.slice(0, 300)}"`,
      )
      .join("\n");

    paperBundles.push(`PAPER ${i + 1}: ${title} (arxiv:${arxivId})\n${excerpts}`);
  }

  if (paperBundles.length === 0) {
    return {
      answer: "No relevant papers found.",
      sources: [],
      synthesis: "No papers were relevant to this query.",
    };
  }

  const context = paperBundles.join("\n\n");
  const model = getSonnet().withStructuredOutput(SynthesisSchema);

  return model.invoke([
    {
      role: "user",
      content: `Here are excerpts from relevant research papers:\n\n${context}\n\nQuestion: ${query}\n\nProvide a direct answer citing each source by [arxivId], list the supporting excerpts, and synthesize how these papers relate — do they agree, disagree, or build on each other?`,
    },
  ]);
}

export async function askKnowledgeBase(query: string): Promise<CrossPaperResult> {
  const kb = await createKnowledgeBase();

  process.stdout.write(`\nRouting query across ${kb.knownArxivIds.size} papers...\n`);
  const arxivIds = await routeQueryToPapers(query, kb);

  if (arxivIds.length === 0) {
    const empty: CrossPaperResult = {
      answer: "No relevant papers found in the knowledge base.",
      sources: [],
      synthesis: "Add papers using addPaperToKnowledgeBase first.",
    };
    process.stdout.write(`${empty.answer}\n`);
    return empty;
  }

  process.stdout.write(`Relevant papers: ${arxivIds.join(", ")}\n\n`);
  const result = await synthesizeAcrossPapers(query, arxivIds, kb);

  process.stdout.write(`Answer:\n${result.answer}\n\n`);
  process.stdout.write(`Synthesis:\n${result.synthesis}\n`);

  return result;
}
