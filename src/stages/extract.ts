import { StateGraph, Annotation, END, START } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { z } from "zod";
import type { ParsedPaper } from "./ingest.js";
import type { VectorStore } from "./embed.js";
import type { Classification, Template } from "./classify.js";
import { askPaper } from "./retrieve.js";

export const AUDIENCE_PIN = `You are writing for a junior software engineer who
is comfortable with basic Python or JavaScript but has never worked with AI or ML.
Avoid all jargon — if you must use a technical term, immediately explain it in one
plain sentence using an everyday analogy. Write short paragraphs. Lead every section
with the single most important takeaway in plain English, then explain the details simply.`;

export interface GlossaryEntry {
  term: string;
  plainEnglish: string;
}
export interface PullQuote {
  quote: string;
  from: string;
}

export const BlogpostState = Annotation.Root({
  paper: Annotation<ParsedPaper>(),
  store: Annotation<VectorStore>(),
  classification: Annotation<Classification>(),
  template: Annotation<Template>(),

  tldr: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  difficulty: Annotation<1 | 2 | 3>({ reducer: (_, b) => b, default: () => 2 }),
  readTimeMinutes: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),
  prerequisites: Annotation<string[]>({
    reducer: (_, b) => b,
    default: () => [],
  }),
  whyItMatters: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  oneLineSummary: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),

  glossary: Annotation<GlossaryEntry[]>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),
  pullQuotes: Annotation<PullQuote[]>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),
  sections: Annotation<Record<string, string>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
});

export type BlogpostStateType = typeof BlogpostState.State;

export async function extractTldrNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  const question =
    "What are the 3 most important takeaways from this paper for an engineer?";
  const result = await askPaper(question, state.store);
  const structuredChat = new ChatAnthropic({
    model: "claude-haiku-4-5-20251001",
  }).withStructuredOutput(z.object({ bullets: z.array(z.string()).length(3) }));
  const response = await structuredChat.invoke([
    { role: "system", content: AUDIENCE_PIN },
    {
      role: "user",
      content: `${question}\n\nContext:\n${result.answer}\n\nWrite exactly 3 TL;DR bullets, each max 20 words.`,
    },
  ]);
  return { tldr: response.bullets };
}

export async function extractWhyItMattersNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  const question =
    "Why does this paper matter? What problem does it solve and what are the practical implications?";
  const result = await askPaper(question, state.store);
  const structuredChat = new ChatAnthropic({
    model: "claude-haiku-4-5-20251001",
  }).withStructuredOutput(z.object({ whyItMatters: z.string() }));
  const response = await structuredChat.invoke([
    { role: "system", content: AUDIENCE_PIN },
    {
      role: "user",
      content: `${question}\n\nContext:\n${result.answer}\n\nWrite exactly 2 sentences. Lead with the engineering implication, then explain the broader significance.`,
    },
  ]);
  return { whyItMatters: response.whyItMatters };
}

export async function extractGlossaryNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  const question =
    "What are the technical terms in this paper that an engineer might not know? For each term, write a plain-English definition.";
  const result = await askPaper(question, state.store);
  const structuredChat = new ChatAnthropic({
    model: "claude-haiku-4-5-20251001",
  }).withStructuredOutput(
    z.object({
      terms: z.array(z.object({ term: z.string(), plainEnglish: z.string() })),
    }),
  );
  const response = await structuredChat.invoke([
    { role: "system", content: AUDIENCE_PIN },
    {
      role: "user",
      content: `${question}\n\nContext:\n${result.answer}\n\nWrite a list of technical terms and their plain-English definitions. For example:\n\n- Term: "Transformer"\n  Plain English: "A type of neural network architecture that uses attention mechanisms to process sequential data."`,
    },
  ]);
  return { glossary: response.terms };
}

export async function extractMetadataNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  const question =
    "Based on the abstract, classify the difficulty of this paper for an engineer (1 = easy, 2 = medium, 3 = hard). Estimate how many minutes it would take to read and understand. List any prerequisites an engineer should know before reading.";
  const structuredChat = new ChatAnthropic({
    model: "claude-haiku-4-5-20251001",
  }).withStructuredOutput(
    z.object({
      difficulty: z.union([z.literal(1), z.literal(2), z.literal(3)]),
      readTimeMinutes: z.number(),
      prerequisites: z.array(z.string()),
    }),
  );
  const response = await structuredChat.invoke([
    { role: "system", content: AUDIENCE_PIN },
    {
      role: "user",
      content: `${question}\n\nAbstract:\n${state.paper.abstract}`,
    },
  ]);
  return {
    difficulty: response.difficulty,
    readTimeMinutes: response.readTimeMinutes,
    prerequisites: response.prerequisites,
  };
}

export async function extractPullQuotesNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  const question =
    "Find 2-3 quotable lines from this paper that an engineer might highlight. For each quote, identify which section it's from.";
  const result = await askPaper(question, state.store);
  const structuredChat = new ChatAnthropic({
    model: "claude-haiku-4-5-20251001",
  }).withStructuredOutput(
    z.object({
      quotes: z.array(
        z.object({
          quote: z.string(),
          from: z.string(),
        }),
      ),
    }),
  );
  const response = await structuredChat.invoke([
    { role: "system", content: AUDIENCE_PIN },
    {
      role: "user",
      content: `${question}\n\nContext:\n${result.answer}\n\nWrite a list of quotable lines and their sections. For example:\n\n- Quote: "Our model achieves state-of-the-art performance on XYZ benchmark."\n  From: "Results" section`,
    },
  ]);
  return { pullQuotes: response.quotes };
}

export async function writeSectionsNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  const sectionsToWrite = state.template.sections.filter(
    (s) => !["TL;DR", "Glossary", "Further Reading", "In Code"].includes(s),
  );
  const sectionQuestions: Record<string, string> = {
    "Problem": "What problem does this paper solve? What gap or limitation motivated this work?",
    "How It Works": "How does the proposed method or system work? What is the core mechanism?",
    "Results": "What results did the paper achieve? How does it compare to baselines?",
    "Limits": "What are the limitations and failure cases of this work?",
    "Why It Matters": "Why does this paper matter? What is the practical impact?",
    "Experimental Setup": "What was the experimental setup? What datasets and metrics were used?",
    "Key Findings": "What are the key findings and takeaways from the experiments?",
    "Background": "What background knowledge is needed to understand this paper?",
    "Core Result": "What is the main theoretical result or contribution?",
    "Proof Sketch": "How is the main result proven or derived?",
    "Implications": "What are the implications of this theoretical result?",
    "Landscape": "What is the landscape of existing work in this area?",
    "Key Themes": "What are the key themes and trends across the surveyed work?",
    "Gaps & Open Problems": "What gaps and open problems does the survey identify?",
    "What's in the Dataset": "What does the dataset contain? What is its size and structure?",
    "Collection Method": "How was the dataset collected and annotated?",
    "Baselines": "What baseline results are reported on this dataset?",
    "The Argument": "What is the central argument or position of this paper?",
    "Evidence": "What evidence supports the paper's argument?",
    "Counterarguments": "What counterarguments does the paper address?",
  };
  const sectionsContent: Record<string, string> = {};
  for (const sectionName of sectionsToWrite) {
    const question = sectionQuestions[sectionName] ?? `What does this paper say about ${sectionName}?`;
    const result = await askPaper(question, state.store);
    const response = await new ChatAnthropic({
      model: "claude-haiku-4-5-20251001",
    }).invoke([
      { role: "system", content: AUDIENCE_PIN },
      {
        role: "user",
        content: `${question}\n\nContext:\n${result.answer}\n\nWrite a 200-400 word section for a blog post. Lead with the main takeaway, then explain the details. Use software engineering analogies to explain technical concepts.`,
      },
    ]);
    sectionsContent[sectionName] = response.content.toString();
  }
  return { sections: sectionsContent };
}

export function buildExtractionGraph() {
  const graph = new StateGraph(BlogpostState)
    .addNode("extractTldr", extractTldrNode)
    .addNode("extractWhyItMatters", extractWhyItMattersNode)
    .addNode("extractGlossary", extractGlossaryNode)
    .addNode("extractMetadata", extractMetadataNode)
    .addNode("extractPullQuotes", extractPullQuotesNode)
    .addNode("writeSections", writeSectionsNode);

  graph.addEdge(START, "extractTldr");
  graph.addEdge(START, "extractWhyItMatters");
  graph.addEdge(START, "extractGlossary");
  graph.addEdge(START, "extractMetadata");
  graph.addEdge(START, "extractPullQuotes");
  graph.addEdge("extractTldr", "writeSections");
  graph.addEdge("extractWhyItMatters", "writeSections");
  graph.addEdge("extractGlossary", "writeSections");
  graph.addEdge("extractMetadata", "writeSections");
  graph.addEdge("extractPullQuotes", "writeSections");
  graph.addEdge("writeSections", END);
  return graph.compile();
}

export async function runExtractionGraph(
  paper: ParsedPaper,
  classification: Classification,
  template: Template,
  store: VectorStore,
): Promise<BlogpostStateType> {
  const graph = buildExtractionGraph();
  return graph.invoke({ paper, classification, template, store });
}
