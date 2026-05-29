import { StateGraph, END, START } from "@langchain/langgraph";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import {
  BlogpostStateV2,
  type BlogpostStateV2Type,
  critiqueNode,
  reviseNode,
  critiqueRouter,
} from "./critique.js";
import {
  extractTldrNode,
  extractWhyItMattersNode,
  extractGlossaryNode,
  extractMetadataNode,
  extractPullQuotesNode,
  writeSectionsNode,
} from "./extract.js";
import { generateCodeNode } from "./codegen.js";
import { ingestPaper } from "./ingest.js";
import { getOrCreateStore } from "./embed.js";
import { classifyPaper, pickTemplate } from "./classify.js";

const MARKDOWN_SKIP = new Set(["TL;DR", "Glossary", "Further Reading"]);

export function assembleMarkdown(state: BlogpostStateV2Type): string {
  const parts: string[] = [
    `# ${state.paper.title}`,
    `*${state.paper.authors.join(", ")} · arxiv:${state.paper.arxivId}*`,
    `**TL;DR**\n${state.tldr.map((b) => `- ${b}`).join("\n")}`,
    `**Difficulty:** ${state.difficulty}/3 · **Read time:** ${state.readTimeMinutes} min  \n**You need:** ${state.prerequisites.join(", ") || "None"}`,
    `---`,
    ...state.template.sections
      .filter((name) => !MARKDOWN_SKIP.has(name))
      .map((name) => `## ${name}\n\n${state.sections[name] ?? ""}`)
      .filter((s) => !s.endsWith("\n\n")),
    `## Glossary\n${state.glossary.map((g) => `- **${g.term}**: ${g.plainEnglish}`).join("\n")}`,
  ];

  return parts.join("\n\n");
}

export async function streamAssembleNode(
  state: BlogpostStateV2Type,
): Promise<Partial<BlogpostStateV2Type>> {
  await mkdir("out", { recursive: true });

  process.stdout.write(`\n# ${state.paper.title}\n`);
  process.stdout.write(
    `*${state.paper.authors.join(", ")} · arxiv:${state.paper.arxivId}*\n\n`,
  );
  process.stdout.write(
    `**TL;DR**\n${state.tldr.map((b) => `- ${b}`).join("\n")}\n\n`,
  );
  process.stdout.write(
    `**Difficulty:** ${state.difficulty}/3 · **Read time:** ${state.readTimeMinutes} min\n\n`,
  );
  process.stdout.write(`---\n\n`);

  for (const name of state.template.sections) {
    if (MARKDOWN_SKIP.has(name)) continue;
    const content = state.sections[name];
    if (!content) continue;
    process.stdout.write(`## ${name}\n\n`);
    process.stdout.write(content + "\n\n");
  }

  process.stdout.write(
    `## Glossary\n${state.glossary.map((g) => `- **${g.term}**: ${g.plainEnglish}`).join("\n")}\n`,
  );

  const markdown = assembleMarkdown(state);
  const outputPath = join("out", `${state.paper.arxivId}.md`);
  await writeFile(outputPath, markdown, "utf-8");

  const wordCount = markdown.split(/\s+/).length;
  process.stdout.write(`\n✓ Wrote ${outputPath} (${wordCount} words)\n`);

  return {};
}

export function buildCheckpointedGraph() {
  const checkpointer = SqliteSaver.fromConnString("./checkpoints.db");

  const graph = new StateGraph(BlogpostStateV2)
    .addNode("extractTldr", extractTldrNode)
    .addNode("extractWhyItMatters", extractWhyItMattersNode)
    .addNode("extractGlossary", extractGlossaryNode)
    .addNode("extractMetadata", extractMetadataNode)
    .addNode("extractPullQuotes", extractPullQuotesNode)
    .addNode("writeSections", writeSectionsNode)
    .addNode("generateCode", generateCodeNode)
    .addNode("critique", critiqueNode)
    .addNode("revise", reviseNode)
    .addNode("streamAssemble", streamAssembleNode);

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
  graph.addEdge("writeSections", "generateCode");
  graph.addEdge("generateCode", "critique");
  graph.addConditionalEdges("critique", critiqueRouter, {
    revise: "revise",
    end: "streamAssemble",
  });
  graph.addEdge("revise", "critique");
  graph.addEdge("streamAssemble", END);

  return graph.compile({ checkpointer });
}

export async function runPipeline(arxivId: string): Promise<string> {
  const paper = await ingestPaper(arxivId);
  const store = await getOrCreateStore(paper);
  const classification = await classifyPaper(paper);
  const template = pickTemplate(classification.type);

  const graph = buildCheckpointedGraph();
  await graph.invoke(
    { paper, store, classification, template },
    { configurable: { thread_id: `arxiv-${arxivId}` } },
  );

  return join(process.cwd(), "out", `${arxivId}.md`);
}
