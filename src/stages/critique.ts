
import { StateGraph, Annotation, END, START } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { z } from "zod";
import {
  BlogpostState,
  AUDIENCE_PIN,
  extractTldrNode,
  extractWhyItMattersNode,
  extractGlossaryNode,
  extractMetadataNode,
  extractPullQuotesNode,
  writeSectionsNode,
} from "./extract.js";

const sonnet = () =>
  new ChatAnthropic({
    model: "claude-sonnet-4-6",
    maxTokens: 8192,
    invocationKwargs: { top_p: undefined },
  });

export const CritiqueSchema = z.object({
  tldrPresent: z.boolean(),
  jargonDefinedFirstUse: z.boolean(),
  analogyCount: z.number(),
  codeRuns: z.boolean(),
  sectionsOpenWithTakeaway: z.boolean(),
  overallScore: z.number().min(1).max(5),
  topIssues: z.array(z.string()).max(3),
  suggestions: z.array(z.string()).max(3),
});

export type Critique = z.infer<typeof CritiqueSchema>;

export const BlogpostStateV2 = Annotation.Root({
  ...BlogpostState.spec,
  critiqueScore: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  topIssues: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  revisionCount: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
});

export type BlogpostStateV2Type = typeof BlogpostStateV2.State;

const DRAFT_SKIP = new Set(["TL;DR", "Glossary", "Further Reading", "In Code"]);

export function assembleDraft(state: BlogpostStateV2Type): string {
  return [
    `# ${state.paper.title}`,
    `**TL;DR**`,
    ...state.tldr.map((b) => `- ${b}`),
    ...state.template.sections
      .filter((name) => !DRAFT_SKIP.has(name))
      .map((name) => `## ${name}\n${state.sections[name] ?? ""}`),
    `**Glossary**`,
    ...state.glossary.map((g) => `- **${g.term}**: ${g.plainEnglish}`),
  ].join("\n\n");
}

export async function critiqueNode(
  state: BlogpostStateV2Type,
): Promise<Partial<BlogpostStateV2Type>> {
  const draft = assembleDraft(state);

  const model = sonnet().withStructuredOutput(CritiqueSchema);

  const result = await model.invoke([
    {
      role: "system",
      content:
        "You are a senior technical editor. Be strict — reject drafts you wouldn't share on your team's Slack.",
    },
    {
      role: "user",
      content: `Review this blogpost draft against the following rubric. Be strict.\n\nRubric:\n- TL;DR present (3 bullets, max 20 words each)\n- Every technical term is defined inline or in the glossary the first time it appears\n- At least 2 software-engineering analogies used\n- Every section opens with the takeaway, not background\n- If code is present, it compiles/runs\n\nDraft:\n${draft}`,
    },
  ]);

  return {
    critiqueScore: result.overallScore,
    topIssues: result.topIssues,
  };
}

const ReviseSchema = z.object({
  revisedSections: z.array(
    z.object({
      name: z.string().describe("Exact section name as it appears in the draft"),
      content: z.string().describe("Full revised content for this section"),
    }),
  ).describe("Only the sections that need changing"),
});

export async function reviseNode(
  state: BlogpostStateV2Type,
): Promise<Partial<BlogpostStateV2Type>> {
  const draft = assembleDraft(state);
  const issueList = state.topIssues.map((issue) => `- ${issue}`).join("\n");

  const model = sonnet().withStructuredOutput(ReviseSchema);

  const result = await model.invoke([
    { role: "system", content: AUDIENCE_PIN },
    {
      role: "user",
      content: `Here is the current blogpost draft:\n\n${draft}\n\nThe critic flagged these top issues:\n${issueList}\n\nRevise only the sections affected by these issues. Return each changed section with its exact name and full new content.`,
    },
  ]);

  const updatedSections: Record<string, string> = {};
  for (const s of result.revisedSections) {
    updatedSections[s.name] = s.content;
  }

  return {
    sections: updatedSections,
    revisionCount: state.revisionCount + 1,
  };
}

export function critiqueRouter(state: BlogpostStateV2Type): "revise" | "end" {
  if (state.critiqueScore >= 4.0 || state.revisionCount >= 2) return "end";
  return "revise";
}

export function buildCritiqueGraph() {
  const graph = new StateGraph(BlogpostStateV2)
    .addNode("extractTldr", extractTldrNode)
    .addNode("extractWhyItMatters", extractWhyItMattersNode)
    .addNode("extractGlossary", extractGlossaryNode)
    .addNode("extractMetadata", extractMetadataNode)
    .addNode("extractPullQuotes", extractPullQuotesNode)
    .addNode("writeSections", writeSectionsNode)
    .addNode("critique", critiqueNode)
    .addNode("revise", reviseNode);

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
  graph.addEdge("writeSections", "critique");
  graph.addConditionalEdges("critique", critiqueRouter, {
    revise: "revise",
    end: END,
  });
  graph.addEdge("revise", "critique");

  return graph.compile();
}
