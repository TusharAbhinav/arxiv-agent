import { ChatAnthropic } from "@langchain/anthropic";
import { z } from "zod";
import type { ParsedPaper } from "./ingest.js";

export type PaperType =
  | "method"
  | "empirical"
  | "theory"
  | "survey"
  | "dataset"
  | "position";

export const ClassificationSchema = z.object({
  type: z.enum(["method", "empirical", "theory", "survey", "dataset", "position"]),
  reason: z.string().describe("One sentence on why this type fits"),
  confidence: z.number().min(0).max(1)
});

export type Classification = z.infer<typeof ClassificationSchema>;

export interface Template {
  sections: string[];
  needsCode: boolean;
  needsDiagram: boolean;
}

export const TEMPLATES: Record<PaperType, Template> = {
  method: {
    sections: ["TL;DR", "Why It Matters", "Problem", "How It Works", "In Code", "Results", "Limits", "Glossary", "Further Reading"],
    needsCode: true,
    needsDiagram: true
  },
  empirical: {
    sections: ["TL;DR", "Why It Matters", "Experimental Setup", "Key Findings", "Limits", "Glossary", "Further Reading"],
    needsCode: false,
    needsDiagram: true
  },
  theory: {
    sections: ["TL;DR", "Why It Matters", "Background", "Core Result", "Proof Sketch", "Implications", "Glossary", "Further Reading"],
    needsCode: false,
    needsDiagram: false
  },
  survey: {
    sections: ["TL;DR", "Why It Matters", "Landscape", "Key Themes", "Gaps & Open Problems", "Glossary", "Further Reading"],
    needsCode: false,
    needsDiagram: true
  },
  dataset: {
    sections: ["TL;DR", "Why It Matters", "What's in the Dataset", "Collection Method", "Baselines", "Limits", "Glossary", "Further Reading"],
    needsCode: false,
    needsDiagram: true
  },
  position: {
    sections: ["TL;DR", "Why It Matters", "The Argument", "Evidence", "Counterarguments", "Glossary", "Further Reading"],
    needsCode: false,
    needsDiagram: false
  }
};

export async function classifyPaper(paper: ParsedPaper): Promise<Classification> {
  const { title, abstract, sections } = paper;
  const intro = sections.length > 0 ? sections[0].content : "";
  const prompt = `Classify the paper with title "${title}" into one of the following types: method, empirical, theory, survey, dataset, position. The abstract is: ${abstract}. The introduction is: ${intro}. Briefly justify your choice.`;
  const model = new ChatAnthropic({ model: "claude-haiku-4-5-20251001" }).withStructuredOutput(ClassificationSchema);
  const response = await model.invoke(prompt);
  return response;
}

export function pickTemplate(type: PaperType): Template {
  return TEMPLATES[type];
}
