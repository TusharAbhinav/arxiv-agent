import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { BlogpostStateV2Type } from "./critique.js";

const SS_BASE = "https://api.semanticscholar.org/graph/v1";

export interface RelatedPaper {
  paperId: string;
  title: string;
  why: string;
  citationCount?: number;
}

export const searchSemanticScholarTool = tool(
  async ({ query, limit }: { query: string; limit: number }) => {
    throw new Error("Not implemented");
  },
  {
    name: "searchSemanticScholar",
    description:
      "Search Semantic Scholar for academic papers by query. " +
      "Returns title, abstract, year, citationCount, paperId.",
    schema: z.object({
      query: z.string().describe("Search query"),
      limit: z.number().default(10).describe("Max results to return")
    })
  }
);

export const getRelatedPapersTool = tool(
  async ({ paperId }: { paperId: string }) => {
    throw new Error("Not implemented");
  },
  {
    name: "getRelatedPapers",
    description:
      "Given a paperId, return papers that cite it (its descendants in the citation graph). " +
      "Useful for finding follow-up work.",
    schema: z.object({ paperId: z.string() })
  }
);

export const getPaperDetailsTool = tool(
  async ({ paperId }: { paperId: string }) => {
    throw new Error("Not implemented");
  },
  {
    name: "getPaperDetails",
    description:
      "Fetch full details (title, abstract, authors, year, citationCount) for a paperId.",
    schema: z.object({ paperId: z.string() })
  }
);

export function buildRelatedPapersAgent() {
  throw new Error("Not implemented");
}

export async function findRelatedPapersNode(
  state: BlogpostStateV2Type
): Promise<Partial<BlogpostStateV2Type>> {
  throw new Error("Not implemented");
}
