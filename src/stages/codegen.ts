import Anthropic from "@anthropic-ai/sdk";
import type { MessageParam } from "@anthropic-ai/sdk/resources/messages.js";
import type { BlogpostStateType } from "./extract.js";
import { execFile } from "child_process";
import { mkdtemp, rm, writeFile } from "fs/promises";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

const SYSTEM_PROMPT =
  "You are an expert software engineer. Write clear, efficient, and well-documented TypeScript code.";

export interface GeneratedCode {
  description: string;
  typescript: string;
  explanation: string;
  isValid: boolean;
  tscErrors: string;
}

export async function validateTypeScript(
  code: string,
): Promise<{ isValid: boolean; tscErrors: string }> {
  const dir = await mkdtemp("/tmp/tsval-");
  try {
    await writeFile(`${dir}/snippet.ts`, code);
    try {
      await execFileAsync("npx", [
        "tsc",
        "--noEmit",
        "--target",
        "es2022",
        "--module",
        "esnext",
        "--moduleResolution",
        "bundler",
        `${dir}/snippet.ts`,
      ]);
      return { isValid: true, tscErrors: "" };
    } catch (error) {
      const err = error as any;
      return { isValid: false, tscErrors: err.stdout || err.message };
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

export function extractCodeBlock(response: string): string {
  const codeBlockRegex = /```typescript\s*([\s\S]*?)\s*```/i;
  const match = response.match(codeBlockRegex);
  return match ? match[1] : response;
}

export async function generateImplementation(
  algorithmDescription: string,
  paperContext: string,
): Promise<Omit<GeneratedCode, "isValid" | "tscErrors">> {
  const client = new Anthropic();
  const messages: MessageParam[] = [];

  messages.push({
    role: "user",
    content: `Here is a description of an algorithm from a research paper:\n\n${algorithmDescription}\n\nAnd here is additional context from the paper:\n\n${paperContext}\n\nFirst, explain in plain English how you would implement this algorithm. No code yet, just a high-level explanation.`,
  });

  const turn1 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    messages,
  });
  const description = (turn1.content[0] as { type: "text"; text: string }).text;

  messages.push({ role: "assistant", content: description });
  messages.push({
    role: "user",
    content: `Great explanation! Now, please write a TypeScript implementation of this algorithm. Focus on correctness and clarity. Don't worry about comments yet, just get the code down. Use inline mock types instead of importing external modules.`,
  });

  const turn2 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    system: SYSTEM_PROMPT,
    messages,
  });
  const rawCode = (turn2.content[0] as { type: "text"; text: string }).text;
  const typescript = extractCodeBlock(rawCode);

  messages.push({ role: "assistant", content: rawCode });
  messages.push({
    role: "user",
    content: `Thanks for the implementation! Now, please add inline comments to the code explaining what each part does and why. This will help others understand your thought process.`,
  });

  const turn3 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    system: SYSTEM_PROMPT,
    messages,
  });
  const explanation = (turn3.content[0] as { type: "text"; text: string }).text;

  return { description, typescript, explanation };
}

export async function validateWithRetry(
  initial: Omit<GeneratedCode, "isValid" | "tscErrors">,
  paperContext: string,
): Promise<GeneratedCode> {
  let code = initial.typescript;
  let validation = await validateTypeScript(code);

  if (validation.isValid) {
    return { ...initial, ...validation };
  }

  const client = new Anthropic();
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    system: SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: `The following TypeScript code has some type errors:\n\n\`\`\`typescript\n${code}\n\`\`\`\n\nHere are the tsc errors:\n\n${validation.tscErrors}\n\nPlease fix the code to resolve these errors. Return only the corrected code in a \`\`\`typescript block. Here's additional context from the paper that might help:\n\n${paperContext}`,
      },
    ],
  });

  const newCode = extractCodeBlock(
    (response.content[0] as { type: "text"; text: string }).text,
  );
  const newValidation = await validateTypeScript(newCode);

  return {
    description: initial.description,
    typescript: newCode,
    explanation: initial.explanation,
    ...newValidation,
  };
}

export async function generateCodeNode(
  state: BlogpostStateType,
): Promise<Partial<BlogpostStateType>> {
  if (!state.template.needsCode) {
    return {};
  }

  const algorithmDescription =
    state.sections["How It Works"] ?? state.paper.abstract;
  const paperContext = state.paper.abstract;

  const initial = await generateImplementation(algorithmDescription, paperContext);
  const result = await validateWithRetry(initial, paperContext);

  return {
    sections: {
      "In Code": result.explanation,
    },
  };
}
