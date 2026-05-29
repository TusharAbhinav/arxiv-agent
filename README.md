# arxiv-agent

Turn any arXiv paper into a structured engineering blogpost — with TL;DR, glossary, runnable TypeScript, and cross-paper synthesis.

```bash
npm run pipeline -- 2005.11401
# → out/2005.11401.md
```

---

## What it produces

Given a paper ID, the pipeline writes a markdown file with:

- **TL;DR** — 3 plain-English bullets, max 20 words each
- **Sections** — problem, how it works, results, limits — written for engineers, not researchers
- **Glossary** — every technical term defined in plain English on first use
- **Runnable TypeScript** — for method papers, a working code example validated with `tsc`
- **Pull quotes** — the 2–3 most quotable lines from the paper

---

## Pipeline

<img width="876" height="502" alt="image" src="https://github.com/user-attachments/assets/aa229b9e-9a19-4049-ba85-b22047c46892" />


## Key concepts

### RAG — Retrieval-Augmented Generation

Feeding an entire paper to an LLM at once is too many tokens and too unfocused. Instead, every question the pipeline asks first retrieves the most relevant chunks from ChromaDB, then sends only those to Claude as context.

```
question → embed → vector search → top-K chunks → Claude → answer
```

This is used everywhere: writing the TL;DR, the glossary, each section. Every generation step is grounded in the actual paper text rather than Claude's training data.

---

### LangGraph — parallel nodes and conditional loops

The extraction step (TL;DR, glossary, metadata, sections) runs as a LangGraph graph where all five nodes fire in parallel and join before writing sections. This cuts wall time significantly compared to sequential calls.

```
START → [tldr, glossary, metadata, whyItMatters, pullQuotes]  ← all parallel
                          ↓ (join)
                     writeSections
                          ↓
                       critique
```

The critique→revise loop uses a **conditional edge** — a router function that decides at runtime whether to loop back or exit:

```ts
function critiqueRouter(state) {
  if (state.critiqueScore >= 4.0) return "end";
  if (state.revisionCount >= 2)   return "end";
  return "revise";
}
```

---

### Structured output — forcing typed JSON from the LLM

Raw LLM text is hard to work with. Every structured step uses Zod schemas with `.withStructuredOutput()` so Claude returns typed objects, not strings.

```ts
const result = await model
  .withStructuredOutput(z.object({
    bullets: z.array(z.string()).length(3)
  }))
  .invoke(prompt);

result.bullets  // string[3] — guaranteed
```

Used for: TL;DR bullets, glossary entries, critique scores, revised sections.

---

### Multi-turn code generation

Asking Claude "write code for X" in one shot produces mediocre results. The codegen stage uses a 3-turn conversation instead:

- **Turn 1** — explain the algorithm in plain English (no code yet)
- **Turn 2** — now write the TypeScript (warmed up from the explanation)
- **Turn 3** — add inline comments explaining each step

Each turn appends the previous response to the message history so Claude has full context. The final code is validated with `tsc --noEmit` and if it fails, the errors are sent back for one retry.

---

### Critique loop — self-evaluation with a rubric

The draft goes through a scoring pass before being written to disk. The critic checks specific yes/no questions rather than asking for a vague "quality score":

- Is TL;DR present with exactly 3 bullets?
- Is every technical term defined on first use?
- Does each section open with the takeaway, not background?
- Are there at least 2 software-engineering analogies?

This produces a score from 1–5 and a list of the top issues. If the score is below 4, the reviser fixes only the affected sections and the critic scores again. Max 2 revision cycles.

---

### Checkpointing — resume from crashes

The pipeline takes ~2 minutes and makes many API calls. If it crashes (network blip, rate limit, laptop sleeps), SqliteSaver has checkpointed every completed node to `checkpoints.db`. Re-running picks up from the last completed node automatically.

```ts
const graph = builder.compile({
  checkpointer: SqliteSaver.fromConnString("./checkpoints.db")
});

await graph.invoke(input, {
  configurable: { thread_id: `arxiv-${arxivId}` }
});
```

---

## Project structure

```
src/
  pipeline.ts       ← entry point
  stages/
    ingest.ts       ← parse PDF with GROBID
    embed.ts        ← chunk + embed into ChromaDB
    retrieve.ts     ← RAG: query expansion, retrieval, rerank
    classify.ts     ← classify paper type, pick template
    extract.ts      ← LangGraph parallel extraction
    codegen.ts      ← multi-turn TypeScript code generation
    critique.ts     ← critique + revise loop
    assemble.ts     ← streaming assembly + pipeline runner
    knowledge.ts    ← cross-paper knowledge base
  helpers.ts
```

---

## Setup

**Prerequisites:** Node.js 20+, Docker

```bash
npm install
docker-compose up -d    # starts ChromaDB and GROBID
cp .env.example .env    # then fill in your keys
```

**.env**
```
ANTHROPIC_API_KEY=your_key
VOYAGE_API_KEY=your_key
GROBID_URL=http://localhost:8070
CHROMA_URL=http://localhost:8000
```

---

## Usage

```bash
# Run the full pipeline
npm run pipeline -- 2005.11401

# View the output
glow out/2005.11401.md

# Type-check
npm run check
```

**Finding paper IDs** — the ID is in the URL: `arxiv.org/abs/2005.11401`

Some papers to try:

| Paper | ID |
|-------|----|
| RAG | `2005.11401` |
| Attention Is All You Need | `1706.03762` |
| BERT | `1810.04805` |
| LoRA | `2106.09685` |
| GPT-3 | `2005.14165` |

---

## Tech stack

| Concern | Tool |
|---------|------|
| LLM | Claude Haiku + Sonnet (Anthropic) |
| Embeddings | Voyage-3 |
| Vector DB | ChromaDB |
| PDF parsing | GROBID |
| Agent orchestration | LangGraph |
| Persistence | SqliteSaver (SQLite) |
| Code validation | `tsc --noEmit` |
| Runtime | Node.js + TypeScript |

---

## License

MIT
