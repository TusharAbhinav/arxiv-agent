import "dotenv/config";
import chalk from "chalk";

const ARXIV_ID = process.argv[2] ?? "2005.11401";

async function main() {
  console.log(chalk.bold.blue("\narxiv-agent pipeline"));
  console.log(chalk.gray(`Paper: ${ARXIV_ID}\n`));

  const { ingestPaper } = await import("./stages/ingest.js");

  console.log(chalk.yellow("Stage 1: Parsing PDF..."));
  const paper = await ingestPaper(ARXIV_ID);
  console.log(chalk.green(`✓ Title: ${paper.title}`));
  console.log(chalk.green(`  Sections: ${paper.sections.map((s) => s.title).join(", ")}`));
  console.log(chalk.green(`  References: ${paper.references.length}\n`));

  const { getOrCreateStore } = await import("./stages/embed.js");

  console.log(chalk.yellow("Stage 2: Building vector store..."));
  const store = await getOrCreateStore(paper);
  console.log(chalk.green(`✓ ${store.chunkCount} chunks indexed\n`));

  const { classifyPaper, pickTemplate } = await import("./stages/classify.js");

  console.log(chalk.yellow("Stage 4: Classifying paper..."));
  const classification = await classifyPaper(paper);
  const template = pickTemplate(
    classification.confidence < 0.6 ? "method" : classification.type,
  );
  console.log(
    chalk.green(
      `✓ Type: ${classification.type} (confidence ${classification.confidence})`,
    ),
  );
  console.log(chalk.gray(`  Sections: ${template.sections.join(", ")}\n`));

  const { buildCheckpointedGraph } = await import("./stages/assemble.js");

  console.log(chalk.yellow("Stages 5–9: Extracting, generating code, critiquing, assembling...\n"));
  const graph = buildCheckpointedGraph();
  await graph.invoke(
    { paper, store, classification, template },
    { configurable: { thread_id: `arxiv-${ARXIV_ID}` } },
  );
  console.log(chalk.green(`\n✓ Markdown written to out/${ARXIV_ID}.md\n`));

  console.log(chalk.bold.green("✓ Pipeline complete\n"));
  console.log(chalk.gray(`Output: out/${ARXIV_ID}.md`));
}

main().catch(console.error);
