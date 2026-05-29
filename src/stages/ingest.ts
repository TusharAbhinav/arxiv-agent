import { XMLParser } from "fast-xml-parser";

const GROBID_URL = process.env.GROBID_URL ?? "http://localhost:8070";

export interface ParsedPaper {
  arxivId: string;
  title: string;
  authors: string[];
  year: number | null;
  abstract: string;
  sections: Array<{ title: string; content: string }>;
  references: string[];
}

function toArray(raw: any): any[] {
  if (Array.isArray(raw)) return raw;
  if (raw) return [raw];
  return [];
}

function getText(node: any): string {
  if (node && typeof node["#text"] === "string") return node["#text"];
  return "";
}

export async function fetchPdf(arxivId: string): Promise<Buffer> {
  const url = `https://arxiv.org/pdf/${arxivId}.pdf`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch PDF: ${res.status} ${res.statusText}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

export async function parseWithGrobid(
  pdfBuffer: Buffer,
  arxivId: string
): Promise<ParsedPaper> {
  const formData = new FormData();
  const pdfBytes = new Uint8Array(pdfBuffer);
  formData.append("input", new Blob([pdfBytes]), `${arxivId}.pdf`);

  const res = await fetch(`${GROBID_URL}/api/processFulltextDocument`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    throw new Error(`Failed to parse with GROBID: ${res.status} ${res.statusText}`);
  }

  const xml = await res.text();
  const parser = new XMLParser({
    ignoreAttributes: false,
    alwaysCreateTextNode: true
  });
  const parsed = parser.parse(xml);
  const tei = parsed.TEI;

  const title = getText(tei.teiHeader.fileDesc.titleStmt.title);

  const authorList = toArray(tei.teiHeader.fileDesc.sourceDesc.biblStruct.analytic.author);

  const authors: string[] = [];
  for (const a of authorList) {
    const forenameList = toArray(a?.persName?.forename);
    const forenameParts: string[] = [];
    for (const f of forenameList) {
      const text = getText(f);
      if (text) forenameParts.push(text);
    }
    const forename = forenameParts.join(" ");
    const surname = getText(a?.persName?.surname);

    const fullName = `${forename} ${surname}`.trim();
    if (fullName) authors.push(fullName);
  }

  const whenAttr =
    tei.teiHeader.fileDesc.publicationStmt?.date?.["@_when"] ??
    tei.teiHeader.fileDesc.sourceDesc?.biblStruct?.monogr?.imprint?.date?.["@_when"];

  let year: number | null = null;
  if (typeof whenAttr === "string") {
    year = parseInt(whenAttr.slice(0, 4), 10);
  }

  const abstractNode = tei.teiHeader.profileDesc?.abstract;
  const abstractParagraphs = [
    ...toArray(abstractNode?.p),
    ...toArray(abstractNode?.div).flatMap((d: any) => toArray(d?.p)),
  ];
  const abstractParts: string[] = [];
  for (const p of abstractParagraphs) {
    const text = getText(p);
    if (text) abstractParts.push(text);
  }
  const abstract = abstractParts.join("\n");

  const sections: Array<{ title: string; content: string }> = [];
  for (const div of toArray(tei.text?.body?.div)) {
    const sectionTitle = getText(div.head);

    const paragraphParts: string[] = [];
    for (const p of toArray(div.p)) {
      const text = getText(p);
      if (text) paragraphParts.push(text);
    }
    const sectionContent = paragraphParts.join("\n");

    if (sectionTitle || sectionContent) {
      sections.push({ title: sectionTitle, content: sectionContent });
    }
  }

  let refsDiv: any = null;
  for (const d of toArray(tei.text?.back?.div)) {
    if (d?.["@_type"] === "references") {
      refsDiv = d;
      break;
    }
  }

  const references: string[] = [];
  for (const ref of toArray(refsDiv?.listBibl?.biblStruct)) {
    const refTitle = getText(ref?.analytic?.title) || getText(ref?.monogr?.title);

    const refAuthorList = toArray(ref?.analytic?.author ?? ref?.monogr?.author);
    const refAuthorParts: string[] = [];
    for (const a of refAuthorList) {
      const surname = getText(a?.persName?.surname);
      if (surname) refAuthorParts.push(surname);
    }
    const refAuthors = refAuthorParts.join(", ");

    let refYear = "";
    const refWhen = ref?.monogr?.imprint?.date?.["@_when"];
    if (typeof refWhen === "string") {
      refYear = refWhen.slice(0, 4);
    }

    const pieces: string[] = [];
    if (refAuthors) pieces.push(refAuthors);
    if (refYear) pieces.push(`(${refYear})`);
    if (refTitle) pieces.push(refTitle);

    const refString = pieces.join(" - ");
    if (refString) references.push(refString);
  }

  return { arxivId, title, authors, year, abstract, sections, references };
}

export async function ingestPaper(
  arxivId: string,
): Promise<ParsedPaper> {
  await new Promise((resolve) => setTimeout(resolve, 1000));

  const pdfBuffer = await fetchPdf(arxivId);
  return parseWithGrobid(pdfBuffer, arxivId);
}
