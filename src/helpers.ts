export function cleanText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export function isValidText(text: string): boolean {
  return text.trim().length > 0;
}
