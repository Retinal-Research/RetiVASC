// The AI report's prompt, from a fact sheet (facts.ts) and the instructions in prompts.json. The
// RetiVASC server builds the same text for the lab's API (retivasc_server/report.py); tests/prompt.ts
// and the server's tests hold both to prompt_fixture.json. No runtime here, so tests run it in Node.

import type { FactSheet, Language } from './facts';
import PROMPTS from './prompts.json';

export function factsText(sheet: FactSheet): string {
  const colon = sheet.language === 'zh' ? '：' : ':';
  return sheet.groups.map((g) => [`${g.title}${g.note ? (sheet.language === 'zh' ? `（${g.note}）` : ` (${g.note})`) : ''}${colon}`,
    ...g.rows.map(([k, v]) => `- ${k}${colon}${sheet.language === 'zh' ? '' : ' '}${v}`)].join('\n')).join('\n\n');
}

export function termsText(sheet: FactSheet): string {
  return sheet.terms.map((t) => `- ${PROMPTS.terms[t][sheet.language]}`).join('\n');
}

// the server writes with this system prompt; the page cannot change it
export const SYSTEM: Record<Language, string> = PROMPTS.system;

/** The editable part of the prompt, as it starts out. */
export const DEFAULT_REQUEST: Record<FactSheet['scope'], string> = PROMPTS.request;

/** What a step after the report can ask of it (prompts.json "needs"): a table, for now. */
export type Need = keyof typeof PROMPTS.needs;
export const NEEDS = Object.keys(PROMPTS.needs) as Need[];

/** The results, their definitions, the request (the default one when it is empty), and what the steps
 *  after the report need from it. */
export function reportPrompt(sheet: FactSheet, request: string, needs: Need[] = []): string {
  const l = sheet.language;
  const parts = [`${PROMPTS.results[sheet.scope][l]}:`, sheet.groups.length ? factsText(sheet) : PROMPTS.none];
  if (sheet.terms.length) parts.push('', `${PROMPTS.definitions[l]}:`, termsText(sheet));
  parts.push('', `${PROMPTS.heading}:`, request.trim() || DEFAULT_REQUEST[sheet.scope]);
  for (const n of needs) parts.push('', PROMPTS.needs[n][l]);
  return parts.join('\n').trim();
}

export function questionPrompt(question: string, language: Language): string {
  return PROMPTS.question[language].replace('{question}', question);
}
