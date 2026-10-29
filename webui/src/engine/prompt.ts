// The AI report's prompt, from a fact sheet (facts.ts) and the instructions in prompts.json. The
// RetiVASC server builds the same text for the lab's API (retivasc_server/report.py); tests/prompt.ts
// and the server's tests hold both to prompt_fixture.json. No runtime here, so tests run it in Node.

import type { FactSheet, Language } from './facts';
import PROMPTS from './prompts.json';

/** Whether a folder sheet compares two steps (its last group). */
export const compares = (sheet: FactSheet) => sheet.groups.some((g) => g.title === 'Comparison between steps' || g.title === '步骤之间的比较');

export function factsText(sheet: FactSheet): string {
  const colon = sheet.language === 'zh' ? '：' : ':';
  return sheet.groups.map((g) => [`${g.title}${g.note ? (sheet.language === 'zh' ? `（${g.note}）` : ` (${g.note})`) : ''}${colon}`,
    ...g.rows.map(([k, v]) => `- ${k}${colon}${sheet.language === 'zh' ? '' : ' '}${v}`)].join('\n')).join('\n\n');
}

export function termsText(sheet: FactSheet): string {
  return sheet.terms.map((t) => `- ${PROMPTS.terms[t][sheet.language]}`).join('\n');
}

// the instructions are shared with the RetiVASC server, which writes reports through the lab's API
export const SYSTEM: Record<Language, string> = PROMPTS.system;

/** The sections asked for, as the prompt lists them. */
function sectionList(sheet: FactSheet): string {
  const l = sheet.language;
  const sections = PROMPTS.sections[sheet.scope][l].filter((_, i) => sheet.scope === 'photo' || i !== 3 || compares(sheet));
  return l === 'zh' ? sections.map((x) => `“## ${x}”`).join('、') : sections.map((x) => `"## ${x}"`).join(', ');
}

export function reportPrompt(sheet: FactSheet, focus: string): string {
  const l = sheet.language, colon = l === 'zh' ? '：' : ':';
  return [
    `${PROMPTS.results[sheet.scope][l]}${colon}`,
    factsText(sheet),
    '',
    `${PROMPTS.definitions[l]}${colon}`,
    termsText(sheet),
    '',
    PROMPTS.write[l].replace('{sections}', sectionList(sheet)),
    PROMPTS.guide[sheet.scope][l],
    focus ? PROMPTS.focus[l].replace('{focus}', focus) : '',
  ].join('\n').trim();
}

export function questionPrompt(question: string, language: Language): string {
  return PROMPTS.question[language].replace('{question}', question);
}
