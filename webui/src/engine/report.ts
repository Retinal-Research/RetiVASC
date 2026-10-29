// The AI report step: the results of the steps connected to it (or, after a folder run, of every
// photo) become a fact sheet (engine/facts.ts), and a language model writes it up: the lab's, through
// the RetiVASC server (which keeps the API key and sees only the sheet), or the small one on this
// computer. Every number in the text is then looked up in the sheet; any it cannot find is shown.

import { llm } from '../llm/llm';
import type { ChatResult } from '../llm/protocol';
import { folderFacts, photoFacts, type Cohort, type FactSheet, type Language, type Source } from './facts';
import { hash } from './folder';
import { questionPrompt, reportPrompt, SYSTEM } from './prompt';

export { reportPrompt, SYSTEM };
import type { RunContext } from './specs';
import type { NodeValue } from './values';

export type { Language };

export type Effort = 'low' | 'medium' | 'high';
export const EFFORTS: Effort[] = ['low', 'medium', 'high'];

/** Who writes: the lab's API. The model on this computer ('local') is frozen: nobody wants a 1.4 GB download to
 *  write a report, so the page never offers it; a workflow file that names it still runs it. */
export type Writer = 'auto' | 'online' | 'local';

// the report service: on the RetiVASC server next to the page, or where VITE_REPORT_URL says (the public site)
const REPORT_URL = import.meta.env.VITE_REPORT_URL || `${import.meta.env.BASE_URL}api/report`;
let service: Promise<{ enabled: boolean; model: string }> | undefined;

/** Whether the lab's API can write reports for this page. */
export function onlineStatus(): Promise<{ enabled: boolean; model: string }> {
  service ??= fetch(REPORT_URL, { cache: 'no-store' })
    .then((r) => (r.ok ? (r.json() as Promise<{ enabled: boolean; model: string }>) : { enabled: false, model: '' }), () => ({ enabled: false, model: '' }));
  return service;
}

let writing: AbortController | undefined;

/** Stop a report or an answer being written, by either model; the text so far is kept. */
export function stopWriting() {
  writing?.abort();
  llm.stop();
}

/** The report service's answer as it arrives. */
async function online(body: object, onText: (text: string) => void): Promise<{ text: string; cut: boolean }> {
  writing = new AbortController();
  let text = '', cut = false;
  try {
    const r = await fetch(REPORT_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: writing.signal });
    if (!r.ok || !r.body) {
      const detail = await r.json().then((j: { detail?: string }) => j.detail, () => undefined);
      throw new Error(detail ?? `the report service answered ${r.status}`);
    }
    const reader = r.body.getReader(), decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      onText(text);
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === 'AbortError')) throw error;
    cut = true;
  } finally {
    writing = undefined;
  }
  const failed = /\n*\[report service error: ([^\]]*)\]\s*$/.exec(text);
  if (failed) throw new Error(`the lab's API: ${failed[1]}`);
  return { text, cut };
}

export interface ReportValue {
  kind: 'report';
  text: string;
  facts: FactSheet;
  language: Language;
  /** the conversation, for follow-up questions */
  chat: string;
  /** the request the person wrote, or '' for the default one */
  request: string;
  /** how hard the lab's model thinks: low, medium or high */
  effort: Effort;
  /** who wrote it: the lab's API through the RetiVASC server, or the model on this computer */
  writer: 'online' | 'local';
  model: string;
  license: string;
  /** the local model's speed and token counts */
  result?: ChatResult;
  ms: number;
  /** Stop was pressed before the end */
  cut: boolean;
  /** numbers in the text that are not in the fact sheet */
  unchecked: string[];
  /** words that judge a value (high, normal…), which no reference range supports */
  judged: string[];
  /** follow-up questions and their answers, in order */
  qa: Array<{ q: string; a: string; unchecked: string[]; judged: string[] }>;
}

/** Numbers as written: 0.673, 151.27, 8.1%, 1,234; years, list markers and zone letters are not numbers here. */
const NUMBER = /(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?/g;

function values(text: string): number[] {
  return [...text.matchAll(NUMBER)].map((m) => Number(m[0].replace(/[,%]/g, ''))).filter(Number.isFinite);
}

/** Numbers in the text that the prompt does not contain, allowing for rounding to the digits written. */
export function uncheckedNumbers(text: string, prompt: string): string[] {
  const known = values(prompt);
  const out: string[] = [];
  for (const m of text.matchAll(NUMBER)) {
    const s = m[0], x = Number(s.replace(/[,%]/g, ''));
    if (!Number.isFinite(x) || (Number.isInteger(x) && Math.abs(x) < 10)) continue;  // counts and list numbers
    const decimals = (s.replace(/%$/, '').split('.')[1] ?? '').length;
    const step = 10 ** -decimals;
    // a probability may be written as a percentage (0.975 as 97.5%)
    const close = (k: number) => Math.abs(k - x) <= step / 2 + 1e-9 || (s.endsWith('%') && Math.abs(k * 100 - x) <= step / 2 + 1e-9);
    if (!known.some(close)) out.push(s);
  }
  return [...new Set(out)];
}

const JUDGE: Record<Language, RegExp> = {
  en: /\b(high|low|normal|abnormal|elevated|reduced|increased|decreased|healthy|unhealthy|pathological|risk|disease)\b/gi,
  zh: /(偏高|偏低|较高|较低|过高|过低|升高|降低|正常|异常|健康|病理|风险|疾病|病变)/g,
};

/** Words that judge a value; without reference ranges nothing supports them. */
export function judgements(text: string, language: Language): string[] {
  return [...new Set([...text.matchAll(JUDGE[language])].map((m) => m[0].toLowerCase()))];
}

/** Text without an empty or stray thinking block (Qwen3 may still write one), nor the replacement
 *  character of a byte the model left without the rest of its Chinese character. */
export const clean = (text: string) => text.replace(/<think>[\s\S]*?<\/think>\s*/g, '').replace(/\uFFFD/g, '').trim();

export async function aiReport(ctx: RunContext): Promise<Record<string, NodeValue>> {
  const language: Language = 'en';  // the public demonstration is in English
  const request = String(ctx.params.prompt ?? '').trim().slice(0, 1500);
  const effort: Effort = EFFORTS.includes(ctx.params.effort as Effort) ? (ctx.params.effort as Effort) : 'medium';
  const sheet = ctx.cohort ? folderFacts(ctx.cohort as Cohort, language) : photoFacts((ctx.many?.results ?? []) as Source[], language);
  // any steps may be connected: those without numbers (an image, a view) just add nothing to the sheet
  const user = reportPrompt(sheet, request);
  const chat = `report:${hash(SYSTEM[language] + user)}`;
  const choice = (ctx.params.engine as Writer | undefined) ?? 'online';
  const status = choice === 'local' ? undefined : await onlineStatus();
  if (choice !== 'local' && !status?.enabled) throw new Error('the lab’s report service cannot be reached right now; please try again later');
  const checks = (text: string) => ({ unchecked: uncheckedNumbers(text, user), judged: judgements(text, language) });
  if (status?.enabled) {
    const t = performance.now();
    const r = await online({ facts: sheet, prompt: request, effort }, (text) => ctx.progress({ stage: 'llm', message: 'The lab’s API is writing the report', text }));
    const text = r.text.trim();
    return { report: { kind: 'report', text, facts: sheet, language, chat, request, effort, writer: 'online', model: status.model || 'the lab’s API', license: '',
      ms: performance.now() - t, cut: r.cut, ...checks(text), qa: [] } };
  }
  const manifest = await llm.manifest;
  let text = '';
  const result = await llm.chat({ chat, system: SYSTEM[language], user, maxTokens: sheet.scope === 'folder' ? 1100 : 900, temperature: 0.3 },
    (delta) => { text += delta; ctx.progress({ stage: 'llm', message: 'Writing the report', text: clean(text) }); }, ctx.progress);
  const final = clean(result.text);
  return {
    report: { kind: 'report', text: final, facts: sheet, language, chat, request, effort, writer: 'local', model: manifest?.name ?? 'language model',
      license: manifest?.license ?? '', result, ms: result.ms, cut: result.cut, ...checks(final), qa: [] },
  };
}

/** A follow-up question about a report; the answer arrives piece by piece. */
export async function askReport(report: ReportValue, question: string, onText: (text: string) => void): Promise<{ a: string; unchecked: string[]; judged: string[] }> {
  let a: string;
  if (report.writer === 'online') {
    const history = report.qa.map(({ q, a: answer }) => ({ q, a: answer }));
    a = (await online({ facts: report.facts, prompt: report.request, effort: report.effort, report: report.text, history, question }, onText)).text.trim();
  } else {
    let text = '';
    const result = await llm.chat({ chat: report.chat, user: questionPrompt(question, report.language), maxTokens: 450, temperature: 0.3 },
      (delta) => { text += delta; onText(clean(text)); });
    a = clean(result.text);
  }
  return { a, unchecked: uncheckedNumbers(a, `${reportPrompt(report.facts, report.request)}\n${report.text}`), judged: judgements(a, report.language) };
}
