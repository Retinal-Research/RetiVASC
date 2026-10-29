// The AI report step: the results of the steps connected to it (or, after a folder run, of every
// photo) become a fact sheet (engine/facts.ts), and a language model writes it up: the lab's, through
// the RetiVASC server (which keeps the API key and sees only the sheet), or the small one on this
// computer. Every number in the text is then looked up in the sheet; any it cannot find is shown.

import { llm } from '../llm/llm';
import type { ChatResult } from '../llm/protocol';
import { folderFacts, photoFacts, tableGroups, type Cohort, type FactSheet, type Language, type Source } from './facts';
import { hash } from './folder';
import { questionPrompt, reportPrompt, SYSTEM, type Need } from './prompt';

export { reportPrompt, SYSTEM };
import type { RunContext } from './specs';
import type { NodeValue, Table } from './values';

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

/** The caller's quota for today, as the report service counts it (dollars of the lab's API, per IP, UTC day). */
export interface Quota { used: number; limit: number; open: boolean }

/** This browser's quota today, or undefined when the service cannot say. */
export async function quota(): Promise<Quota | undefined> {
  try {
    const r = await fetch(REPORT_URL, { cache: 'no-store' });
    const j = r.ok ? ((await r.json()) as { today?: Quota }) : undefined;
    return j?.today;
  } catch {
    return undefined;
  }
}

let writing: AbortController | undefined;

/** Stop a report or an answer being written, by either model; the text so far is kept. */
export function stopWriting() {
  writing?.abort();
  llm.stop();
}

/** The report service's answer as it arrives, and how long the model thought before its first word. */
async function online(body: object, onText: (text: string) => void): Promise<{ text: string; cut: boolean; thinkMs?: number }> {
  writing = new AbortController();
  const t = performance.now();
  let text = '', cut = false, thinkMs: number | undefined;
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
      if (text) thinkMs ??= performance.now() - t;
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
  return { text, cut, thinkMs };
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
  /** the lab's model: how long it thought before its first word (it does not send its reasoning) */
  thinkMs?: number;
  /** Stop was pressed before the end */
  cut: boolean;
  /** numbers in the text that are not in the fact sheet */
  unchecked: string[];
  /** words that judge a value (high, normal…), which no reference range supports */
  judged: string[];
  /** follow-up questions and their answers, in order */
  qa: Array<{ q: string; a: string; unchecked: string[]; judged: string[] }>;
  /** what the steps after it needed (a results table), and the table it wrote for them */
  needs?: Need[];
  table?: Table;
  /** the table was asked for but could not be read */
  tableError?: string;
}

/** The table the model wrote after its report (prompts.json needs.table): the last json code block. Its
 *  numbers are checked like the text's. */
export function takeTable(text: string): { text: string; table?: Table; error?: string } {
  const m = /```json\s*([\s\S]*?)```\s*$/.exec(text.trim());
  if (!m) return { text: text.replace(/```json[\s\S]*$/, '').trim(), error: /```json/.test(text) ? 'the table was cut off before its end: the answer reached the length limit' : 'the model wrote no table' };
  const rest = text.trim().slice(0, m.index).trim();
  try {
    const j = JSON.parse(m[1]) as { title?: unknown; columns?: unknown; rows?: unknown };
    const cells = (r: unknown) => (Array.isArray(r) ? r.map((c) => (c == null ? '' : String(c))) : null);
    const columns = cells(j.columns);
    const rows = Array.isArray(j.rows) ? j.rows.map(cells) : null;
    if (!columns?.length || !rows || rows.some((r) => !r) || rows.length > 2000 || columns.length > 200) throw new Error('not a table');
    const width = columns.length;
    return { text: rest, table: { title: typeof j.title === 'string' && j.title ? j.title : 'Table written by the AI', note: 'written by the AI from the results; numbers checked', columns,
      rows: (rows as string[][]).map((r) => [...r.slice(0, width), ...new Array(Math.max(0, width - r.length)).fill('')]) } };
  } catch {
    return { text: rest, error: 'the model’s table could not be read' };
  }
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
  const request = String(ctx.params.prompt ?? '').trim();
  const effort: Effort = EFFORTS.includes(ctx.params.effort as Effort) ? (ctx.params.effort as Effort) : 'medium';
  const sheet = ctx.cohort ? folderFacts(ctx.cohort as Cohort, language) : photoFacts((ctx.many?.context ?? []) as Source[], language);
  if (ctx.cohort) {
    // a Results table after the folder is a step of its own: its rows come in as its value, not as records
    const terms = new Set(sheet.terms);
    for (const { value: v } of ctx.many?.context ?? []) if (v.kind === 'table') for (const t of v.tables) sheet.groups.push(...tableGroups(t, language, terms));
    sheet.terms = [...terms];
  }
  // any steps may be connected: those without numbers (an image, a view) just add nothing to the sheet.
  // The steps after it say what they need: a results table gets a table as well as the report.
  const needs: Need[] = ctx.downstream?.includes('view.table') ? ['table'] : [];
  const user = reportPrompt(sheet, request, needs);
  const chat = `report:${hash(SYSTEM[language] + user)}`;
  const choice = (ctx.params.engine as Writer | undefined) ?? 'online';
  const status = choice === 'local' ? undefined : await onlineStatus();
  if (choice !== 'local' && !status?.enabled) throw new Error('the lab’s report service cannot be reached right now; please try again later');
  const checks = (text: string) => ({ unchecked: uncheckedNumbers(text, user), judged: judgements(text, language) });
  if (status?.enabled) {
    const t = performance.now();
    ctx.progress({ stage: 'llm', message: 'The lab’s model is thinking' });
    const shown = (text: string) => text.replace(/```json[\s\S]*$/, '').trim();  // the table goes to the table step
    const r = await online({ facts: sheet, prompt: request, effort, needs }, (text) => ctx.progress({ stage: 'llm', message: needs.length ? 'The lab’s model is writing the report and the table' : 'The lab’s model is writing the report', text: shown(text) }));
    const written = needs.includes('table') ? takeTable(r.text) : { text: r.text };
    const text = written.text.trim();
    const table = written.table;
    const tableChecks = table ? uncheckedNumbers([table.columns, ...table.rows].flat().join('\n'), user) : [];
    return { report: { kind: 'report', text, facts: sheet, language, chat, request, effort, writer: 'online', model: status.model || 'the lab’s API', license: '',
      ms: performance.now() - t, thinkMs: r.thinkMs, cut: r.cut, unchecked: [...new Set([...checks(text).unchecked, ...tableChecks])], judged: checks(text).judged,
      qa: [], needs, table, tableError: 'error' in written ? written.error : undefined } };
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
