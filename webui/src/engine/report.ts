// The AI report step: the results of the steps connected to it (or, after a folder run, of every
// photo) become a fact sheet (engine/facts.ts), and the language model on this computer writes it
// up. Every number in the text is then looked up in the sheet; any it cannot find is shown.

import { llm } from '../llm/llm';
import type { ChatResult } from '../llm/protocol';
import { compares, factsText, folderFacts, photoFacts, termsText, type Cohort, type FactSheet, type Language, type Source } from './facts';
import { hash } from './folder';
import type { RunContext } from './specs';
import type { NodeValue } from './values';

export type { Language };

export interface ReportValue {
  kind: 'report';
  text: string;
  facts: FactSheet;
  language: Language;
  /** the conversation, for follow-up questions */
  chat: string;
  model: string;
  license: string;
  result: ChatResult;
  /** numbers in the text that are not in the fact sheet */
  unchecked: string[];
  /** words that judge a value (high, normal…), which no reference range supports */
  judged: string[];
  /** follow-up questions and their answers, in order */
  qa: Array<{ q: string; a: string; unchecked: string[]; judged: string[] }>;
}

export const SYSTEM: Record<Language, string> = {
  en: `You write the reports of RetiVASC, a research platform that measures the retinal blood vessels in colour fundus photographs, for researchers and clinicians.
Rules:
- Use only the results you are given. Never invent numbers, normal ranges, risks or diagnoses.
- Copy numbers exactly as they are written in the results, with their units.
- No reference ranges are given, so never call a value high, low, normal or abnormal.
- These measurements are research markers, not a diagnosis. Do not give medical advice.
- When image quality is poor or a value is missing, say what that means for the results.
- Be concise and factual.`,
  zh: `你为 RetiVASC 撰写报告。RetiVASC 是测量彩色眼底照片中视网膜血管的研究平台，读者是研究人员和临床医生。
规则：
- 只使用给出的结果。绝不编造数字、正常范围、风险或诊断。
- 数字按结果中的写法原样抄写，带上单位。
- 没有给出参考范围，所以绝不说某个数值偏高、偏低、正常或异常。
- 这些测量是研究指标，不是诊断。不要给医疗建议。
- 图像质量差或缺少数值时，说明这对结果意味着什么。
- 简洁、客观。全部用简体中文书写，术语用结果中的中文名称。`,
};

const SECTIONS: Record<FactSheet['scope'], Record<Language, string[]>> = {
  photo: { en: ['Summary', 'Image quality', 'Vessels', 'Notes'], zh: ['概要', '图像质量', '血管', '说明'] },
  folder: { en: ['Summary', 'Image quality', 'Measurements across the folder', 'Comparison between steps', 'Notes'], zh: ['概要', '图像质量', '整个文件夹的测量', '步骤之间的比较', '说明'] },
};

const GUIDE: Record<FactSheet['scope'], Record<Language, string>> = {
  photo: {
    en: 'Use short bullet points. The summary is two or three bullets: the image quality, then the vessel values you find most informative, with their numbers. In the vessels section, group the values by calibre, complexity and density, and tortuosity.',
    zh: '使用简短的要点。概要两三条：图像质量，以及你认为最有信息量的血管数值（带上数字）。血管部分按管径、复杂度和密度、迂曲度分组。',
  },
  folder: {
    en: 'Use short bullet points. The summary gives how many photographs were analysed and their quality; then the measurements, grouped; then the comparison between steps, if there is one.',
    zh: '使用简短的要点。概要写分析了多少张照片及其质量；然后按类别写测量结果；如果有步骤之间的比较，再写比较。',
  },
};

export function reportPrompt(sheet: FactSheet, focus: string): string {
  const l = sheet.language, zh = l === 'zh';
  const sections = SECTIONS[sheet.scope][l].filter((_, i) => sheet.scope === 'photo' || i !== 3 || compares(sheet));
  const what = sheet.scope === 'photo' ? (zh ? '一张照片的结果' : 'Results for one photograph') : (zh ? '一个文件夹中照片的结果' : 'Results for a folder of photographs');
  return [
    `${what}${zh ? '：' : ':'}`,
    factsText(sheet),
    '',
    zh ? '定义：' : 'Definitions:',
    termsText(sheet),
    '',
    zh
      ? `请写一份 Markdown 格式的报告，包含这些小节：${sections.map((x) => `“## ${x}”`).join('、')}。`
      : `Write a report in English, in Markdown, with these sections: ${sections.map((x) => `"## ${x}"`).join(', ')}.`,
    GUIDE[sheet.scope][l],
    focus ? (zh ? `特别关注：${focus}` : `Pay particular attention to: ${focus}`) : '',
  ].join('\n').trim();
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
  const language: Language = ctx.params.language === 'zh' ? 'zh' : 'en';
  const focus = String(ctx.params.focus ?? '').trim();
  const sheet = ctx.cohort ? folderFacts(ctx.cohort as Cohort, language) : photoFacts((ctx.many?.results ?? []) as Source[], language);
  if (!sheet.groups.length || (sheet.scope === 'folder' && sheet.groups.length < 2)) {
    throw new Error('the connected steps have no numbers to report; connect quality, segmentation or feature steps');
  }
  const user = reportPrompt(sheet, focus);
  const chat = `report:${hash(SYSTEM[language] + user)}`;
  const manifest = await llm.manifest;
  let text = '';
  const result = await llm.chat({ chat, system: SYSTEM[language], user, maxTokens: sheet.scope === 'folder' ? 1100 : 900, temperature: 0.3 },
    (delta) => { text += delta; ctx.progress({ stage: 'llm', message: 'Writing the report', text: clean(text) }); }, ctx.progress);
  const final = clean(result.text);
  return {
    report: { kind: 'report', text: final, facts: sheet, language, chat, model: manifest?.name ?? 'language model', license: manifest?.license ?? '', result,
      unchecked: uncheckedNumbers(final, user), judged: judgements(final, language), qa: [] },
  };
}

/** A follow-up question about a report; the answer arrives piece by piece. */
export async function askReport(report: ReportValue, question: string, onText: (text: string) => void): Promise<{ a: string; unchecked: string[]; judged: string[] }> {
  let text = '';
  const user = report.language === 'zh' ? `${question}\n\n请用简体中文简短回答，只依据上面的结果。` : `${question}\n\nAnswer in English, briefly, from the results above only.`;
  const result = await llm.chat({ chat: report.chat, user, maxTokens: 450, temperature: 0.3 }, (delta) => { text += delta; onText(clean(text)); });
  const a = clean(result.text);
  return { a, unchecked: uncheckedNumbers(a, `${reportPrompt(report.facts, '')}\n${report.text}`), judged: judgements(a, report.language) };
}
