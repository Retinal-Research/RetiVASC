// The AI report's prompt: the browser (src/engine/report.ts) and the RetiVASC server
// (retivasc_server/report.py, which writes through the lab's API) must build the same text from the
// same fact sheet. prompt_fixture.json holds sheets and the prompt the browser builds; the server's
// tests check theirs against it. Run: npx tsx tests/prompt.ts [--write]
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { FactSheet } from '../src/engine/facts';
import { reportPrompt } from '../src/engine/prompt';

const FILE = join(import.meta.dirname, 'prompt_fixture.json');
const sheets: Array<{ sheet: FactSheet; focus: string }> = [
  { focus: 'arteries and veins', sheet: { scope: 'photo', language: 'en', terms: ['quality', 'cdr', 'avr', 'zones', 'microns'], groups: [
    { title: 'AutoMorph M1 · Image quality', rows: [['Grade', 'Good'], ['Probability of Good', '0.975'], ['Gradable by AutoMorph’s rule', 'yes']] },
    { title: 'AutoMorph M3 · Vascular features', note: 'photograph centred on the macula', rows: [['Cup-to-disc ratio, vertical', '0.405'], ['CRAE, Zone B', '336.5 µm'], ['AVR, Zone B', '0.604']] },
  ] } },
  { focus: '', sheet: { scope: 'folder', language: 'zh', terms: ['statistics', 'share'], groups: [
    { title: '文件夹', rows: [['文件夹中的照片', '3'], ['已分析', '3']] },
    { title: 'AutoMorph M2 · 血管', rows: [['血管占视野比例', 'n 3，均值 8.1%，标准差 0.5%，中位数 8%，范围 7.6% 至 8.6%']] },
    { title: '步骤之间的比较', rows: [['血管占视野比例：VascX · 血管 减 AutoMorph M2 · 血管', 'n 3，平均差 1.2%，标准差 0.3%，范围 0.9% 至 1.5%，相关系数 r 0.97']] },
  ] } },
];
const rendered = sheets.map(({ sheet, focus }) => ({ sheet, focus, prompt: reportPrompt(sheet, focus) }));
if (process.argv.includes('--write')) {
  writeFileSync(FILE, `${JSON.stringify(rendered, null, 1)}\n`);
  console.log(`wrote ${FILE}`);
} else {
  const fixture = JSON.parse(readFileSync(FILE, 'utf8')) as typeof rendered;
  const same = fixture.length === rendered.length && fixture.every((f, i) => f.prompt === rendered[i].prompt);
  console.log(`${same ? 'PASS' : 'FAIL'}  the browser builds the prompts in prompt_fixture.json`);
  process.exit(same ? 0 : 1);
}
