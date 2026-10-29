// The AI report's prompt: the browser (src/engine/report.ts) and the RetiVASC server
// (retivasc_server/report.py, which writes through the lab's API) must build the same text from the
// same fact sheet. prompt_fixture.json holds sheets and the prompt the browser builds; the server's
// tests check theirs against it. Run: npx tsx tests/prompt.ts [--write]
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { FactSheet } from '../src/engine/facts';
import { reportPrompt, type Need } from '../src/engine/prompt';

const FILE = join(import.meta.dirname, 'prompt_fixture.json');
const sheets: Array<{ sheet: FactSheet; request: string; needs?: Need[] }> = [
  { request: 'Compare the arteries with the veins in two sentences.', sheet: { scope: 'photo', language: 'en', terms: ['quality', 'cdr', 'avr', 'zones', 'microns'], groups: [
    { title: 'AutoMorph M1 · Image quality', rows: [['Grade', 'Good'], ['Probability of Good', '0.975'], ['Gradable by AutoMorph’s rule', 'yes']] },
    { title: 'AutoMorph M3 · Vascular features', note: 'photograph centred on the macula', rows: [['Cup-to-disc ratio, vertical', '0.405'], ['CRAE, Zone B', '336.5 µm'], ['AVR, Zone B', '0.604']] },
  ] } },
  { request: '', sheet: { scope: 'folder', language: 'en', terms: ['statistics', 'share'], groups: [
    { title: 'Folder', rows: [['Photos in the folder', '3'], ['Analysed', '3']] },
    { title: 'AutoMorph M2 · Vessels', rows: [['Vessels, share of the field of view', 'n 3, mean 8.1%, SD 0.5%, median 8%, range 7.6% to 8.6%']] },
    { title: 'Comparison between steps', rows: [['Vessels, share of the field of view: VascX · Vessels minus AutoMorph M2 · Vessels', 'n 3, mean difference 1.2%, SD 0.3%, range 0.9% to 1.5%, correlation r 0.97']] },
  ] } },
  { request: '', sheet: { scope: 'photo', language: 'en', terms: [], groups: [] } },
  // a table step after the report
  { request: '', needs: ['table'], sheet: { scope: 'photo', language: 'en', terms: [], groups: [
    { title: 'RetiVASC · Branching angles · RetiFlow', rows: [['Branching points, arteries', '42'], ['Mean branching angle, arteries', '80.14°']] },
  ] } },
];
const rendered = sheets.map(({ sheet, request, needs }) => ({ sheet, request, needs: needs ?? [], prompt: reportPrompt(sheet, request, needs) }));
if (process.argv.includes('--write')) {
  writeFileSync(FILE, `${JSON.stringify(rendered, null, 1)}\n`);
  console.log(`wrote ${FILE}`);
} else {
  const fixture = JSON.parse(readFileSync(FILE, 'utf8')) as typeof rendered;
  const same = fixture.length === rendered.length && fixture.every((f, i) => f.prompt === rendered[i].prompt);
  console.log(`${same ? 'PASS' : 'FAIL'}  the browser builds the prompts in prompt_fixture.json`);
  process.exit(same ? 0 : 1);
}
