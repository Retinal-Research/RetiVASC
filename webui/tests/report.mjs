// The AI report through the real page: the RetiVASC pipeline on a photo ends with a report written by
// the language model on this computer, from numbers it can be checked against; a follow-up question;
// the report in Chinese; and after a folder of photos, one report on all of them.
//   node tests/report.mjs   (dev server on :5178, or RETIVASC_URL=http://localhost:8080/; models from tools/release.py fetch, --llm, and TPOT)
import { copyFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const SHOTS = join(import.meta.dirname, '..', '..', 'tools', 'shots');
mkdirSync(SHOTS, { recursive: true });
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
page.on('dialog', async (d) => { problems.push(`dialog: ${d.message()}`); await d.dismiss(); });
let failed = 0;
const ok = (cond, label) => { if (!cond) failed++; console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`); };
const show = (text) => console.log(text.split('\n').map((l) => `    | ${l}`).join('\n'));

await page.goto(`${process.env.RETIVASC_URL ?? 'http://localhost:5178/'}#studio`);
await page.waitForFunction(() => window.__studio?.engine());
await page.evaluate(() => window.__studio.load('retivasc'));
await page.waitForTimeout(300);

// 1. one photo, in English
let t = Date.now();
let summary = await page.evaluate(() => window.__studio.run());
let r = await page.evaluate(() => window.__studio.result('report'));
ok(summary.errors === 0 && r?.kind === 'report', `RetiVASC pipeline with the AI report: ${((Date.now() - t) / 1000).toFixed(1)} s, ${summary.ran} steps`);
show(r.text);
const by = (x) => (x.writer === 'online' ? `${x.model} through the lab's gateway, ${(x.ms / 1000).toFixed(1)} s` : `${x.model} on this computer, ${x.result.newTokens} tokens at ${x.result.tokensPerSecond.toFixed(1)}/s`);
ok(/##/.test(r.text) && r.text.length > 300, `report has sections (${r.text.length} characters; ${by(r)})`);
ok(r.facts.length >= 2, `it was given ${r.facts.length} groups of numbers: ${r.facts.map((g) => g.title).join('; ')}`);
console.log(`     numbers not found in the results: ${r.unchecked.length ? r.unchecked.join(', ') : 'none'}; judging words: ${r.judged.length ? r.judged.join(', ') : 'none'}`);

// 2. a follow-up question in the inspector
await page.click('.react-flow__node[data-id="report"]');
await page.fill('.ask textarea', 'Is the arteriole-to-venule ratio in Zone B below 1? Quote the value.');
await page.click('.ask button:has-text("Ask")');
await page.waitForFunction(() => window.__studio.result('report').qa.length === 1, null, { timeout: 180_000 });
r = await page.evaluate(() => window.__studio.result('report'));
show(r.qa[0].a);
ok(r.qa[0].a.length > 10, `follow-up answered (${r.qa[0].a.length} characters)`);
await page.screenshot({ path: join(SHOTS, 'report_inspector.png') });

// 3. the same in Chinese
await page.evaluate(() => window.__studio.setParams('report', { language: 'zh' }));
t = Date.now();
summary = await page.evaluate(() => window.__studio.run());
r = await page.evaluate(() => window.__studio.result('report'));
show(r.text);
ok(summary.errors === 0 && (r.text.match(/\p{Script=Han}/gu) ?? []).length > 60, `Chinese report in ${((Date.now() - t) / 1000).toFixed(1)} s (only the report step ran: ${summary.ran} step)`);
console.log(`     numbers not found in the results: ${r.unchecked.length ? r.unchecked.join(', ') : 'none'}; judging words: ${r.judged.length ? r.judged.join(', ') : 'none'}`);
await page.evaluate(() => window.__studio.setParams('report', { language: 'en' }));

// 4. a folder of three photos: one report on all of them, after the last
const dir = mkdtempSync(join(tmpdir(), 'retivasc-report-'));
for (const f of ['cc0_left_3744.jpg', 'normal_left_1411.jpg', 'pd_retinopathy_900.jpg']) copyFileSync(join(import.meta.dirname, '..', 'public', 'samples', f), join(dir, f));
await page.getByRole('radio', { name: /A folder/ }).click();
await page.waitForSelector('.folder input[type=file]', { state: 'attached' });
await page.locator('.folder input[type=file]').setInputFiles(dir);
await page.waitForTimeout(300);
t = Date.now();
summary = await page.evaluate(() => window.__studio.run());
r = await page.evaluate(() => window.__studio.result('report'));
show(r?.text ?? '(none)');
ok(summary.errors === 0 && r?.scope === 'folder', `folder report after ${summary.photos} photos, ${((Date.now() - t) / 1000).toFixed(1)} s`);
console.log(`     numbers not found in the results: ${r?.unchecked.length ? r.unchecked.join(', ') : 'none'}; judging words: ${r?.judged.length ? r.judged.join(', ') : 'none'}`);
await page.click('.react-flow__node[data-id="report"]');
await page.screenshot({ path: join(SHOTS, 'report_folder.png') });

console.log(problems.length ? `problems: ${problems.join(' | ')}` : 'no page errors');
await browser.close();
process.exit(failed ? 1 : 0);
