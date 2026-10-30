// When steps of the RetiVASC pipeline fail, the AI report and the results table still run, on whatever
// finished; the steps that need a failed one wait.
//   node tests/fallback.mjs   (dev server on :5178; models for AutoMorph, TPOT, RRWNet and VascX; the report service)
import { chromium } from 'playwright-core';

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
let failed = 0;
const ok = (cond, label) => { if (!cond) failed++; console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`); };
const URL = `${process.env.RETIVASC_URL ?? 'http://localhost:5178/'}#studio`;

/** A fresh page with the RetiVASC pipeline; break the given step types, run, and return the states. */
async function run(broken) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const problems = [];
  page.on('pageerror', (e) => problems.push(String(e)));
  await page.goto(URL);
  await page.waitForFunction(() => window.__studio?.engine());
  await page.evaluate((types) => { for (const t of types) window.__studio.breakStep(t); window.__studio.load('retivasc'); }, broken);
  await page.waitForTimeout(500);
  const t = Date.now();
  await page.evaluate(() => window.__studio.run());
  await page.waitForFunction(() => Object.values(window.__studio.states()).every((x) => x.status !== 'running'), null, { timeout: 900_000, polling: 500 });
  const s = await page.evaluate(() => window.__studio.states());
  const r = (id) => page.evaluate((x) => window.__studio.result(x), id);
  return { page, s, r, problems, seconds: ((Date.now() - t) / 1000).toFixed(0) };
}
const status = (s, ids) => ids.map((id) => `${id} ${s[id]?.status}`).join(', ');

// 1. RRWNet fails (a weak graphics card): M3 and RetiFlow wait; the AI report and the table run on the rest
{
  const { page, s, r, problems, seconds } = await run(['retivasc.rrwnet']);
  ok(s.rrwnet.status === 'error', `RRWNet broken (${s.rrwnet.message})`);
  ok(s.m3.status === 'blocked' && s.retiflow.status === 'blocked', `the steps that need it wait (${status(s, ['m3', 'retiflow'])}; ${seconds} s)`);
  const table = await r('table'), report = await r('report');
  ok(s.table.status === 'done' && table?.tables?.[0]?.columns.includes('Quality'), `the table ran on what finished (${s.table.status}: ${table?.tables?.[0]?.columns.length} columns)`);
  ok(s.report.status === 'done' && report?.facts?.length > 0, `and the AI report (${s.report.status}: ${report?.facts?.map((g) => g.title).join('; ')})`);
  ok(!problems.length, 'no page errors');
  await page.close();
}

// 2. nothing measures: the AI report and the table still run
{
  const { page, s, r, problems } = await run(['retivasc.enhance']);
  const table = await r('table');
  ok(s.enhance.status === 'error' && s.table.status === 'done' && table?.tables?.length === 0, `with the enhancement broken, the table ran with nothing to show (${status(s, ['enhance', 'm1', 'table'])})`);
  ok(s.report.status === 'done', `and the AI report ran (${s.report.status})`);
  ok(!problems.length, 'no page errors');
  await page.close();
}

await browser.close();
process.exit(failed ? 1 : 0);
