// The RetiVASC pipeline's fallbacks: VascX stays on standby while RRWNet works, stands in when RRWNet
// fails, the cropped photo stands in when the enhancement fails, and the AI report and the results table
// run whatever failed before them.
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

// 1. all working: the VascX steps stay on standby
{
  const { page, s, problems, seconds } = await run([]);
  ok(['vx_vessel', 'vx_av', 'vx_disc'].every((id) => s[id].status === 'idle' && /Standby/.test(s[id].message ?? '')), `all working (${seconds} s): VascX on standby (${status(s, ['vx_vessel', 'vx_av', 'vx_disc'])})`);
  ok(['rrwnet', 'm3', 'retiflow', 'report', 'table'].every((id) => s[id].status === 'done'), `and the rest done (${status(s, ['rrwnet', 'm3', 'retiflow', 'report', 'table'])})`);
  ok(!problems.length, 'no page errors');
  await page.close();
}

// 2. RRWNet fails (a weak graphics card): VascX stands in for M3, RetiFlow and the view
{
  const { page, s, r, problems, seconds } = await run(['retivasc.rrwnet']);
  ok(s.rrwnet.status === 'error', `RRWNet broken (${s.rrwnet.message})`);
  ok(s.vx_vessel.status === 'done' && s.vx_av.status === 'done' && s.vx_disc.status === 'idle', `VascX vessels and arteries/veins ran in its place, the disc stayed on standby (${status(s, ['vx_vessel', 'vx_av', 'vx_disc'])}; ${seconds} s)`);
  const m3 = await r('m3'), flow = await r('retiflow');
  ok(s.m3.status === 'done' && Number.isFinite(m3?.features?.Fractal_dimension), `M3 measured VascX's maps (fractal dimension ${m3?.features?.Fractal_dimension})`);
  ok(s.retiflow.status === 'done' && flow?.artery?.length > 0 && /binary/.test(flow?.warning ?? ''), `RetiFlow measured VascX's artery/vein map (${flow?.artery?.length} + ${flow?.vein?.length} branching points; "${flow?.warning}")`);
  ok(s.report.status === 'done' && s.table.status === 'done', `the AI report and the table ran (${status(s, ['report', 'table'])})`);
  ok(!problems.length, 'no page errors');
  await page.close();
}

// 3. the enhancement fails too: the steps use the cropped photo
{
  const { page, s, problems } = await run(['retivasc.rrwnet', 'retivasc.enhance']);
  ok(s.enhance.status === 'error' && ['m1', 'disc', 'vx_vessel', 'vx_av', 'm3', 'retiflow', 'report', 'table'].every((id) => s[id].status === 'done'),
    `without the enhancement, everything ran on the cropped photo (${status(s, ['enhance', 'm1', 'disc', 'm3', 'retiflow', 'report', 'table'])})`);
  ok(!problems.length, 'no page errors');
  await page.close();
}

// 4. nothing measures: the AI report and the table still run, on what there is
{
  const broken = ['retivasc.enhance', 'retivasc.rrwnet', 'automorph.m1', 'automorph.disc', 'vascx.vessel', 'vascx.av', 'vascx.disc'];
  const { page, s, r, problems } = await run(broken);
  const table = await r('table');
  ok(s.table.status === 'done' && table?.tables?.length === 0, `the table ran with nothing to show (${s.table.status}, ${table?.tables?.length} tables)`);
  ok(s.report.status === 'done', `and the AI report ran (${s.report.status}${s.report.message ? `: ${s.report.message}` : ''})`);
  ok(!problems.length, 'no page errors');
  await page.close();
}

await browser.close();
process.exit(failed ? 1 : 0);
