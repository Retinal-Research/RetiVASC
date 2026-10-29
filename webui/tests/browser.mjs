// End-to-end check in a real browser: runs the full AutoMorph workflow on every
// sample through the page itself (WebGPU when available) and compares the
// results with tools/make_reference.py. Needs the dev server on :5178.
//   node tests/browser.mjs [--headed] [--shots]
import { readFileSync, readdirSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const REF = join(import.meta.dirname, '..', '..', 'tools', 'ref');
const SHOTS = join(import.meta.dirname, '..', '..', 'tools', 'shots');
const URL = process.env.STUDIO_URL ?? 'http://localhost:5178/';
const headed = process.argv.includes('--headed');
const shots = process.argv.includes('--shots');

const browser = await chromium.launch({
  channel: process.env.BROWSER_CHANNEL ?? 'chrome',
  headless: !headed,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 }, deviceScaleFactor: 1 });
// the sidebar folds everything but the pipelines; open the sections these checks click into
await page.addInitScript(() => { for (const k of ['singles', 'server', 'skills']) localStorage.setItem(`retivasc-studio-fold-${k}`, 'true'); });
const problems = [];
page.on('console', (m) => { if (m.type() === 'error') problems.push(m.text()); });
page.on('pageerror', (e) => problems.push(String(e)));

await page.goto(URL + '#studio');
await page.waitForFunction(() => window.__studio?.engine(), null, { timeout: 120_000 });
const engine = await page.evaluate(() => {
  const e = window.__studio.engine();
  return { backend: e.backend, adapter: e.adapter, threads: e.threads, crossOriginIsolated };
});
console.log('engine', engine);

// nothing runs by itself: open AutoMorph, run it once (models load here) and wait for it to settle
const settle = () => page.waitForFunction(() => {
  const s = window.__studio.states();
  return Object.values(s).every((x) => x.status !== 'running') && Object.values(s).some((x) => x.status === 'done' || x.status === 'error');
}, null, { timeout: 600_000, polling: 250 });
await page.evaluate(() => window.__studio.load('automorph'));
await page.evaluate(() => window.__studio.run());
await settle();

const decode = (b64) => Buffer.from(b64, 'base64');
function compare(got, want, classes) {
  let same = 0;
  const inter = new Array(classes).fill(0), a = new Array(classes).fill(0), b = new Array(classes).fill(0);
  for (let i = 0; i < want.length; i++) {
    if (got[i] === want[i]) same++;
    a[got[i]]++;
    b[want[i]]++;
    if (got[i] === want[i]) inter[got[i]]++;
  }
  const dice = [];
  for (let c = 1; c < classes; c++) dice.push(a[c] + b[c] ? (2 * inter[c]) / (a[c] + b[c]) : 1);
  return { agree: same / want.length, dice };
}

if (shots) mkdirSync(SHOTS, { recursive: true });
const rows = [];
for (const stem of readdirSync(REF).sort()) {
  const meta = JSON.parse(readFileSync(join(REF, stem, 'meta.json'), 'utf8'));
  await page.evaluate((s) => window.__studio.setSource('photo', `sample:samples/${s}.jpg`), stem);
  const t0 = Date.now();
  const summary = await page.evaluate(() => window.__studio.run());
  const wall = Date.now() - t0;
  const r = await page.evaluate(() => ({
    m0: window.__studio.result('m0'),
    q: window.__studio.result('m1'),
    v: window.__studio.result('vessel'),
    av: window.__studio.result('av'),
    d: window.__studio.result('disc'),
    states: window.__studio.states(),
  }));
  const errors = Object.entries(r.states).filter(([, s]) => s.status === 'error').map(([k, s]) => `${k}: ${s.message}`);
  if (errors.length) { console.log(stem, 'ERRORS', errors); rows.push({ stem, errors }); continue; }
  const g = r.m0.geometry;
  const m0ok = g.radius === meta.m0.radius && g.bbox.join() === meta.m0.bbox.join() && g.centreRow === meta.m0.centreRow && g.centreCol === meta.m0.centreCol;
  const v = compare(decode(r.v.labels), readFileSync(join(REF, stem, 'l_vessel.u8')), 2);
  const av = compare(decode(r.av.labels), readFileSync(join(REF, stem, 'l_artery_vein.u8')), 4);
  const d = compare(decode(r.d.labels), readFileSync(join(REF, stem, 'l_disc_cup.u8')), 3);
  const qp = r.q.result.probs.map((p, i) => Math.abs(p - meta.quality.probs[i]));
  const row = {
    stem,
    wallMs: wall,
    runMs: Math.round(summary.ms),
    m0: m0ok ? 'same' : `DIFF r=${g.radius} vs ${meta.m0.radius}`,
    quality: `${r.q.result.prediction}${r.q.result.prediction === meta.quality.prediction ? '' : ` (python ${meta.quality.prediction})`} max|dp|=${Math.max(...qp).toExponential(1)}`,
    vessel: `agree ${(v.agree * 100).toFixed(3)}% dice ${v.dice[0].toFixed(4)}`,
    av: `agree ${(av.agree * 100).toFixed(3)}% dice A ${av.dice[0].toFixed(3)} V ${av.dice[1].toFixed(3)}`,
    disc: `agree ${(d.agree * 100).toFixed(3)}% dice disc ${d.dice[0].toFixed(3)} cup ${d.dice[1].toFixed(3)}`,
    inferMs: { q: Math.round(r.q.timing.inferenceMs), v: Math.round(r.v.timing.inferenceMs), av: Math.round(r.av.timing.inferenceMs), d: Math.round(r.d.timing.inferenceMs) },
    backend: r.v.timing.backend,
  };
  rows.push(row);
  console.log(JSON.stringify(row));
  if (shots) await page.screenshot({ path: join(SHOTS, `${stem}.png`) });
}

if (shots) {
  // the inspector for a few steps, then another workflow
  await page.evaluate(() => window.__studio.setSource('photo', 'sample:samples/cc0_left_3744.jpg'));
  await page.evaluate(() => window.__studio.run());
  for (const id of ['view', 'disc', 'm1']) {
    await page.click(`.react-flow__node[data-id="${id}"] .node-head`);
    await page.waitForTimeout(400);
    await page.screenshot({ path: join(SHOTS, `ui_${id}.png`) });
  }
  await page.click('.wf:has-text("Artery / vein and optic disc")');
  await page.evaluate(() => window.__studio.run());  // nothing runs by itself
  await settle();
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(SHOTS, 'ui_av_disc_workflow.png') });
}

if (problems.length) console.log('console problems:', problems.slice(0, 10));
await browser.close();
