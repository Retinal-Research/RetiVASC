// The whole pipeline in the browser (M0–M2 complete ensembles + M3) against the official AutoMorph v2
// run on the sample photos.   M3_REF=<prep_ref.py folder> STUDIO_URL=http://localhost:8081/ node tests/m3browser.mjs
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const REF = process.env.M3_REF;
const URL = process.env.STUDIO_URL ?? 'http://localhost:8081/';
const cases = JSON.parse(readFileSync(join(REF, 'cases.json'), 'utf8'));

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
await page.goto(URL + '#studio');
await page.waitForFunction(() => window.__studio?.engine(), null, { timeout: 120_000 });
const engine = await page.evaluate(() => window.__studio.engine());
console.log(`engine ${engine.backend} · ${engine.adapter}`);
await page.evaluate(() => window.__studio.load('automorph'));

const all = { n: 0, same: 0, tiny: 0, pct1: 0, pct5: 0 };
for (const c of cases) {
  const sample = c.stem.replace(/^\d+__/, '');
  await page.evaluate((s) => window.__studio.setSource('photo', `sample:samples/${s}.jpg`), sample);
  const t0 = Date.now();
  await page.evaluate(() => window.__studio.run());
  const r = await page.evaluate(() => ({ f: window.__studio.result('m3'), states: window.__studio.states() }));
  if (!r.f) { console.log(sample, 'no result', JSON.stringify(r.states.m3)); continue; }
  const want = c.expected.values;
  const rows = { n: 0, same: 0, tiny: 0, pct1: 0, pct5: 0, worst: ['', 0] };
  for (const [k, w] of Object.entries(want)) {
    const g = r.f.features[k];
    if (w === null && g === null) { rows.n++; rows.same++; continue; }
    rows.n++;
    if (g === w) { rows.same++; continue; }
    const rel = g === null || w === null ? Infinity : Math.abs(g - w) / Math.max(Math.abs(w), 1e-12);
    if (rel <= 1e-9) rows.tiny++;
    else if (rel <= 0.01) rows.pct1++;
    else if (rel <= 0.05) rows.pct5++;
    if (rel > rows.worst[1]) rows.worst = [k, rel];
  }
  for (const k of ['n', 'same', 'tiny', 'pct1', 'pct5']) all[k] += rows[k];
  console.log(`${sample.padEnd(20)} ${((Date.now() - t0) / 1000).toFixed(1).padStart(5)} s  ${r.f.centredOn === (c.expected.table.startsWith('Disc') ? 'optic disc' : 'macula') ? 'same table' : 'TABLE DIFFERS'}`
    + `  identical ${rows.same}/${rows.n}, ≤1e-9 ${rows.tiny}, ≤1% ${rows.pct1}, 1–5% ${rows.pct5}, >5% ${rows.n - rows.same - rows.tiny - rows.pct1 - rows.pct5}`
    + `  (largest: ${rows.worst[0]} ${(rows.worst[1] * 100).toFixed(2)}%)`);
}
console.log(`all: ${all.same}/${all.n} identical, ${all.tiny} within 1e-9, ${all.pct1} within 1%, ${all.pct5} within 5%, ${all.n - all.same - all.tiny - all.pct1 - all.pct5} beyond`);
await browser.close();
