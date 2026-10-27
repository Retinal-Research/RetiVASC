// The browser's complete ensembles against the official AutoMorph v2 run on the sample photos.
//   OFFICIAL=<folder> STUDIO_URL=http://localhost:8081/ node tests/official.mjs [--quick]
// OFFICIAL holds M1_quality_final.csv and, per photo, <stem>.vessel.u8 (912²), <stem>.av.u8 (720²,
// 1 artery 2 vein 3 crossing) and <stem>.disc.u8 (512², 1 disc 2 cup): the official M2 images of
// `python -m automorph_v2 run` on webui/public/samples, decoded to labels.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const OFFICIAL = process.env.OFFICIAL;
const URL = process.env.STUDIO_URL ?? 'http://localhost:8081/';
const quick = process.argv.includes('--quick');
const CLASSES = ['Good', 'Usable', 'Reject'];

const m1 = {};
for (const line of readFileSync(join(OFFICIAL, 'M1_quality_final.csv'), 'utf8').trim().split(/\r?\n/).slice(1)) {
  const f = line.split(',');
  m1[f[0].replace(/\\/g, '/').split('/').pop().replace(/\.png$/, '')] = { probs: f.slice(1, 4).map(Number), prediction: CLASSES[+f[7]], gradable: f[8] === 'True' };
}

const labels = (stem, kind) => readFileSync(join(OFFICIAL, `${stem}.${kind}.u8`));

function agree(got, want, classes) {
  let same = 0;
  const inter = new Array(classes).fill(0), a = new Array(classes).fill(0), b = new Array(classes).fill(0);
  for (let i = 0; i < want.length; i++) {
    if (got[i] === want[i]) { same++; inter[got[i]]++; }
    a[got[i]]++;
    b[want[i]]++;
  }
  return { differ: want.length - same, of: want.length, dice: inter.map((x, k) => (a[k] + b[k] ? (2 * x) / (a[k] + b[k]) : 1)) };
}

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
await page.goto(URL);
await page.waitForFunction(() => window.__studio?.engine(), null, { timeout: 120_000 });
const engine = await page.evaluate(() => window.__studio.engine());
console.log(`engine ${engine.backend} · ${engine.adapter} · ${quick ? 'quick (1 model each)' : 'complete ensembles'}`);
await page.evaluate(() => window.__studio.load(0));
if (quick) for (const id of ['m1', 'vessel', 'av']) await page.evaluate((n) => window.__studio.setParams(n, { quick: true }), id);

let failed = 0;
for (const stem of Object.keys(m1).sort()) {
  await page.evaluate((s) => window.__studio.setSource('photo', `sample:samples/${s}.jpg`), stem);
  const t0 = Date.now();
  await page.evaluate(() => window.__studio.run());
  const r = await page.evaluate(() => ({ q: window.__studio.result('m1'), v: window.__studio.result('vessel'), av: window.__studio.result('av'), d: window.__studio.result('disc') }));
  const o = m1[stem];
  const dp = Math.max(...r.q.result.probs.map((p, i) => Math.abs(p - o.probs[i])));
  const sameGrade = r.q.result.prediction === o.prediction && r.q.result.gradable === o.gradable;
  let line = `${stem.padEnd(20)} ${((Date.now() - t0) / 1000).toFixed(1).padStart(5)} s  quality ${r.q.result.prediction}${sameGrade ? '' : ` (official ${o.prediction})`} max|dp| ${dp.toExponential(1)}`;
  if (!sameGrade) failed++;
  if (!existsSync(join(OFFICIAL, `${stem}.vessel.u8`))) { console.log(line + '  (official M2 skipped this photo)'); continue; }
  const v = agree(Buffer.from(r.v.labels, 'base64'), labels(stem, 'vessel'), 2);
  const av = agree(Buffer.from(r.av.labels, 'base64'), labels(stem, 'av'), 4);
  const d = agree(Buffer.from(r.d.labels, 'base64'), labels(stem, 'disc'), 3);
  line += `  vessel ${v.differ} px differ (dice ${v.dice[1].toFixed(4)})  artery/vein ${av.differ} px (A ${av.dice[1].toFixed(4)} V ${av.dice[2].toFixed(4)})  disc ${d.differ} px (disc ${d.dice[1].toFixed(4)} cup ${d.dice[2].toFixed(4)})`;
  console.log(line);
}
await browser.close();
process.exit(failed ? 1 : 0);
