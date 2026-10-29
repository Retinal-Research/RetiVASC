// The home page wording, and the parts of the canvas page you arrange or operate: folding sidebar,
// red Stop, inspector width and image height, the compute device, and Preload.
//   node tests/layout.mjs   (dev server on :5178)
import { mkdirSync } from 'node:fs';
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
const idle = () => page.waitForFunction(() => {
  const s = Object.values(window.__studio.states());
  return s.length && s.every((x) => x.status !== 'running');
}, null, { timeout: 300_000, polling: 200 });
const width = (sel) => page.$eval(sel, (el) => Math.round(el.getBoundingClientRect().width));
const height = (sel) => page.$eval(sel, (el) => Math.round(el.getBoundingClientRect().height));
async function drag(sel, dx, dy) {
  const box = await (await page.$(sel)).boundingBox();
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx / 2, y + dy / 2, { steps: 4 });
  await page.mouse.move(x + dx, y + dy, { steps: 4 });
  await page.mouse.up();
}

// 1. home page: Prof. Wang's wording
await page.goto('http://localhost:5178/');
const features = await page.$$eval('.feature p', (ps) => ps.map((p) => p.textContent));
ok(features.some((t) => t.includes('Images are stored and processed locally.')) && !features.some((t) => t.includes('never leave')), 'features: "Images are stored and processed locally."');
ok(features.some((t) => t.includes('Combined with an LLM for report generation and analysis.')), 'features: "Combined with an LLM for report generation and analysis."');
ok(features.some((t) => t.startsWith('RetiVASC, AutoMorph and VascX')), 'pipeline library starts with RetiVASC');
ok((await page.textContent('.hero-what')).trim().startsWith('RetiVASC'), 'the introduction starts with RetiVASC');

// 2. sidebar: the pipelines always show; the rest folds, and stays as it was left
await page.goto('http://localhost:5178/#studio');
await page.waitForFunction(() => window.__studio?.engine());
const pipelines = await page.$$eval('.sidebar > section .wf-name', (n) => n.map((e) => e.textContent));
ok(pipelines.join(',') === 'RetiVASC,AutoMorph,VascX,Custom', `pipelines shown unfolded, ours first (${pipelines.join(', ')})`);
ok(!(await page.isVisible('.wf:has-text("Vessel segmentation")')) && !(await page.isVisible('.lib-item')), 'single steps and skills start folded');
await page.click('.fold-head:has-text("Single steps")');
await page.click('.fold-head:has-text("Skills")');
ok(await page.isVisible('.wf:has-text("Vessel segmentation")') && (await page.$$('.lib-item:visible')).length > 10, 'clicking a section opens it');
await page.click('.fold-head:has-text("AutoMorph M2")');
ok(!(await page.isVisible('.lib-item:has-text("M2 · Vessels")')) && await page.isVisible('.lib-item:has-text("M1 · Image quality")'), 'a skill group folds on its own');
await page.reload();
await page.waitForFunction(() => window.__studio?.engine());
ok(await page.isVisible('.wf:has-text("Vessel segmentation")') && !(await page.isVisible('.lib-item:has-text("M2 · Vessels")')), 'folding is remembered after a reload');

// 3. inspector width and image height: drag, remembered, double-click resets
const w0 = await width('.inspector');
await drag('.grip-x', -120, 0);
const w1 = await width('.inspector');
ok(Math.abs(w1 - w0 - 120) <= 2, `dragging the inspector's edge widens it (${w0} → ${w1} px)`);
await page.click('.wf:has-text("Image quality check")');
await page.click('.react-flow__node[data-id="photo"]');
const h0 = await height('.imageview');
await drag('.grip-y', 0, 90);
const h1 = await height('.imageview');
ok(Math.abs(h1 - h0 - 90) <= 2, `dragging below the image makes it taller (${h0} → ${h1} px)`);
await page.reload();
await page.waitForFunction(() => window.__studio?.engine());
ok(Math.abs((await width('.inspector')) - w1) <= 1, 'the width is remembered after a reload');
await page.dblclick('.grip-x');
ok(Math.abs((await width('.inspector')) - 392) <= 1, 'double-click puts the default width back');

// 4. Preload: the models of the workflow, then Run needs no loading
await page.click('.wf:has-text("Image quality check")');
await page.waitForSelector('.btn.preload');
ok((await page.textContent('.btn.preload')).includes('Preload'), `Preload offered: ${await page.getAttribute('.btn.preload', 'title')}`);
await page.click('.btn.preload');
await page.waitForSelector('.btn.preload.ready', { timeout: 300_000 });
ok(true, 'Preload finished: "Models ready"');
await page.click('.btn.primary');
await idle();
const q = await page.evaluate(() => window.__studio.result('m1'));
ok(q && q.timing.loadMs === undefined, `after Preload, Run loads nothing (${q?.timing.inferenceMs.toFixed(0)} ms inference)`);

// 5. Stop is red while a run is going
await page.evaluate(() => window.__studio.setParams('photo', { source: 'sample:samples/pd_cmv_retinitis.jpg' }));
await page.click('.btn.primary');
await page.waitForSelector('.btn.danger');
const red = await page.$eval('.btn.danger', (el) => getComputedStyle(el).backgroundColor);
ok(red === 'rgb(220, 38, 38)', `Stop is red (${red})`);
await page.screenshot({ path: join(SHOTS, 'layout_stop.png') });
await idle();

// 6. compute device: CPU only restarts the engine; back to the faster card
await page.click('button.engine');
ok((await page.$$('.choice')).length === 3, 'the compute menu offers three devices');
await page.screenshot({ path: join(SHOTS, 'layout_compute.png') });
await page.click('.choice:has-text("CPU only")');
await page.waitForFunction(() => window.__studio.engine()?.backend === 'wasm', null, { timeout: 60_000 });
ok((await page.textContent('button.engine')).includes('CPU only'), `CPU only: ${await page.textContent('button.engine')}`);
const after = await page.evaluate(() => window.__studio.states().m1);
ok(after.status === 'idle' && after.message.includes('restarted'), `results are cleared (${after.message})`);
await page.click('button.engine');
await page.click('.choice:has-text("Faster graphics card")');
await page.waitForFunction(() => window.__studio.engine()?.backend === 'webgpu', null, { timeout: 60_000 });
ok((await page.textContent('button.engine')).includes('WebGPU'), `back on the graphics card: ${await page.textContent('button.engine')}`);

console.log(problems.length ? `problems: ${problems.join(' | ')}` : 'no page errors');
await browser.close();
process.exit(failed ? 1 : 0);
