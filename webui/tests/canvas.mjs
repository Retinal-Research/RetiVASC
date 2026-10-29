// What you do with a step on the canvas: run it alone (the steps after it wait), duplicate it,
// disconnect it, delete it, and remove a connection with its ×.
//   node tests/canvas.mjs   (dev server on :5178)
import { chromium } from 'playwright-core';

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
page.on('dialog', async (d) => { problems.push(`dialog: ${d.message()}`); await d.dismiss(); });
let failed = 0;
const ok = (cond, label) => { if (!cond) failed++; console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`); };
const states = () => page.evaluate(() => window.__studio.states());
const idle = () => page.waitForFunction(() => Object.values(window.__studio.states()).every((x) => x.status !== 'running'), null, { timeout: 300_000, polling: 200 });
const node = (id) => `.react-flow__node[data-id="${id}"]`;
const menu = async (id, label) => {
  await page.click(`${node(id)} .ntool[aria-label="More"]`);
  await page.click(`${node(id)} .nmenu-item:has-text("${label}")`);
};

await page.goto(`${process.env.RETIVASC_URL ?? 'http://localhost:5178/'}#studio`);
await page.waitForFunction(() => window.__studio?.engine());
await page.evaluate(() => window.__studio.load('automorph'));
await page.waitForTimeout(500);

// 1. run one step: it and what it needs run; the steps after it do not
await page.click(`${node('vessel')} .ntool[aria-label="Run this step"]`);
await idle();
let s = await states();
ok(s.photo.status === 'done' && s.m0.status === 'done' && s.vessel.status === 'done', 'Run this step: the step and the steps before it ran');
ok(s.m3.status === 'idle' && s.m1.status === 'idle' && s.av.status === 'idle', `the steps after it and beside it did not (m3 ${s.m3.status}, m1 ${s.m1.status}, av ${s.av.status})`);

// 2. duplicate
const before = Object.keys(s).length;
await menu('m1', 'Duplicate');
s = await states();
ok(Object.keys(s).length === before + 1 && Object.keys(s).some((id) => id.startsWith('m1-')), 'Duplicate adds a copy');

// 3. disconnect inputs: the step now waits for a connection
await menu('m3', 'Disconnect inputs');
await page.click(`${node('m3')} .ntool[aria-label="Run this step"]`);
await idle();
s = await states();
ok(s.m3.status === 'blocked', `Disconnect inputs: the step is not connected any more (${s.m3.message})`);

// 4. a connection's ×: select it, remove it
const edges = () => page.$$eval('.react-flow__edge', (es) => es.length);
const n0 = await edges();
await page.click('.react-flow__edge[data-id="m0.image->vessel.image"] .react-flow__edge-interaction', { force: true });
await page.click('.edge-x');
ok((await edges()) === n0 - 1, `the × removes the connection (${n0} → ${await edges()})`);
s = await states();
ok(s.vessel.status === 'idle' && (s.vessel.message ?? '').includes('Changed'), 'the step it fed waits for Run again');

// 5. delete
await menu('view', 'Delete');
s = await states();
ok(!('view' in s), 'Delete removes the step');

console.log(problems.length ? `problems: ${problems.join(' | ')}` : 'no page errors');
await browser.close();
process.exit(failed ? 1 : 0);
