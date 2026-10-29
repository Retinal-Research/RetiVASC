// Interaction checks through the real UI: open a local photo, add a step from
// the library, see a missing connection reported, save and reopen a workflow.
//   node tests/ui.mjs   (dev server on :5178)
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const SHOTS = join(import.meta.dirname, '..', '..', 'tools', 'shots');
mkdirSync(SHOTS, { recursive: true });
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 }, acceptDownloads: true });
// the sidebar folds everything but the pipelines; open the sections these checks click into
await page.addInitScript(() => { for (const k of ['singles', 'server', 'skills']) localStorage.setItem(`retivasc-studio-fold-${k}`, 'true'); });
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
page.on('dialog', async (d) => { problems.push(`dialog: ${d.message()}`); await d.dismiss(); });

const idle = () => page.waitForFunction(() => {
  const s = Object.values(window.__studio.states());
  return s.length && s.every((x) => x.status !== 'running');
}, null, { timeout: 300_000, polling: 200 });
const states = () => page.evaluate(() => window.__studio.states());
const ok = (cond, label) => console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`);

await page.goto('http://localhost:5178/#studio');
await page.waitForFunction(() => window.__studio?.engine());

// 1. open a local photo through the node's file input (a workflow runs only when Run is pressed)
await page.click('.wf:has-text("Vessel segmentation")');
await page.click('.btn.primary');
await idle();
const file = join(import.meta.dirname, '..', 'public', 'samples', 'pd_cmv_retinitis.jpg');
await page.setInputFiles('.react-flow__node[data-id="photo"] input[type=file]', file);
// the file is read asynchronously before the node changes
await page.waitForFunction(() => window.__studio.states().vessel.status === 'idle', null, { timeout: 10_000 }).catch(() => {});
let s = await states();
ok(s.vessel.status === 'idle' && s.vessel.message?.includes('Changed'), `after opening a file, downstream steps wait for Run (${s.vessel.message})`);
await page.click('.btn.primary');
await idle();
s = await states();
const photo = await page.evaluate(() => window.__studio.result('photo'));
ok(s.vessel.status === 'done' && photo.width === 1800, `opened file ran end to end (${photo.width}×${photo.height})`);
const option = await page.$eval('.react-flow__node[data-id="photo"] select', (el) => el.selectedOptions[0].textContent);
ok(option === 'pd_cmv_retinitis.jpg', `photo node shows the opened file name (${option})`);

// 2. add a step from the library; it should report the missing connection
await page.click('.lib-item:has-text("M1 · Image quality")');
await page.click('.btn.primary');
await idle();
s = await states();
const added = Object.entries(s).find(([id]) => id.startsWith('m1-'));
ok(added && added[1].status === 'blocked' && added[1].message.includes('cropped'), `new unconnected step is blocked: ${added?.[1].message}`);
await page.screenshot({ path: join(SHOTS, 'ui_added_step.png') });

// 3. save the workflow, open it again
const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
const saved = join(SHOTS, 'saved_workflow.json');
await download.saveAs(saved);
const wf = JSON.parse(readFileSync(saved, 'utf8'));
ok(wf.format === 'retivasc-studio/1' && wf.nodes.length === 5 && !wf.nodes.find((n) => n.id === 'photo').params.source, `saved ${wf.nodes.length} steps; the opened photo itself is not written to the file`);
await page.click('.wf:has-text("Image quality check")');
await page.click('.btn.primary');
await idle();
await page.setInputFiles('label.btn input[type=file]', saved);
await page.waitForTimeout(500);
const count = await page.$$eval('.react-flow__node', (n) => n.length);
ok(count === 5, `reopened workflow has ${count} steps`);

console.log(problems.length ? `problems: ${problems.join(' | ')}` : 'no page errors');
await browser.close();
