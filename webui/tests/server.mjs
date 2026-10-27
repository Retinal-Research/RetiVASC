// End to end against a running RetiVASC server (python -m retivasc_server serve):
// sign in, full analysis of one sample photo on the server, then a small study sheet from an
// uploaded folder. Usage:
//   RETIVASC_URL=http://localhost:8080/ RETIVASC_USER=… RETIVASC_PASSWORD=… RETIVASC_STUDY=<dir with photos/ and template.xlsx> node tests/server.mjs
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const URL = process.env.RETIVASC_URL ?? 'http://localhost:8080/';
const STUDY = process.env.RETIVASC_STUDY;
const SHOTS = join(import.meta.dirname, '..', '..', 'tools', 'shots');
let failed = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(URL);

// 1. sign in
await page.waitForSelector('.login');
ok(!(await page.isVisible('.wf:has-text("Full analysis on the server")')), 'server workflows hidden before sign-in');
await page.fill('.login input[autocomplete=username]', process.env.RETIVASC_USER);
await page.fill('.login input[type=password]', 'wrong-password');
await page.click('.login button');
await page.waitForSelector('.login .error');
ok(true, 'a wrong password is refused');
await page.fill('.login input[type=password]', process.env.RETIVASC_PASSWORD);
await page.click('.login button');
await page.waitForSelector('.login', { state: 'detached' });
ok(await page.isVisible('.wf:has-text("Full analysis on the server")'), 'signed in; server workflows shown');

// 2. one photo, full AutoMorph v2 on the server
await page.click('.wf:has-text("Full analysis on the server")');
await page.waitForSelector('.react-flow__node[data-id="full"]');  // a preset workflow runs as soon as it opens
await page.waitForSelector('.react-flow__node[data-id="full"] .sprog', { timeout: 30000 });
const step = await page.textContent('.react-flow__node[data-id="full"] .sprog-head span');
ok(true, `progress shown while it runs ("${step}")`);
await page.waitForSelector('.react-flow__node[data-id="full"] .ana', { timeout: 600000 });
const grade = await page.textContent('.react-flow__node[data-id="full"] .ana-grade b');
const imgs = await page.$$eval('.react-flow__node[data-id="full"] .ana-imgs img', (els) => els.map((e) => e.naturalWidth));
ok(['Good', 'Usable', 'Reject'].includes(grade), `quality grade from the full ensemble: ${grade}`);
ok(imgs.length === 4 && imgs.every((w) => w > 0), `4 result images load (${imgs.join(', ')} px wide)`);
await page.click('.react-flow__node[data-id="full"] .node-head');
await page.waitForTimeout(800);
const rows = await page.$$eval('.inspector .ana-group tr', (r) => r.length);
ok(rows === 72, `inspector lists all 72 features (${rows})`);
await page.screenshot({ path: join(SHOTS, 'server_analysis.png') });

// 3. a small study: upload a folder, fill and audit the sheet
if (STUDY) {
  await page.click('.wf:has-text("Excel sheet")');
  await page.waitForSelector('.react-flow__node[data-id="sheet"]');
  await page.setInputFiles('.react-flow__node[data-id="run"] input[type=file]', join(STUDY, 'photos'));
  const picked = await page.textContent('.react-flow__node[data-id="run"] .meta');
  ok(/3 photos/.test(picked), `folder picked: ${picked}`);
  const [tpl] = await page.$$('.react-flow__node[data-id="sheet"] input[type=file]');
  await tpl.setInputFiles(join(STUDY, 'template.xlsx'));
  await page.fill('.react-flow__node[data-id="sheet"] .sfield input.select', 'automorph_results_v1.xlsx');
  await page.click('.btn.primary');
  await page.waitForSelector('.react-flow__node[data-id="sheet"] .sverdict', { timeout: 900000 });
  const verdict = await page.textContent('.react-flow__node[data-id="sheet"] .sverdict');
  ok(/deliverable/.test(verdict) && !/NOT/.test(verdict), `study sheet audited: ${verdict.replace(/\s+/g, ' ').trim()}`);
  const links = await page.$$eval('.react-flow__node[data-id="sheet"] .sfile', (a) => a.length);
  ok(links >= 4, `${links} files to download`);
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('.react-flow__node[data-id="sheet"] .sfile >> nth=0')]);
  ok(download.suggestedFilename().endsWith('.xlsx'), `downloaded ${download.suggestedFilename()}`);
  await page.click('.react-flow__node[data-id="sheet"] .node-head');
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(SHOTS, 'server_sheet.png') });
}

// 4. the jobs list, sign out
ok((await page.$$('.jobs .job')).length >= 1, 'jobs listed in the sidebar');
await page.click('.linkbtn:has-text("Sign out")');
await page.waitForSelector('.login');
ok(true, 'signed out');
ok(errors.length === 0, errors.length ? `page errors: ${errors.join(' | ')}` : 'no page errors');
await browser.close();
process.exit(failed ? 1 : 0);
