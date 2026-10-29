// Public-beta (open) mode against a running server: no sign-in, and two browsers do not see
// each other's jobs.   RETIVASC_URL=http://localhost:8080/ node tests/open.mjs
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const URL = process.env.RETIVASC_URL ?? 'http://localhost:8080/';
let failed = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++; };

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const a = await (await browser.newContext({ viewport: { width: 1600, height: 1000 } })).newPage();
const b = await (await browser.newContext({ viewport: { width: 1600, height: 1000 } })).newPage();
await a.goto(URL + '#studio');
await a.waitForSelector('.wf:has-text("Full analysis on the server")');
ok(!(await a.isVisible('.login')), 'no sign-in screen');
ok((await a.textContent('.status .private')).includes('Public beta'), 'top bar says public beta');
await a.click('.wf:has-text("Full analysis on the server")');
await a.click('.btn.primary');  // nothing runs until Run
await a.waitForSelector('.react-flow__node[data-id="full"] .ana', { timeout: 600000 });
ok(true, `browser A ran a full analysis (${await a.textContent('.react-flow__node[data-id="full"] .ana-grade b')})`);
await a.waitForSelector('.jobs .job', { timeout: 10000 });
ok((await a.$$('.jobs .job')).length >= 1, 'browser A sees its job');
await a.screenshot({ path: join(import.meta.dirname, '..', '..', 'tools', 'shots', 'open_mode.png') });

await b.goto(URL + '#studio');
await b.waitForSelector('.wf:has-text("Full analysis on the server")');
await b.waitForTimeout(5000);
ok((await b.$$('.jobs .job')).length === 0, "browser B does not see browser A's job");
await browser.close();
process.exit(failed ? 1 : 0);
