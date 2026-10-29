// The tutorial: the Tutorial button opens it on the home page, it lights up each part in turn and goes on
// to the canvas, Escape or Skip closes it for good, and it opens by itself only on a first visit.
//   node tests/tour.mjs   (dev server on :5178)
import { chromium } from 'playwright-core';

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
let failed = 0;
const ok = (cond, label) => { if (!cond) failed++; console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`); };
const card = () => page.evaluate(() => ({
  title: document.querySelector('#tour-title')?.textContent ?? null,
  lit: !!document.querySelector('.tour-hole.lit'),
  view: location.hash === '#studio' ? 'studio' : 'home',
}));
const URL = process.env.RETIVASC_URL ?? 'http://localhost:5178/';

await page.goto(URL);
await page.waitForSelector('.tour-btn');
ok((await card()).title === null, 'an automated browser gets no tour by itself');
await page.click('.tour-btn');
let c = await card();
ok(c.title === 'Welcome to RetiVASC' && !c.lit, 'the Tutorial button opens the welcome card');
const seen = [];
for (let i = 0; i < 20 && c.title; i++) {
  seen.push(`${c.view}:${c.title}${c.lit ? '' : ' (no light)'}`);
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(700);
  c = await card();
}
console.log(`     ${seen.join(' → ')}`);
ok(seen.length === 13, `thirteen cards from the home page (${seen.length})`);
ok(seen.slice(1).every((s) => !s.endsWith('(no light)')), 'every card after the welcome lights up its part');
ok(seen[3]?.startsWith('studio:'), 'the tour goes on to the canvas');
ok(seen.at(-1)?.startsWith('home:') && (await card()).view === 'home', 'and ends on the home page, for people to start on their own');
ok(c.title === null && (await page.evaluate(() => localStorage.getItem('retivasc-studio-tour'))) === '"done"', 'Done closes it and remembers');

// from the canvas: Escape closes it at once
await page.goto(`${URL}#studio`);
await page.waitForSelector('.topbar .tour-btn');
await page.click('.topbar .tour-btn');
await page.keyboard.press('ArrowRight');
await page.waitForTimeout(500);
ok((await card()).title === 'This is your pipeline', 'from the canvas it starts there');
await page.keyboard.press('Escape');
ok((await card()).title === null, 'Escape closes it');

// a first visit by a person: it opens by itself, and Skip ends it
const person = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await person.addInitScript(() => Object.defineProperty(navigator, 'webdriver', { get: () => false }));
await person.goto(URL);
await person.waitForSelector('#tour-title', { timeout: 10_000 }).catch(() => {});
ok((await person.evaluate(() => document.querySelector('#tour-title')?.textContent)) === 'Welcome to RetiVASC', 'a first visit opens the tour by itself');
await person.click('.tour-skip');
ok(!(await person.$('.tour')), 'Skip closes it');
await person.reload();
await person.waitForTimeout(800);
ok(!(await person.$('.tour')), 'and it does not come back');

ok(problems.length === 0, `no page errors${problems.length ? `: ${problems.join(' | ')}` : ''}`);
await browser.close();
process.exit(failed ? 1 : 0);
