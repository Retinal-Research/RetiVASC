// Day/night theme and accent colour through the UI, including persistence.
//   node tests/theme.mjs   (dev server on :5178)
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

const SHOTS = join(import.meta.dirname, '..', '..', 'tools', 'shots');
mkdirSync(SHOTS, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const context = await browser.newContext({ viewport: { width: 1680, height: 1000 }, colorScheme: 'dark' });
const page = await context.newPage();
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
const ok = (cond, label) => console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`);
const theme = () => page.evaluate(() => document.documentElement.dataset.theme);
const accent = () => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim());
const idle = () => page.waitForFunction(() => {
  const s = Object.values(window.__studio?.states() ?? {});
  return s.length && s.every((x) => x.status !== 'running') && s.some((x) => x.status === 'done');
}, null, { timeout: 300_000, polling: 200 });

await page.goto('http://localhost:5178/');
await page.evaluate(() => localStorage.removeItem('retivasc-studio-appearance'));
await page.reload();
await idle();
ok((await theme()) === 'dark', 'follows a dark system setting by default');

await page.click('.appearance .icon-btn:first-child');
ok((await theme()) === 'light', 'sun button switches to the day theme');
await page.click('.react-flow__node[data-id="view"] .node-head');
await page.waitForTimeout(300);
await page.screenshot({ path: join(SHOTS, 'theme_day.png') });

await page.click('.appearance .icon-btn:nth-child(2)');
await page.click('.swatch[title="Indigo"]');
ok((await accent()) === '#6366f1', `preset accent applied (${await accent()})`);
await page.screenshot({ path: join(SHOTS, 'theme_day_menu.png') });
await page.fill('.hex', '#e11d48');
ok((await accent()) === '#e11d48', `typed accent applied (${await accent()})`);
const ink = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--accent-ink').trim());
ok(ink === '#ffffff', `white text on a dark red accent (${ink})`);
await page.keyboard.press('Escape');

await page.reload();
const early = await page.evaluate(() => [document.documentElement.dataset.theme, document.documentElement.style.getPropertyValue('--accent')]);
ok(early[0] === 'light' && early[1] === '#e11d48', `remembered after reload (${early.join(', ')})`);
await idle();

await page.click('.appearance .icon-btn:nth-child(2)');
await page.click('.segmented button:has-text("System")');
ok((await theme()) === 'dark', 'System mode follows the dark setting');
await page.emulateMedia({ colorScheme: 'light' });
await page.waitForTimeout(100);
ok((await theme()) === 'light', 'System mode switches live when the OS goes light');
await page.emulateMedia({ colorScheme: 'dark' });
await page.click('.segmented button:has-text("Night")');
await page.click('.swatch[title="Amber"]');
await page.keyboard.press('Escape');
const amberInk = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--accent-ink').trim());
ok(amberInk === '#0b1210', `dark text on a light amber accent (${amberInk})`);
await page.click('.react-flow__node[data-id="av"] .node-head');
await page.waitForTimeout(300);
await page.screenshot({ path: join(SHOTS, 'theme_night_amber.png') });

await page.click('.appearance .icon-btn:nth-child(2)');
await page.click('.custom-accent .btn:has-text("Reset")');
ok((await accent()) === '#2dd4bf', 'Reset restores the default accent');

console.log(problems.length ? `problems: ${problems.join(' | ')}` : 'no page errors');
await browser.close();
