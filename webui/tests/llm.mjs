// The language model in the browser: it loads, writes, keeps a conversation's cache for a follow-up
// question, gives the same text for the same request, and reads a long prompt in chunks as it would
// in one pass.
//   node tests/llm.mjs   (dev server on :5178; models/llm from tools/release.py fetch --llm)
import { chromium } from 'playwright-core';

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
let failed = 0;
const ok = (cond, label) => { if (!cond) failed++; console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`); };

await page.goto('http://localhost:5178/#studio');
await page.waitForFunction(() => window.__studio?.engine());
const info = await page.evaluate(() => window.__studio.llm.info());
console.log(`model ${info.manifest.name} on ${info.backend} · ${info.adapter}`);

const t0 = Date.now();
await page.evaluate(() => window.__studio.llm.load());
ok(true, `loaded in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

const ask = (req) => page.evaluate(async (r) => {
  let streamed = '';
  const result = await window.__studio.llm.chat(r, (d) => { streamed += d; });
  return { ...result, streamed };
}, req);
const system = 'You are a careful assistant. Answer briefly.';
const a = await ask({ chat: 'test', system, user: 'In two sentences: what does the retina do?', maxTokens: 120 });
console.log(`  → ${JSON.stringify(a.text.slice(0, 200))}`);
ok(a.text.length > 40 && /retina|light|eye/i.test(a.text) && a.streamed === a.text, `wrote ${a.newTokens} tokens at ${a.tokensPerSecond.toFixed(1)} tokens/s; read ${a.promptTokens} in ${a.readMs.toFixed(0)} ms; streamed text equals the result`);

const b = await ask({ chat: 'test', user: 'And the optic disc? One sentence.', maxTokens: 80 });
console.log(`  → ${JSON.stringify(b.text.slice(0, 200))}`);
ok(b.reusedTokens >= a.promptTokens + a.newTokens && b.text.length > 10, `follow-up reused ${b.reusedTokens} of ${b.promptTokens} prompt tokens`);

const c = await ask({ chat: 'test', system, user: 'In two sentences: what does the retina do?', maxTokens: 120 });
ok(c.text === a.text, 'the same request gives the same text');

// a prompt of several hundred tokens, read in chunks of 128 and in one pass
const long = Array.from({ length: 40 }, (_, i) => `Photo ${i + 1}: vessel density ${(0.07 + i * 0.001).toFixed(3)}, AVR ${(0.62 + (i % 7) * 0.01).toFixed(2)}.`).join('\n');
const user = `${long}\n\nWhich photo has the highest vessel density? Answer with the photo number only.`;
const d1 = await ask({ chat: 'long1', system, user, maxTokens: 30, chunk: 128 });
const d2 = await ask({ chat: 'long2', system, user, maxTokens: 30, chunk: 4096 });
console.log(`  → chunks: ${JSON.stringify(d1.text)} · one pass: ${JSON.stringify(d2.text)} (${d1.promptTokens} prompt tokens)`);
ok(d1.text === d2.text, 'a prompt read in chunks gives the same answer as one pass');

console.log(problems.length ? `problems: ${problems.join(' | ')}` : 'no page errors');
await browser.close();
process.exit(failed ? 1 : 0);
