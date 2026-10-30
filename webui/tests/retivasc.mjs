// The RetiVASC pipeline end to end in a browser: TPOT first, then on the enhanced photo the quality grade,
// RRWNet, the optic disc, M3 on RRWNet's maps, RetiFlow's branching angles, and the AI report on them.
//   node tests/retivasc.mjs   (dev server on :5178; the report needs the RetiVASC server's report service)
import { chromium } from 'playwright-core';

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
const problems = [];
page.on('pageerror', (e) => problems.push(String(e)));
let failed = 0;
const ok = (cond, label) => { if (!cond) failed++; console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}`); };

await page.goto(`${process.env.RETIVASC_URL ?? 'http://localhost:5178/'}#studio`);
await page.waitForFunction(() => window.__studio?.engine());
await page.evaluate(() => window.__studio.load('retivasc'));
await page.waitForTimeout(500);
const t = Date.now();
await page.evaluate(() => window.__studio.run());
await page.waitForFunction(() => Object.values(window.__studio.states()).every((x) => x.status !== 'running'), null, { timeout: 900_000, polling: 500 });
const s = await page.evaluate(() => window.__studio.states());
console.log(`ran in ${((Date.now() - t) / 1000).toFixed(0)} s`);
for (const [id, st] of Object.entries(s)) ok(st.status === 'done', `${id}: ${st.status}${st.message ? ` (${st.message})` : ''}`);

const r = (id) => page.evaluate((x) => window.__studio.result(x), id);
const enhanced = await r('enhance');
ok(enhanced?.kind === 'm0', 'the enhanced photo feeds the later steps');
const flow = await r('retiflow');
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
ok(flow?.kind === 'bifurcations' && flow.artery.length > 5 && flow.vein.length > 5,
  `RetiFlow: ${flow?.artery.length} artery and ${flow?.vein.length} vein branching points, mean ${mean(flow?.artery ?? []).toFixed(1)}° and ${mean(flow?.vein ?? []).toFixed(1)}°, root [${flow?.root}], ${(flow?.ms / 1000).toFixed(1)} s`);
ok(!flow?.warning && !flow?.failed && flow?.gradable === true, `a good photo: no warning, graded gradable (${flow?.warning ?? flow?.failed ?? 'ok'})`);
ok(flow?.artery.every((a) => a >= 20 && a <= 120) && flow?.vein.every((a) => a >= 20 && a <= 120), 'every angle between 20° and 120°');
const m3 = await r('m3');
ok(m3?.kind === 'features' && Number.isFinite(m3.features.Fractal_dimension), `M3 on RRWNet's maps: fractal dimension ${m3?.features.Fractal_dimension}`);
const report = await r('report');
const facts = (report?.facts ?? []).map((g) => g.title);
ok(report?.kind === 'report' && facts.some((x) => x.includes('RetiFlow')), `the report is given RetiFlow's numbers (${facts.join('; ')})`);
ok(report && /branch/i.test(report.text), 'and writes about branching');
if (report) console.log(`\n${report.text.slice(0, 1200)}\n`);
// the Results table: the numbers of M1, M3 and RetiFlow, one row, and its CSV named after the photo, pipeline and date
const table = await r('table');
const t0 = table?.tables?.[0];
ok(t0 && t0.rows.length === 1 && t0.columns.includes('Fractal_dimension') && t0.columns.includes('retiflow_Artery_branching_angle'),
  `Results table: ${t0?.rows.length} row, ${t0?.columns.length} columns`);
await page.click('.react-flow__node[data-id="table"] .node-title');
const [download] = await Promise.all([page.waitForEvent('download'), page.click('.inspector .tbl .btn:has-text("Download CSV")')]);
const csv = await (await import('node:fs/promises')).readFile(await download.path(), 'utf8');
const [head, row] = csv.trim().split('\n');
ok(/_RetiVASC_\d{4}-\d\d-\d\d\.csv$/.test(download.suggestedFilename()) && head.split(',').length === t0?.columns.length && row.split(',').length === head.split(',').length,
  `Download CSV: ${download.suggestedFilename()}, ${head.split(',').length} columns`);

// the AI report connected to the table: it sees the table after it and writes a table too, numbers checked
await page.evaluate(() => {
  const wf = window.__studio.workflow('retivasc');
  wf.edges.push({ source: 'report', sourceHandle: 'report', target: 'table', targetHandle: 'results' });
  window.__studio.loadFile(wf);
});
await page.waitForTimeout(500);
await page.evaluate(() => window.__studio.run());
await page.waitForFunction(() => Object.values(window.__studio.states()).every((x) => x.status !== 'running'), null, { timeout: 900_000, polling: 500 });
const rep2 = await r('report'), table2 = await r('table');
ok(rep2?.table && !rep2.tableError && !/```/.test(rep2.text), `the report wrote a table for the table step (${rep2?.table?.rows.length} rows: ${rep2?.table?.columns.join(' | ')})${rep2?.tableError ? ` ${rep2.tableError}` : ''}`);
ok(rep2 && rep2.unchecked.length === 0, `its numbers are all in the results (${rep2?.unchecked.join(', ') || 'none missing'})`);
ok(table2?.tables?.length === 2, `the table step shows both tables (${table2?.tables?.map((x) => x.title).join('; ')})`);
if (rep2?.table) console.log(rep2.table.rows.slice(0, 6).map((x) => '    | ' + x.join(' | ')).join('\n'));
// the other way round: the AI report whose only context is the Results table reads all of its numbers
await page.evaluate(() => {
  const wf = window.__studio.workflow('retivasc');
  wf.edges = wf.edges.filter((e) => e.target !== 'report');
  wf.edges.push({ source: 'table', sourceHandle: 'table', target: 'report', targetHandle: 'context' });
  window.__studio.loadFile(wf);
});
await page.waitForTimeout(500);
await page.evaluate(() => window.__studio.run());
await page.waitForFunction(() => Object.values(window.__studio.states()).every((x) => x.status !== 'running'), null, { timeout: 900_000, polling: 500 });
const rep3 = await r('report');
const rows3 = (rep3?.facts ?? []).flatMap((g) => g.rows);
// every value the table has; AutoMorph leaves the disc and zone columns empty when its disc test fails
const filled = t0 ? t0.rows[0].filter((v) => v !== '').length : 0;
ok(rows3.length >= filled - 3 && rows3.some(([k]) => /Fractal dimension/.test(k)) && rows3.some(([k]) => /branching angle, arteries/i.test(k)) && /\d/.test(rep3?.text ?? ''),
  `the report reads the whole Results table (${rows3.length} of ${filled} values; numbers not in them: ${rep3?.unchecked.join(', ') || 'none'})`);
if (rep3) console.log(rep3.text.split('\n').slice(0, 10).map((x) => '    | ' + x).join('\n'));
ok(problems.length === 0, `no page errors${problems.length ? `: ${problems.join(' | ')}` : ''}`);
await page.screenshot({ path: 'tests/retivasc.png' });
await browser.close();
process.exit(failed ? 1 : 0);
