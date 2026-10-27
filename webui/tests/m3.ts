// Browser M3 against the official AutoMorph v2 run, fed with the same M2 maps.
//   M3_REF=<folder from prep_ref.py> npx tsx tests/m3.ts
// The folder holds cases.json (radius, gradability, published features per photo), the official M2
// maps as label files (<stem>.vessel.u8 / .av.u8 / .disc.u8) and floats.json (repr / pandas vectors).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { runM3 } from '../src/automorph/m3/m3';
import { pandasParse, pyRepr } from '../src/automorph/m3/pyfloat';

const REF = process.env.M3_REF!;
let failures = 0;
const fail = (msg: string) => { failures++; console.log('FAIL', msg); };

const fromHex = (hex: string) => { const b = Buffer.from(hex, 'hex'); return b.readDoubleBE(0); };

// 1. number formatting and parsing
const vectors = JSON.parse(readFileSync(join(REF, 'floats.json'), 'utf8')) as Array<{ bits: string; repr: string; pandas_bits: string }>;
let reprBad = 0, parseBad = 0;
for (const v of vectors) {
  const x = fromHex(v.bits);
  if (pyRepr(x) !== v.repr) { if (reprBad++ < 3) console.log('  repr', v.repr, 'got', pyRepr(x)); }
  const p = pandasParse(v.repr);
  if (!Object.is(p, fromHex(v.pandas_bits))) { if (parseBad++ < 3) console.log('  parse', v.repr, 'got', p, 'want', fromHex(v.pandas_bits)); }
}
console.log(`repr: ${vectors.length - reprBad}/${vectors.length} identical · pandas parse: ${vectors.length - parseBad}/${vectors.length} identical`);
if (reprBad || parseBad) fail('number formatting');

// 2. the 72 features
interface Case { name: string; stem: string; radius: number; gradable: boolean; expected?: { table: string; values: Record<string, number | null> } }
const cases = JSON.parse(readFileSync(join(REF, 'cases.json'), 'utf8')) as Case[];
const u8 = (stem: string, kind: string) => new Uint8Array(readFileSync(join(REF, `${stem}.${kind}.u8`)));
let exact = 0, total = 0, worst = 0;
for (const c of cases) {
  const r = runM3({ vessel: u8(c.stem, 'vessel'), arteryVein: u8(c.stem, 'av'), discCup: u8(c.stem, 'disc'), radius: c.radius });
  const want = c.expected!;
  const table = r.centredOn === 'optic disc' ? 'Disc_Features.csv' : 'Macular_Features.csv';
  if (table !== want.table) fail(`${c.stem}: table ${table}, official ${want.table}`);
  const diffs: string[] = [];
  for (const [k, w] of Object.entries(want.values)) {
    const g = c.gradable || k.startsWith('Disc_') || k.startsWith('Cup_') || k.startsWith('CDR_') ? r.features[k] : null;
    total++;
    if (g === w || (g !== null && w !== null && Object.is(g, w))) { exact++; continue; }
    const rel = g === null || w === null ? Infinity : Math.abs(g - w) / Math.max(Math.abs(w), 1e-300);
    worst = Math.max(worst, rel);
    diffs.push(`${k}: ${g} vs ${w}`);
  }
  console.log(`${c.stem.padEnd(28)} ${table.padEnd(20)} ${r.ms.toFixed(0).padStart(5)} ms  ${72 - diffs.length}/72 identical${r.failed.length ? `  (AutoMorph -1 rows: ${r.failed.join(', ')})` : ''}`);
  for (const d of diffs.slice(0, 6)) console.log('    ', d);
}
console.log(`features: ${exact}/${total} identical, largest relative difference ${worst.toExponential(2)}`);
if (exact !== total) fail('features differ');
process.exit(failures ? 1 : 0);
