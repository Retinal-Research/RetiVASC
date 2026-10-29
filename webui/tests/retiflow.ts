// RetiFlow in the browser's terms (src/retiflow) against the Python original, step by step, each step fed
// the Python result of the step before it, then the whole of process_prob.
//   1. npx tsx tests/retiflow.ts      writes tools/retiflow_ref/av3.rgb: RRWNet's map of the RRWNet sample
//   2. python tools/retiflow_reference.py <folder holding the RetiFlow package>
//   3. npx tsx tests/retiflow.ts      compares
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Bifurcation } from '../src/engine/values';
import { processProb, rootFromBv } from '../src/retiflow';
import { bridgeEndpoints } from '../src/retiflow/bridge';
import { extractCenterline } from '../src/retiflow/centerline';
import { completeProbabilityMasks } from '../src/retiflow/completion';
import { runOriginal } from '../src/retiflow/rbad';
import { av3, circleMask, rrwnetInput, RRW_SIDE } from '../src/rrwnet/prep';

const ROOT = join(import.meta.dirname, '..', '..');
const REF = join(ROOT, 'tools', 'retiflow_ref');
const N = RRW_SIDE;
const f = Math.fround;

if (!existsSync(join(REF, 'result.json'))) {
  // the AV3 map of RRWNet's PyTorch probabilities for its sample (tools/export_rrwnet.py)
  const RRW = join(ROOT, 'tools', 'rrwnet_ref');
  const b = readFileSync(join(RRW, 'probs.f32'));
  const probs = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const input = rrwnetInput({ width: N, height: N, data: new Uint8Array(readFileSync(join(RRW, 'input.rgb'))) }, circleMask(N));
  mkdirSync(REF, { recursive: true });
  writeFileSync(join(REF, 'av3.rgb'), av3(probs, input));
  console.log('wrote tools/retiflow_ref/av3.rgb; now: python tools/retiflow_reference.py <folder holding RetiFlow>');
  process.exit(0);
}

const map = new Uint8Array(readFileSync(join(REF, 'av3.rgb')));
const u8 = (name: string) => new Uint8Array(readFileSync(join(REF, name)));
const b = readFileSync(join(REF, 'conditional_a.f32'));
const refCa = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
type Found = { x: number; y: number; angle: number; children: Array<[number, number]> };
const result = JSON.parse(readFileSync(join(REF, 'result.json'), 'utf8')) as {
  root: [number, number];
  a: { before: Found[]; after: Found[]; root: [number, number] };
  v: { before: Found[]; after: Found[]; root: [number, number] };
  summary: { RBAD: Record<'A' | 'V', { count: number; angles: number[] }> };
};

let failed = 0;
const check = (what: string, problem: string) => {
  if (problem) failed++;
  console.log(`${problem ? 'FAIL' : 'PASS'}  ${what}${problem ? `: ${problem}` : ''}`);
};
const pixels = (ts: Uint8Array, py: Uint8Array) => {
  let n = 0, on = 0;
  for (let i = 0; i < ts.length; i++) { if (!!ts[i] !== !!py[i]) n++; if (py[i]) on++; }
  return n ? `${n} pixels differ (of ${on})` : '';
};
const points = (ts: Bifurcation[], py: Found[], tolerance = 1e-9) => {
  if (ts.length !== py.length) return `${ts.length} bifurcations, Python ${py.length}`;
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i], p = py[i];
    if (t.x !== p.x || t.y !== p.y || Math.abs(t.angle - p.angle) > tolerance || JSON.stringify(t.children) !== JSON.stringify(p.children)) {
      return `#${i}: (${t.x}, ${t.y}) ${t.angle} against (${p.x}, ${p.y}) ${p.angle}`;
    }
  }
  return '';
};
const timed = <T>(run: () => T): [T, string] => {
  const t = performance.now();
  const r = run();
  return [r, `${((performance.now() - t) / 1000).toFixed(2)} s`];
};

const a = new Float32Array(N * N), v = new Float32Array(N * N), bv = new Float32Array(N * N);
for (let i = 0; i < N * N; i++) { a[i] = f(map[3 * i] / 255); v[i] = f(map[3 * i + 1] / 255); bv[i] = f(map[3 * i + 2] / 255); }

const [masks, tc] = timed(() => completeProbabilityMasks(a, v, bv, N, N));
let d = 0;
for (let i = 0; i < refCa.length; i++) d = Math.max(d, Math.abs(masks.conditionalA[i] - refCa[i]));
check('completion: conditional artery probability', d ? `max difference ${d}` : '');
check(`completion: artery mask (${tc})`, pixels(masks.a, u8('a_mask.u8')));
check('completion: vein mask', pixels(masks.v, u8('v_mask.u8')));

const pa = new Float64Array(N * N), pv = new Float64Array(N * N);
for (let i = 0; i < N * N; i++) {
  pa[i] = f(bv[i] * f(f(0.65) + f(f(0.35) * refCa[i])));
  pv[i] = f(bv[i] * f(f(0.65) + f(f(0.35) * f(1 - refCa[i]))));
}
const [sa, ts] = timed(() => extractCenterline(pa, u8('a_mask.u8'), N, N));
check(`centreline: arteries (${ts})`, pixels(sa, u8('a_centerline.u8')));
check('centreline: veins', pixels(extractCenterline(pv, u8('v_mask.u8'), N, N), u8('v_centerline.u8')));

const root = rootFromBv(bv, N, N);
check('root', root.join() === result.root.join() ? '' : `[${root}] against [${result.root}]`);

const support = Float64Array.from(bv);
for (const name of ['a', 'v'] as const) {
  const skel = u8(`${name}_centerline.u8`), ref = result[name];
  const before = runOriginal(skel, N, N, result.root);
  check(`RBAD before repair, ${name === 'a' ? 'arteries' : 'veins'}`, points(before.angles, ref.before));
  const [repaired, tb] = timed(() => bridgeEndpoints(skel, support, N, N));
  check(`endpoint repair (${tb})`, pixels(repaired, u8(`${name}_repaired.u8`)));
  check('RBAD after repair', points(runOriginal(u8(`${name}_repaired.u8`), N, N, ref.root).angles, ref.after));
}

const [all, tp] = timed(() => processProb(map, N, N));
for (const [name, mine] of [['A', all.artery], ['V', all.vein]] as const) {
  const py = result.summary.RBAD[name];
  const ours = mine.points.map((p) => Math.round(p.angle * 1000) / 1000);
  const same = ours.length === py.angles.length && ours.every((x, i) => Math.abs(x - py.angles[i]) <= 1e-3);
  check(`process_prob, ${name} (${tp} in all)`, same ? '' : `${ours.length} angles against ${py.count}`);
  console.log(`      ${mine.points.length} bifurcations, mean ${(ours.reduce((s, x) => s + x, 0) / (ours.length || 1)).toFixed(2)} degrees`);
}
process.exit(failed ? 1 : 0);
