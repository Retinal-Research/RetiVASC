// Compares the browser port of AutoMorph v2 pre/post-processing with the
// Python reference written by tools/make_reference.py. Run: npm run test:parity
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { runM0 } from '../src/automorph/m0';
import { decodeArteryVein, decodeDiscCup, decodeQuality, decodeVessel } from '../src/automorph/postprocess';
import { arteryVeinInput, discCupInput, qualityInput, vesselInput, type ModelInput } from '../src/automorph/preprocess';
import { resizePIL } from '../src/automorph/resample';
import type { RGBImage } from '../src/automorph/types';

const REF = join(import.meta.dirname, '..', '..', 'tools', 'ref');

const u8 = (p: string) => new Uint8Array(readFileSync(p));
const f32 = (p: string) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

function countDiff(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

function maxAbs(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

const tasks: Array<[string, (img: RGBImage) => ModelInput, number, 'bicubic' | 'bilinear']> = [
  ['quality', qualityInput, 512, 'bicubic'],
  ['vessel', vesselInput, 912, 'bicubic'],
  ['artery_vein', arteryVeinInput, 720, 'bicubic'],
  ['disc_cup', discCupInput, 512, 'bilinear'],
];

let failures = 0;
const check = (ok: boolean, label: string) => {
  if (!ok) failures++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}`);
};

for (const stem of readdirSync(REF).sort()) {
  const dir = join(REF, stem);
  const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
  console.log(`${stem} (${meta.width}x${meta.height})`);
  const input: RGBImage = { width: meta.width, height: meta.height, data: u8(join(dir, 'input.rgb')) };

  let t = performance.now();
  const m0 = runM0(input);
  const m0ms = performance.now() - t;
  const g = m0.geometry;
  const r = meta.m0;
  check(
    g.centreRow === r.centreRow && g.centreCol === r.centreCol && g.radius === r.radius && g.bbox.join() === r.bbox.join(),
    `M0 geometry centre=(${g.centreRow}, ${g.centreCol}) r=${g.radius} bbox=${g.bbox} vs python (${r.centreRow}, ${r.centreCol}) r=${r.radius} bbox=${r.bbox}  [${m0ms.toFixed(0)} ms]`,
  );
  const refM0 = u8(join(dir, 'm0.rgb'));
  check(countDiff(m0.image.data, refM0) === 0, `M0 image ${m0.image.width}x${m0.image.height}, differing bytes: ${countDiff(m0.image.data, refM0)}`);

  // From here on use the reference M0 image so each stage is checked on its own.
  const ref: RGBImage = { width: r.side, height: r.side, data: refM0 };
  for (const [task, build, size, filter] of tasks) {
    t = performance.now();
    const resized = resizePIL(ref, size, size, filter);
    const rms = performance.now() - t;
    const dr = countDiff(resized.data, u8(join(dir, `r_${task}.rgb`)));
    const x = build(ref);
    const dx = maxAbs(x.data, f32(join(dir, `x_${task}.f32`)));
    check(dr === 0 && dx === 0, `${task.padEnd(11)} resize ${size} ${filter}: differing bytes ${dr} [${rms.toFixed(0)} ms]; model input max |diff| ${dx}`);
  }

  const q = decodeQuality(f32(join(dir, 'y_quality.f32')));
  check(q.prediction === meta.quality.prediction && q.gradable === meta.quality.gradable, `quality decode ${q.prediction} gradable=${q.gradable}`);
  const v = decodeVessel(f32(join(dir, 'y_vessel.f32')), 912);
  check(countDiff(v, u8(join(dir, 'l_vessel.u8'))) === 0, `vessel decode, differing pixels ${countDiff(v, u8(join(dir, 'l_vessel.u8')))}`);
  const av = decodeArteryVein(f32(join(dir, 'y_artery_vein.f32')), 720);
  check(countDiff(av.labels, u8(join(dir, 'l_artery_vein.u8'))) === 0, `artery/vein decode, differing pixels ${countDiff(av.labels, u8(join(dir, 'l_artery_vein.u8')))}`);
  const dc = decodeDiscCup(f32(join(dir, 'y_disc_cup.f32')), 512);
  check(countDiff(dc.labels, u8(join(dir, 'l_disc_cup.u8'))) === 0, `disc/cup decode, differing pixels ${countDiff(dc.labels, u8(join(dir, 'l_disc_cup.u8')))}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
