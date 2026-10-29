// RRWNet in the browser's terms against the Python reference (tools/export_rrwnet.py writes tools/rrwnet_ref):
// the preprocessing (src/rrwnet/prep.ts) in float64, then the ONNX model in onnxruntime-web (WebAssembly).
// Run: npx tsx tests/rrwnet.ts   (after tools/export_rrwnet.py)
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { av3, circleMask, rrwnetInput, RRW_SIDE } from '../src/rrwnet/prep';

const ROOT = join(import.meta.dirname, '..', '..');
const REF = join(ROOT, 'tools', 'rrwnet_ref');
const read = (name: string, T: Float32ArrayConstructor | Float64ArrayConstructor) => {
  const b = readFileSync(join(REF, name));
  return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

// the 584 x 584 RGB image the reference was made from (input.rgb, raw)
const rgb = new Uint8Array(readFileSync(join(REF, 'input.rgb')));
let failed = 0;
const check = (what: string, diff: number, limit: number) => {
  const ok = diff <= limit;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}: ${diff.toExponential(2)} (limit ${limit.toExponential(0)})`);
};

const mask = circleMask(RRW_SIDE);
const input = rrwnetInput({ width: RRW_SIDE, height: RRW_SIDE, data: rgb }, mask);
const enhanced = read('enhanced.f64', Float64Array);
let d = 0;
for (let i = 0; i < enhanced.length; i++) d = Math.max(d, Math.abs(enhanced[i] - input.enhanced[i]));
check('preprocessing (enhance_image), max |TS - Python|', d, 1e-9);
const x = read('x.f32', Float32Array);
d = 0;
for (let i = 0; i < x.length; i++) d = Math.max(d, Math.abs(x[i] - input.x[i]));
check('padded network input, max |TS - Python|', d, 1e-6);

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(readFileSync(join(ROOT, 'webui', 'public', 'models', 'rrwnet', 'rrwnet.onnx')));
const t = performance.now();
const out = await session.run({ image: new ort.Tensor('float32', input.x, [1, 3, input.side, input.side]) });
const probs = out.probabilities.data as Float32Array;
const expected = read('probs.f32', Float32Array);
d = 0;
for (let i = 0; i < probs.length; i++) d = Math.max(d, Math.abs(probs[i] - expected[i]));
check(`ONNX (float16 weights) vs PyTorch, ${((performance.now() - t) / 1000).toFixed(0)} s on one CPU thread`, d, 1e-2);
const map = av3(probs, input), ref = av3(expected, input);
let flips = 0;
for (let i = 0; i < map.length; i++) if ((map[i] > 127) !== (ref[i] > 127)) flips++;
console.log(`      AV3 pixels on the other side of 0.5: ${flips} of ${map.length}`);
process.exit(failed ? 1 : 0);
