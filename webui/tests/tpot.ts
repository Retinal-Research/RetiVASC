// Runs the exported TPOT enhancement and its fusion head in onnxruntime-web (WebAssembly) and
// compares them with PyTorch, using the reference tensors tools/export_tpot.py writes to tools/tpot_ref.
// Run: npx tsx tests/tpot.ts
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';

const ROOT = join(import.meta.dirname, '..', '..');
const REF = join(ROOT, 'tools', 'tpot_ref');
const MODELS = join(ROOT, 'webui', 'public', 'models', 'tpot');

const f32 = (name: string) => {
  const b = readFileSync(join(REF, `${name}.f32`));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

function maxAbs(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

ort.env.wasm.numThreads = 1;
let failed = 0;
const check = (what: string, diff: number, limit: number) => {
  const ok = diff <= limit;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}: max |ONNX - PyTorch| = ${diff.toExponential(2)} (limit ${limit.toExponential(0)})`);
};

const tpot = await ort.InferenceSession.create(readFileSync(join(MODELS, 'tpot.onnx')));
const out = await tpot.run({ image: new ort.Tensor('float32', f32('x_lr'), [1, 3, 256, 256]) });
check('TPOT 256 x 256', maxAbs(out.output.data as Float32Array, f32('e_lr')), 1e-4);

const fusion = await ort.InferenceSession.create(readFileSync(join(MODELS, 'fusion.onnx')));
const tile = await fusion.run({
  image: new ort.Tensor('float32', f32('tile_x'), [1, 3, 522, 522]),
  enhanced: new ort.Tensor('float32', f32('tile_e'), [1, 3, 522, 522]),
});
check('fusion head, one 522 x 522 tile', maxAbs(tile.output.data as Float32Array, f32('tile_out')), 1e-4);

process.exit(failed ? 1 : 0);
