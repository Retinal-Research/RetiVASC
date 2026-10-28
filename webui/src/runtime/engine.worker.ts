/// <reference lib="webworker" />
// All pixel work and inference runs here so the canvas stays responsive.
// Full-resolution images stay in this worker; the page only gets previews.

import * as ort from 'onnxruntime-web/webgpu';
// ort's glue script and wasm, emitted as separate files (see init): ort's worker threads must
// load the glue itself, never this bundle
import ortGlueUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url';
import ortWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url';
import { runM0 } from '../automorph/m0';
import { decodeArteryVein, decodeDiscCup, decodeQuality, decodeVessel } from '../automorph/postprocess';
import { arteryVeinInput, discCupInput, qualityInput, vesselInput, type ModelInput } from '../automorph/preprocess';
import { runM3 } from '../automorph/m3/m3';
import { resizePIL } from '../automorph/resample';
import type { RGBImage } from '../automorph/types';
import type { Backend, EngineInfo, EngineMessage, EngineOp, ImageHandle, Manifest, ModelKey, Progress, Timing } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

const PREVIEW_SIDE = 1024;
const CACHE_NAME = 'retivasc-studio-models';

const images = new Map<string, RGBImage>();
// Sessions are kept per part ("quality#3") up to a memory budget, least recently used out first,
// so a full ensemble also runs on a GPU that cannot hold every member at once.
interface Loaded { session: ort.InferenceSession; backend: Backend; bytes: number }
const sessions = new Map<string, Loaded>();
const loading = new Map<string, Promise<Loaded & { loadMs: number }>>();
const downloads = new Map<string, Promise<Uint8Array>>();
// bytes of weights held at once (float32 once loaded): every member of all four models (about 1.7 GB),
// so a batch loads each model once for the whole folder
const SESSION_BUDGET = 2.0 * 2 ** 30;
let manifest: Manifest | undefined;
let base = '/';
// where the .onnx files are; a static site keeps them elsewhere (GitHub Pages limits a site to 1 GB)
let modelsUrl = '';
let backend: Backend = 'wasm';
let nextImage = 1;

function post(message: EngineMessage, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}

self.onmessage = async (event: MessageEvent<{ id: number } & EngineOp>) => {
  const { id } = event.data;
  const progress = (p: Progress) => post({ id, kind: 'progress', progress: p });
  try {
    const [result, transfer] = await handle(event.data, progress);
    post({ id, kind: 'result', result }, transfer);
  } catch (error) {
    post({ id, kind: 'error', error: error instanceof Error ? error.message : String(error) });
  }
};

async function handle(req: EngineOp, progress: (p: Progress) => void): Promise<[unknown, Transferable[]]> {
  switch (req.op) {
    case 'init':
      return [await init(req.base, req.models), []];
    case 'decode': {
      progress({ stage: 'decode', message: 'Decoding image' });
      const handle = await decode(req.bytes);
      return [handle, [handle.preview]];
    }
    case 'm0': {
      progress({ stage: 'm0', message: 'Finding the field of view' });
      const started = performance.now();
      const { image, geometry } = runM0(take(req.imageId));
      const handle = await store(image);
      return [{ ...handle, geometry, ms: performance.now() - started }, [handle.preview]];
    }
    case 'quality': {
      const run = await infer('quality', req.imageId, qualityInput, progress, req.quick);
      const t = performance.now();
      const result = decodeQuality(run.output);
      return [{ result, timing: { ...run.timing, postprocessMs: performance.now() - t } }, []];
    }
    case 'vessel': {
      const run = await infer('vessel', req.imageId, vesselInput, progress, req.quick);
      progress({ stage: 'postprocess', message: 'Removing specks' });
      const t = performance.now();
      const mask = decodeVessel(run.output, 912);
      return [{ size: 912, mask, timing: { ...run.timing, postprocessMs: performance.now() - t } }, [mask.buffer]];
    }
    case 'artery_vein': {
      const run = await infer('artery_vein', req.imageId, arteryVeinInput, progress, req.quick);
      progress({ stage: 'postprocess', message: 'Labelling arteries and veins' });
      const t = performance.now();
      const map = decodeArteryVein(run.output, 720);
      return [{ map, timing: { ...run.timing, postprocessMs: performance.now() - t } }, [map.labels.buffer]];
    }
    case 'disc_cup': {
      const run = await infer('disc_cup', req.imageId, discCupInput, progress);
      const t = performance.now();
      const map = decodeDiscCup(run.output, 512);
      return [{ map, timing: { ...run.timing, postprocessMs: performance.now() - t } }, [map.labels.buffer]];
    }
    case 'm3': {
      progress({ stage: 'measure', message: 'Skeletons, zones and 72 vascular features' });
      return [runM3({ vessel: req.vessel, arteryVein: req.arteryVein, discCup: req.discCup, radius: req.radius }), []];
    }
    case 'release':
      for (const id of req.ids) images.delete(id);
      return [null, []];
  }
}

async function init(baseUrl: string, models = ''): Promise<EngineInfo> {
  base = baseUrl;
  modelsUrl = models && !models.endsWith('/') ? `${models}/` : models;
  manifest = (await (await fetch(`${base}models/manifest.json`, { cache: 'no-cache' })).json()) as Manifest;
  const threads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmPaths = { mjs: new URL(ortGlueUrl, self.location.href).href, wasm: new URL(ortWasmUrl, self.location.href).href };
  let adapterName = 'CPU';
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        backend = 'webgpu';
        const info = adapter.info;
        // browsers usually hide the exact model; vendor + architecture is what is left
        const title = (s = '') => (['nvidia', 'amd', 'arm'].includes(s) ? s.toUpperCase() : s.charAt(0).toUpperCase() + s.slice(1));
        const vendor = title(info?.vendor);
        const arch = title(info?.architecture);
        adapterName = info?.description || [vendor, arch].filter(Boolean).join(' ') || 'WebGPU adapter';
        (ort.env as unknown as { webgpu: { adapter?: GPUAdapter } }).webgpu.adapter = adapter;
      }
    } catch {
      backend = 'wasm';
    }
  }
  return { backend, adapter: adapterName, threads, manifest };
}

function take(imageId: string): RGBImage {
  const img = images.get(imageId);
  if (!img) throw new Error('image is no longer in memory; run the workflow again');
  return img;
}

async function previewOf(img: RGBImage): Promise<ImageBitmap> {
  const scale = Math.min(1, PREVIEW_SIDE / Math.max(img.width, img.height));
  const small = scale < 1 ? resizePIL(img, Math.round(img.width * scale), Math.round(img.height * scale), 'bilinear') : img;
  const rgba = new Uint8ClampedArray(small.width * small.height * 4);
  for (let i = 0, p = 0; p < small.data.length; i += 4, p += 3) {
    rgba[i] = small.data[p];
    rgba[i + 1] = small.data[p + 1];
    rgba[i + 2] = small.data[p + 2];
    rgba[i + 3] = 255;
  }
  return createImageBitmap(new ImageData(rgba, small.width, small.height));
}

async function store(img: RGBImage): Promise<ImageHandle> {
  const imageId = `img${nextImage++}`;
  images.set(imageId, img);
  return { imageId, width: img.width, height: img.height, preview: await previewOf(img) };
}

async function decode(bytes: ArrayBuffer): Promise<ImageHandle> {
  // No colour management: AutoMorph reads raw pixel values with OpenCV.
  const bitmap = await createImageBitmap(new Blob([bytes]), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas is not available in this browser');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const rgb = new Uint8Array(canvas.width * canvas.height * 3);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 3) {
    rgb[p] = rgba[i];
    rgb[p + 1] = rgba[i + 1];
    rgb[p + 2] = rgba[i + 2];
  }
  return store({ width: canvas.width, height: canvas.height, data: rgb });
}

function part(key: ModelKey, index: number) {
  if (!manifest) throw new Error('engine not initialised');
  return manifest.models[key].parts[index];
}

async function download(key: ModelKey, index: number, progress?: (p: Progress) => void): Promise<Uint8Array> {
  const info = part(key, index);
  // a model host keeps one folder per model (quality/, vessel/, artery_vein/, disc_cup/); models/ here is flat
  const url = new URL(modelsUrl ? `${modelsUrl}${key}/${info.file.split('/').pop()}` : `${base}${info.file}`, self.location.href).href;
  const cacheKey = `${url}?sha256=${info.sha256.slice(0, 16)}`;
  const cache = 'caches' in self ? await caches.open(CACHE_NAME) : undefined;
  const hit = await cache?.match(cacheKey);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`model download failed (${response.status})`);
  const total = info.bytes;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    const now = performance.now();
    if (progress && now - lastReport > 100) {
      lastReport = now;
      progress({ stage: 'download', message: `Downloading ${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB`, fraction: loaded / total });
    }
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.byteLength; }
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) => b.toString(16).padStart(2, '0')).join('');
  if (digest !== info.sha256) throw new Error(`model file ${info.file} does not match its manifest checksum`);
  await cache?.put(cacheKey, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
  return bytes;
}

/** Start downloading a part in the background (the next member while this one runs). */
function prefetch(key: ModelKey, index: number) {
  const id = `${key}#${index}`;
  if (sessions.has(id) || loading.has(id) || downloads.has(id)) return;
  const d = download(key, index);
  downloads.set(id, d);
  d.catch(() => downloads.delete(id));  // a failed prefetch is retried when the part is needed
}

function evict(keep: string) {
  let held = 0;
  for (const s of sessions.values()) held += s.bytes;
  for (const [id, s] of sessions) {  // Map order = least recently used first
    if (held <= SESSION_BUDGET) break;
    if (id === keep) continue;
    void s.session.release();
    sessions.delete(id);
    held -= s.bytes;
  }
}

async function session(key: ModelKey, index: number, progress: (p: Progress) => void) {
  const id = `${key}#${index}`;
  const ready = sessions.get(id);
  if (ready) {
    sessions.delete(id);
    sessions.set(id, ready);  // most recently used
    return { ...ready, loadMs: undefined };
  }
  let pending = loading.get(id);
  if (!pending) {
    pending = (async () => {
      const started = performance.now();
      const bytes = await (downloads.get(id) ?? download(key, index, progress));
      downloads.delete(id);
      progress({ stage: 'compile', message: backend === 'webgpu' ? 'Preparing the model on the GPU' : 'Preparing the model' });
      let used: Backend = backend;
      let s: ort.InferenceSession;
      try {
        s = await ort.InferenceSession.create(bytes, { executionProviders: [used], graphOptimizationLevel: 'all' });
      } catch (error) {
        if (used !== 'webgpu') throw error;
        used = 'wasm';
        s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      }
      const loadedPart = { session: s, backend: used, bytes: part(key, index).bytes * 2 };
      sessions.set(id, loadedPart);
      evict(id);
      return { ...loadedPart, loadMs: performance.now() - started };
    })();
    loading.set(id, pending);
    const clear = () => loading.delete(id);
    pending.then(clear, clear);
  }
  return pending;
}

/** Run a model on one image. An ensemble runs every member on the same input and sums the
 *  outputs in AutoMorph v2's order before dividing by their number (float32 throughout);
 *  "quick" runs the first member only. */
async function infer(key: ModelKey, imageId: string, build: (img: RGBImage) => ModelInput, progress: (p: Progress) => void, quick = false) {
  const img = take(imageId);
  const n = quick ? 1 : manifest!.models[key].parts.length;
  const say = (i: number) => (p: Progress) => progress({ ...p, message: n > 1 ? `Model ${i + 1} of ${n} · ${p.message}` : p.message,
    fraction: n > 1 ? (i + (p.fraction ?? 0)) / n : p.fraction });
  prefetch(key, 0);
  progress({ stage: 'preprocess', message: 'Resizing and normalising' });
  let t = performance.now();
  const input = build(img);
  const preprocessMs = performance.now() - t;
  let inferenceMs = 0;
  let loadMs = 0;
  let used: Backend = backend;
  let sum: Float32Array | undefined;
  for (let i = 0; i < n; i++) {
    if (i + 1 < n) prefetch(key, i + 1);  // download the next member while this one runs
    const loaded = await session(key, i, say(i));
    loadMs += loaded.loadMs ?? 0;
    used = loaded.backend;
    say(i)({ stage: 'inference', message: loaded.backend === 'webgpu' ? 'Running on the GPU' : 'Running on the CPU', fraction: 0.5 });
    t = performance.now();
    const results = await loaded.session.run({ input: new ort.Tensor('float32', input.data, input.dims) });
    const tensor = results.output;
    const out = (await tensor.getData()) as Float32Array;
    tensor.dispose();
    inferenceMs += performance.now() - t;
    if (!sum) sum = Float32Array.from(out);
    else for (let j = 0; j < out.length; j++) sum[j] += out[j];  // Float32Array rounds each step to float32, as torch does
  }
  if (n > 1) for (let j = 0; j < sum!.length; j++) sum![j] /= n;
  const timing: Timing = { preprocessMs, inferenceMs, postprocessMs: 0, loadMs: loadMs || undefined, backend: used };
  return { output: sum!, timing };
}
