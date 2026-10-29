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
import type { M0Geometry, RGBImage } from '../automorph/types';
import { cropFundus, fundusBounds } from '../vascx/prep';
import type { Backend, EngineInfo, EngineMessage, EngineOp, GpuChoice, ImageHandle, Manifest, ModelInfo, ModelKey, ModelNeed, ModelSet, ModelStatus, Progress, Timing } from './protocol';

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
// bytes of weights held at once (float32 once loaded): every member of AutoMorph's four models
// (about 1.7 GB), so a folder run loads each model once; VascX (0.56 GB) and TPOT come and go
const SESSION_BUDGET = 2.0 * 2 ** 30;
let manifest: Manifest | undefined;
let base = '/';
// where the .onnx files are; a static site keeps them elsewhere (GitHub Pages limits a site to 1 GB).
// Empty: models/ (AutoMorph), models/vascx/ and models/tpot/ next to the page
let modelsUrl = '';
let vascxUrl = '';
let tpotUrl = '';
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
      return [await init(req.base, req.models, req.vascx, req.tpot, req.gpu), []];
    case 'status':
      return [await status(), []];
    case 'preload':
      return [await preload(req.models, progress), []];
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
    case 'vx_prep': {
      progress({ stage: 'm0', message: 'Finding the field of view (VascX)' });
      const started = performance.now();
      const src = take(req.imageId);
      const bounds = fundusBounds(src);
      const handle = await store(cropFundus(src, bounds, VX));
      // geometry in the photo's pixels, like M0, so measurements keep their physical scale
      const r = bounds.radius;
      const geometry: M0Geometry = { centreRow: bounds.cy, centreCol: bounds.cx, radius: r, bbox: [bounds.cy - r, bounds.cx - r, 2 * r, 2 * r] };
      return [{ ...handle, bounds, geometry, ms: performance.now() - started }, [handle.preview]];
    }
    case 'vx_vessel':
    case 'vx_av':
    case 'vx_disc':
    case 'vx_fovea':
      return vascx(req.op, take(req.imageId), progress);
    case 'enhance':
      return enhance(take(req.imageId), !!req.quick, progress);
  }
}

const slash = (url = '') => (url && !url.endsWith('/') ? `${url}/` : url);

async function init(baseUrl: string, models = '', vascx = '', tpot = '', gpu: GpuChoice = 'high-performance'): Promise<EngineInfo> {
  base = baseUrl;
  modelsUrl = slash(models);
  vascxUrl = slash(vascx);
  tpotUrl = slash(tpot);
  manifest = (await (await fetch(`${base}models/manifest.json`, { cache: 'no-cache' })).json()) as Manifest;
  for (const info of Object.values(manifest.models)) if (info) info.set = 'automorph';
  // VascX and TPOT are optional: a deployment without them still runs AutoMorph
  for (const set of ['vascx', 'tpot'] as const) {
    try {
      const r = await fetch(`${base}models/${set}/manifest.json`, { cache: 'no-cache' });
      if (!r.ok) continue;
      const m = (await r.json()) as Manifest & { source?: string; revision?: string; license?: string };
      for (const [key, info] of Object.entries(m.models)) if (info) manifest.models[key as ModelKey] = { ...info, set };
      if (set === 'vascx') manifest.vascx = { source: m.source ?? '', revision: m.revision ?? '', license: m.license ?? '' };
    } catch { /* not installed here */ }
  }
  const threads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmPaths = { mjs: new URL(ortGlueUrl, self.location.href).href, wasm: new URL(ortWasmUrl, self.location.href).href };
  let adapterName = 'CPU';
  // "cpu" leaves the graphics card alone; otherwise the browser picks the faster or the power-saving one
  const webgpu = gpu === 'cpu' ? undefined : (navigator as Navigator & { gpu?: GPU }).gpu;
  if (webgpu) {
    try {
      const adapter = await webgpu.requestAdapter({ powerPreference: gpu === 'low-power' ? 'low-power' : 'high-performance' });
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
  return { backend, gpu, adapter: adapterName, threads, manifest };
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

const MISSING: Record<ModelSet, string> = {
  automorph: 'the AutoMorph models are missing; run tools/release.py fetch',
  vascx: 'the VascX models are not installed here; run tools/release.py fetch --vascx, or set VITE_VASCX_URL',
  tpot: 'the TPOT enhancement models are not installed here; run tools/export_tpot.py, or set VITE_TPOT_URL',
};

function info(key: ModelKey): ModelInfo {
  if (!manifest) throw new Error('engine not initialised');
  const m = manifest.models[key];
  if (!m) throw new Error(MISSING[key.startsWith('vx_') ? 'vascx' : key.startsWith('tpot') ? 'tpot' : 'automorph']);
  return m;
}

function part(key: ModelKey, index: number) {
  return info(key).parts[index];
}

/** Where a model file is: its pipeline's folder next to the page, or the host its URL names. */
function partUrl(key: ModelKey, file: string): string {
  switch (info(key).set) {
    case 'vascx': return `${vascxUrl || `${base}models/vascx/`}${file}`;
    case 'tpot': return `${tpotUrl || `${base}models/tpot/`}${file}`;
    default: return modelsUrl ? `${modelsUrl}${key}/${file.split('/').pop()}` : `${base}${file}`;
  }
}

/** A model file's URL, and its key in the browser's cache (a new checksum is a new file). */
function where(key: ModelKey, index: number) {
  const info = part(key, index);
  // an AutoMorph model host keeps one folder per model (quality/, vessel/, artery_vein/, disc_cup/); models/ here is flat
  const url = new URL(partUrl(key, info.file), self.location.href).href;
  return { url, cacheKey: `${url}?sha256=${info.sha256.slice(0, 16)}` };
}

async function download(key: ModelKey, index: number, progress?: (p: Progress) => void): Promise<Uint8Array> {
  const info = part(key, index);
  const { url, cacheKey } = where(key, index);
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
  try {
    await cache?.put(cacheKey, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
  } catch {
    // the browser's storage quota is full (private windows have little): use the model anyway, uncached
  }
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
      const loadedPart = { session: s, backend: used, bytes: part(key, index).bytes * (info(key).float32 ? 1 : 2) };
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

/** For each model this site has: its parts, how many are in this browser's cache, how many are ready to run. */
async function isCached(key: ModelKey, index: number): Promise<boolean> {
  if (!('caches' in self)) return false;
  return !!(await (await caches.open(CACHE_NAME)).match(where(key, index).cacheKey));
}

async function status(): Promise<ModelStatus> {
  if (!manifest) throw new Error('engine not initialised');
  const out: ModelStatus = {};
  for (const [key, info] of Object.entries(manifest.models) as Array<[ModelKey, ModelInfo | undefined]>) {
    if (!info) continue;
    let cached = 0, loaded = 0;
    for (let i = 0; i < info.parts.length; i++) {
      if (sessions.has(`${key}#${i}`)) loaded++;
      if (await isCached(key, i)) cached++;
    }
    out[key] = { parts: info.parts.length, cached, loaded };
  }
  return out;
}

/**
 * Download and prepare the models a workflow uses, before Run, so that Run starts at once. Parts are
 * made ready while they fit in the memory budget; the rest are only downloaded, which is also what
 * makes a model available without the internet later.
 */
async function preload(needs: ModelNeed[], progress: (p: Progress) => void): Promise<ModelStatus> {
  const list = needs.flatMap(({ key, parts }) => Array.from({ length: Math.min(parts, info(key).parts.length) }, (_, i) => ({ key, i })));
  let room = SESSION_BUDGET;
  for (const [n, { key, i }] of list.entries()) {
    const p = part(key, i);
    const bytes = p.bytes * (info(key).float32 ? 1 : 2);
    const say = (q: Progress) => progress({ ...q, message: `Model ${n + 1} of ${list.length} · ${info(key).label} · ${q.message}`, fraction: (n + (q.fraction ?? 0)) / list.length });
    if (bytes <= room) {
      room -= bytes;
      await session(key, i, say);
    } else if (!(await isCached(key, i))) {
      await download(key, i, say);
    }
  }
  return status();
}

/** Run a model on one image. An ensemble runs every member on the same input and sums the
 *  outputs in AutoMorph v2's order before dividing by their number (float32 throughout);
 *  "quick" runs the first member only. */
async function infer(key: ModelKey, imageId: string, build: (img: RGBImage) => ModelInput, progress: (p: Progress) => void, quick = false) {
  const img = take(imageId);
  const n = quick ? 1 : info(key).parts.length;
  const say = (i: number) => (p: Progress) => progress({ ...p, message: n > 1 ? `Model ${i + 1} of ${n} · ${p.message}` : p.message,
    fraction: n > 1 ? (i + (p.fraction ?? 0)) / n : p.fraction });
  prefetch(key, 0);
  progress({ stage: 'preprocess', message: 'Resizing and normalising' });  // (AutoMorph's four models)
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

// ---------------------------------------------------------------- one pass of one model

interface Feed { data: Float32Array; dims: number[] }

/** Run a single-file model once; its first output, as floats. */
async function runOnce(key: ModelKey, feeds: Record<string, Feed>, progress: (p: Progress) => void, message: string) {
  const loaded = await session(key, 0, progress);
  progress({ stage: 'inference', message: `${message} · ${loaded.backend === 'webgpu' ? 'GPU' : 'CPU'}` });
  const t = performance.now();
  const inputs: Record<string, ort.Tensor> = {};
  for (const [name, f] of Object.entries(feeds)) inputs[name] = new ort.Tensor('float32', f.data, f.dims);
  const results = await loaded.session.run(inputs);
  const tensor = results[loaded.session.outputNames[0]];
  const data = (await tensor.getData()) as Float32Array;
  const dims = [...tensor.dims];
  tensor.dispose();
  return { data, dims, ms: performance.now() - t, loadMs: loaded.loadMs ?? 0, backend: loaded.backend };
}

// ---------------------------------------------------------------- VascX

const VX = 1024;  // VascX works on the field of view cropped to 1024 x 1024
const VX_MEAN = [0.485, 0.456, 0.406];
const VX_STD = [0.229, 0.224, 0.225];

/** A 1024 x 1024 VascX image: as it is, or resized when another step's image comes in. */
function vxImage(img: RGBImage): RGBImage {
  return img.width === VX && img.height === VX ? img : resizePIL(img, VX, VX, 'bilinear');
}

/** cv2.resize(INTER_LINEAR) to half size on 8-bit pixels: the mean of each 2 x 2 block, rounded. */
function halve(img: RGBImage): RGBImage {
  const w = img.width >> 1, h = img.height >> 1, out = new Uint8Array(w * h * 3), src = img.data, sw = img.width;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        const a = ((2 * y) * sw + 2 * x) * 3 + c, b = ((2 * y + 1) * sw + 2 * x) * 3 + c;
        out[(y * w + x) * 3 + c] = (src[a] + src[a + 3] + src[b] + src[b + 3] + 2) >> 2;
      }
    }
  }
  return { width: w, height: h, data: out };
}

/** ImageNet normalisation, (v / 255 - mean) / std, channels first. */
function normalise(img: RGBImage): Float32Array {
  const plane = img.width * img.height, out = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    const m = VX_MEAN[c], s = VX_STD[c];
    for (let i = 0; i < plane; i++) out[c * plane + i] = (img.data[i * 3 + c] / 255 - m) / s;
  }
  return out;
}

/** Flip channels-first data upside down (axis 2) and/or left to right (axis 3); its own inverse. */
function flip(src: Float32Array, c: number, h: number, w: number, vertical: boolean, horizontal: boolean): Float32Array {
  if (!vertical && !horizontal) return src;
  const out = new Float32Array(src.length);
  for (let k = 0; k < c; k++) {
    for (let y = 0; y < h; y++) {
      const sy = vertical ? h - 1 - y : y;
      for (let x = 0; x < w; x++) out[(k * h + y) * w + x] = src[(k * h + sy) * w + (horizontal ? w - 1 - x : x)];
    }
  }
  return out;
}

/** The flips VascX averages (retinalysis-inference: tta_flips [2], [3], [2, 3], plus the image itself). */
const FLIPS: Array<[boolean, boolean]> = [[false, false], [true, false], [false, true], [true, true]];

/** MONAI's Gaussian importance map for a window: sigma = 0.125 * size, centred, at least 1e-3. */
function gaussianWindow(n: number): Float32Array {
  const g = new Float32Array(n), sigma = 0.125 * n, out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) { const x = i - (n - 1) / 2; g[i] = Math.exp((x * x) / (-2 * sigma * sigma)); }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[y * n + x] = Math.max(g[y] * g[x], 1e-3);
  return out;
}

/** Index of the largest channel per pixel (the first on ties). */
function argmax(logits: Float32Array, c: number, plane: number): Uint8Array {
  const out = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    let best = logits[i], arg = 0;
    for (let k = 1; k < c; k++) { const v = logits[k * plane + i]; if (v > best) { best = v; arg = k; } }
    out[i] = arg;
  }
  return out;
}

/** Bilinear resize of channels-first floats, half-pixel centres, edges clamped (torch align_corners=False). */
function upsample(src: Float32Array, c: number, h: number, w: number, oh: number, ow: number): Float32Array {
  const out = new Float32Array(c * oh * ow);
  const axis = (n: number, on: number) => {
    const i0 = new Int32Array(on), i1 = new Int32Array(on), f = new Float32Array(on);
    for (let o = 0; o < on; o++) {
      const s = Math.max(0, ((o + 0.5) * n) / on - 0.5), lo = Math.min(Math.floor(s), n - 1);
      i0[o] = lo; i1[o] = Math.min(lo + 1, n - 1); f[o] = s - lo;
    }
    return { i0, i1, f };
  };
  const ax = axis(w, ow), ay = axis(h, oh);
  for (let k = 0; k < c; k++) {
    const p = k * h * w;
    for (let y = 0; y < oh; y++) {
      const r0 = p + ay.i0[y] * w, r1 = p + ay.i1[y] * w, fy = ay.f[y];
      for (let x = 0; x < ow; x++) {
        const a = src[r0 + ax.i0[x]], b = src[r0 + ax.i1[x]], cc = src[r1 + ax.i0[x]], d = src[r1 + ax.i1[x]], fx = ax.f[x];
        out[(k * oh + y) * ow + x] = (a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy;
      }
    }
  }
  return out;
}

/** Keep only the largest 4-connected region of a binary mask (VascX keeps one optic disc). */
function largestRegion(mask: Uint8Array, w: number, h: number): Uint8Array {
  const label = new Int32Array(w * h), stack: number[] = [];
  let best = 0, bestSize = 0, next = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || label[i]) continue;
    next++;
    let size = 0;
    label[i] = next; stack.push(i);
    while (stack.length) {
      const j = stack.pop()!, x = j % w, y = (j - x) / w;
      size++;
      for (const k of [x > 0 ? j - 1 : -1, x < w - 1 ? j + 1 : -1, y > 0 ? j - w : -1, y < h - 1 ? j + w : -1]) {
        if (k >= 0 && mask[k] && !label[k]) { label[k] = next; stack.push(k); }
      }
    }
    if (size > bestSize) { bestSize = size; best = next; }
  }
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < out.length; i++) out[i] = best && label[i] === best ? 1 : 0;
  return out;
}

async function vascx(op: 'vx_vessel' | 'vx_av' | 'vx_disc' | 'vx_fovea', src: RGBImage, progress: (p: Progress) => void): Promise<[unknown, Transferable[]]> {
  const t0 = performance.now();
  const img = vxImage(src);
  let inferenceMs = 0, loadMs = 0, used: Backend = backend;
  const run = async (key: ModelKey, data: Float32Array, dims: number[], message: string) => {
    const r = await runOnce(key, { image: { data, dims } }, progress, message);
    inferenceMs += r.ms; loadMs += r.loadMs; used = r.backend;
    return r.data;  // [1, batch, C, H, W]: the leading axis is the (single) ensemble member
  };
  const timing = (pre: number, post: number): Timing => ({ preprocessMs: pre, inferenceMs, postprocessMs: post, loadMs: loadMs || undefined, backend: used });

  if (op === 'vx_vessel' || op === 'vx_av') {
    const C = op === 'vx_vessel' ? 2 : 4, plane = VX * VX;
    progress({ stage: 'preprocess', message: 'Normalising' });
    const x = normalise(img);
    const pre = performance.now() - t0;
    const total = new Float32Array(C * plane);
    for (const [fi, [v, h]] of FLIPS.entries()) {
      const xi = flip(x, 3, VX, VX, v, h);
      let logits: Float32Array;
      if (op === 'vx_vessel') {
        // the whole 1024 image is one window
        logits = await run('vx_vessel', xi, [1, 3, VX, VX], `Flip ${fi + 1} of 4`);
      } else {
        // 9 windows of 512 with stride 256, blended with a Gaussian, 3 windows per run
        const P = 512, starts = [0, 256, 512], wts = gaussianWindow(P);
        const acc = new Float32Array(C * plane), count = new Float32Array(plane);
        const tiles = starts.flatMap((ty) => starts.map((tx) => [ty, tx] as const));
        for (let b = 0; b < tiles.length; b += 3) {
          const group = tiles.slice(b, b + 3), batch = new Float32Array(group.length * 3 * P * P);
          group.forEach(([ty, tx], t) => {
            for (let k = 0; k < 3; k++) for (let y = 0; y < P; y++) {
              const from = (k * VX + ty + y) * VX + tx;
              batch.set(xi.subarray(from, from + P), ((t * 3 + k) * P + y) * P);
            }
          });
          const out = await run('vx_av', batch, [group.length, 3, P, P], `Flip ${fi + 1} of 4 · windows ${b + 1}–${b + group.length} of 9`);
          group.forEach(([ty, tx], t) => {
            for (let y = 0; y < P; y++) for (let xx = 0; xx < P; xx++) {
              const wv = wts[y * P + xx], o = (ty + y) * VX + tx + xx;
              count[o] += wv;
              for (let k = 0; k < C; k++) acc[k * plane + o] += wv * out[((t * C + k) * P + y) * P + xx];
            }
          });
        }
        for (let k = 0; k < C; k++) for (let i = 0; i < plane; i++) acc[k * plane + i] /= count[i];
        logits = acc;
      }
      const back = flip(logits, C, VX, VX, v, h);
      for (let i = 0; i < total.length; i++) total[i] += back[i];
    }
    const t1 = performance.now();
    for (let i = 0; i < total.length; i++) total[i] /= FLIPS.length;
    const labels = argmax(total, C, plane);
    const post = performance.now() - t1;
    if (op === 'vx_vessel') return [{ size: VX, mask: labels, timing: timing(pre, post) }, [labels.buffer]];
    // VascX's classes are AutoMorph's: 0 background, 1 artery, 2 vein, 3 both
    return [{ map: { width: VX, height: VX, labels }, timing: timing(pre, post) }, [labels.buffer]];
  }

  // disc and fovea: the 1024 image halved to 512, one window
  progress({ stage: 'preprocess', message: 'Resizing to 512 and normalising' });
  const small = normalise(halve(img));
  const pre = performance.now() - t0;
  const S = 512;
  if (op === 'vx_disc') {
    const logits = await run('vx_disc', small, [1, 3, S, S], 'Optic disc');
    const t1 = performance.now();
    const probs = new Float32Array(2 * S * S);
    for (let i = 0; i < S * S; i++) {
      const a = logits[i], b = logits[S * S + i], m = Math.max(a, b), ea = Math.exp(a - m), eb = Math.exp(b - m);
      probs[i] = ea / (ea + eb);
      probs[S * S + i] = eb / (ea + eb);
    }
    const up = upsample(probs, 2, S, S, VX, VX);
    const labels = largestRegion(argmax(up, 2, VX * VX), VX, VX);  // 1 = disc, as AutoMorph's disc label
    return [{ map: { width: VX, height: VX, labels }, timing: timing(pre, performance.now() - t1) }, [labels.buffer]];
  }
  const a = await run('vx_fovea', small, [1, 3, S, S], 'Fovea');
  const b = flip(await run('vx_fovea', flip(small, 3, S, S, false, true), [1, 3, S, S], 'Fovea, flipped'), 1, S, S, false, true);
  const t1 = performance.now();
  let best = -Infinity, at = 0;
  for (let i = 0; i < S * S; i++) { const v = (a[i] + b[i]) / 2; if (v > best) { best = v; at = i; } }
  const col = at % S, row = (at - col) / S;
  return [{ x: (col + 0.5) * (VX / S), y: (row + 0.5) * (VX / S), size: VX, timing: timing(pre, performance.now() - t1) }, []];
}

// ---------------------------------------------------------------- TPOT enhancement

const TPOT = 256;      // TPOT enhances at 256 x 256
const TPOT_HR = 2048;  // the fusion head restores detail at 2048 x 2048
const TILE = 512;      // fusion tiles, with HALO pixels of context: 5 convolutions of 3 x 3
const HALO = 5;

const toCHW = (img: RGBImage): Float32Array => {
  const plane = img.width * img.height, out = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) for (let i = 0; i < plane; i++) out[c * plane + i] = img.data[i * 3 + c] / 255;
  return out;
};

/** [0, 1] floats to 8 bits the way TPOT-SR's enhance.py saves them: (x * 255).astype(uint8). */
const fromCHW = (src: Float32Array, side: number): RGBImage => {
  const plane = side * side, data = new Uint8Array(3 * plane);
  for (let c = 0; c < 3; c++) for (let i = 0; i < plane; i++) data[i * 3 + c] = Math.min(255, Math.max(0, Math.floor(src[c * plane + i] * 255)));
  return { width: side, height: side, data };
};

/**
 * TPOT-SR's enhance.py on the square M0 image: TPOT at 256, then (unless quick) the fusion head at
 * 2048 with the photo, and the result resized back to the M0 size (bicubic), so every later step
 * sees the same size and field of view as without enhancement.
 */
async function enhance(src: RGBImage, quick: boolean, progress: (p: Progress) => void): Promise<[unknown, Transferable[]]> {
  const t0 = performance.now();
  progress({ stage: 'enhance', message: 'Resizing to 256' });
  const lr = toCHW(resizePIL(src, TPOT, TPOT, 'bicubic'));
  const pre = performance.now() - t0;
  const first = await runOnce('tpot', { image: { data: lr, dims: [1, 3, TPOT, TPOT] } }, progress, 'TPOT at 256');
  let inferenceMs = first.ms, loadMs = first.loadMs;
  let side = TPOT, result: Float32Array = first.data;
  if (!quick) {
    progress({ stage: 'enhance', message: 'Resizing to 2048' });
    const hr = toCHW(resizePIL(src, TPOT_HR, TPOT_HR, 'bicubic'));
    const ehr = upsample(first.data, 3, TPOT, TPOT, TPOT_HR, TPOT_HR);
    const out = new Float32Array(3 * TPOT_HR * TPOT_HR);
    const n = TPOT_HR / TILE;
    for (let ty = 0; ty < n; ty++) {
      for (let tx = 0; tx < n; tx++) {
        const y0 = Math.max(0, ty * TILE - HALO), y1 = Math.min(TPOT_HR, (ty + 1) * TILE + HALO);
        const x0 = Math.max(0, tx * TILE - HALO), x1 = Math.min(TPOT_HR, (tx + 1) * TILE + HALO);
        const th = y1 - y0, tw = x1 - x0, xt = new Float32Array(3 * th * tw), et = new Float32Array(3 * th * tw);
        for (let k = 0; k < 3; k++) for (let y = 0; y < th; y++) {
          const from = (k * TPOT_HR + y0 + y) * TPOT_HR + x0;
          xt.set(hr.subarray(from, from + tw), (k * th + y) * tw);
          et.set(ehr.subarray(from, from + tw), (k * th + y) * tw);
        }
        const r = await runOnce('tpot_fusion', { image: { data: xt, dims: [1, 3, th, tw] }, enhanced: { data: et, dims: [1, 3, th, tw] } },
          progress, `Full resolution, tile ${ty * n + tx + 1} of ${n * n}`);
        inferenceMs += r.ms; loadMs += r.loadMs;
        // keep the tile's own pixels; the halo only gave the convolutions their context
        for (let k = 0; k < 3; k++) for (let y = ty * TILE; y < (ty + 1) * TILE; y++) {
          const from = (k * th + y - y0) * tw + tx * TILE - x0;
          out.set(r.data.subarray(from, from + TILE), (k * TPOT_HR + y) * TPOT_HR + tx * TILE);
        }
      }
    }
    side = TPOT_HR;
    result = out;
  }
  const t1 = performance.now();
  progress({ stage: 'postprocess', message: 'Resizing back to the photo' });
  const enhanced = resizePIL(fromCHW(result, side), src.width, src.height, 'bicubic');
  const handle = await store(enhanced);
  const timing: Timing = { preprocessMs: pre, inferenceMs, postprocessMs: performance.now() - t1, loadMs: loadMs || undefined, backend: first.backend };
  return [{ ...handle, ms: performance.now() - t0, timing }, [handle.preview]];
}
