/// <reference lib="webworker" />
// The language model behind the AI report step, in a worker of its own so that the image steps keep
// running while it writes: Qwen3-1.7B with 4-bit weights in onnxruntime-web, on the graphics card when
// there is one. A conversation keeps the model's attention cache, so a follow-up question only reads
// the new words. Nothing leaves this computer.

import * as ort from 'onnxruntime-web/webgpu';
// the same glue script and wasm as the engine worker (see engine.worker.ts)
import ortGlueUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url';
import ortWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url';
import type { Backend, GpuChoice, Progress } from '../runtime/protocol';
import type { ChatRequest, ChatResult, LlmFile, LlmInfo, LlmManifest, LlmMessage, LlmOp, LlmStatus } from './protocol';
import { splitWeights } from './onnxsplit';
import { Tokenizer } from './tokenizer';

declare const self: DedicatedWorkerGlobalScope;

const CACHE_NAME = 'retivasc-studio-models';  // the image models' cache: one place to clear
// prompt tokens per pass; each pass also makes 151936 logits per token, which are thrown away
const CHUNK = 128;
// sampling as Qwen3 recommends without thinking, with a fixed seed: the same results give the same text
const TOP_K = 20, TOP_P = 0.8, REPEAT = 1.05;

let manifest: LlmManifest | undefined;
let base = '/';
let url = '';
let backend: Backend = 'wasm';
let adapterName = 'CPU';
let session: ort.InferenceSession | undefined;
let tokenizer: Tokenizer | undefined;
let loading: Promise<void> | undefined;
// every conversation's tokens so far, to read again if its cache was given to another one
const histories = new Map<string, number[]>();
// the attention cache (keys and values of every layer) of one conversation, covering exactly `ids`
let kv: { chat: string; ids: number[]; past: ort.Tensor[] } | undefined;
let stopRequested = false;
// one load or answer at a time, in the order asked
let queue: Promise<unknown> = Promise.resolve();
const exclusive = <T>(work: () => Promise<T>): Promise<T> => {
  const run = queue.then(work, work);
  queue = run.catch(() => undefined);
  return run;
};

function post(message: LlmMessage) {
  self.postMessage(message);
}

self.onmessage = async (event: MessageEvent<{ id: number } & LlmOp>) => {
  const req = event.data;
  const progress = (p: Progress) => post({ id: req.id, kind: 'progress', progress: p });
  try {
    let result: unknown;
    switch (req.op) {
      case 'init': result = await init(req.base, req.url, req.gpu); break;
      case 'status': result = await status(); break;
      case 'load': result = await exclusive(() => load(progress)); break;
      case 'chat': result = await exclusive(() => chat(req.req, progress, (text) => post({ id: req.id, kind: 'token', text }))); break;
      case 'stop': stopRequested = true; result = null; break;
    }
    post({ id: req.id, kind: 'result', result });
  } catch (error) {
    post({ id: req.id, kind: 'error', error: error instanceof Error ? error.message : String(error) });
  }
};

const slash = (u = '') => (u && !u.endsWith('/') ? `${u}/` : u);

async function init(baseUrl: string, modelUrl: string | undefined, gpu: GpuChoice): Promise<LlmInfo> {
  base = baseUrl;
  url = slash(modelUrl);
  const r = await fetch(`${base}models/llm/manifest.json`, { cache: 'no-cache' });
  if (!r.ok) throw new Error('the language model is not installed here; run tools/release.py fetch --llm');
  manifest = (await r.json()) as LlmManifest;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  ort.env.wasm.wasmPaths = { mjs: new URL(ortGlueUrl, self.location.href).href, wasm: new URL(ortWasmUrl, self.location.href).href };
  const webgpu = gpu === 'cpu' ? undefined : (navigator as Navigator & { gpu?: GPU }).gpu;
  if (webgpu) {
    try {
      const adapter = await webgpu.requestAdapter({ powerPreference: gpu === 'low-power' ? 'low-power' : 'high-performance' });
      if (adapter) {
        backend = 'webgpu';
        adapterName = adapter.info?.description || [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ') || 'WebGPU adapter';
        (ort.env as unknown as { webgpu: { adapter?: GPUAdapter } }).webgpu.adapter = adapter;
      }
    } catch {
      backend = 'wasm';
    }
  }
  return { manifest, backend, adapter: adapterName };
}

function where(f: LlmFile) {
  const href = new URL(`${url || `${base}models/llm/`}${f.file}`, self.location.href).href;
  return { href, cacheKey: `${href}?sha256=${f.sha256.slice(0, 16)}` };
}

async function cached(f: LlmFile): Promise<Response | undefined> {
  if (!('caches' in self)) return undefined;
  return (await caches.open(CACHE_NAME)).match(where(f).cacheKey);
}

async function status(): Promise<LlmStatus> {
  if (!manifest) throw new Error('language model not initialised');
  return { cached: !!(await cached(manifest.files.model)) && !!(await cached(manifest.files.tokenizer)), loaded: !!session };
}

/** A file from this browser's cache, or downloaded (into a buffer of the known size: the model is 1.4 GB) and checked. */
async function download(f: LlmFile, what: string, progress: (p: Progress) => void): Promise<Uint8Array> {
  const hit = await cached(f);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const { href, cacheKey } = where(f);
  const response = await fetch(href);
  if (!response.ok || !response.body) throw new Error(`${what} download failed (${response.status})`);
  const bytes = new Uint8Array(f.bytes);
  const reader = response.body.getReader();
  let loaded = 0, lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.byteLength > f.bytes) throw new Error(`${what} is larger than its manifest says`);
    bytes.set(value, loaded);
    loaded += value.byteLength;
    const now = performance.now();
    if (now - lastReport > 150) {
      lastReport = now;
      progress({ stage: 'download', message: `Downloading the ${what} · ${(loaded / 2 ** 20).toFixed(0)} / ${(f.bytes / 2 ** 20).toFixed(0)} MB`, fraction: loaded / f.bytes });
    }
  }
  if (loaded !== f.bytes) throw new Error(`${what} download was cut short`);
  progress({ stage: 'download', message: `Checking the ${what}` });
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) => b.toString(16).padStart(2, '0')).join('');
  if (digest !== f.sha256) throw new Error(`the ${what} does not match its manifest checksum`);
  try {
    await (await caches.open(CACHE_NAME)).put(cacheKey, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
  } catch {
    // no room in the browser's storage: use it anyway, uncached
  }
  return bytes;
}

async function load(progress: (p: Progress) => void): Promise<LlmStatus> {
  if (!manifest) throw new Error('language model not initialised');
  if (!session) {
    const m = manifest;
    loading ??= (async () => {
      const json = await download(m.files.tokenizer, 'tokenizer', progress);
      tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(json)));
      let model: ReturnType<typeof splitWeights> | undefined = splitWeights(await download(m.files.model, 'language model', progress));
      progress({ stage: 'compile', message: backend === 'webgpu' ? 'Preparing the language model on the GPU' : 'Preparing the language model' });
      const options: ort.InferenceSession.SessionOptions = { executionProviders: [backend], graphOptimizationLevel: 'all', externalData: model.files };
      // keep the attention cache (and the logits, until read) on the graphics card between passes
      if (backend === 'webgpu') options.preferredOutputLocation = 'gpu-buffer';
      try {
        session = await ort.InferenceSession.create(model.graph, options);
      } catch (error) {
        if (backend !== 'webgpu') throw error;
        backend = 'wasm';
        session = await ort.InferenceSession.create(model.graph, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', externalData: model.files });
      }
      model = undefined;  // the weights are on the graphics card (or in onnxruntime's memory) now
    })();
    try { await loading; } finally { loading = undefined; }
  }
  return status();
}

function emptyPast(): ort.Tensor[] {
  const { layers, kvHeads, headDim } = manifest!.config;
  return Array.from({ length: 2 * layers }, () => new ort.Tensor('float16', new Uint16Array(0), [1, kvHeads, 0, headDim]));
}

function release(tensors: ort.Tensor[]) {
  for (const t of tensors) if (t.location === 'gpu-buffer') t.dispose();
}

/** One pass of the model over `ids`, which follow the `pos` tokens the cache holds. */
async function pass(ids: number[], pos: number, past: ort.Tensor[]) {
  const n = ids.length, layers = manifest!.config.layers;
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, (v) => BigInt(v)), [1, n]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(pos + n).fill(1n), [1, pos + n]),
    position_ids: new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, i) => BigInt(pos + i)), [1, n]),
  };
  for (let l = 0; l < layers; l++) {
    feeds[`past_key_values.${l}.key`] = past[2 * l];
    feeds[`past_key_values.${l}.value`] = past[2 * l + 1];
  }
  const out = await session!.run(feeds);
  const present: ort.Tensor[] = [];
  for (let l = 0; l < layers; l++) present.push(out[`present.${l}.key`], out[`present.${l}.value`]);
  return { logits: out.logits, present };
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The next token: a light penalty on tokens already written, then top-k, temperature and top-p. */
function sample(logits: Float32Array, written: Set<number>, rand: () => number, temperature: number): number {
  for (const t of written) logits[t] = logits[t] > 0 ? logits[t] / REPEAT : logits[t] * REPEAT;
  const top: number[] = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (top.length === TOP_K && v <= logits[top[TOP_K - 1]]) continue;
    let j = Math.min(top.length, TOP_K - 1);
    while (j > 0 && logits[top[j - 1]] < v) j--;
    top.splice(j, 0, i);
    if (top.length > TOP_K) top.pop();
  }
  const weights = top.map((i) => Math.exp((logits[i] - logits[top[0]]) / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  let kept = 0, mass = 0;
  while (kept < top.length && mass < TOP_P) mass += weights[kept++] / total;
  let r = rand() * mass;
  for (let i = 0; i < kept; i++) {
    r -= weights[i] / total;
    if (r <= 0) return top[i];
  }
  return top[kept - 1];
}

async function chat(req: ChatRequest, progress: (p: Progress) => void, onText: (delta: string) => void): Promise<ChatResult> {
  await load(progress);
  const tk = tokenizer!, m = manifest!;
  const started = performance.now();
  stopRequested = false;

  // the conversation in Qwen's chat format, without the thinking phase (an empty <think> block)
  let history = req.system === undefined ? histories.get(req.chat) : undefined;
  if (!history) {
    if (req.system === undefined) throw new Error('this conversation is no longer in memory; write the report again');
    history = [...tk.encode('<|im_start|>system\n'), ...tk.encode(req.system, false), ...tk.encode('<|im_end|>')];
  }
  const prompt = [...history, ...tk.encode('\n<|im_start|>user\n'), ...tk.encode(req.user, false),
    ...tk.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')];
  if (prompt.length + req.maxTokens > m.config.context) throw new Error('the conversation is too long; write the report again');

  // keep what the model already read of this conversation; otherwise start from nothing
  let past: ort.Tensor[], pos = 0;
  const reuse = kv && kv.chat === req.chat && kv.ids.length < prompt.length && kv.ids.every((t, i) => prompt[i] === t);
  if (reuse) { past = kv!.past; pos = kv!.ids.length; }
  else { if (kv) release(kv.past); past = emptyPast(); }
  const reused = pos;
  kv = undefined;

  const feed = async (ids: number[]) => {
    const out = await pass(ids, pos, past);
    release(past);
    past = out.present;
    pos += ids.length;
    return out.logits;
  };
  try {
    // read the prompt (all but its last token) in chunks; their logits are not needed
    while (pos < prompt.length - 1) {
      progress({ stage: 'llm', message: `Reading the results · ${pos} of ${prompt.length} tokens`, fraction: pos / prompt.length });
      const logits = await feed(prompt.slice(pos, Math.min(prompt.length - 1, pos + (req.chunk ?? CHUNK))));
      if (logits.location === 'gpu-buffer') logits.dispose();
    }
    const readMs = performance.now() - started;
    progress({ stage: 'llm', message: 'Writing' });

    const eos = new Set(m.config.eos);
    const rand = mulberry32(req.seed ?? 7);
    const written: number[] = [], seen = new Set<number>();
    const decoder = new TextDecoder();
    let next = prompt[prompt.length - 1], cut = false;
    for (;;) {
      const logits = await feed([next]);
      const data = (await logits.getData()) as Float32Array;
      if (logits.location === 'gpu-buffer') logits.dispose();
      next = sample(data, seen, rand, req.temperature ?? 0.7);
      if (eos.has(next)) break;
      if (stopRequested || written.length >= req.maxTokens) { cut = true; break; }
      written.push(next);
      seen.add(next);
      const delta = decoder.decode(tk.bytes([next]), { stream: true });
      if (delta) onText(delta);
    }
    const tail = decoder.decode();
    if (tail) onText(tail);

    // the cache now holds the prompt and every token written; the history ends the answer as the format does
    kv = { chat: req.chat, ids: [...prompt, ...written], past };
    histories.set(req.chat, [...prompt, ...written, tk.id('<|im_end|>')]);
    const ms = performance.now() - started;
    return {
      text: tk.decode(written), promptTokens: prompt.length, reusedTokens: reused, newTokens: written.length,
      readMs, ms, tokensPerSecond: written.length / Math.max(1e-3, (ms - readMs) / 1000), backend, cut,
    };
  } catch (error) {
    release(past);
    throw error;
  }
}
