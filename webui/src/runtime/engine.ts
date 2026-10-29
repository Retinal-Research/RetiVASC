import { recall, remember } from '../remember';
import { isGpuChoice, type EngineInfo, type EngineMessage, type EngineOp, type GpuChoice, type Progress } from './protocol';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (p: Progress) => void };

/** Promise wrapper around the engine worker. It starts when the canvas opens (boot), not with the page:
 *  someone who only reads the home page, on a phone say, loads no worker, WebAssembly or graphics card. */
export class Engine {
  private worker!: Worker;
  private pending = new Map<number, Pending>();
  private next = 1;
  private started = false;
  ready!: Promise<EngineInfo>;
  /** the device this browser computes on; remembered for the next visit */
  gpu: GpuChoice = recall('gpu', 'high-performance', isGpuChoice);

  /** Start the worker once; later calls give the same start. */
  boot(): Promise<EngineInfo> {
    if (!this.started) {
      this.started = true;
      this.start();
    }
    return this.ready;
  }

  private start() {
    this.worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<EngineMessage>) => {
      const msg = event.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.kind === 'progress') p.onProgress?.(msg.progress);
      else {
        this.pending.delete(msg.id);
        if (msg.kind === 'result') p.resolve(msg.result);
        else p.reject(new Error(msg.error));
      }
    };
    this.ready = this.call<EngineInfo>({
      op: 'init',
      base: import.meta.env.BASE_URL,
      models: import.meta.env.VITE_MODELS_URL,
      vascx: import.meta.env.VITE_VASCX_URL,
      tpot: import.meta.env.VITE_TPOT_URL,
      rrwnet: import.meta.env.VITE_RRWNET_URL,
      gpu: this.gpu,
    });
  }

  /** Start again on another device. The images and models the engine held are gone; run again. */
  restart(gpu: GpuChoice): Promise<EngineInfo> {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('the engine was restarted'));
    this.pending.clear();
    this.gpu = gpu;
    remember('gpu', gpu);
    this.start();
    return this.ready;
  }

  call<T>(op: EngineOp, onProgress?: (p: Progress) => void, transfer: Transferable[] = []): Promise<T> {
    if (!this.started) void this.boot();
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
      this.worker.postMessage({ id, ...op }, transfer);
    });
  }
}

export const engine = new Engine();

/** The model keys this site has, from the manifests the engine reads, without starting the engine (the
 *  home page greys out a pipeline whose models are missing). */
export async function siteModels(): Promise<Set<string>> {
  const keys = new Set<string>();
  const base = import.meta.env.BASE_URL;
  await Promise.all(['', 'vascx/', 'tpot/', 'rrwnet/'].map(async (set) => {
    try {
      const r = await fetch(`${base}models/${set}manifest.json`, { cache: 'no-cache' });
      if (r.ok) for (const k of Object.keys(((await r.json()) as { models: object }).models)) keys.add(k);
    } catch { /* not installed here */ }
  }));
  return keys;
}
