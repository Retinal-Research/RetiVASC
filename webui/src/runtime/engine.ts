import { recall, remember } from '../remember';
import { isGpuChoice, type EngineInfo, type EngineMessage, type EngineOp, type GpuChoice, type Progress } from './protocol';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (p: Progress) => void };

/** Promise wrapper around the engine worker. */
export class Engine {
  private worker!: Worker;
  private pending = new Map<number, Pending>();
  private next = 1;
  ready!: Promise<EngineInfo>;
  /** the device this browser computes on; remembered for the next visit */
  gpu: GpuChoice = recall('gpu', 'high-performance', isGpuChoice);

  constructor() {
    this.start();
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
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
      this.worker.postMessage({ id, ...op }, transfer);
    });
  }
}

export const engine = new Engine();
