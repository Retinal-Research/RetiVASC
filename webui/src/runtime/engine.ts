import type { EngineInfo, EngineMessage, EngineOp, Progress } from './protocol';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (p: Progress) => void };

/** Promise wrapper around the engine worker. */
export class Engine {
  private worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
  private pending = new Map<number, Pending>();
  private next = 1;
  readonly ready: Promise<EngineInfo>;

  constructor() {
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
    this.ready = this.call<EngineInfo>({ op: 'init', base: import.meta.env.BASE_URL });
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
