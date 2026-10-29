// The page's handle on the language model worker. The worker starts on first use, so a page that
// never writes a report never loads the model; it uses the device chosen for the engine.

import { engine } from '../runtime/engine';
import type { Progress } from '../runtime/protocol';
import type { ChatRequest, ChatResult, LlmInfo, LlmManifest, LlmMessage, LlmOp, LlmStatus } from './protocol';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (p: Progress) => void; onText?: (t: string) => void };

class Llm {
  private worker?: Worker;
  private pending = new Map<number, Pending>();
  private next = 1;
  private started?: Promise<LlmInfo>;
  /** the model this site offers, or undefined when it has none (the AI report step is then greyed out) */
  readonly manifest: Promise<LlmManifest | undefined> = fetch(`${import.meta.env.BASE_URL}models/llm/manifest.json`, { cache: 'no-cache' })
    .then((r) => (r.ok ? (r.json() as Promise<LlmManifest>) : undefined), () => undefined);

  private send<T>(op: LlmOp, onProgress?: (p: Progress) => void, onText?: (t: string) => void): Promise<T> {
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress, onText });
      this.worker!.postMessage({ id, ...op });
    });
  }

  private start(): Promise<LlmInfo> {
    if (!this.started) {
      this.worker = new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (event: MessageEvent<LlmMessage>) => {
        const msg = event.data;
        const p = this.pending.get(msg.id);
        if (!p) return;
        if (msg.kind === 'progress') p.onProgress?.(msg.progress);
        else if (msg.kind === 'token') p.onText?.(msg.text);
        else {
          this.pending.delete(msg.id);
          if (msg.kind === 'result') p.resolve(msg.result);
          else p.reject(new Error(msg.error));
        }
      };
      this.started = this.send<LlmInfo>({ op: 'init', base: import.meta.env.BASE_URL, url: import.meta.env.VITE_LLM_URL, gpu: engine.gpu });
    }
    return this.started;
  }

  info(): Promise<LlmInfo> {
    return this.start();
  }

  async status(): Promise<LlmStatus> {
    await this.start();
    return this.send<LlmStatus>({ op: 'status' });
  }

  /** Download (once) and prepare the model, before it is needed. */
  async load(onProgress?: (p: Progress) => void): Promise<LlmStatus> {
    await this.start();
    return this.send<LlmStatus>({ op: 'load' }, onProgress);
  }

  /** Ask the model; its text arrives piece by piece in onText. */
  async chat(req: ChatRequest, onText: (delta: string) => void, onProgress?: (p: Progress) => void): Promise<ChatResult> {
    await this.start();
    return this.send<ChatResult>({ op: 'chat', req }, onProgress, onText);
  }

  /** Stop writing; the text so far is kept. */
  stop() {
    if (this.worker) void this.send({ op: 'stop' });
  }

  /** Forget the worker (another device was chosen); the next use starts a new one. */
  restart() {
    this.worker?.terminate();
    for (const p of this.pending.values()) p.reject(new Error('the language model was restarted'));
    this.pending.clear();
    this.worker = undefined;
    this.started = undefined;
  }
}

export const llm = new Llm();
