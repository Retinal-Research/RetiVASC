import type { Backend, GpuChoice, Progress } from '../runtime/protocol';

export interface LlmFile {
  file: string;
  bytes: number;
  sha256: string;
}

/** The language model's files and shape: models/llm/manifest.json. */
export interface LlmManifest {
  name: string;
  label: string;
  /** where the files come from, at a fixed revision */
  source: string;
  revision: string;
  license: string;
  files: { model: LlmFile; tokenizer: LlmFile };
  config: { layers: number; kvHeads: number; headDim: number; eos: number[]; context: number };
}

export interface LlmInfo {
  manifest: LlmManifest;
  backend: Backend;
  adapter: string;
}

export interface LlmStatus {
  /** both files are in this browser's cache */
  cached: boolean;
  loaded: boolean;
}

export interface ChatRequest {
  /** the conversation; a follow-up question continues it and reuses what the model has read */
  chat: string;
  /** starts the conversation afresh with these instructions */
  system?: string;
  user: string;
  maxTokens: number;
  /** the same results and seed give the same text */
  seed?: number;
  /** lower keeps closer to the given text (default 0.7, Qwen3's recommendation) */
  temperature?: number;
  /** prompt tokens per pass (tests compare passes of different sizes) */
  chunk?: number;
}

export interface ChatResult {
  text: string;
  /** tokens the model read for this answer, and how many of them it had already read before */
  promptTokens: number;
  reusedTokens: number;
  newTokens: number;
  /** time to read the prompt, and in all */
  readMs: number;
  ms: number;
  tokensPerSecond: number;
  backend: Backend;
  /** Stop was pressed, or the length limit reached */
  cut: boolean;
}

export type LlmOp =
  | { op: 'init'; base: string; url?: string; gpu: GpuChoice }
  | { op: 'status' }
  | { op: 'load' }
  | { op: 'chat'; req: ChatRequest }
  | { op: 'stop' };

export type LlmMessage =
  | { id: number; kind: 'progress'; progress: Progress }
  | { id: number; kind: 'token'; text: string }
  | { id: number; kind: 'result'; result: unknown }
  | { id: number; kind: 'error'; error: string };
