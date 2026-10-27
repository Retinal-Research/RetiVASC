import type { LabelMap, M0Geometry, QualityResult } from '../automorph/types';

export type ModelKey = 'quality' | 'vessel' | 'artery_vein' | 'disc_cup';
export type Backend = 'webgpu' | 'wasm';

/** One file of a model: an ensemble member, or the whole model when it is a single file. */
export interface ModelPart {
  file: string;
  bytes: number;
  sha256: string;
  member: string;
}

export interface ModelInfo {
  label: string;
  /** total download, all parts */
  bytes: number;
  input: number[];
  /** what the parts are, e.g. "all 8 members (mean, AutoMorph v2 order)" */
  members: string;
  /** ensemble members in AutoMorph v2's order; the browser averages them in this order */
  parts: ModelPart[];
}

export interface Manifest {
  version: number;
  opset: number;
  models: Record<ModelKey, ModelInfo>;
}

export interface EngineInfo {
  backend: Backend;
  adapter: string;
  threads: number;
  manifest: Manifest;
}

export interface Timing {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  /** first use only: download (or cache read) + session creation */
  loadMs?: number;
  backend: Backend;
}

export interface ImageHandle {
  imageId: string;
  width: number;
  height: number;
  preview: ImageBitmap;
}

export interface M0Output extends ImageHandle {
  geometry: M0Geometry;
  ms: number;
}

export interface QualityOutput { result: QualityResult; timing: Timing }
export interface VesselOutput { size: number; mask: Uint8Array; timing: Timing }
export interface LabelOutput { map: LabelMap; timing: Timing }
export type M3Output = import('../automorph/m3/m3').M3Result;

export interface Progress {
  stage: 'download' | 'compile' | 'preprocess' | 'inference' | 'postprocess' | 'm0' | 'decode' | 'compare' | 'server' | 'measure';
  message: string;
  /** 0..1 when known */
  fraction?: number;
}

export type EngineOp =
  | { op: 'init'; base: string }
  | { op: 'decode'; bytes: ArrayBuffer; name: string }
  | { op: 'm0'; imageId: string }
  | { op: 'quality' | 'vessel' | 'artery_vein' | 'disc_cup'; imageId: string; quick?: boolean }
  | { op: 'm3'; vessel: Uint8Array; arteryVein: Uint8Array; discCup: Uint8Array; radius: number }
  | { op: 'release'; ids: string[] };

export type EngineMessage =
  | { id: number; kind: 'progress'; progress: Progress }
  | { id: number; kind: 'result'; result: unknown }
  | { id: number; kind: 'error'; error: string };
