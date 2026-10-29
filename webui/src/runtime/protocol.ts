import type { LabelMap, M0Geometry, QualityResult } from '../automorph/types';

/** AutoMorph's four models, the four VascX models, and the lab's TPOT enhancement (with its fusion head). */
export type ModelKey = 'quality' | 'vessel' | 'artery_vein' | 'disc_cup'
  | 'vx_av' | 'vx_vessel' | 'vx_disc' | 'vx_fovea'
  | 'tpot' | 'tpot_fusion';
/** Each pipeline's models come from their own folder or host (their licences differ). */
export type ModelSet = 'automorph' | 'vascx' | 'tpot';
export type Backend = 'webgpu' | 'wasm';
/** Which device the browser computes on. A web page cannot list the graphics cards; it can only ask
 *  for the faster one (high-performance) or the power-saving one (usually the integrated GPU). */
export type GpuChoice = 'high-performance' | 'low-power' | 'cpu';
export const GPU_CHOICES: GpuChoice[] = ['high-performance', 'low-power', 'cpu'];
export const isGpuChoice = (v: unknown): v is GpuChoice => GPU_CHOICES.includes(v as GpuChoice);

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
  /** weights stored as float32 (AutoMorph's are float16 and double in size once loaded) */
  float32?: boolean;
  /** set by the engine: which pipeline's folder the files are in */
  set?: ModelSet;
}

export interface Manifest {
  version: number;
  opset: number;
  /** every model this deployment has; VascX and TPOT are optional */
  models: Partial<Record<ModelKey, ModelInfo>>;
  /** where VascX's files come from (licence and revision), when it is there */
  vascx?: { source: string; revision: string; license: string };
}

export interface EngineInfo {
  backend: Backend;
  /** the device asked for; backend and adapter say what the browser gave */
  gpu: GpuChoice;
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

/** VascX preprocessing: the field of view found in the photo, cropped to 1024 x 1024. */
export interface VascxPrepOutput extends ImageHandle {
  /** the circle in the photo's pixels */
  bounds: { cx: number; cy: number; radius: number; fullFrame: boolean; fit: number };
  geometry: M0Geometry;
  ms: number;
}

/** The fovea in the 1024 x 1024 VascX image. */
export interface FoveaOutput { x: number; y: number; size: number; timing: Timing }

/** An enhanced copy of an M0 image, the same size; geometry is unchanged. */
export interface EnhanceOutput extends ImageHandle { ms: number; timing: Timing }

export interface QualityOutput { result: QualityResult; timing: Timing }
export interface VesselOutput { size: number; mask: Uint8Array; timing: Timing }
export interface LabelOutput { map: LabelMap; timing: Timing }
export type M3Output = import('../automorph/m3/m3').M3Result;

/** How far each model is: files in this browser's cache, members ready to run. */
export type ModelStatus = Partial<Record<ModelKey, { parts: number; cached: number; loaded: number }>>;

/** A model a workflow uses: every ensemble member, or the first only (quick). */
export interface ModelNeed { key: ModelKey; parts: number }

export interface Progress {
  stage: 'download' | 'compile' | 'preprocess' | 'inference' | 'postprocess' | 'm0' | 'decode' | 'compare' | 'server' | 'measure' | 'enhance' | 'llm';
  message: string;
  /** 0..1 when known */
  fraction?: number;
  /** text written so far (the AI report, as the language model writes it) */
  text?: string;
}

export type EngineOp =
  | { op: 'init'; base: string; models?: string; vascx?: string; tpot?: string; gpu?: GpuChoice }
  | { op: 'status' }
  | { op: 'preload'; models: ModelNeed[] }
  | { op: 'decode'; bytes: ArrayBuffer; name: string }
  | { op: 'm0'; imageId: string }
  | { op: 'quality' | 'vessel' | 'artery_vein' | 'disc_cup'; imageId: string; quick?: boolean }
  | { op: 'm3'; vessel: Uint8Array; arteryVein: Uint8Array; discCup: Uint8Array; radius: number }
  | { op: 'vx_prep'; imageId: string }
  | { op: 'vx_vessel' | 'vx_av' | 'vx_disc' | 'vx_fovea'; imageId: string }
  | { op: 'enhance'; imageId: string; quick?: boolean }
  | { op: 'release'; ids: string[] };

export type EngineMessage =
  | { id: number; kind: 'progress'; progress: Progress }
  | { id: number; kind: 'result'; result: unknown }
  | { id: number; kind: 'error'; error: string };
