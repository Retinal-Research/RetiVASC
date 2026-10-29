import type { LabelMap, M0Geometry, QualityResult } from '../automorph/types';
import type { Timing } from '../runtime/protocol';
import type { PhotoResult } from '../server/api';

/** Values that travel along canvas edges. Pixel data stays in the worker. */
export interface FundusValue {
  kind: 'fundus';
  imageId: string;
  name: string;
  width: number;
  height: number;
  preview: ImageBitmap;
  /** where the bytes came from (sample or opened file), so a server step can send the original */
  source: string;
}

export interface M0Value {
  kind: 'm0';
  imageId: string;
  side: number;
  geometry: M0Geometry;
  preview: ImageBitmap;
  ms: number;
  source: string;
  /** sharper picture of the same square, for display only (a 256 canvas keeps its M0 preview here) */
  hires?: ImageBitmap;
  /** set when a model produced the image (enhancement) */
  timing?: Timing;
  /** the field-of-view radius in this image's pixels, when the image is not at the photo's scale (VascX: 512 of 1024);
   *  geometry stays in the photo's pixels so measurements keep their physical scale */
  fovRadius?: number;
  /** what made this image, for the inspector: M0's crop, VascX's crop, or an enhancement */
  made?: 'm0' | 'vascx' | 'enhanced';
  /** the image before enhancement, to compare */
  before?: ImageBitmap;
}

/** The fovea found by VascX, in the pixels of its base image. */
export interface FoveaValue {
  kind: 'fovea';
  x: number;
  y: number;
  size: number;
  timing: Timing;
  base: M0Value;
}

export interface QualityValue {
  kind: 'quality';
  result: QualityResult;
  timing: Timing;
  base: M0Value;
}

export interface VesselValue {
  kind: 'vessel';
  size: number;
  mask: Uint8Array;
  timing: Timing;
  base: M0Value;
}

export interface ArteryVeinValue {
  kind: 'av';
  map: LabelMap;
  timing: Timing;
  base: M0Value;
}

export interface DiscCupValue {
  kind: 'disc';
  map: LabelMap;
  timing: Timing;
  base: M0Value;
}

/** Which photos a study sheet uses: an uploaded folder, a folder on the server,
 *  or the AutoMorph run of an earlier finished sheet job. */
export interface RunValue {
  kind: 'run';
  source: 'upload' | 'server' | 'reuse';
  pickId?: string;
  folder?: string;
  reuseJob?: string;
  photos: number;
  label: string;
}

/** Full AutoMorph v2 (all ensemble members, M0–M3) on the RetiVASC server, for one photo. */
export interface AnalysisValue {
  kind: 'analysis';
  jobId: string;
  photo: PhotoResult;
}

export type CheckResult = 'PASS' | 'FAIL' | 'BLOCKED' | 'WARN' | 'DEFERRED';

export interface SheetSummary {
  /** what the build decided: file / none / blocked / gradable / picked … */
  built: Record<string, number>;
  /** change-log categories against the previous version */
  changes: Record<string, number>;
  /** findings per severity */
  counts: Partial<Record<CheckResult, number>>;
  checks: Array<{ id: string; result: CheckResult; n: number; rule: string }>;
}

/** The Excel results sheet built from a run, with its audit verdict. */
export interface SheetValue {
  kind: 'sheet';
  result: string;
  jobId: string;
  /** download links */
  files: Array<{ name: string; url: string }>;
  summary?: SheetSummary;
}

/** AutoMorph v2 M3 in the browser: the 72 vascular features of one photo. */
export interface FeaturesValue {
  kind: 'features';
  result: import('../automorph/m3/m3').M3Result;
  base: M0Value;
  /** AutoMorph measures vessels on gradable photos only; false when this one was not */
  gradable?: boolean;
  /** measured although not gradable (the node's "measure anyway" box) */
  forced: boolean;
}

export type NodeValue = FundusValue | M0Value | QualityValue | VesselValue | ArteryVeinValue | DiscCupValue | FoveaValue | RunValue | SheetValue | AnalysisValue | FeaturesValue;
export type PortType = NodeValue['kind'];
