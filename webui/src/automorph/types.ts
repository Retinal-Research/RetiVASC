/** 8-bit RGB image, pixels interleaved (R, G, B), row-major. */
export interface RGBImage {
  width: number;
  height: number;
  data: Uint8Array;
}

export function createRGB(width: number, height: number): RGBImage {
  return { width, height, data: new Uint8Array(width * height * 3) };
}

/** Field-of-view geometry found by AutoMorph M0 (legacy_geometry). */
export interface M0Geometry {
  centreRow: number;
  centreCol: number;
  radius: number;
  /** [startH, startW, cropH, cropW] */
  bbox: [number, number, number, number];
}

/** AutoMorph M1 classes, in model output order. */
export const QUALITY_CLASSES = ['Good', 'Usable', 'Reject'] as const;
export type QualityClass = (typeof QUALITY_CLASSES)[number];

export interface QualityResult {
  probs: [number, number, number];
  prediction: QualityClass;
  /** AutoMorph rule: Prediction == Good, or Usable with P(reject) < 0.25. */
  gradable: boolean;
}

/** Per-pixel class labels at model resolution. */
export interface LabelMap {
  width: number;
  height: number;
  labels: Uint8Array;
}
