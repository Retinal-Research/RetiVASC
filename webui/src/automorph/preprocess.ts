// Model inputs from the M0 square image, mirroring AutoMorph v2's datasets.
// Statistics are computed in float64 and the result cast to float32, as numpy
// followed by torch.FloatTensor does.

import { resizePIL } from './resample';
import type { RGBImage } from './types';

export interface ModelInput {
  data: Float32Array;
  dims: [number, number, number, number];
}

function toCHW(values: Float64Array | Float32Array, size: number): Float32Array {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = values[i * 3];
    out[plane + i] = values[i * 3 + 1];
    out[2 * plane + i] = values[i * 3 + 2];
  }
  return out;
}

/** Population mean/std per channel over pixels whose first channel passes `keep`. */
function channelStats(px: Uint8Array, keep: (r: number) => boolean): { mean: number[]; std: number[] } {
  const sum = [0, 0, 0];
  let n = 0;
  for (let p = 0; p < px.length; p += 3) {
    if (!keep(px[p])) continue;
    sum[0] += px[p];
    sum[1] += px[p + 1];
    sum[2] += px[p + 2];
    n++;
  }
  const mean = sum.map((s) => s / n);
  const sq = [0, 0, 0];
  for (let p = 0; p < px.length; p += 3) {
    if (!keep(px[p])) continue;
    for (let c = 0; c < 3; c++) {
      const d = px[p + c] - mean[c];
      sq[c] += d * d;
    }
  }
  return { mean, std: sq.map((s) => Math.sqrt(s / n)) };
}

/** M1 QualityDataset: PIL bicubic 512, one mean/std over all non-zero values. */
export function qualityInput(m0: RGBImage): ModelInput {
  const size = 512;
  const px = resizePIL(m0, size, size, 'bicubic').data;
  let sum = 0;
  let n = 0;
  for (const v of px) if (v > 0) { sum += v; n++; }
  if (!n) throw new Error('image has no positive pixels');
  const mean = sum / n;
  let sq = 0;
  for (const v of px) if (v > 0) sq += (v - mean) * (v - mean);
  const std = Math.sqrt(sq / n);
  if (!Number.isFinite(std) || std === 0) throw new Error('image has invalid non-zero standard deviation');
  const values = new Float64Array(px.length);
  for (let i = 0; i < px.length; i++) values[i] = (px[i] - mean) / std;
  return { data: toCHW(values, size), dims: [1, 3, size, size] };
}

/** M2 vessel preprocess_image: PIL bicubic 912, per-channel stats over R > 40. */
export function vesselInput(m0: RGBImage): ModelInput {
  const size = 912;
  const px = resizePIL(m0, size, size, 'bicubic').data;
  let blue = 0;
  for (let p = 2; p < px.length; p += 3) blue += px[p];
  if (blue === 0) {
    for (let p = 0; p < px.length; p += 3) { px[p] = px[p + 1]; px[p + 2] = px[p + 1]; }
  }
  const { mean, std } = channelStats(px, (r) => r > 40.0);
  const values = new Float64Array(px.length);
  for (let i = 0; i < px.length; i++) values[i] = (px[i] - mean[i % 3]) / std[i % 3];
  return { data: toCHW(values, size), dims: [1, 3, size, size] };
}

/**
 * M2 artery/vein AVImageDataset: PIL bicubic 720, per-channel stats over R > 0,
 * then the legacy `(image - mean) / 1.0 * std` (a multiply, kept on purpose).
 */
export function arteryVeinInput(m0: RGBImage): ModelInput {
  const size = 720;
  const px = resizePIL(m0, size, size, 'bicubic').data;
  const { mean, std } = channelStats(px, (r) => r > 0.0);
  const values = new Float64Array(px.length);
  for (let i = 0; i < px.length; i++) values[i] = ((px[i] - 1.0 * mean[i % 3]) / 1.0) * std[i % 3];
  return { data: toCHW(values, size), dims: [1, 3, size, size] };
}

/** M2 disc/cup: PIL bilinear 512 then torchvision to_tensor (float32 x / 255). */
export function discCupInput(m0: RGBImage): ModelInput {
  const size = 512;
  const px = resizePIL(m0, size, size, 'bilinear').data;
  const lut = new Float32Array(256);
  for (let v = 0; v < 256; v++) lut[v] = Math.fround(v) / 255;
  const values = new Float32Array(px.length);
  for (let i = 0; i < px.length; i++) values[i] = lut[px[i]];
  return { data: toCHW(values, size), dims: [1, 3, size, size] };
}
