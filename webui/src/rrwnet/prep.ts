// RRWNet's preprocessing (preprocessing.py enhance_image, utils.py padding; github.com/j-morano/rrwnet),
// operation for operation in float64, so the browser feeds the network what the authors' code does:
// the field of view's rim filled from a 1.15x zoom, a Gaussian background (sigma 10, scipy's
// reflect mode) subtracted, divided by the standard deviation and scaled to [0, 1].
// tests/rrwnet.ts compares it with the Python reference in tools/rrwnet_ref.

import { resizePIL } from '../automorph/resample';
import type { RGBImage } from '../automorph/types';
import { npSum } from '../retiflow/ndi';

/** The side RRWNet works at in RetiVASC, as in RetiFlow's examples; padded to 608 for the network. */
export const RRW_SIDE = 584;

/** A disc of radius r (skimage.morphology.disk). */
function disk(r: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (y * y + x * x <= r * r) out.push([y, x]);
  return out;
}

/** scipy.ndimage.binary_erosion with a structure, the border counting as 0. */
export function erode(mask: Uint8Array, w: number, h: number, structure: Array<[number, number]>): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      let keep = 1;
      for (const [dy, dx] of structure) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= h || xx < 0 || xx >= w || !mask[yy * w + xx]) { keep = 0; break; }
      }
      out[y * w + x] = keep;
    }
  }
  return out;
}

/** scipy.ndimage.gaussian_filter1d weights (truncate 4.0). */
export function gaussianWeights(sigma: number): Float64Array {
  const radius = Math.floor(4.0 * sigma + 0.5);
  const w = new Float64Array(2 * radius + 1);
  for (let i = -radius; i <= radius; i++) w[i + radius] = Math.exp((-0.5 / (sigma * sigma)) * (i * i));
  const sum = npSum(w);  // phi_x.sum(), pairwise
  for (let i = 0; i < w.length; i++) w[i] /= sum;
  return w;
}

/** scipy's 'reflect' (d c b a | a b c d | d c b a) index into [0, n). */
function reflect(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * n;
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - 1 - i;
}

/** One axis of scipy.ndimage.gaussian_filter on a float64 plane (rows when axis 0, columns when 1). */
export function gaussian1d(src: Float64Array, w: number, h: number, weights: Float64Array, axis: 0 | 1): Float64Array {
  const out = new Float64Array(w * h);
  const r = (weights.length - 1) / 2;
  const n = axis === 0 ? h : w;
  const line = new Float64Array(n);
  for (let k = 0; k < (axis === 0 ? w : h); k++) {
    for (let i = 0; i < n; i++) line[i] = axis === 0 ? src[i * w + k] : src[k * w + i];
    for (let i = 0; i < n; i++) {
      // scipy's symmetric-kernel path (ni_filters.c): the centre, then the pairs either side, outermost first
      let acc = line[i] * weights[r];
      for (let j = r; j >= 1; j--) acc += (line[reflect(i - j, n)] + line[reflect(i + j, n)]) * weights[r - j];
      if (axis === 0) out[i * w + k] = acc; else out[k * w + i] = acc;
    }
  }
  return out;
}

export interface RRWInput {
  /** the padded network input, CHW float32, 3 x side x side */
  x: Float32Array;
  /** the image's side, and the padded side and padding the network sees */
  size: number;
  side: number;
  pad: number;
  /** RRWNet's eroded field of view at RRW_SIDE: predictions outside it are zero */
  eroded: Uint8Array;
  /** the enhanced image before padding, HWC float64 (for tests) */
  enhanced: Float64Array;
}

/** enhance_image(img, mask) and the network's padding, for a square RGB image and its field-of-view mask. */
export function rrwnetInput(rgb: RGBImage, mask: Uint8Array): RRWInput {
  const n = rgb.width, plane = n * n;
  if (rgb.height !== n) throw new Error('RRWNet expects a square image');
  // the zoom of np.uint8(img * 255), img = rgb / 255, which truncates
  const back: RGBImage = { width: n, height: n, data: new Uint8Array(3 * plane) };
  for (let i = 0; i < 3 * plane; i++) back.data[i] = Math.trunc((rgb.data[i] / 255) * 255);
  const zs = Math.trunc(n * 1.15);
  const zoom = resizePIL(back, zs, zs, 'bicubic');
  const off = Math.floor(zs / 2) - Math.floor(n / 2);
  const eroded = erode(Uint8Array.from(mask, (m) => (m > 0.5 ? 1 : 0)), n, n, disk(5));

  // composed: the image inside the eroded field, the zoom outside it; then subtract its Gaussian background
  const composed = [0, 1, 2].map(() => new Float64Array(plane));
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = y * n + x, z = ((y + off) * zs + (x + off)) * 3;
      for (let c = 0; c < 3; c++) composed[c][i] = eroded[i] ? rgb.data[i * 3 + c] / 255 : zoom.data[z + c] / 255.0;
    }
  }
  const weights = gaussianWeights(10);
  const sub = composed.map((p) => {
    const g = gaussian1d(gaussian1d(p, n, n, weights, 0), n, n, weights, 1);
    const s = new Float64Array(plane);
    for (let i = 0; i < plane; i++) s[i] = eroded[i] ? p[i] - g[i] : 0;
    return s;
  });
  // np.std over the (h, w, 3) array, summed pairwise in its memory order, then interp1d([min, max], [0, 1])
  const flat = new Float64Array(3 * plane);
  for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) flat[i * 3 + c] = sub[c][i];
  const mean = npSum(flat) / flat.length;
  const squares = Float64Array.from(flat, (v) => (v - mean) * (v - mean));
  const std = Math.sqrt(npSum(squares) / flat.length);
  let lo = Infinity, hi = -Infinity;
  for (const s of sub) for (let i = 0; i < plane; i++) { const v = s[i] / std; if (v < lo) lo = v; if (v > hi) hi = v; }
  const slope = (1.0 - 0.0) / (hi - lo);
  const enhanced = new Float64Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < plane; i++) enhanced[i * 3 + c] = eroded[i] ? 0.0 + slope * (sub[c][i] / std - lo) : 0;
  }

  // utils.get_unet_padding_np: up to the next multiple of 32 (a full 32 when already a multiple)
  const pad = Math.floor((32 - (n % 32)) / 2), side = n + 32 - (n % 32);
  const x = new Float32Array(3 * side * side);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < n; y++) for (let xx = 0; xx < n; xx++) x[(c * side + y + pad) * side + xx + pad] = enhanced[(y * n + xx) * 3 + c];
  }
  return { x, size: n, side, pad, eroded, enhanced };
}

/** The circular field of view of a square image whose circle touches its edges (as in tools/export_rrwnet.py). */
export function circleMask(n: number): Uint8Array {
  const m = new Uint8Array(n * n), c = (n - 1) / 2, r2 = (n / 2 - 1) ** 2;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) m[y * n + x] = (y - c) ** 2 + (x - c) ** 2 <= r2 ? 255 : 0;
  return m;
}

/** The network's output, cropped and outside the eroded field set to zero, quantized as torchvision's
 *  save_image writes it (round(p * 255)): the AV3 map RetiFlow reads, R arteries, G veins, B vessels. */
export function av3(probs: Float32Array, input: RRWInput): Uint8Array {
  const n = input.size, side = input.side, pad = input.pad;
  const out = new Uint8Array(3 * n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (!input.eroded[y * n + x]) continue;
      for (let c = 0; c < 3; c++) {
        const p = probs[(c * side + y + pad) * side + x + pad];
        out[(y * n + x) * 3 + c] = Math.min(255, Math.max(0, Math.floor(p * 255 + 0.5)));
      }
    }
  }
  return out;
}
