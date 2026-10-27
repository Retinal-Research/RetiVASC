// Pillow's 8-bit two-pass resampler (libImaging/Resample.c), reproduced
// operation-for-operation so browser inputs match AutoMorph's PIL resize.
// Image.resize() with no filter uses BICUBIC; torchvision's PIL path uses BILINEAR.

import { createRGB, type RGBImage } from './types';

export type ResampleFilter = 'bicubic' | 'bilinear';

const PRECISION_BITS = 32 - 8 - 2;
const HALF = 1 << (PRECISION_BITS - 1);
const ONE = 1 << PRECISION_BITS;
const CLIP_HIGH = ONE * 256; // 1 << PRECISION_BITS << 8

function bicubic(x: number): number {
  const a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

function bilinear(x: number): number {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

interface Coefficients {
  ksize: number;
  bounds: Int32Array;
  kk: Float64Array; // fixed-point integers stored exactly in doubles
}

function precompute(inSize: number, outSize: number, filter: ResampleFilter): Coefficients {
  const fn = filter === 'bicubic' ? bicubic : bilinear;
  const baseSupport = filter === 'bicubic' ? 2.0 : 1.0;
  const scale = inSize / outSize;
  const filterscale = scale < 1.0 ? 1.0 : scale;
  const support = baseSupport * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const pre = new Float64Array(outSize * ksize);
  const ss = 1.0 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = 0.0 + (xx + 0.5) * scale;
    let ww = 0.0;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const base = xx * ksize;
    for (let x = 0; x < xmax; x++) {
      const w = fn((x + xmin - center + 0.5) * ss);
      pre[base + x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      if (ww !== 0.0) pre[base + x] /= ww;
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  const kk = new Float64Array(pre.length);
  for (let i = 0; i < pre.length; i++) {
    const v = pre[i];
    kk[i] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
  }
  return { ksize, bounds, kk };
}

function clip8(v: number): number {
  if (v >= CLIP_HIGH) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

function horizontal(src: RGBImage, outWidth: number, c: Coefficients): RGBImage {
  const { width, height, data } = src;
  const out = createRGB(outWidth, height);
  const o = out.data;
  const { ksize, bounds, kk } = c;
  for (let y = 0; y < height; y++) {
    const row = y * width * 3;
    const orow = y * outWidth * 3;
    for (let xx = 0; xx < outWidth; xx++) {
      const xmin = bounds[xx * 2];
      const xmax = bounds[xx * 2 + 1];
      const kb = xx * ksize;
      let s0 = HALF;
      let s1 = HALF;
      let s2 = HALF;
      let p = row + xmin * 3;
      for (let x = 0; x < xmax; x++, p += 3) {
        const k = kk[kb + x];
        s0 += data[p] * k;
        s1 += data[p + 1] * k;
        s2 += data[p + 2] * k;
      }
      const q = orow + xx * 3;
      o[q] = clip8(s0);
      o[q + 1] = clip8(s1);
      o[q + 2] = clip8(s2);
    }
  }
  return out;
}

function vertical(src: RGBImage, outHeight: number, c: Coefficients): RGBImage {
  const { width, data } = src;
  const out = createRGB(width, outHeight);
  const o = out.data;
  const { ksize, bounds, kk } = c;
  const stride = width * 3;
  for (let yy = 0; yy < outHeight; yy++) {
    const ymin = bounds[yy * 2];
    const ymax = bounds[yy * 2 + 1];
    const kb = yy * ksize;
    const orow = yy * stride;
    for (let x = 0; x < width; x++) {
      let s0 = HALF;
      let s1 = HALF;
      let s2 = HALF;
      let p = ymin * stride + x * 3;
      for (let y = 0; y < ymax; y++, p += stride) {
        const k = kk[kb + y];
        s0 += data[p] * k;
        s1 += data[p + 1] * k;
        s2 += data[p + 2] * k;
      }
      const q = orow + x * 3;
      o[q] = clip8(s0);
      o[q + 1] = clip8(s1);
      o[q + 2] = clip8(s2);
    }
  }
  return out;
}

/** PIL Image.resize((width, height), filter) for an RGB image. */
export function resizePIL(src: RGBImage, width: number, height: number, filter: ResampleFilter): RGBImage {
  if (src.width === width && src.height === height) {
    return { width, height, data: src.data.slice() };
  }
  let img = src;
  if (width !== src.width) img = horizontal(img, width, precompute(src.width, width, filter));
  if (height !== src.height) img = vertical(img, height, precompute(src.height, height, filter));
  return img;
}
