// VascX preprocessing in the browser: find the circular field of view and crop it to a 1024 x 1024
// square, as retinalysis-fundusprep 1.3.0 does before the VascX models (FundusPreprocessor with
// square_size=1024, no contrast enhancement). The steps follow its mask_extraction.py and
// cfi_bounds.py (AGPL-3.0, github.com/Eyened/retinalysis-fundusprep):
//
//   red channel -> 256 x 256 (bilinear warp, no anti-aliasing) -> full-frame test
//   -> polar unwrap (nearest) -> Sobel x (1 - Gaussian 31) -> minimum-cost path = 256 edge points
//   -> RANSAC circle (1000 tries, 1 px inliers, radius 64..142) -> corner fit when it fails
//   -> back to the photo's coordinates -> crop the circle to 1024 x 1024 (bilinear warp)
//
// Two differences, both small: the RANSAC samples come from a fixed-seed generator here instead of
// numpy's (the final circle is a least-squares fit to the inliers, so it moves by a fraction of a
// pixel at most), and the straight top/bottom/left/right edges are not fitted, because they only
// limit the crop metadata, never the 1024 image the models see. The bilinear warps reproduce
// OpenCV's fixed-point arithmetic (1/32 pixel positions, 15-bit weights), so pixels match cv2.

import type { RGBImage } from '../automorph/types';

const RES = 256;
const CENTER = RES / 2;          // 128
const MAX_R = Math.floor(RES / 1.8);  // 142 (Python: RESOLUTION // 1.8)
const MIN_R = RES / 4;           // 64
const N = RES - MIN_R;           // 192 radii in the polar strip

export interface FundusBounds {
  /** circle centre and radius in the photo's pixels */
  cx: number;
  cy: number;
  radius: number;
  /** no circular aperture was found: the circle encloses the whole frame */
  fullFrame: boolean;
  /** share of the 256 edge points on the fitted circle (1 for a clean round field of view) */
  fit: number;
}

// ---------------------------------------------------------------- OpenCV-exact bilinear warp

/** Round half to even, like OpenCV's cvRound (lrint). */
function rint(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * cv2.warpAffine/warpPerspective with INTER_LINEAR and a black border, for an affine map given by
 * its inverse: source x = ax * x' + bx, source y = ay * y' + by (no rotation). 8-bit, any channels.
 */
export function warpLinear(src: Uint8Array, w: number, h: number, ch: number, outW: number, outH: number,
                           ax: number, bx: number, ay: number, by: number): Uint8Array {
  const out = new Uint8Array(outW * outH * ch);
  const fx = new Int32Array(outW), sxs = new Int32Array(outW);
  for (let x = 0; x < outW; x++) {
    const X = rint((ax * x + bx) * 32);
    sxs[x] = X >> 5;
    fx[x] = X & 31;
  }
  for (let y = 0; y < outH; y++) {
    const Y = rint((ay * y + by) * 32);
    const sy = Y >> 5, ty = Y & 31;
    const wy0 = 32 - ty, wy1 = ty;
    for (let x = 0; x < outW; x++) {
      const sx = sxs[x], tx = fx[x];
      const o = (y * outW + x) * ch;
      if (sx >= w || sx + 1 < 0 || sy >= h || sy + 1 < 0) continue;  // all four outside: border (0)
      // weights in units of 1/1024, scaled by 32 to OpenCV's 1/32768: exact, they sum to 32768
      const w00 = (32 - tx) * wy0 * 32, w01 = tx * wy0 * 32, w10 = (32 - tx) * wy1 * 32, w11 = tx * wy1 * 32;
      const in0 = sy >= 0 && sy < h, in1 = sy + 1 >= 0 && sy + 1 < h;
      const ix0 = sx >= 0 && sx < w, ix1 = sx + 1 >= 0 && sx + 1 < w;
      const r0 = sy * w, r1 = (sy + 1) * w;
      for (let c = 0; c < ch; c++) {
        const p00 = in0 && ix0 ? src[(r0 + sx) * ch + c] : 0;
        const p01 = in0 && ix1 ? src[(r0 + sx + 1) * ch + c] : 0;
        const p10 = in1 && ix0 ? src[(r1 + sx) * ch + c] : 0;
        const p11 = in1 && ix1 ? src[(r1 + sx + 1) * ch + c] : 0;
        const v = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + (1 << 14)) >> 15;
        out[o + c] = v > 255 ? 255 : v;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- full-frame test

function percentile(sorted: Float64Array, q: number): number {
  const pos = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

const median = (values: number[]) => percentile(Float64Array.from(values).sort(), 50);

/** True when the photograph fills the frame, with no dark corners around a circular aperture. */
function isFullFrame(img: Uint8Array): boolean {
  const h = RES, w = RES, cy = h >> 1, cx = w >> 1, d = (Math.min(h, w) / 8) | 0;
  const at = (y: number, x: number) => img[y * w + x];
  const edge: number[] = [];
  for (let y = 0; y < 4; y++) for (let x = 0; x < w; x++) edge.push(at(y, x));
  for (let y = h - 4; y < h; y++) for (let x = 0; x < w; x++) edge.push(at(y, x));
  for (let y = 0; y < h; y++) for (let x = 0; x < 4; x++) edge.push(at(y, x));
  for (let y = 0; y < h; y++) for (let x = w - 4; x < w; x++) edge.push(at(y, x));
  const corners: number[] = [];
  for (const [y0, x0] of [[0, 0], [0, w - d], [h - d, 0], [h - d, w - d]]) {
    for (let y = y0; y < y0 + d; y++) for (let x = x0; x < x0 + d; x++) corners.push(at(y, x));
  }
  const center: number[] = [];
  for (let y = cy - d; y < cy + d; y++) for (let x = cx - d; x < cx + d; x++) center.push(at(y, x));
  const thr = percentile(Float64Array.from(center).sort(), 5);
  const cornerDark = corners.filter((v) => v < thr).length / corners.length;
  const centerMed = median(center);
  const cornerRatio = median(corners) / (centerMed + 1e-6);
  const edgeRatio = median(edge) / (centerMed + 1e-6);
  return cornerDark < 0.95 || (cornerRatio > 0.145 && edgeRatio > 0.1);
}

// ---------------------------------------------------------------- edge points

/** cv2.GaussianBlur(31 x 31, sigma 0) on a float image: sigma 5, BORDER_REFLECT_101. */
function gaussian31(src: Float64Array, w: number, h: number): Float64Array {
  const k = new Float64Array(31);
  let sum = 0;
  for (let i = 0; i < 31; i++) { k[i] = Math.exp(-((i - 15) ** 2) / (2 * 5 * 5)); sum += k[i]; }
  for (let i = 0; i < 31; i++) k[i] /= sum;
  const reflect = (i: number, n: number) => (i < 0 ? -i : i >= n ? 2 * n - 2 - i : i);
  const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < 31; i++) s += k[i] * src[y * w + reflect(x + i - 15, w)];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < 31; i++) s += k[i] * tmp[reflect(y + i - 15, h) * w + x];
      out[y * w + x] = s;
    }
  }
  return out;
}

/** 256 points on the edge of the field of view, one per angle, from a minimum-cost path in polar coordinates. */
function edgePoints(img: Uint8Array): { xs: Float64Array; ys: Float64Array } {
  // cv2.linearPolar(img, (128, 128), 142, WARP_FILL_OUTLIERS): rows are angles, columns radii, nearest pixel
  const polar = new Float64Array(RES * N);
  for (let phi = 0; phi < RES; phi++) {
    const a = ((2 * Math.PI) / RES) * phi, cp = Math.cos(a), sp = Math.sin(a);
    for (let rho = MIN_R; rho < RES; rho++) {
      const r = (MAX_R / RES) * rho;
      const x = rint(Math.fround(r * cp + CENTER)), y = rint(Math.fround(r * sp + CENTER));
      polar[phi * N + rho - MIN_R] = x >= 0 && x < RES && y >= 0 && y < RES ? img[y * RES + x] : 0;
    }
  }
  let max = 0;
  for (const v of polar) if (v > max) max = v;
  if (max === 0) throw new Error('blank image');
  for (let i = 0; i < polar.length; i++) polar[i] /= max;

  // scipy.ndimage.sobel(axis=1), mode "reflect" (the edge pixel is repeated)
  const rf = (i: number, n: number) => (i < 0 ? -i - 1 : i >= n ? 2 * n - 1 - i : i);
  const dx = new Float64Array(RES * N), gx = new Float64Array(RES * N);
  for (let y = 0; y < RES; y++) for (let x = 0; x < N; x++) dx[y * N + x] = polar[y * N + rf(x + 1, N)] - polar[y * N + rf(x - 1, N)];
  for (let y = 0; y < RES; y++) {
    for (let x = 0; x < N; x++) gx[y * N + x] = dx[rf(y - 1, RES) * N + x] + 2 * dx[y * N + x] + dx[rf(y + 1, RES) * N + x];
  }
  const blur = gaussian31(polar, N, RES);
  const cost = new Float64Array(RES * N);
  for (let i = 0; i < cost.length; i++) cost[i] = gx[i] * (1 - blur[i]);

  // dynamic programming from the first angle to the last, moving 0.02 * (step in radius)^2
  const from = new Int32Array(RES * N);
  for (let y = 1; y < RES; y++) {
    for (let i = 0; i < N; i++) {
      let best = Infinity, arg = 0;
      for (let j = 0; j < N; j++) {
        const c = cost[(y - 1) * N + j] + 0.02 * ((i - j) * (i - j));  // numpy: 0.02 * d**2, in this order
        if (c < best) { best = c; arg = j; }
      }
      cost[y * N + i] += best;
      from[y * N + i] = arg;
    }
  }
  let end = 0;
  for (let i = 1; i < N; i++) if (cost[(RES - 1) * N + i] < cost[(RES - 1) * N + end]) end = i;
  const path = new Int32Array(RES);
  path[RES - 1] = end;
  for (let y = RES - 1; y > 0; y--) path[y - 1] = from[y * N + path[y]];

  const xs = new Float64Array(RES), ys = new Float64Array(RES);
  for (let k = 0; k < RES; k++) {
    const r = (MAX_R * (MIN_R + path[k])) / RES, th = (k * 2 * Math.PI) / RES;
    xs[k] = CENTER + r * Math.cos(th);
    ys[k] = CENTER + r * Math.sin(th);
  }
  return { xs, ys };
}

// ---------------------------------------------------------------- circle fits

/** Least-squares circle (x^2 + y^2 = 2a x + 2b y + c) through the given points. */
function circleFit(xs: ArrayLike<number>, ys: ArrayLike<number>, idx: ArrayLike<number>): { cx: number; cy: number; r: number } {
  // normal equations of B y = d, B = [x y 1], d = x^2 + y^2, solved by Cramer's rule
  let sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, n = 0, sxd = 0, syd = 0, sd = 0;
  for (let k = 0; k < idx.length; k++) {
    const x = xs[idx[k]], y = ys[idx[k]], d = x * x + y * y;
    sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y; n++; sxd += x * d; syd += y * d; sd += d;
  }
  if (n < 3) throw new Error('too few points');
  const det = (a: number[]) => a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  const A = [sxx, sxy, sx, sxy, syy, sy, sx, sy, n];
  const D = det(A);
  if (Math.abs(D) < 1e-9) throw new Error('degenerate points');
  const p = det([sxd, sxy, sx, syd, syy, sy, sd, sy, n]) / D;
  const q = det([sxx, sxd, sx, sxy, syd, sy, sx, sd, n]) / D;
  const c = det([sxx, sxy, sxd, sxy, syy, syd, sx, sy, sd]) / D;
  const cx = p / 2, cy = q / 2;
  return { cx, cy, r: Math.sqrt(c + cx * cx + cy * cy) };
}

/** mulberry32: a small fixed-seed generator, so the same photo always gives the same circle. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** circle_fit.find_circle: RANSAC over 3-point circles, then least squares on the best inliers. */
function findCircle(xs: Float64Array, ys: Float64Array): { cx: number; cy: number; r: number; inliers: number } {
  const n = xs.length, rand = rng(42);
  const dist = (c: { cx: number; cy: number; r: number }, k: number) => Math.abs(Math.hypot(xs[k] - c.cx, ys[k] - c.cy) - c.r);
  let best: number[] | undefined;
  for (let it = 0; it < 1000; it++) {
    let circle: { cx: number; cy: number; r: number } | undefined;
    for (let attempt = 0; attempt < 100; attempt++) {
      const i = Math.floor(rand() * n);
      let j = Math.floor(rand() * (n - 1)); if (j >= i) j++;
      let k = Math.floor(rand() * (n - 2)); if (k >= Math.min(i, j)) k++; if (k >= Math.max(i, j)) k++;
      try { circle = circleFit(xs, ys, [i, j, k]); } catch { circle = undefined; continue; }
      if (MIN_R < circle.r && circle.r < MAX_R) break;
    }
    if (!circle) continue;
    const inliers: number[] = [];
    for (let k = 0; k < n; k++) if (dist(circle, k) < 1) inliers.push(k);
    if (!best || inliers.length > best.length) best = inliers;
    if (inliers.length > n / 2) break;
  }
  if (!best || best.length < 0.2 * n) throw new Error('no circle found');
  const c = circleFit(xs, ys, best);
  let inliers = 0;
  for (let k = 0; k < n; k++) if (dist(c, k) < 1) inliers++;
  return { ...c, inliers };
}

/** Directions on the 256-point path that lie between the four straight sides of a rectangular frame. */
function cornerPoints(): number[] {
  const q = RES / 8, f = 0.3, idx: number[] = [];
  for (let r = 0; r < RES; r++) {
    const bottom = q < r && r < 3 * q, top = 5 * q < r && r < 7 * q;
    const right = r < (1 - f) * q || r > (7 + f) * q, left = (3 + f) * q < r && r < (5 - f) * q;
    if (!bottom && !top && !right && !left) idx.push(r);
  }
  return idx;
}

// ---------------------------------------------------------------- the bounds and the crop

/** The circular field of view of a colour fundus photograph, in its pixels. */
export function fundusBounds(img: RGBImage): FundusBounds {
  const { width: w, height: h } = img;
  const fullFrame = (): FundusBounds => ({ cx: w / 2, cy: h / 2, radius: Math.hypot(w / 2, h / 2) + Math.max(1, 0.001 * Math.max(h, w)), fullFrame: true, fit: 0 });
  try {
    // red channel, scaled by min(256/h, 256/w) about (h // 2, w // 2) into 256 x 256
    const red = new Uint8Array(w * h);
    for (let i = 0; i < red.length; i++) red[i] = img.data[i * 3];
    const s = Math.min(RES / h, RES / w), ox = Math.floor(w / 2), oy = Math.floor(h / 2);
    const small = warpLinear(red, w, h, 1, RES, RES, 1 / s, ox - CENTER / s, 1 / s, oy - CENTER / s);
    const back = (c: { cx: number; cy: number; r: number }, full: boolean, fit: number): FundusBounds =>
      ({ cx: (c.cx - CENTER) / s + ox, cy: (c.cy - CENTER) / s + oy, radius: c.r / s, fullFrame: full, fit });

    if (isFullFrame(small)) return back({ cx: CENTER, cy: CENTER, r: Math.hypot(CENTER, CENTER) + Math.max(1, 0.001 * RES) }, true, 0);

    const { xs, ys } = edgePoints(small);
    let circle: { cx: number; cy: number; r: number } | undefined;
    let fraction = 0;
    try {
      const c = findCircle(xs, ys);
      circle = c;
      fraction = c.inliers / RES;
    } catch { /* fraction stays 0 */ }
    if (fraction < 0.3) {
      // a rectangular frame with soft corners: fit the corners only, and allow for the soft edge
      const c = circleFit(xs, ys, cornerPoints());
      circle = { ...c, r: 0.95 * c.r };
    }
    return back(circle!, false, fraction);
  } catch {
    return fullFrame();
  }
}

/** The field of view cropped to size x size: its circle touches all four sides. */
export function cropFundus(img: RGBImage, b: FundusBounds, size = 1024): RGBImage {
  const s = size / (2 * b.radius);
  const data = warpLinear(img.data, img.width, img.height, 3, size, size, 1 / s, b.cx - size / 2 / s, 1 / s, b.cy - size / 2 / s);
  return { width: size, height: size, data };
}
