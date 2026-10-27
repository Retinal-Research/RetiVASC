// AutoMorph v2 M3 vessel metrics (a derivative of Retipy, GPL-3.0-or-later; see
// automorph_v2/src/v2/M3/metrics.py, traversal.py and width.py). Arithmetic and accumulation
// order follow the Python code line by line; where Python would raise, this throws PyError,
// and the caller turns that into AutoMorph's all -1 row.

import { circleOffsets } from './cv';
import { pairwiseSum } from './pyfloat';

export class PyError extends Error {}

export type VesselType = 'binary' | 'artery' | 'vein';

export interface MetricResult {
  fractalDimension: number;
  vesselDensity: number;
  averageWidth: number;
  distanceTortuosity: number;
  squaredCurvatureTortuosity: number;
  tortuosityDensity: number;
  craeHubbard: number;
  crveHubbard: number;
  craeKnudtson: number;
  crveKnudtson: number;
}

export const SENTINEL: MetricResult = {
  fractalDimension: -1, vesselDensity: -1, averageWidth: -1, distanceTortuosity: -1, squaredCurvatureTortuosity: -1,
  tortuosityDensity: -1, craeHubbard: -1, crveHubbard: -1, craeKnudtson: -1, crveKnudtson: -1,
};

const div = (a: number, b: number) => {
  if (b === 0) throw new PyError('division by zero');
  return a / b;
};

// ---------------------------------------------------------------- global metrics

/** Least-squares slope of y on x, as np.polyfit(x, y, 1)[0] (scaled Vandermonde, then solve). */
function polyfitSlope(x: number[], y: number[]): number {
  let s0 = 0;
  for (const v of x) s0 += v * v;
  const scale0 = Math.sqrt(s0);
  const scale1 = Math.sqrt(x.length);
  // normal equations of the scaled problem [x/scale0, 1/scale1] c = y
  let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
  for (let i = 0; i < x.length; i++) {
    const u = x[i] / scale0;
    const v = 1 / scale1;
    a00 += u * u; a01 += u * v; a11 += v * v; b0 += u * y[i]; b1 += v * y[i];
  }
  const det = a00 * a11 - a01 * a01;
  const c0 = (b0 * a11 - a01 * b1) / det;
  return c0 / scale0;
}

function boxcount(img: Uint8Array, h: number, w: number, size: number): number {
  let count = 0;
  for (let r0 = 0; r0 < h; r0 += size) {
    for (let c0 = 0; c0 < w; c0 += size) {
      let sum = 0;
      const r1 = Math.min(r0 + size, h), c1 = Math.min(c0 + size, w);
      for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) sum += img[r * w + c];
      if (sum > 0 && sum < size * size) count++;
    }
  }
  return count;
}

/** Box-counting fractal dimension of a 0/1 image. */
export function fractalDimension(img01: Uint8Array, h: number, w: number): number {
  const side = Math.min(h, w);
  const top = 2 ** Math.floor(Math.log(side) / Math.log(2));
  const exponent = Math.trunc(Math.log(top) / Math.log(2));
  const sizes: number[] = [];
  for (let k = exponent; k > 1; k--) sizes.push(2 ** k);
  const counts = sizes.map((s) => boxcount(img01, h, w, s));
  // log(0) = -inf makes numpy's least squares fail ("SVD did not converge")
  if (counts.some((c) => c === 0)) throw new PyError('empty box count');
  return -polyfitSlope(sizes.map(Math.log), counts.map(Math.log));
}

// ---------------------------------------------------------------- vessel traversal

/** Retipy's detect_vessel_border on a skeleton: junctions removed, then FIFO traversal per seed. */
export function extractVessels(skeleton: Uint8Array, h: number, w: number, ignored = 1): Array<[number[], number[]]> {
  const active = new Uint8Array(h * w);
  for (let i = 0; i < active.length; i++) active[i] = skeleton[i] > 0 ? 1 : 0;
  const inside = (x: number, y: number) => x >= ignored && x < h - ignored && y >= ignored && y < w - ignored;
  // neighbour counts with edge-clamped coordinates, as Retipy's max/min clamping does
  const clampedCount = (x: number, y: number) => {
    let n = 0;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (!dx && !dy) continue;
        const xx = Math.min(Math.max(x + dx, 0), h - 1), yy = Math.min(Math.max(y + dy, 0), w - 1);
        n += active[xx * w + yy];
      }
    }
    return n;
  };
  const work = active.slice();
  for (let x = 0; x < h; x++) {
    for (let y = 0; y < w; y++) {
      if (!active[x * w + y] || !inside(x, y) || clampedCount(x, y) <= 2) continue;
      // a filled radius-1 OpenCV circle: the pixel and its four axial neighbours
      work[x * w + y] = 0;
      if (x > 0) work[(x - 1) * w + y] = 0;
      if (x < h - 1) work[(x + 1) * w + y] = 0;
      if (y > 0) work[x * w + y - 1] = 0;
      if (y < w - 1) work[x * w + y + 1] = 0;
    }
  }
  const seeds: number[] = [];
  for (let x = 0; x < h; x++) for (let y = 0; y < w; y++) if (work[x * w + y] && inside(x, y)) seeds.push(x * w + y);
  const vessels: Array<[number[], number[]]> = [];
  let queue = new Int32Array(1024);
  for (const seed of seeds) {
    if (!work[seed]) continue;
    let head = 0, tail = 0;
    queue[tail++] = seed;
    const xs: number[] = [], ys: number[] = [];
    while (head < tail) {
      const p = queue[head++];
      if (!work[p]) continue;
      const x = Math.floor(p / w), y = p - x * w;
      xs.push(x);
      ys.push(y);
      work[p] = 0;
      const xl = Math.max(0, x - 1), yl = Math.max(0, y - 1), xm = Math.min(h - 1, x + 1), ym = Math.min(w - 1, y + 1);
      // Retipy's neighbour order; clamped duplicates are appended too
      const cand = [xl * w + yl, xl * w + y, xl * w + ym, x * w + yl, x * w + ym, xm * w + yl, xm * w + y, xm * w + ym];
      if (tail + 8 > queue.length) {
        const bigger = new Int32Array(queue.length * 2);
        bigger.set(queue.subarray(0, tail));
        queue = bigger;
      }
      for (const c of cand) if (work[c]) queue[tail++] = c;
    }
    vessels.push([xs, ys]);
  }
  return vessels;
}

// ---------------------------------------------------------------- tortuosity

const dist = (x1: number, y1: number, x2: number, y2: number) => Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);

function curveLength(x: number[], y: number[], from = 0, to = x.length): number {
  let d = 0;
  for (let i = from; i < to - 1; i++) d += dist(x[i], y[i], x[i + 1], y[i + 1]);
  return d;
}

const chordLength = (x: number[], y: number[], from = 0, to = x.length) => dist(x[from], y[from], x[to - 1], y[to - 1]);

function distanceTortuosity(x: number[], y: number[]): number {
  if (x.length < 2) throw new PyError('curve too short');
  return div(curveLength(x, y), chordLength(x, y));
}

function inflectionPoints(y: number[]): number[] {
  const n = y.length;
  const conv = (k: number) => (k === 0 ? y[0] : k === n ? -y[n - 1] : y[k] - y[k - 1]);
  const points: number[] = [];
  for (let it = 2; it < n; it++) if (Math.sign(conv(it)) !== Math.sign(conv(it - 1))) points.push(it - 1);
  return points;
}

function tortuosityDensity(x: number[], y: number[]): number {
  const points = inflectionPoints(y);
  const count = points.length;
  if (!count) return 0;
  let start = 0;
  let sum = 0;
  for (const p of points) {
    // Python slices x[start:point]; an empty slice would raise IndexError
    if (p <= start) throw new PyError('empty segment');
    const chord = chordLength(x, y, start, p);
    if (chord) sum += curveLength(x, y, start, p) / chordLength(x, y, start, p) - 1;
    start = p;
  }
  return (count - 1) / count + div(1, curveLength(x, y)) * sum;
}

function squaredCurvatureTortuosity(x: number[], y: number[]): number {
  const curv: number[] = [];
  for (let i = 1; i < x.length - 1; i++) {
    const x1 = (x[i + 1] - x[i - 1]) / 2;
    const x2 = (x[i + 1] - 2 * x[i] + x[i - 1]) / 4;
    const y1 = (y[i + 1] - y[i - 1]) / 2;
    const y2 = (y[i + 1] - 2 * y[i] + y[i - 1]) / 4;
    curv.push(div(x1 * y2 - x2 * y1, (y1 * y1 + x1 * x1) ** 1.5));
  }
  // np.trapz with unit spacing: pairwise sum of (c[k+1] + c[k]) / 2
  const terms = new Float64Array(Math.max(0, curv.length - 1));
  for (let k = 0; k < terms.length; k++) terms[k] = (curv[k + 1] + curv[k]) / 2;
  return Math.abs(pairwiseSum(terms));
}

// ---------------------------------------------------------------- calibre

/** Retipy width_measurement: grow OpenCV circles at each point until one touches background. */
function widths(x: number[], y: number[], vessel: Uint8Array, h: number, w: number, resolution: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < x.length - 1; i++) {
    const cr = x[i], cc = y[i];
    for (let radius = 1; ; radius++) {
      const o = circleOffsets(radius);
      let zeros = 0;
      for (let k = 0; k < o.length; k += 2) {
        const r = o[k] + cr, c = o[k + 1] + cc;
        if (r < 0 || r >= h || c < 0 || c >= w) continue;
        if (vessel[r * w + c] === 0) zeros++;
      }
      if (zeros === 0) continue;
      const cal = zeros === 2 || zeros === 3 ? radius * 2 - 1 : radius * 2;
      out.push(cal * resolution);
      break;
    }
  }
  return out;
}

function hubbard(w1: number, w2: number): [number, number] {
  return [Math.sqrt(0.87 * (w1 * w1) + 1.01 * (w2 * w2) - 0.22 * w1 * w2 - 10.76), Math.sqrt(0.72 * (w1 * w1) + 0.91 * (w2 * w2) + 450.05)];
}

function knudtson(w1: number, w2: number): [number, number] {
  return [0.88 * Math.sqrt(w1 * w1 + w2 * w2), 0.95 * Math.sqrt(w1 * w1 + w2 * w2)];
}

const sorted = (a: number[]) => [...a].sort((p, q) => p - q);

function calibre(averages: number[], type: VesselType): [number, number, number, number] {
  const selected = sorted(averages).slice(-6);
  if (selected.length < 6) return [-1, -1, -1, -1];  // Python: IndexError, caught
  const pairs: Array<[number, number]> = [[0, 5], [1, 4], [2, 3]];
  const run = (cal: (a: number, b: number) => [number, number]): [number, number] => {
    const firstA = sorted(pairs.map(([l, r]) => cal(selected[l], selected[r])[0]));
    const firstV = sorted(pairs.map(([l, r]) => cal(selected[l], selected[r])[1]));
    if (type === 'artery') {
      const second = cal(firstA[0], firstA[2])[0];
      const s = sorted([second, firstA[1]]);
      return cal(s[0], s[1]);
    }
    const second = cal(firstV[0], firstV[2])[1];
    const s = sorted([second, firstV[1]]);
    return cal(s[0], s[1]);
  };
  const [craeH, crveH] = run(hubbard);
  const [craeK, crveK] = run(knudtson);
  return [craeH, crveH, craeK, crveK];
}

// ---------------------------------------------------------------- one image, one vessel type, one region

/** evaluate_arrays: skeleton and vessel images are 912 x 912, values 0/255. */
export function evaluate(skeleton: Uint8Array, vessel: Uint8Array, h: number, w: number, resolution: number, isZone: boolean,
  type: VesselType, minPixels = 15): MetricResult {
  let skelSum = 0, skelMax = 0, vesselMax = 0;
  for (let i = 0; i < skeleton.length; i++) { skelSum += skeleton[i]; if (skeleton[i] > skelMax) skelMax = skeleton[i]; if (vessel[i] > vesselMax) vesselMax = vessel[i]; }
  if (skelSum < minPixels) throw new PyError('No windows were created for the given retinal image');
  // global metrics on the image divided by 255 (when its maximum is above one)
  const v01 = new Uint8Array(vessel.length);
  let vesselSum = 0, vesselOnes = 0, skelCount = 0;
  for (let i = 0; i < vessel.length; i++) {
    const v = vesselMax > 1 ? vessel[i] / 255 : vessel[i];
    v01[i] = v === 1 ? 1 : 0;
    vesselSum += v;
    if (v === 1) vesselOnes++;
    skelCount += skelMax > 1 ? skeleton[i] / 255 : skeleton[i];
  }
  const fd = fractalDimension(v01, h, w);
  const vd = vesselOnes / (h * w);
  const averageWidth = (vesselSum / skelCount) * resolution;

  const vessels = extractVessels(skeleton, h, w);
  let count = 0, t2 = 0, t4 = 0, td = 0;
  const averages: number[] = [];
  for (const [x, y] of vessels) {
    if (x.length <= minPixels) continue;
    count++;
    t2 += distanceTortuosity(x, y);
    t4 += squaredCurvatureTortuosity(x, y);
    td += tortuosityDensity(x, y);
    if (isZone) {
      const ws = widths(x, y, vessel, h, w, resolution);
      let s = 0;
      for (const v of ws) s += v;
      averages.push(s / ws.length);
    }
  }
  if (count > 0) { t2 /= count; t4 /= count; td /= count; }
  const [craeH, crveH, craeK, crveK] = isZone ? calibre(averages, type) : [-1, -1, -1, -1];
  return { fractalDimension: fd, vesselDensity: vd, averageWidth, distanceTortuosity: t2, squaredCurvatureTortuosity: t4,
    tortuosityDensity: td, craeHubbard: craeH, crveHubbard: crveH, craeKnudtson: craeK, crveKnudtson: crveK };
}
