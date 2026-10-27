// AutoMorph v2 M0 (src/v2/M0/core.py): estimate the circular field of view,
// zero everything outside it, crop and pad to a square. Each OpenCV call is
// reproduced with OpenCV's own integer/rounding rules.

import { createRGB, type M0Geometry, type RGBImage } from './types';

/** cv2.cvtColor(RGB2GRAY) for 8-bit input: 14-bit fixed-point BT.601 weights. */
export function rgbToGray(img: RGBImage): Uint8Array {
  const n = img.width * img.height;
  const gray = new Uint8Array(n);
  const d = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 3) {
    gray[i] = (d[p] * 4899 + d[p + 1] * 9617 + d[p + 2] * 1868 + 8192) >> 14;
  }
  return gray;
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** cv2.normalize(src, None, 0, 255, NORM_MINMAX) for 8-bit input. */
export function normalizeMinMax(gray: Uint8Array): Uint8Array {
  let mn = 255;
  let mx = 0;
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const scale = 255 * (mx - mn > 0 ? 1 / (mx - mn) : 0);
  const shift = 0 - mn * scale;
  const a = Math.fround(scale);
  const b = Math.fround(shift);
  const lut = new Uint8Array(256);
  for (let v = 0; v < 256; v++) {
    const r = roundHalfEven(Math.fround(Math.fround(v * a) + b));
    lut[v] = r < 0 ? 0 : r > 255 ? 255 : r;
  }
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) out[i] = lut[gray[i]];
  return out;
}

/**
 * cv2.floodFill(img, mask, seed, newVal=0, loDiff=FLOODFILL_MASK_ONLY) as the
 * legacy code calls it: the flag lands in loDiff (saturated to 255), upDiff is
 * 0 and flags stay 4-connected, so a neighbour joins when its value is <= the
 * value of the pixel it is reached from. Filled pixels become 0 afterwards.
 */
function floodFillLegacy(values: Uint8Array, visited: Uint8Array, w: number, h: number, sx: number, sy: number): void {
  const seed = sy * w + sx;
  if (visited[seed]) return;
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;
  visited[seed] = 1;
  queue[tail++] = seed;
  while (head < tail) {
    const u = queue[head++];
    const vu = values[u];
    const x = u % w;
    const y = (u - x) / w;
    if (x > 0) {
      const v = u - 1;
      if (!visited[v] && values[v] <= vu) { visited[v] = 1; queue[tail++] = v; }
    }
    if (x < w - 1) {
      const v = u + 1;
      if (!visited[v] && values[v] <= vu) { visited[v] = 1; queue[tail++] = v; }
    }
    if (y > 0) {
      const v = u - w;
      if (!visited[v] && values[v] <= vu) { visited[v] = 1; queue[tail++] = v; }
    }
    if (y < h - 1) {
      const v = u + w;
      if (!visited[v] && values[v] <= vu) { visited[v] = 1; queue[tail++] = v; }
    }
  }
  for (let i = 0; i < tail; i++) values[queue[i]] = 0;
}

/**
 * Binary erode/dilate with a kw x kh rectangle anchored at (kw/2, kh/2), as
 * cv2.erode/cv2.dilate with the default border (pixels outside are ignored).
 */
function rectMorph(src: Uint8Array, w: number, h: number, kw: number, kh: number, op: 'erode' | 'dilate'): Uint8Array {
  const ax = kw >> 1;
  const ay = kh >> 1;
  const want = op === 'erode' ? 0 : 1; // erode looks for any 0, dilate for any 1
  const tmp = new Uint8Array(w * h);
  const prefix = new Int32Array(w + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) prefix[x + 1] = prefix[x] + (src[row + x] === want ? 1 : 0);
    for (let x = 0; x < w; x++) {
      const a = Math.max(0, x - ax);
      const b = Math.min(w - 1, x + (kw - 1 - ax));
      const hit = prefix[b + 1] - prefix[a] > 0;
      tmp[row + x] = op === 'erode' ? (hit ? 0 : 1) : hit ? 1 : 0;
    }
  }
  const out = new Uint8Array(w * h);
  const colPrefix = new Int32Array(h + 1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) colPrefix[y + 1] = colPrefix[y] + (tmp[y * w + x] === want ? 1 : 0);
    for (let y = 0; y < h; y++) {
      const a = Math.max(0, y - ay);
      const b = Math.min(h - 1, y + (kh - 1 - ay));
      const hit = colPrefix[b + 1] - colPrefix[a] > 0;
      out[y * w + x] = op === 'erode' ? (hit ? 0 : 1) : hit ? 1 : 0;
    }
  }
  return out;
}

/** M0 _get_mask_bz: threshold, legacy flood fill from two corners, 20x20 opening. */
export function roughMask(gray: Uint8Array, w: number, h: number): Uint8Array {
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const threshold = sum / gray.length / 3.0 - 5.0;
  const ithresh = Math.floor(Math.max(5.0, threshold));
  const mask = new Uint8Array(gray.length);
  const fill = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    mask[i] = gray[i] > ithresh ? 1 : 0;
    fill[i] = 1 - mask[i];
  }
  const visited = new Uint8Array(gray.length);
  floodFillLegacy(fill, visited, w, h, 0, 0);
  floodFillLegacy(fill, visited, w, h, w - 1, h - 1);
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i] + fill[i];
  const eroded = rectMorph(mask, w, h, 20, 20, 'erode');
  return rectMorph(eroded, w, h, 20, 20, 'dilate');
}

/** cv2.getStructuringElement(MORPH_ELLIPSE, (k, k)) as per-row [j1, j2) spans. */
function ellipseSpans(k: number): Array<[number, number]> {
  const r = k >> 1;
  const c = k >> 1;
  const invR2 = r ? 1 / (r * r) : 0;
  const spans: Array<[number, number]> = [];
  for (let i = 0; i < k; i++) {
    const dy = i - r;
    let j1 = 0;
    let j2 = 0;
    if (Math.abs(dy) <= r) {
      const dx = roundHalfEven(c * Math.sqrt((r * r - dy * dy) * invR2));
      j1 = Math.max(c - dx, 0);
      j2 = Math.min(c + dx + 1, k);
    }
    spans.push([j1, j2]);
  }
  return spans;
}

/**
 * Pixels where cv2.morphologyEx(mask, MORPH_GRADIENT, ellipse(k)) is non-zero,
 * i.e. whose kernel window holds both 0 and 1. Only pixels near a 0/1
 * transition can qualify, so those are found first and then checked exactly.
 */
function gradientEdges(mask: Uint8Array, w: number, h: number, k: number): Int32Array {
  const c = k >> 1;
  const spans = ellipseSpans(k);
  // transition pixels (4-neighbour differs)
  const boundary = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x + 1 < w && mask[i] !== mask[i + 1]) { boundary[i] = 1; boundary[i + 1] = 1; }
      if (y + 1 < h && mask[i] !== mask[i + w]) { boundary[i] = 1; boundary[i + w] = 1; }
    }
  }
  // candidates: within Chebyshev distance c of a transition pixel
  const near = rectMorph(boundary, w, h, 2 * c + 1, 2 * c + 1, 'dilate');
  // per-row prefix counts of ones
  const prefix = new Int32Array((w + 1) * h);
  for (let y = 0; y < h; y++) {
    const base = y * (w + 1);
    const row = y * w;
    for (let x = 0; x < w; x++) prefix[base + x + 1] = prefix[base + x] + mask[row + x];
  }
  const edges: number[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!near[y * w + x]) continue;
      let anyOne = false;
      let anyZero = false;
      for (let i = 0; i < k && !(anyOne && anyZero); i++) {
        const [j1, j2] = spans[i];
        if (j2 <= j1) continue;
        const yy = y + i - c;
        if (yy < 0 || yy >= h) continue;
        const a = Math.max(0, x + j1 - c);
        const b = Math.min(w - 1, x + j2 - 1 - c);
        if (b < a) continue;
        const ones = prefix[yy * (w + 1) + b + 1] - prefix[yy * (w + 1) + a];
        if (ones > 0) anyOne = true;
        if (ones < b - a + 1) anyZero = true;
      }
      if (anyOne && anyZero) edges.push(y * w + x);
    }
  }
  return Int32Array.from(edges);
}

/** M0 legacy_geometry: centre from row/column sums, radius from the edge-distance histogram. */
export function legacyGeometry(img: RGBImage): M0Geometry {
  const { width: w, height: h } = img;
  const normalized = normalizeMinMax(rgbToGray(img));
  const mask = roughMask(normalized, w, h);

  const rowSums = new Float64Array(h);
  const colSums = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = mask[y * w + x];
      rowSums[y] += v;
      colSums[x] += v;
    }
  }
  const meanAbove = (sums: Float64Array): number => {
    let max = 0;
    for (const s of sums) if (s > max) max = s;
    let total = 0;
    let count = 0;
    for (let i = 0; i < sums.length; i++) {
      if (sums[i] > max * 0.95) { total += i; count++; }
    }
    if (!count) throw new Error('fundus centre could not be estimated');
    return total / count;
  };
  const centreRow = meanAbove(rowSums);
  const centreCol = meanAbove(colSums);

  const ksize = Math.max(Math.floor(w / 400) * 2 + 1, 3);
  const edges = gradientEdges(mask, w, h, ksize);
  if (!edges.length) throw new Error('fundus radius could not be estimated');
  const bins: number[] = [];
  for (const e of edges) {
    const col = e % w;
    const row = (e - col) / w;
    const d = Math.ceil(Math.sqrt((row - centreRow) ** 2 + (col - centreCol) ** 2));
    bins[d] = (bins[d] ?? 0) + 1;
  }
  let maxCount = 0;
  for (const c of bins) if (c !== undefined && c > maxCount) maxCount = c;
  let radius = -1;
  for (let i = 0; i < bins.length; i++) if ((bins[i] ?? 0) > maxCount * 0.995) radius = i;
  if (radius < 0) throw new Error('fundus radius histogram is empty');

  const startH = Math.max(0, Math.trunc(centreRow - radius));
  const startW = Math.max(0, Math.trunc(centreCol - radius));
  return {
    centreRow,
    centreCol,
    radius,
    bbox: [startH, startW, Math.min(h - startH, 2 * radius), Math.min(w - startW, 2 * radius)],
  };
}

/** cv2.circle(img, center, radius, 1, thickness=-1) with LINE_8: OpenCV's midpoint fill. */
export function filledCircle(w: number, h: number, cx: number, cy: number, radius: number): Uint8Array {
  const out = new Uint8Array(w * h);
  const hline = (y: number, x1: number, x2: number) => {
    if (y < 0 || y >= h) return;
    const a = Math.max(x1, 0);
    const b = Math.min(x2, w - 1);
    for (let x = a; x <= b; x++) out[y * w + x] = 1;
  };
  let err = 0;
  let dx = radius;
  let dy = 0;
  let plus = 1;
  let minus = (radius << 1) - 1;
  while (dx >= dy) {
    const y11 = cy - dy, y12 = cy + dy, y21 = cy - dx, y22 = cy + dx;
    const x11 = cx - dx, x12 = cx + dx, x21 = cx - dy, x22 = cx + dy;
    if (x11 < w && x12 >= 0 && y21 < h && y22 >= 0) {
      hline(y11, x11, x12);
      hline(y12, x11, x12);
      if (x21 < w && x22 >= 0) {
        hline(y21, x21, x22);
        hline(y22, x21, x22);
      }
    }
    dy++;
    err += plus;
    plus += 2;
    const m = (err <= 0 ? 1 : 0) - 1; // 0 when err <= 0, else -1
    err -= minus & m;
    dx += m;
    minus -= m & 2;
  }
  return out;
}

/** M0 _mask_crop_pad_cpu: zero outside the circle, crop the bbox, centre-pad to a square. */
export function maskCropPad(img: RGBImage, g: M0Geometry): RGBImage {
  const { width: w, height: h, data } = img;
  const circle = filledCircle(w, h, Math.trunc(g.centreCol), Math.trunc(g.centreRow), g.radius);
  const [startH, startW, cropH, cropW] = g.bbox;
  const side = Math.max(cropH, cropW);
  const out = createRGB(side, side);
  const top = Math.trunc(side / 2 - cropH / 2);
  const left = Math.trunc(side / 2 - cropW / 2);
  for (let y = 0; y < cropH; y++) {
    const sy = startH + y;
    for (let x = 0; x < cropW; x++) {
      const sx = startW + x;
      const si = sy * w + sx;
      if (!circle[si]) continue;
      const p = si * 3;
      const q = ((top + y) * side + left + x) * 3;
      out.data[q] = data[p];
      out.data[q + 1] = data[p + 1];
      out.data[q + 2] = data[p + 2];
    }
  }
  return out;
}

export function runM0(img: RGBImage): { image: RGBImage; geometry: M0Geometry } {
  const geometry = legacyGeometry(img);
  return { image: maskCropPad(img, geometry), geometry };
}
