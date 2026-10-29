// The image operations RetiFlow takes from SciPy, scikit-image and OpenCV, written to give the same
// results: labels in SciPy's order, the exact Euclidean distance transform, binary dilation, hysteresis
// thresholds and small-object removal as scikit-image 0.26 does them, Zhang-Suen skeletons with
// scikit-image's lookup table, and NumPy's pairwise summation where RetiFlow sums lengths.
// Images are row-major (C order); masks are Uint8Array of 0 and 1.

/** 4- or 8-connected components, numbered 1.. in the order their first pixel appears (SciPy's order). */
export function label(mask: Uint8Array, w: number, h: number, eight: boolean): { labels: Int32Array; count: number } {
  const parent = new Int32Array(w * h).fill(-1);
  const find = (i: number) => {
    let r = i;
    while (parent[r] !== r) r = parent[r];
    while (parent[i] !== r) { const n = parent[i]; parent[i] = r; i = n; }
    return r;
  };
  const union = (a: number, b: number) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) { if (ra < rb) parent[rb] = ra; else parent[ra] = rb; }
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      parent[i] = i;
      if (x > 0 && mask[i - 1]) union(i, i - 1);
      if (y > 0) {
        if (mask[i - w]) union(i, i - w);
        if (eight && x > 0 && mask[i - w - 1]) union(i, i - w - 1);
        if (eight && x + 1 < w && mask[i - w + 1]) union(i, i - w + 1);
      }
    }
  }
  const labels = new Int32Array(w * h);
  const id = new Map<number, number>();
  let count = 0;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) continue;
    const r = find(i);
    let l = id.get(r);
    if (l === undefined) { l = ++count; id.set(r, l); }
    labels[i] = l;
  }
  return { labels, count };
}

/** Bounding boxes of labels 1..count: [y0, x0, y1, x1) (scipy.ndimage.find_objects). */
export function boxes(labels: Int32Array, w: number, h: number, count: number): Array<[number, number, number, number] | null> {
  const out: Array<[number, number, number, number] | null> = Array.from({ length: count }, () => null);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const l = labels[y * w + x];
      if (!l) continue;
      const b = out[l - 1];
      if (!b) out[l - 1] = [y, x, y + 1, x + 1];
      else { b[0] = Math.min(b[0], y); b[1] = Math.min(b[1], x); b[2] = Math.max(b[2], y + 1); b[3] = Math.max(b[3], x + 1); }
    }
  }
  return out;
}

/** Exact Euclidean distance from each foreground pixel to the nearest background pixel
 *  (scipy.ndimage.distance_transform_edt; pixels outside the image do not count). */
export function edt(mask: Uint8Array, w: number, h: number): Float64Array {
  const INF = 1e20;
  const f = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) f[i] = mask[i] ? INF : 0;
  const pass = (n: number, get: (i: number) => number, set: (i: number, v: number) => void) => {
    // Felzenszwalb and Huttenlocher's 1-D squared distance transform: exact for integer positions
    const v = new Int32Array(n), z = new Float64Array(n + 1), d = new Float64Array(n), g = new Float64Array(n);
    for (let i = 0; i < n; i++) g[i] = get(i);
    let k = 0;
    v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < n; q++) {
      let s = ((g[q] + q * q) - (g[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) { k--; s = ((g[q] + q * q) - (g[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
      k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q++) {
      while (z[k + 1] < q) k++;
      d[q] = (q - v[k]) ** 2 + g[v[k]];
    }
    for (let i = 0; i < n; i++) set(i, d[i]);
  };
  for (let x = 0; x < w; x++) pass(h, (i) => f[i * w + x], (i, v) => { f[i * w + x] = v; });
  for (let y = 0; y < h; y++) pass(w, (i) => f[y * w + i], (i, v) => { f[y * w + i] = v; });
  const out = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = mask[i] ? Math.sqrt(f[i] >= INF ? Infinity : Math.round(f[i])) : 0;
  return out;
}

/** Binary dilation with the full 3 x 3 square (scipy.ndimage.binary_dilation, structure ones((3, 3))). */
export function dilate8(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < w) out[yy * w + xx] = 1;
        }
      }
    }
  }
  return out;
}

/** skimage.filters.apply_hysteresis_threshold: pixels above low, in 4-connected regions with a pixel above high. */
export function hysteresis(img: Float32Array, w: number, h: number, low: number, high: number): Uint8Array {
  const lo = Math.fround(Math.min(low, high)), hi = Math.fround(high);
  const above = Uint8Array.from(img, (v) => (v > lo ? 1 : 0));
  const { labels, count } = label(above, w, h, false);
  const keep = new Uint8Array(count + 1);
  for (let i = 0; i < w * h; i++) if (img[i] > hi && labels[i]) keep[labels[i]] = 1;
  return Uint8Array.from(labels, (l) => (l && keep[l] ? 1 : 0));
}

/** remove_small_objects(mask, min_size=minSize), 4-connected, as scikit-image 0.26 runs it: its deprecation
 *  wrapper passes min_size on as max_size, so pieces of minSize pixels or fewer go (its documentation says
 *  fewer than minSize; earlier versions did that). */
export function removeSmall(mask: Uint8Array, w: number, h: number, minSize: number): Uint8Array {
  const { labels, count } = label(mask, w, h, false);
  const size = new Int32Array(count + 1);
  for (let i = 0; i < w * h; i++) size[labels[i]]++;
  return Uint8Array.from(labels, (l) => (l && size[l] > minSize ? 1 : 0));
}

// scikit-image's Zhang-Suen lookup table (_skeletonize_various_cy.pyx, _fast_skeletonize)
const LUT = [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3,
  1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0,
  1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0];

/** skimage.morphology.skeletonize for 2-D images (Zhang and Suen, scikit-image's fast loop). */
export function skeletonize(mask: Uint8Array, w: number, h: number): Uint8Array {
  const W = w + 2, H = h + 2;
  const skel = new Uint8Array(W * H);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) skel[(y + 1) * W + x + 1] = mask[y * w + x] ? 1 : 0;
  const clean = skel.slice();
  let removed = true;
  while (removed) {
    removed = false;
    for (let pass = 0; pass < 2; pass++) {
      const first = pass === 0;
      for (let r = 1; r < H - 1; r++) {
        for (let c = 1; c < W - 1; c++) {
          if (!skel[r * W + c]) continue;
          const n = LUT[skel[(r - 1) * W + c - 1] + 2 * skel[(r - 1) * W + c] + 4 * skel[(r - 1) * W + c + 1] + 8 * skel[r * W + c + 1]
            + 16 * skel[(r + 1) * W + c + 1] + 32 * skel[(r + 1) * W + c] + 64 * skel[(r + 1) * W + c - 1] + 128 * skel[r * W + c - 1]];
          if (n === 0) continue;
          if (n === 3 || (n === 1 && first) || (n === 2 && !first)) { clean[r * W + c] = 0; removed = true; }
        }
      }
      skel.set(clean);
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = skel[(y + 1) * W + x + 1];
  return out;
}

/** OpenCV's GaussianBlur(src, (0, 0), sigma) on a float32 image (kernel size round(8 sigma + 1) | 1,
 *  BORDER_REFLECT_101), computed in float64; RetiFlow only takes the position of its maximum. */
export function cvGaussian(src: Float32Array, w: number, h: number, sigma: number): Float64Array {
  const ksize = (Math.round(sigma * 4 * 2 + 1)) | 1, r = (ksize - 1) / 2;
  const k = new Float64Array(ksize);
  let sum = 0;
  for (let i = 0; i < ksize; i++) { k[i] = Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)); sum += k[i]; }
  const scale = 1 / sum;  // getGaussianKernel: in double, times the reciprocal, stored as float
  for (let i = 0; i < ksize; i++) k[i] = Math.fround(k[i] * scale);
  const reflect101 = (i: number, n: number) => {
    if (n === 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  };
  const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let j = 0; j < ksize; j++) acc += src[y * w + reflect101(x + j - r, w)] * k[j];
      tmp[y * w + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let j = 0; j < ksize; j++) acc += tmp[reflect101(y + j - r, h) * w + x] * k[j];
      out[y * w + x] = acc;
    }
  }
  return out;
}

/** The index of the first maximum (NumPy's argmax). */
export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** NumPy's pairwise summation of float64 values (np.sum on an array), so lengths compare as RetiFlow's do. */
export function npSum(a: ArrayLike<number>, start = 0, n = a.length): number {
  if (n < 8) {
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[start + i];
    return s;
  }
  if (n <= 128) {
    const r = new Float64Array(8);
    for (let j = 0; j < 8; j++) r[j] = a[start + j];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[start + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[start + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return npSum(a, start, n2) + npSum(a, start + n2, n - n2);
}

/** The length of a pixel path: the sum of its steps' lengths, summed as NumPy does. */
export function pathLength(path: Array<[number, number]>): number {
  const steps = new Float64Array(Math.max(0, path.length - 1));
  for (let i = 1; i < path.length; i++) steps[i - 1] = Math.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2);
  return npSum(steps);
}

/** Python's round-half-to-even (numpy.rint). */
export function rint(v: number): number {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
