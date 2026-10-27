// skimage.morphology.skeletonize for 2-D images (Zhang-Suen variant with skimage's own lookup
// table, copied from scikit-image 0.25.2 _skeletonize_various_cy.pyx). Identical to skimage on
// every image tried, including real vessel maps.

const LUT = Uint8Array.from([
  0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
  0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
  0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
  3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
]);

/** Skeleton of a binary image (non-zero = foreground); returns 0/1. */
export function skeletonize(image: Uint8Array, w: number, h: number): Uint8Array {
  const W = w + 2;
  const H = h + 2;
  const sk = new Uint8Array(W * H);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) sk[(r + 1) * W + c + 1] = image[r * w + c] ? 1 : 0;
  const clean = sk.slice();
  // candidate list: only foreground pixels can be removed, and they never come back
  let active: number[] = [];
  for (let i = 0; i < sk.length; i++) if (sk[i]) active.push(i);
  let removed = true;
  while (removed) {
    removed = false;
    for (let pass = 0; pass < 2; pass++) {
      const first = pass === 0;
      for (const i of active) {
        if (!sk[i]) continue;
        const v = LUT[sk[i - W - 1] + 2 * sk[i - W] + 4 * sk[i - W + 1] + 8 * sk[i + 1] + 16 * sk[i + W + 1]
          + 32 * sk[i + W] + 64 * sk[i + W - 1] + 128 * sk[i - 1]];
        if (v === 3 || (v === 1 && first) || (v === 2 && !first)) {
          clean[i] = 0;
          removed = true;
        }
      }
      sk.set(clean);
    }
    active = active.filter((i) => sk[i]);
  }
  const out = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) out[r * w + c] = sk[(r + 1) * W + c + 1];
  return out;
}
