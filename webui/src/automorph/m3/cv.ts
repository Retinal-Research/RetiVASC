// The OpenCV operations M3 depends on, reproduced exactly (checked against cv2 4.12):
// filled circles (zone rings, junction removal, vessel width) and nearest-neighbour resizing.

/** cv2.circle(img, (cx, cy), radius, value, thickness=-1): OpenCV's integer midpoint fill, clipped. */
export function circleFill(img: Uint8Array, w: number, h: number, cx: number, cy: number, radius: number, value: number): void {
  let err = 0;
  let dx = radius;
  let dy = 0;
  let plus = 1;
  let minus = (radius << 1) - 1;
  const inside = cx >= radius && cx < w - radius && cy >= radius && cy < h - radius;
  const hline = (y: number, x1: number, x2: number) => {
    if (y < 0 || y >= h) return;
    const a = Math.max(x1, 0);
    const b = Math.min(x2, w - 1);
    if (a <= b) img.fill(value, y * w + a, y * w + b + 1);
  };
  while (dx >= dy) {
    const y11 = cy - dy, y12 = cy + dy, y21 = cy - dx, y22 = cy + dx;
    const x11 = cx - dx, x12 = cx + dx, x21 = cx - dy, x22 = cx + dy;
    if (inside) {
      img.fill(value, y11 * w + x11, y11 * w + x12 + 1);
      img.fill(value, y12 * w + x11, y12 * w + x12 + 1);
      img.fill(value, y21 * w + x21, y21 * w + x22 + 1);
      img.fill(value, y22 * w + x21, y22 * w + x22 + 1);
    } else if (x11 < w && x12 >= 0 && y21 < h && y22 >= 0) {
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
    const mask = (err <= 0 ? 1 : 0) - 1;
    err -= minus & mask;
    dx += mask;
    minus -= mask & 2;
  }
}

const offsets = new Map<number, Int32Array>();

/** Row/column offsets of a filled radius-r OpenCV circle, row-major (cached). */
export function circleOffsets(radius: number): Int32Array {
  let o = offsets.get(radius);
  if (o) return o;
  const size = radius * 2 + 1;
  const canvas = new Uint8Array(size * size);
  circleFill(canvas, size, size, radius, radius, radius, 1);
  const list: number[] = [];
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (canvas[r * size + c]) list.push(r - radius, c - radius);
  o = Int32Array.from(list);
  if (radius <= 128) offsets.set(radius, o);
  return o;
}

/** Source index for each destination index of cv2.resize(..., INTER_NEAREST). */
export function nearestIndex(src: number, dst: number): Int32Array {
  const ifx = 1 / (dst / src);
  const out = new Int32Array(dst);
  for (let x = 0; x < dst; x++) out[x] = Math.min(Math.floor(x * ifx), src - 1);
  return out;
}

/** Nearest-neighbour resize of a square single-channel map, as cv2.resize does. */
export function resizeNearest(src: Uint8Array, srcSide: number, dstSide: number): Uint8Array {
  const idx = nearestIndex(srcSide, dstSide);
  const out = new Uint8Array(dstSide * dstSide);
  for (let y = 0; y < dstSide; y++) {
    const row = idx[y] * srcSide;
    for (let x = 0; x < dstSide; x++) out[y * dstSide + x] = src[row + idx[x]];
  }
  return out;
}
