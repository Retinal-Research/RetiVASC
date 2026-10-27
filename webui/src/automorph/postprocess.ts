// AutoMorph v2 M1/M2 output decoding. Small-object removal follows
// skimage.morphology.remove_small_objects: components smaller than minSize go.

import { QUALITY_CLASSES, type LabelMap, type QualityResult } from './types';

/** Remove connected components of `binary` smaller than minSize pixels (in place). */
export function removeSmallObjects(binary: Uint8Array, w: number, h: number, minSize: number, connectivity: 4 | 8): Uint8Array {
  const seen = new Uint8Array(binary.length);
  const stack = new Int32Array(binary.length);
  const members: number[] = [];
  for (let start = 0; start < binary.length; start++) {
    if (!binary[start] || seen[start]) continue;
    let top = 0;
    stack[top++] = start;
    seen[start] = 1;
    members.length = 0;
    while (top > 0) {
      const u = stack[--top];
      members.push(u);
      const x = u % w;
      const y = (u - x) / w;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          if (connectivity === 4 && dx && dy) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const v = yy * w + xx;
          if (binary[v] && !seen[v]) { seen[v] = 1; stack[top++] = v; }
        }
      }
    }
    if (members.length < minSize) for (const m of members) binary[m] = 0;
  }
  return binary;
}

/** Argmax over channels of a [1, C, H, W] tensor; ties keep the first class. */
export function argmaxChannels(data: Float32Array, channels: number, plane: number): Uint8Array {
  const out = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    let best = 0;
    let bestValue = data[i];
    for (let c = 1; c < channels; c++) {
      const v = data[c * plane + i];
      if (v > bestValue) { bestValue = v; best = c; }
    }
    out[i] = best;
  }
  return out;
}

export function decodeQuality(probs: Float32Array): QualityResult {
  const p: [number, number, number] = [probs[0], probs[1], probs[2]];
  let k = 0;
  if (p[1] > p[k]) k = 1;
  if (p[2] > p[k]) k = 2;
  return { probs: p, prediction: QUALITY_CLASSES[k], gradable: k === 0 || (k === 1 && p[2] < 0.25) };
}

/** Vessel probability -> binary (>= 0.5) -> remove objects < 30 px (8-connected). */
export function decodeVessel(prob: Float32Array, size: number): Uint8Array {
  const binary = new Uint8Array(prob.length);
  for (let i = 0; i < prob.length; i++) binary[i] = prob[i] >= 0.5 ? 1 : 0;
  return removeSmallObjects(binary, size, size, 30, 8);
}

export const AV = { background: 0, artery: 1, vein: 2, overlap: 3 } as const;

/** Artery/vein softmax -> labels; arteries and veins under 30 px removed (8-connected). */
export function decodeArteryVein(probs: Float32Array, size: number): LabelMap {
  const plane = size * size;
  const labels = argmaxChannels(probs, 4, plane);
  const artery = new Uint8Array(plane);
  const vein = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    artery[i] = labels[i] === AV.artery ? 1 : 0;
    vein[i] = labels[i] === AV.vein ? 1 : 0;
  }
  removeSmallObjects(artery, size, size, 30, 8);
  removeSmallObjects(vein, size, size, 30, 8);
  for (let i = 0; i < plane; i++) {
    if (labels[i] === AV.artery && !artery[i]) labels[i] = AV.background;
    if (labels[i] === AV.vein && !vein[i]) labels[i] = AV.background;
  }
  return { width: size, height: size, labels };
}

export const DISC = { background: 0, disc: 1, cup: 2 } as const;

/** Disc/cup mean softmax -> labels; disc < 100 px and cup < 50 px removed (4-connected). */
export function decodeDiscCup(probs: Float32Array, size: number): LabelMap {
  const plane = size * size;
  const labels = argmaxChannels(probs, 3, plane);
  const disc = new Uint8Array(plane);
  const cup = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    disc[i] = labels[i] === DISC.disc ? 1 : 0;
    cup[i] = labels[i] === DISC.cup ? 1 : 0;
  }
  removeSmallObjects(disc, size, size, 100, 4);
  removeSmallObjects(cup, size, size, 50, 4);
  for (let i = 0; i < plane; i++) labels[i] = cup[i] ? DISC.cup : disc[i] ? DISC.disc : DISC.background;
  return { width: size, height: size, labels };
}
