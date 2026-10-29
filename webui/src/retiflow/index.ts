// RetiFlow's infer.py process_prob: from an AV3 map (R arteries, G veins, B vessels, 8-bit) to the
// branching points and angles of the arteries and of the veins, measured separately: masks completed in
// the probability field, a probability-guided centreline for each class, and the two-pass RBAD.
// RetiFlow is MIT-licensed (Retinal-Research); its notice and scikit-image's are in NOTICE.md.

import type { Bifurcation } from '../engine/values';
import { extractCenterline } from './centerline';
import { completeProbabilityMasks } from './completion';
import { argmax, cvGaussian, skeletonize } from './ndi';
import { detectTwoPass } from './rbad';

const f = Math.fround;

export interface RetiFlowResult {
  /** where the traversal starts, [y, x]: the densest part of the vessel skeleton */
  root: [number, number];
  artery: { points: Bifurcation[]; centerline: Uint8Array };
  vein: { points: Bifurcation[]; centerline: Uint8Array };
}

/** _root_from_bv: the peak of the vessel skeleton (vessels > 0.5) blurred with sigma 21. */
export function rootFromBv(bv: Float32Array, w: number, h: number): [number, number] {
  const skel = skeletonize(Uint8Array.from(bv, (p) => (p > 0.5 ? 1 : 0)), w, h);
  const i = argmax(cvGaussian(Float32Array.from(skel), w, h, 21));
  return [Math.floor(i / w), i % w];
}

/** process_prob(a, v, bv): av3 holds the map's pixels as RGB triples, row by row. */
export function processProb(av3: Uint8Array, w: number, h: number, step: (message: string) => void = () => {}): RetiFlowResult {
  const n = w * h;
  const a = new Float32Array(n), v = new Float32Array(n), bv = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = f(av3[3 * i] / 255);
    v[i] = f(av3[3 * i + 1] / 255);
    bv[i] = f(av3[3 * i + 2] / 255);
  }
  step('Completing the artery and vein masks');
  const masks = completeProbabilityMasks(a, v, bv, w, h);
  // bv * (0.65 + 0.35 * conditional_a), in float32; extract_centerline reads it as float64
  const pa = new Float64Array(n), pv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ca = masks.conditionalA[i];
    pa[i] = f(bv[i] * f(f(0.65) + f(f(0.35) * ca)));
    pv[i] = f(bv[i] * f(f(0.65) + f(f(0.35) * f(1 - ca))));
  }
  step('Tracing the centrelines');
  const sa = extractCenterline(pa, masks.a, w, h), sv = extractCenterline(pv, masks.v, w, h);
  step('Measuring the branching angles');
  const root = rootFromBv(bv, w, h);
  const support = Float64Array.from(bv);
  const artery = detectTwoPass(sa, support, w, h, root), vein = detectTwoPass(sv, support, w, h, root);
  return { root, artery: { points: artery.angles, centerline: sa }, vein: { points: vein.angles, centerline: sv } };
}
