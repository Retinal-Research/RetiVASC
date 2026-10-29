// RetiFlow's completion/centerline.py: a probability-guided geodesic centreline forest inside a vessel
// mask, one tree per 8-connected component, grown from the component's widest point; no thinning.

import { gaussian1d, gaussianWeights } from '../rrwnet/prep';
import { MCPGeometric } from './mcp';
import { argmax, boxes, edt, label, pathLength } from './ndi';

/** _cover_path: mark the vessel cross-sections a path already represents as no longer available. */
function cover(available: Uint8Array, w: number, h: number, path: Array<[number, number]>, radius: Float64Array) {
  for (const [y, x] of path) {
    const distance = 1.25 * radius[y * w + x] + 0.5;
    const extent = Math.ceil(distance), d2 = distance * distance;
    const y0 = Math.max(0, y - extent), y1 = Math.min(h, y + extent + 1), x0 = Math.max(0, x - extent), x1 = Math.min(w, x + extent + 1);
    for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) if ((yy - y) ** 2 + (xx - x) ** 2 <= d2) available[yy * w + xx] = 0;
  }
}

/** scipy.ndimage.gaussian_filter(img, 0.8): both axes, axis 0 first, mode 'reflect'. */
function smooth(img: Float64Array, w: number, h: number): Float64Array {
  const k = gaussianWeights(0.8);
  return gaussian1d(gaussian1d(img, w, h, k, 0), w, h, k, 1);
}

function oneComponent(probability: Float64Array, domain: Uint8Array, w: number, h: number, minBranchLength: number): Uint8Array {
  // the distance radius on the domain padded by one background pixel, so an all-true crop has edges
  const pw = w + 2, ph = h + 2, padded = new Uint8Array(pw * ph);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) padded[(y + 1) * pw + x + 1] = domain[y * w + x];
  const full = edt(padded, pw, ph);
  const radius = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) radius[y * w + x] = full[(y + 1) * pw + x + 1];

  const masked = Float64Array.from(probability, (p, i) => p * domain[i]);
  const sp = smooth(masked, w, h), mass = smooth(Float64Array.from(domain), w, h);
  const soft = new Float64Array(w * h), cost = new Float64Array(w * h), geometric = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    sp[i] /= Math.max(mass[i], 1e-12);
    soft[i] = radius[i] * Math.sqrt(0.15 + 0.85 * sp[i]);
    cost[i] = domain[i] ? (0.15 + (1.0 - sp[i]) * (1.0 - sp[i])) / ((0.5 + soft[i]) * (0.5 + soft[i])) : Infinity;
    geometric[i] = domain[i] ? 1.0 : Infinity;
  }
  const r0 = argmax(soft), root: [number, number] = [Math.floor(r0 / w), r0 % w];
  const solver = new MCPGeometric(cost, w, h);
  solver.findCosts([root]);
  const distance = new MCPGeometric(geometric, w, h).findCosts([root]);
  const priority = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) priority[i] = domain[i] ? distance[i] + 0.5 * soft[i] : -Infinity;

  const available = domain.slice();
  const result = new Uint8Array(w * h);
  result[r0] = 1;
  cover(available, w, h, [root], radius);
  let accepted = 0;
  const masked2 = new Float64Array(w * h);
  for (;;) {
    let any = false;
    for (let i = 0; i < w * h; i++) {
      masked2[i] = available[i] ? priority[i] : -Infinity;
      if (available[i]) any = true;
    }
    if (!any) break;
    const t = argmax(masked2);
    const path = solver.traceback([Math.floor(t / w), t % w]);
    let last = 0;
    for (let i = 0; i < path.length; i++) if (result[path[i][0] * w + path[i][1]]) last = i;
    const branch = path.slice(last);
    const length = pathLength(branch);
    if (!accepted || length >= minBranchLength) {
      for (const [y, x] of branch) result[y * w + x] = 1;
      accepted++;
    }
    // short twigs are covered too, so every round removes at least its target
    cover(available, w, h, branch, radius);
  }
  return result;
}

/** extract_centerline(probability, domain): a connected path forest inside the domain, row-major. */
export function extractCenterline(probability: Float64Array, domain: Uint8Array, w: number, h: number, minBranchLength = 6.0): Uint8Array {
  const result = new Uint8Array(w * h);
  const { labels, count } = label(domain, w, h, true);
  const regions = boxes(labels, w, h, count);
  regions.forEach((region, k) => {
    if (!region) return;
    const [y0, x0, y1, x1] = region, rw = x1 - x0, rh = y1 - y0, id = k + 1;
    const local = new Uint8Array(rw * rh), prob = new Float64Array(rw * rh);
    for (let y = 0; y < rh; y++) {
      for (let x = 0; x < rw; x++) {
        local[y * rw + x] = labels[(y + y0) * w + x + x0] === id ? 1 : 0;
        prob[y * rw + x] = probability[(y + y0) * w + x + x0];
      }
    }
    const part = oneComponent(prob, local, rw, rh, minBranchLength);
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) if (part[y * rw + x]) result[(y + y0) * w + x + x0] = 1;
  });
  return result;
}
