// RetiFlow's completion/completion.py: complementary artery and vein masks inside the vessel (BV)
// probability field. Class confidence makes anchors; ambiguous vessel pixels go to the class whose
// anchors are geodesically closer; short gaps between facing pieces of one class are bridged along
// vessel paths. Float32 where NumPy computes in float32 (NumPy 2 treats Python scalars as weak).

import { MCPGeometric } from './mcp';
import { dilate8, edt, hysteresis, label, npSum, removeSmall } from './ndi';

const f = Math.fround;

export interface CompletionConfig {
  lowBv: number;
  highBv: number;
  minBvArea: number;
  semanticMargin: number;
  minAnchorArea: number;
  maxBridgeLength: number;
  maxOppositeRun: number;
  minFacingCosine: number;
  directionRadius: number;
  bridgePasses: number;
}

/** config.py CompletionConfig, as infer.py passes it. */
export const COMPLETION: CompletionConfig = {
  lowBv: 0.25, highBv: 0.5, minBvArea: 20, semanticMargin: 0.12, minAnchorArea: 4, maxBridgeLength: 50,
  maxOppositeRun: 50, minFacingCosine: 0.1, directionRadius: 10, bridgePasses: 5,
};

// for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx, with math.hypot
const STEPS: Array<[number, number, number]> = [];
for (const dy of [-1, 0, 1]) for (const dx of [-1, 0, 1]) if (dy || dx) STEPS.push([dy, dx, Math.hypot(dy, dx)]);

function geodesic(cost: Float64Array, seeds: Uint8Array, w: number, h: number): Float64Array {
  const starts: Array<[number, number]> = [];
  for (let i = 0; i < w * h; i++) if (seeds[i]) starts.push([Math.floor(i / w), i % w]);
  if (!starts.length) return new Float64Array(w * h).fill(Infinity);
  return new MCPGeometric(cost, w, h).findCosts(starts);
}

/** The unit vector pointing away from a component near a point, or null (_outward_direction, not strict). */
function outward(labels: Int32Array, w: number, h: number, component: number, py: number, px: number, radius: number): [number, number] | null {
  const r = Math.ceil(radius);
  const y0 = Math.max(0, py - r), y1 = Math.min(h, py + r + 1), x0 = Math.max(0, px - r), x1 = Math.min(w, px + r + 1);
  let n = 0, sy = 0, sx = 0;
  const lo = Math.max(2, radius * 0.3);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (labels[y * w + x] !== component) continue;
      const oy = y - py, ox = x - px, len = Math.sqrt(oy * oy + ox * ox);
      if (len >= lo && len <= radius) { n++; sy += oy; sx += ox; }
    }
  }
  if (n < 3) return null;
  const vy = -(sy / n), vx = -(sx / n);  // integer sums are exact in float64, whatever the order
  const norm = Math.sqrt(vy * vy + vx * vx);
  if (norm < 0.5) return null;
  return [vy / norm, vx / norm];
}

function longestRun(values: boolean[]): number {
  let longest = 0, run = 0;
  for (const v of values) { run = v ? run + 1 : 0; longest = Math.max(longest, run); }
  return longest;
}

/** Binary min-heap of (distance, index) pairs, ordered as Python's heapq orders tuples. */
class TupleHeap {
  private d: number[] = [];
  private i: number[] = [];
  get size() { return this.d.length; }
  private less(a: number, b: number) { return this.d[a] < this.d[b] || (this.d[a] === this.d[b] && this.i[a] < this.i[b]); }
  private swap(a: number, b: number) {
    [this.d[a], this.d[b]] = [this.d[b], this.d[a]];
    [this.i[a], this.i[b]] = [this.i[b], this.i[a]];
  }
  push(d: number, i: number) {
    this.d.push(d); this.i.push(i);
    let c = this.d.length - 1;
    while (c > 0) { const p = (c - 1) >> 1; if (!this.less(c, p)) break; this.swap(c, p); c = p; }
  }
  pop(): [number, number] {
    const top: [number, number] = [this.d[0], this.i[0]];
    const ld = this.d.pop()!, li = this.i.pop()!;
    if (this.d.length) {
      this.d[0] = ld; this.i[0] = li;
      let p = 0;
      for (;;) {
        const l = 2 * p + 1, r = l + 1;
        let m = p;
        if (l < this.d.length && this.less(l, m)) m = l;
        if (r < this.d.length && this.less(r, m)) m = r;
        if (m === p) break;
        this.swap(p, m); p = m;
      }
    }
    return top;
  }
}

/** _front_meetings: a multi-source Dijkstra from each component's boundary, recording where fronts of
 *  different components meet. Distances accumulate in float32, as they do in Python. */
function frontMeetings(mask: Uint8Array, domain: Uint8Array, cost: Float32Array, w: number, h: number, maxDistance: number) {
  const { labels, count } = label(mask, w, h, true);
  const owner = Int32Array.from(labels);
  const distance = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) distance[i] = mask[i] ? 0.0 : Infinity;
  const parent = new Int32Array(w * h).fill(-1);
  const free = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) free[i] = domain[i] && !mask[i] ? 1 : 0;
  const near = dilate8(free, w, h);
  const queue = new TupleHeap();
  // heapq.heapify of [(0.0, i) for i in boundary]: the heap's order matters only through (distance, index)
  for (let i = 0; i < w * h; i++) if (mask[i] && near[i]) queue.push(0.0, i);
  const max32 = f(maxDistance);
  const meetings = new Map<string, [number, number, number]>();
  while (queue.size) {
    const [current, at] = queue.pop();
    if (current !== distance[at] || current > max32) continue;
    const y = Math.floor(at / w), x = at % w;
    for (const [dy, dx, step] of STEPS) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
      const nb = yy * w + xx;
      if (!domain[nb]) continue;
      const edge = f(f(f(step) * f(cost[at] + cost[nb])) * 0.5);
      const candidate = f(current + edge);
      if (owner[nb] && owner[nb] !== owner[at]) {
        const a = Math.min(owner[at], owner[nb]), b = Math.max(owner[at], owner[nb]);
        const key = `${a},${b}`;
        const total = candidate + distance[nb];
        const best = meetings.get(key);
        if (total < (best ? best[0] : Infinity)) meetings.set(key, [total, at, nb]);
      }
      if (candidate < distance[nb] && candidate <= max32) {
        distance[nb] = candidate;
        owner[nb] = owner[at];
        parent[nb] = at;
        queue.push(candidate, nb);
      }
    }
  }
  const trace = (i: number) => {
    const path = [i];
    while (parent[i] >= 0) { i = parent[i]; path.push(i); }
    return path;
  };
  const sorted = [...meetings.values()].sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
  const paths = sorted.map(([score, left, right]) => {
    const idx = [...trace(left).reverse(), ...trace(right)];
    return { score, route: idx.map((i): [number, number] => [Math.floor(i / w), i % w]) };
  });
  return { labels, count, paths };
}

function bridge(mask: Uint8Array, domain: Uint8Array, bv: Float32Array, cls: Float32Array, w: number, h: number, cfg: CompletionConfig) {
  // cost = 1.0 + 1.5 * (1.0 - bv) + 1.5 * np.maximum(0, 0.5 - class_probability), in float32
  const cost = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const t = f(1.0 + f(1.5 * f(1.0 - bv[i])));
    const u = f(1.5 * Math.max(0, f(0.5 - cls[i])));
    cost[i] = f(t + u);
  }
  const { labels, count, paths } = frontMeetings(mask, domain, cost, w, h, cfg.maxBridgeLength * 3.0);
  const result = mask.slice();
  let current = labels;
  let accepted = 0;
  const routes: Array<Array<[number, number]>> = [];
  const roots = Array.from({ length: count + 1 }, (_, i) => i);
  const root = (i: number): number => {
    while (roots[i] !== i) { roots[i] = roots[roots[i]]; i = roots[i]; }
    return i;
  };
  const t30 = f(0.30);
  for (const { route } of paths) {
    const [fy, fx] = route[0], [ly, lx] = route[route.length - 1];
    const left = labels[fy * w + fx], right = labels[ly * w + lx];
    if (!left || !right || root(left) === root(right)) continue;
    const steps = new Float64Array(route.length - 1);
    for (let i = 1; i < route.length; i++) steps[i - 1] = Math.sqrt((route[i][0] - route[i - 1][0]) ** 2 + (route[i][1] - route[i - 1][1]) ** 2);
    const length = npSum(steps);
    const gap = route.reduce((n, [y, x]) => n + (result[y * w + x] ? 0 : 1), 0);
    const sy = ly - fy, sx = lx - fx, separation = Math.sqrt(sy * sy + sx * sx);
    const opposite = longestRun(route.map(([y, x]) => cls[y * w + x] < t30));
    let reason = '';
    if (length > cfg.maxBridgeLength) reason = 'too_long';
    else if (separation === 0 || length > 2.5 * separation + 2) reason = 'detour';
    else if (opposite > cfg.maxOppositeRun) reason = 'long_opposite_vessel';
    else {
      // earlier repairs can turn an apparent endpoint into the side of a vessel: judge on the current geometry
      const dl = outward(current, w, h, current[fy * w + fx], fy, fx, cfg.directionRadius);
      const dr = outward(current, w, h, current[ly * w + lx], ly, lx, cfg.directionRadius);
      const den = Math.max(separation, 1e-6), uy = sy / den, ux = sx / den;
      const al = dl ? dl[0] * uy + dl[1] * ux : null;
      const ar = dr ? dr[0] * -uy + dr[1] * -ux : null;
      if (gap > 2) {
        if (al === null || ar === null) reason = 'insufficient_direction';
        else if (Math.min(al, ar) < cfg.minFacingCosine) reason = 'not_facing';
      }
    }
    if (reason) continue;
    for (const [y, x] of route) result[y * w + x] = 1;
    current = label(result, w, h, true).labels;
    roots[root(right)] = root(left);
    accepted++;
    routes.push(route);
  }
  return { result, accepted, routes };
}

export interface Completed {
  a: Uint8Array;
  v: Uint8Array;
  /** (a + 1e-5) / (a + v + 2e-5), float32: the centrelines weight the vessel probability with it */
  conditionalA: Float32Array;
  bvMask: Uint8Array;
  /** for checks against the Python: the anchors, the masks before bridging, and the bridges made */
  debug: { aSeeds: Uint8Array; vSeeds: Uint8Array; initialA: Uint8Array; initialV: Uint8Array; bridges: { a: number; v: number }; routes: { a: Array<[number, number][]>; v: Array<[number, number][]> } };
}

/** complete_probability_masks(a, v, bv): the artery and vein masks. */
export function completeProbabilityMasks(a: Float32Array, v: Float32Array, bv: Float32Array, w: number, h: number,
                                         cfg: CompletionConfig = COMPLETION): Completed {
  const n = w * h;
  const domain = removeSmall(hysteresis(bv, w, h, cfg.lowBv, cfg.highBv), w, h, cfg.minBvArea);
  const ca = new Float32Array(n);
  const evidence = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    ca[i] = f(f(a[i] + f(1e-5)) / f(f(a[i] + v[i]) + f(2e-5)));
    evidence[i] = f(a[i] + v[i]) >= f(0.25) ? 1 : 0;
  }
  const hiA = f(0.5 + cfg.semanticMargin), loA = f(0.5 - cfg.semanticMargin);
  const aSeeds = removeSmall(Uint8Array.from(ca, (c, i) => (domain[i] && evidence[i] && c >= hiA ? 1 : 0)), w, h, cfg.minAnchorArea);
  const vSeeds = removeSmall(Uint8Array.from(ca, (c, i) => (domain[i] && evidence[i] && c <= loA ? 1 : 0)), w, h, cfg.minAnchorArea);
  const radius = edt(domain, w, h);
  const spatial = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    spatial[i] = domain[i] ? f(1.0 + f(2.0 * f(1.0 - bv[i]))) / Math.sqrt(0.5 + radius[i]) : Infinity;
  }
  const da = geodesic(spatial, aSeeds, w, h), dv = geodesic(spatial, vSeeds, w, h);
  let am = new Uint8Array(n), vm = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!domain[i]) continue;
    const reachable = da[i] !== Infinity || dv[i] !== Infinity;
    if (!reachable) { am[i] = 1; vm[i] = 1; continue; }  // unseeded vessel pixels: both
    if (da[i] <= dv[i]) am[i] = 1;
    if (dv[i] <= da[i]) vm[i] = 1;
  }
  const initialA = am.slice(), initialV = vm.slice(), bridges = { a: 0, v: 0 };
  const routes: { a: Array<[number, number][]>; v: Array<[number, number][]> } = { a: [], v: [] };
  const cv = Float32Array.from(ca, (c) => f(1 - c));
  for (const [cls, which] of [[ca, 'a'], [cv, 'v']] as const) {
    let target = which === 'a' ? am : vm;
    for (let pass = 0; pass < cfg.bridgePasses; pass++) {
      const r = bridge(target, domain, bv, cls, w, h, cfg);
      target = r.result;
      bridges[which] += r.accepted;
      routes[which].push(...r.routes);
      if (!r.accepted) break;
    }
    if (which === 'a') am = target; else vm = target;
  }
  return { a: am, v: vm, conditionalA: ca, bvMask: domain, debug: { aSeeds, vSeeds, initialA, initialV, bridges, routes } };
}
