// RetiFlow's detect/endpoint_bridge.py: trace each skeleton end back along its parent arm, extend its
// direction, and join another component, end to end or end onto a branch body, where the vessel
// probability supports the join. Original pixels are never removed. Points are [y, x].

import { dilate8, label, npSum, pathLength, rint } from './ndi';

type P = [number, number];

/** config.py EndpointBridgeConfig */
export const BRIDGE = {
  maxDistance: 40.0, maxAngle: 35.0, lookback: 12.0, minParentLength: 4.0, minBvMean: 0.25, lowBv: 0.10,
  maxUnsupportedRun: 8, passes: 2, bodyLanding: true, minBodyAngle: 25.0,
};

const OFFSETS: P[] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

/** neighbours: the skeleton's 8-neighbours, without a diagonal that an edge neighbour already reaches. */
function neighbours(skel: Uint8Array, w: number, h: number, [y, x]: P): P[] {
  const result: P[] = [];
  for (const [dy, dx] of OFFSETS) {
    const yy = y + dy, xx = x + dx;
    if (!(yy >= 0 && yy < h && xx >= 0 && xx < w && skel[yy * w + xx])) continue;
    if (dy && dx && (skel[y * w + xx] || skel[yy * w + x])) continue;
    result.push([yy, xx]);
  }
  return result;
}

const same = (a: P, b: P) => a[0] === b[0] && a[1] === b[1];
const norm = (v: P) => Math.sqrt(v[0] * v[0] + v[1] * v[1]);

/** _walk: follow the single continuation from start, away from point, for about distance pixels. */
function walk(skel: Uint8Array, w: number, h: number, point: P, start: P, distance: number): { path: P[]; length: number } {
  const path: P[] = [point, start];
  let length = norm([start[0] - point[0], start[1] - point[1]]);
  while (length < distance) {
    let choices = neighbours(skel, w, h, path[path.length - 1]);
    if (choices.length !== 2) break;
    const back = path[path.length - 2];
    choices = choices.filter((p) => !same(p, back) && !path.some((q) => same(p, q)));
    if (!choices.length) break;
    const next = choices[0], last = path[path.length - 1];
    length += norm([next[0] - last[0], next[1] - last[1]]);
    path.push(next);
  }
  return { path, length };
}

function unit(v: P): P | null {
  const n = norm(v);
  return n < 1e-8 ? null : [v[0] / n, v[1] / n];
}

/** _angle: degrees between two unit vectors (np.degrees multiplies by 180 / pi). */
function angle(a: P, b: P): number {
  return Math.acos(Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1]))) * (180.0 / Math.PI);
}

interface End { point: P; direction: P }

/** endpoints: each end pixel with its outward direction, from the mean of the far half of its parent arm. */
function endpoints(skel: Uint8Array, w: number, h: number): End[] {
  const result: End[] = [];
  for (let i = 0; i < w * h; i++) {
    if (!skel[i]) continue;
    const point: P = [Math.floor(i / w), i % w];
    const adjacent = neighbours(skel, w, h, point);
    if (adjacent.length !== 1) continue;
    const { path, length } = walk(skel, w, h, point, adjacent[0], BRIDGE.lookback);
    if (length < BRIDGE.minParentLength) continue;
    const from = Math.max(1, Math.floor(path.length / 2)), n = path.length - from;
    let sy = 0, sx = 0;
    for (let k = from; k < path.length; k++) { sy += path[k][0]; sx += path[k][1]; }
    const direction = unit([point[0] - sy / n, point[1] - sx / n]);
    if (direction) result.push({ point, direction });
  }
  return result;
}

/** _path: a cubic Bezier from first to last, leaving along firstTangent, sampled four times a pixel
 *  and rounded half to even, without consecutive repeats. */
function bezier(first: P, last: P, firstTangent: P, lastTangent?: P): P[] {
  const span: P = [last[0] - first[0], last[1] - first[1]];
  const distance = norm(span);
  const end = lastTangent ?? unit(span)!;
  const c1: P = [first[0] + firstTangent[0] * distance / 3, first[1] + firstTangent[1] * distance / 3];
  const c2: P = [last[0] - end[0] * distance / 3, last[1] - end[1] * distance / 3];
  const num = Math.max(3, Math.ceil(distance * 4) + 1), step = 1 / (num - 1);
  const route: P[] = [];
  for (let k = 0; k < num; k++) {
    const t = k === num - 1 ? 1 : k * step, s = 1 - t;  // np.linspace
    const b0 = Math.pow(s, 3), b1 = 3 * (s * s) * t, b2 = 3 * s * (t * t), b3 = Math.pow(t, 3);
    const p: P = [
      rint(b0 * first[0] + b1 * c1[0] + b2 * c2[0] + b3 * last[0]),
      rint(b0 * first[1] + b1 * c1[1] + b2 * c2[1] + b3 * last[1]),
    ];
    if (!route.length || !same(p, route[route.length - 1])) route.push(p);
  }
  return route;
}

function longestRun(values: boolean[]): number {
  let best = 0, run = 0;
  for (const v of values) { run = v ? run + 1 : 0; best = Math.max(best, run); }
  return best;
}

interface Candidate { score: number; i: number; j: number | null; route: P[] }

/** bridge_endpoints(skeleton, bv_probability): the repaired skeleton. bv holds float32 values.
 *  Where several branch-body landings tie on score, the first in row order wins; SciPy takes the first
 *  in its k-d tree's order, which depends on the platform's C++ library. */
export function bridgeEndpoints(skeleton: Uint8Array, bv: Float64Array, w: number, h: number): Uint8Array {
  let result = Uint8Array.from(skeleton, (v) => (v ? 1 : 0));
  const r2 = BRIDGE.maxDistance * BRIDGE.maxDistance;
  for (let pass = 0; pass < (BRIDGE.maxDistance ? BRIDGE.passes : 0); pass++) {
    const { labels, count } = label(result, w, h, true);
    const ends = endpoints(result, w, h);
    if (count < 2 || !ends.length) break;
    const candidates: Candidate[] = [];
    // end to end: cKDTree.query_pairs, sorted
    for (let i = 0; i < ends.length; i++) {
      for (let j = i + 1; j < ends.length; j++) {
        const p = ends[i].point, q = ends[j].point;
        const span: P = [q[0] - p[0], q[1] - p[1]];
        if (span[0] * span[0] + span[1] * span[1] > r2) continue;
        if (labels[p[0] * w + p[1]] === labels[q[0] * w + q[1]]) continue;
        const u = unit(span)!;
        const alpha = angle(ends[i].direction, u), beta = angle(ends[j].direction, [-u[0], -u[1]]);
        if (Math.max(alpha, beta) > BRIDGE.maxAngle) continue;
        const route = bezier(p, q, ends[i].direction, [-ends[j].direction[0], -ends[j].direction[1]]);
        candidates.push({ score: norm(span) * (1 + (alpha + beta) / 90), i, j, route });
      }
    }
    // end onto the body of another component: the best landing for each component
    if (BRIDGE.bodyLanding) {
      const body: P[] = [];
      for (let k = 0; k < w * h; k++) {
        if (!result[k]) continue;
        const p: P = [Math.floor(k / w), k % w];
        if (neighbours(result, w, h, p).length === 2) body.push(p);
      }
      ends.forEach((end, i) => {
        const p = end.point, best = new Map<number, Candidate>();
        for (const q of body) {
          const span: P = [q[0] - p[0], q[1] - p[1]];
          if (span[0] * span[0] + span[1] * span[1] > r2) continue;
          const component = labels[q[0] * w + q[1]];
          if (component === labels[p[0] * w + p[1]]) continue;
          const direction = unit(span)!;
          const alpha = angle(end.direction, direction);
          if (alpha > BRIDGE.maxAngle) continue;
          const arms = neighbours(result, w, h, q).map((start) => {
            const { path } = walk(result, w, h, q, start, BRIDGE.lookback);
            return path[path.length - 1];
          });
          const tangent = unit([arms[1][0] - arms[0][0], arms[1][1] - arms[0][1]]);
          if (!tangent) continue;
          const bodyAngle = Math.min(angle(direction, tangent), angle(direction, [-tangent[0], -tangent[1]]));
          if (bodyAngle < BRIDGE.minBodyAngle) continue;
          const score = norm(span) * (1 + alpha / 45) + 5;
          const old = best.get(component);
          if (!old || score < old.score) best.set(component, { score, i, j: null, route: bezier(p, q, end.direction) });
        }
        candidates.push(...best.values());
      });
    }
    candidates.sort((a, b) => {
      if (a.score !== b.score) return a.score < b.score ? -1 : 1;
      if (a.i !== b.i) return a.i - b.i;
      const ea = a.route[a.route.length - 1], eb = b.route[b.route.length - 1];
      return ea[0] - eb[0] || ea[1] - eb[1];
    });
    const used = new Set<number>();
    let joined = 0;
    let current = label(result, w, h, true);  // relabelled after every join, as the original does per candidate
    for (const c of candidates) {
      if (used.has(c.i) || (c.j !== null && used.has(c.j))) continue;
      const p = c.route[0], q = c.route[c.route.length - 1];
      if (current.labels[p[0] * w + p[1]] === current.labels[q[0] * w + q[1]]) continue;
      if (neighbours(result, w, h, p).length !== 1 || (c.j !== null && neighbours(result, w, h, q).length !== 1)) continue;
      if (c.route.some(([y, x]) => y < 0 || x < 0 || y >= h || x >= w)) continue;
      if (pathLength(c.route) > BRIDGE.maxDistance) continue;
      const added = new Uint8Array(w * h);
      let any = false;
      for (const [y, x] of c.route) if (!result[y * w + x]) { added[y * w + x] = 1; any = true; }
      if (!any) continue;
      // the new pixels may touch the skeleton only near their two ends
      const near = dilate8(added, w, h);
      let far = false;
      for (let k = 0; k < w * h && !far; k++) {
        if (!near[k] || !result[k]) continue;
        const y = Math.floor(k / w), x = k % w;
        far = (y - p[0]) ** 2 + (x - p[1]) ** 2 > 9 && (y - q[0]) ** 2 + (x - q[1]) ** 2 > 9;
      }
      if (far) continue;
      const support = Float64Array.from(c.route, ([y, x]) => bv[y * w + x]);
      const low = Array.from(support, (s) => s < BRIDGE.lowBv);
      if (npSum(support) / support.length < BRIDGE.minBvMean || longestRun(low) > BRIDGE.maxUnsupportedRun) continue;
      const trial = result.slice();
      for (const [y, x] of c.route) trial[y * w + x] = 1;
      if (label(trial, w, h, true).count !== current.count - 1) continue;
      result = trial;
      current = label(result, w, h, true);
      used.add(c.i);
      if (c.j !== null) used.add(c.j);
      joined++;
    }
    if (!joined) break;
  }
  return result;
}
