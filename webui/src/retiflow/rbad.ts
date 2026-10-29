// RetiFlow's detect/utils.py (the original RBAD fast_keypoints breadth-first traversal and its angle)
// and detect/two_pass.py (RBAD on the centreline, endpoint repair, RBAD again on the repaired one).

import type { Bifurcation } from '../engine/values';
import { bridgeEndpoints } from './bridge';
import { label } from './ndi';

/** A keypoint: node (x, y), type (0 root, 1 bifurcation, 2 end, 4 tail), level, step, parent. */
type Key = [[number, number], number, number, number, [number, number]];

/** fast_keypoints(image, [root]): RBAD's traversal of a (padded) skeleton from its root (y, x). */
function fastKeypoints(img: Uint8Array, w: number, root: [number, number], tail: number): Key[] {
  const plate = img.slice();
  const keypoints: Key[] = [];
  const center: [number, number] = root;  // roots[0], kept as (y, x), as the original does
  const queue: Key[] = [];
  let head = 0;
  const first: Key = [[root[1], root[0]], 0, 0, 0, center];
  queue.push(first);
  keypoints.push(first);
  while (head < queue.length) {
    const [[dx, dy], , parentLevel, step, parent] = queue[head++];
    plate[dy * w + dx] = 0;
    const box: Array<[number, number]> = [];  // np.where order: row by row
    for (let y = dy - 1; y <= dy + 1; y++) for (let x = dx - 1; x <= dx + 1; x++) if (plate[y * w + x]) box.push([x, y]);
    if (box.length === 0) {
      keypoints.push([[dx, dy], 2, parentLevel + 1, step, parent]);
    } else if (box.length === 1) {
      const [x, y] = box[0];
      plate[y * w + x] = 0;
      queue.push([[x, y], 3, parentLevel, step + 1, parent]);
      if (step === tail) keypoints.push([[x, y], 4, parentLevel, step + 1, parent]);
    } else {
      keypoints.push([[dx, dy], 1, parentLevel + 1, 0, parent]);
      for (const [x, y] of box) {
        plate[y * w + x] = 0;
        queue.push([[x, y], 3, parentLevel + 1, 0, [dx, dy]]);
      }
    }
  }
  return keypoints;
}

/** calculate_angle(a, b, c): the angle at b between b - a and b - c, in degrees (math.degrees). */
function calculateAngle(a: [number, number], b: [number, number], c: [number, number]): number | null {
  const bax = b[0] - a[0], bay = b[1] - a[1], bcx = b[0] - c[0], bcy = b[1] - c[1];
  const dot = bax * bcx + bay * bcy;
  const mag = Math.sqrt(bax * bax + bay * bay) * Math.sqrt(bcx * bcx + bcy * bcy);
  if (mag === 0) return null;  // ZeroDivisionError
  const cos = dot / mag;
  if (cos > 1 || cos < -1) return null;  // ValueError
  return Math.acos(cos) * (180.0 / Math.PI);  // math.degrees multiplies by 180 / pi
}

interface Record { point: [number, number]; type: number; parent: [number, number]; component: number }

export interface RbadResult {
  angles: Bifurcation[];
  /** the snapped root, [y, x] */
  root: [number, number] | null;
}

export const RBAD = { tail: 15, childMinDist: 5, angleMin: 20.0, angleMax: 120.0 };

/** run_original(skeleton, root_yx): RBAD from the main root, then every other island from its pixel
 *  nearest to that root. */
export function runOriginal(skel: Uint8Array, w: number, h: number, rootYX: [number, number]): RbadResult {
  const { labels, count } = label(skel, w, h, true);
  if (!count) return { angles: [], root: null };
  let root: [number, number] = [0, 0], best = Infinity;
  for (let i = 0; i < w * h; i++) {
    if (!skel[i]) continue;
    const y = Math.floor(i / w), x = i % w;
    const d = Math.abs(y - rootYX[0]) + Math.abs(x - rootYX[1]);
    if (d < best) { best = d; root = [y, x]; }
  }
  const primary = labels[root[0] * w + root[1]];
  const order = [primary, ...Array.from({ length: count }, (_, i) => i + 1).filter((c) => c !== primary)];
  const angles: Bifurcation[] = [];
  const pw = w + 2;
  for (const component of order) {
    let local: [number, number] = [0, 0], lb = Infinity;
    const padded = new Uint8Array(pw * (h + 2));
    for (let i = 0; i < w * h; i++) {
      if (labels[i] !== component) continue;
      const y = Math.floor(i / w), x = i % w;
      padded[(y + 1) * pw + x + 1] = 1;
      const d = Math.abs(y - root[0]) + Math.abs(x - root[1]);
      if (d < lb) { lb = d; local = [y, x]; }
    }
    const keys = fastKeypoints(padded, pw, [local[0] + 1, local[1] + 1], RBAD.tail);
    const records: Record[] = keys.map(([node, type, , , parent]) => ({
      point: [node[0] - 1, node[1] - 1], type, parent: [parent[0] - 1, parent[1] - 1], component,
    }));
    for (const r of records) {
      if (r.type !== 1) continue;
      const node = r.point;
      const children = records.filter((c) => c.parent[0] === node[0] && c.parent[1] === node[1]
        && Math.abs(c.point[0] - node[0]) + Math.abs(c.point[1] - node[1]) >= RBAD.childMinDist).map((c) => c.point);
      if (children.length < 2) continue;
      const angle = calculateAngle(children[0], node, children[1]);
      if (angle === null) continue;
      if (RBAD.angleMin <= angle && angle <= RBAD.angleMax) {
        angles.push({ x: node[0], y: node[1], angle, children: [children[0], children[1]] });
      }
    }
  }
  return { angles, root };
}

/** detect_two_pass: RBAD, endpoint repair (with the vessel probability), RBAD again from the same root.
 *  Returns the repaired skeleton and the second pass's bifurcations. */
export function detectTwoPass(skel: Uint8Array, bv: Float64Array, w: number, h: number, rootYX: [number, number]) {
  const before = runOriginal(skel, w, h, rootYX);
  const repaired = bridgeEndpoints(skel, bv, w, h);
  const after = runOriginal(repaired, w, h, before.root ?? rootYX);
  return { repaired, angles: after.angles };
}
