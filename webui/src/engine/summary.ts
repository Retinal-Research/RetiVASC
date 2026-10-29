// Simple measures of one segmentation or landmark result, the same for AutoMorph and VascX steps,
// so a folder's CSV can compare the two pipelines: how much of the field of view the vessels,
// arteries and veins cover, the optic disc's size, and where the fovea is in the photograph;
// and RetiFlow's branching points and their mean angle, for arteries and veins.

import type { Bifurcation, M0Value, NodeValue } from './values';

/** Share of the circular field of view (at a model's resolution) that a pixel count covers. */
export function fovShare(count: number, size: number, base: M0Value): number {
  const r = ((base.fovRadius ?? base.geometry.radius) * size) / base.side;
  return count / (Math.PI * r * r);
}

/** A point of the base image in the photograph's pixels. */
export function toPhoto(base: M0Value, x: number, y: number): [number, number] {
  const g = base.geometry, k = g.radius / (base.fovRadius ?? g.radius);
  return [g.centreCol + (x - base.side / 2) * k, g.centreRow + (y - base.side / 2) * k];
}

/** Named numbers for a node's output, or undefined when it has none. */
export function measures(v: NodeValue): Record<string, number> | undefined {
  switch (v.kind) {
    case 'vessel': {
      let n = 0;
      for (const m of v.mask) n += m;
      return { Vessel_share: fovShare(n, v.size, v.base) };
    }
    case 'av': {
      let a = 0, b = 0;
      for (const l of v.map.labels) { if (l === 1 || l === 3) a++; if (l === 2 || l === 3) b++; }  // 3: both
      return { Artery_share: fovShare(a, v.map.width, v.base), Vein_share: fovShare(b, v.map.width, v.base) };
    }
    case 'disc': {
      let n = 0;
      for (const l of v.map.labels) if (l) n++;
      return { Disc_share: fovShare(n, v.map.width, v.base) };
    }
    case 'fovea': {
      const [x, y] = toPhoto(v.base, (v.x * v.base.side) / v.size, (v.y * v.base.side) / v.size);
      return { Fovea_x: x, Fovea_y: y };
    }
    case 'bifurcations': {
      if (v.failed) return undefined;  // not measured: no numbers rather than zeros
      const out: Record<string, number> = {};
      for (const [name, points] of [['Artery', v.artery.points], ['Vein', v.vein.points]] as Array<[string, Bifurcation[]]>) {
        out[`${name}_branching_points`] = points.length;
        if (points.length) out[`${name}_branching_angle`] = points.reduce((s, p) => s + p.angle, 0) / points.length;
      }
      return out;
    }
    default:
      return undefined;
  }
}
