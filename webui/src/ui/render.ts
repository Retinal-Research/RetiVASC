// Turns results into drawable layers and paints them over the M0 photo.

import { AV, DISC } from '../automorph/postprocess';
import type { ArteryVeinValue, BifurcationsValue, DiscCupValue, FoveaValue, NodeValue, VesselValue } from '../engine/values';
import { coloursVersion, rgb, seen, vesselColours } from './vesselColours';

export type LayerKind = 'vessel' | 'av' | 'disc' | 'fovea';
export type LayerValue = VesselValue | ArteryVeinValue | DiscCupValue | FoveaValue;

export const LAYER_LABEL: Record<LayerKind, string> = { vessel: 'Vessels', av: 'Artery / vein', disc: 'Disc & cup', fovea: 'Fovea' };

/** The legend of a layer, in the colours it shows (vesselColours.ts) */
export function legend(kind: LayerKind): Array<[string, string]> {
  if (kind === 'vessel') return [['Vessel', seen('vessel')]];
  if (kind === 'av') return [['Artery', seen('artery')], ['Vein', seen('vein')], ['Crossing', seen('crossing')]];
  return LEGEND[kind];
}

export const LEGEND: Record<LayerKind, Array<[string, string]>> = {
  vessel: [['Vessel', '#2ee6f2']],
  av: [['Artery', '#ff4b4b'], ['Vein', '#4b8bff'], ['Crossing', '#3ddc84']],
  disc: [['Optic disc', '#ffd84a'], ['Cup', '#ff6fc1']],
  fovea: [['Fovea', '#ffffff']],
};

// drawn layers, per result and per set of vessel colours
const cache = new WeakMap<object, { v: number; canvas: HTMLCanvasElement }>();

function canvasFrom(width: number, height: number, paint: (px: Uint8ClampedArray) => void): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(width, height);
  paint(img.data);
  ctx.putImageData(img, 0, 0);
  return canvas;
}

function put(px: Uint8ClampedArray, i: number, r: number, g: number, b: number, a: number) {
  const p = i * 4;
  px[p] = r;
  px[p + 1] = g;
  px[p + 2] = b;
  px[p + 3] = a;
}

/** RGBA layer at model resolution, built once per result. */
export function layerCanvas(value: LayerValue): HTMLCanvasElement {
  const hit = cache.get(value);
  if (hit && hit.v === coloursVersion()) return hit.canvas;
  let canvas: HTMLCanvasElement;
  const colours = vesselColours();
  if (value.kind === 'fovea') {
    // a ring and a cross at the fovea, on a 512 canvas over the whole base image
    const n = 512, x = (value.x / value.size) * n, y = (value.y / value.size) * n;
    canvas = document.createElement('canvas');
    canvas.width = n;
    canvas.height = n;
    const g = canvas.getContext('2d')!;
    g.strokeStyle = 'rgba(0, 0, 0, 0.55)';
    g.lineWidth = 5;
    g.beginPath(); g.arc(x, y, 13, 0, 2 * Math.PI); g.stroke();
    g.strokeStyle = '#ffffff';
    g.lineWidth = 2.5;
    g.beginPath(); g.arc(x, y, 13, 0, 2 * Math.PI); g.moveTo(x - 22, y); g.lineTo(x - 6, y); g.moveTo(x + 6, y); g.lineTo(x + 22, y);
    g.moveTo(x, y - 22); g.lineTo(x, y - 6); g.moveTo(x, y + 6); g.lineTo(x, y + 22); g.stroke();
    cache.set(value, { v: coloursVersion(), canvas });
    return canvas;
  }
  if (value.kind === 'vessel') {
    const { size, mask } = value;
    canvas = canvasFrom(size, size, (px) => {
      const [r, g, b] = rgb(colours.vessel);
      for (let i = 0; i < mask.length; i++) if (mask[i]) put(px, i, r, g, b, 235);
    });
  } else if (value.kind === 'av') {
    const { width, height, labels } = value.map;
    canvas = canvasFrom(width, height, (px) => {
      const [a, v, x] = [rgb(colours.artery), rgb(colours.vein), rgb(colours.crossing)];
      for (let i = 0; i < labels.length; i++) {
        const l = labels[i];
        if (l === AV.artery) put(px, i, a[0], a[1], a[2], 235);
        else if (l === AV.vein) put(px, i, v[0], v[1], v[2], 235);
        else if (l === AV.overlap) put(px, i, x[0], x[1], x[2], 235);
      }
    });
  } else {
    const { width: w, height: h, labels } = value.map;
    canvas = canvasFrom(w, h, (px) => {
      const inDisc = (i: number) => labels[i] !== DISC.background;
      const inCup = (i: number) => labels[i] === DISC.cup;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (!inDisc(i)) continue;
          const edge = (test: (j: number) => boolean) =>
            x === 0 || y === 0 || x === w - 1 || y === h - 1 || !test(i - 1) || !test(i + 1) || !test(i - w) || !test(i + w);
          if (inCup(i) && edge(inCup)) put(px, i, 255, 111, 193, 255);
          else if (edge(inDisc)) put(px, i, 255, 216, 74, 255);
          else if (inCup(i)) put(px, i, 255, 111, 193, 70);
          else put(px, i, 255, 216, 74, 55);
        }
      }
    });
  }
  cache.set(value, { v: coloursVersion(), canvas });
  return canvas;
}

export type BranchKind = 'artery' | 'vein';
export const BRANCH_LABEL: Record<BranchKind, string> = { artery: 'Arteries', vein: 'Veins' };
export const branchLegend = (): Array<[string, string]> => [['Artery centreline', seen('artery')], ['Vein centreline', seen('vein')], ['Branching point', '#ffd84a'], ['Root', '#3ddc84']];

const branchCache = new WeakMap<BifurcationsValue, { v: number } & Partial<Record<BranchKind, HTMLCanvasElement>>>();

/** RetiFlow's result for one class: its centreline, each branching point with the directions of its two
 *  daughter vessels, and the root the traversal starts from; at twice the map's resolution, so the marks
 *  stay sharp. */
export function branchCanvas(value: BifurcationsValue, which: BranchKind): HTMLCanvasElement {
  const cached = branchCache.get(value);
  const held = cached && cached.v === coloursVersion() ? cached : { v: coloursVersion() };
  const hit = held[which];
  if (hit) return hit;
  const n = value.size, s = 2, { centerline, points } = value[which];
  const [r, g, b] = rgb(seen(which));
  const canvas = canvasFrom(n * s, n * s, (px) => {
    for (let i = 0; i < centerline.length; i++) {
      if (!centerline[i]) continue;
      const y = Math.floor(i / n) * s, x = (i % n) * s;
      for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) put(px, (y + dy) * n * s + x + dx, r, g, b, 240);
    }
  });
  const ctx = canvas.getContext('2d')!;
  for (const p of points) {
    const cx = (p.x + 0.5) * s, cy = (p.y + 0.5) * s;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 1.5;
    for (const [x, y] of p.children) {
      const dx = x - p.x, dy = y - p.y, len = Math.hypot(dx, dy) || 1;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + (dx / len) * 16, cy + (dy / len) * 16); ctx.stroke();
    }
    ctx.fillStyle = '#ffd84a';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  const [ry, rx] = value.root;
  ctx.strokeStyle = '#3ddc84';
  ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.arc((rx + 0.5) * s, (ry + 0.5) * s, 9, 0, 2 * Math.PI); ctx.stroke();
  branchCache.set(value, { ...held, [which]: canvas });
  return canvas;
}

export interface Placement {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Largest rectangle with the image's aspect ratio that fits the box. */
export function fit(imgW: number, imgH: number, boxW: number, boxH: number): Placement {
  const s = Math.min(boxW / imgW, boxH / imgH);
  const w = imgW * s;
  const h = imgH * s;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

/** Paint the photo and its layers into `rect` (CSS pixels of an already scaled context). */
export function paint(ctx: CanvasRenderingContext2D, base: ImageBitmap, layers: HTMLCanvasElement[], rect: Placement, opacity: number) {
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(base, rect.x, rect.y, rect.w, rect.h);
  ctx.globalAlpha = opacity;
  if (vesselColours().mix && layers.length > 1) {
    // the layers' light added where they overlap (vessels + arteries = magenta, + veins = cyan), then laid on the photo
    // at the layers' own resolution (a zoomed view would otherwise ask for a huge canvas)
    const w = Math.max(...layers.map((l) => l.width)), h = Math.max(...layers.map((l) => l.height));
    const mix = document.createElement('canvas');
    mix.width = w;
    mix.height = h;
    const m = mix.getContext('2d')!;
    m.imageSmoothingEnabled = true;
    m.globalCompositeOperation = 'lighter';
    for (const layer of layers) m.drawImage(layer, 0, 0, w, h);
    ctx.drawImage(mix, rect.x, rect.y, rect.w, rect.h);
  } else {
    for (const layer of layers) ctx.drawImage(layer, rect.x, rect.y, rect.w, rect.h);
  }
  ctx.globalAlpha = 1;
}

export function isLayer(v: NodeValue | undefined): v is LayerValue {
  return !!v && (v.kind === 'vessel' || v.kind === 'av' || v.kind === 'disc' || v.kind === 'fovea');
}

export function countLabels(labels: Uint8Array, classes: number): number[] {
  const counts = new Array<number>(classes).fill(0);
  for (const l of labels) counts[l]++;
  return counts;
}
