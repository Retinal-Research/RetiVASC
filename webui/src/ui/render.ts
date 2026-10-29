// Turns results into drawable layers and paints them over the M0 photo.

import { AV, DISC } from '../automorph/postprocess';
import type { ArteryVeinValue, DiscCupValue, FoveaValue, NodeValue, VesselValue } from '../engine/values';

export type LayerKind = 'vessel' | 'av' | 'disc' | 'fovea';
export type LayerValue = VesselValue | ArteryVeinValue | DiscCupValue | FoveaValue;

export const LAYER_LABEL: Record<LayerKind, string> = { vessel: 'Vessels', av: 'Artery / vein', disc: 'Disc & cup', fovea: 'Fovea' };

export const LEGEND: Record<LayerKind, Array<[string, string]>> = {
  vessel: [['Vessel', '#2ee6f2']],
  av: [['Artery', '#ff4b4b'], ['Vein', '#4b8bff'], ['Crossing', '#3ddc84']],
  disc: [['Optic disc', '#ffd84a'], ['Cup', '#ff6fc1']],
  fovea: [['Fovea', '#ffffff']],
};

const cache = new WeakMap<object, HTMLCanvasElement>();

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
  if (hit) return hit;
  let canvas: HTMLCanvasElement;
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
    cache.set(value, canvas);
    return canvas;
  }
  if (value.kind === 'vessel') {
    const { size, mask } = value;
    canvas = canvasFrom(size, size, (px) => {
      for (let i = 0; i < mask.length; i++) if (mask[i]) put(px, i, 46, 230, 242, 235);
    });
  } else if (value.kind === 'av') {
    const { width, height, labels } = value.map;
    canvas = canvasFrom(width, height, (px) => {
      for (let i = 0; i < labels.length; i++) {
        const l = labels[i];
        if (l === AV.artery) put(px, i, 255, 75, 75, 235);
        else if (l === AV.vein) put(px, i, 75, 139, 255, 235);
        else if (l === AV.overlap) put(px, i, 61, 220, 132, 235);
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
  cache.set(value, canvas);
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
  for (const layer of layers) ctx.drawImage(layer, rect.x, rect.y, rect.w, rect.h);
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
