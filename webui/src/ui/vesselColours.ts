// The colours of the vessel layers: vessels, arteries, veins and their crossings, for every view of a
// segmentation (thumbnails, the inspector, the overlay, RetiFlow's centrelines, the legends). Two presets
// and the person's own. "Mix" adds the layers' light where they overlap, as RRWNet shows its AV3 maps:
// vessels blue, arteries red, veins green, so an artery reads magenta, a vein cyan, a crossing white.
// Kept per browser.

import { useSyncExternalStore } from 'react';
import { recall, remember } from '../remember';

export interface VesselColours {
  preset: 'rrwnet' | 'automorph' | 'custom';
  vessel: string;
  artery: string;
  vein: string;
  crossing: string;
  /** add the layers' light where they overlap */
  mix: boolean;
}

export const PRESETS: Record<'rrwnet' | 'automorph', VesselColours> = {
  rrwnet: { preset: 'rrwnet', vessel: '#0000ff', artery: '#ff0000', vein: '#00ff00', crossing: '#ffff00', mix: true },
  automorph: { preset: 'automorph', vessel: '#2ee6f2', artery: '#ff4b4b', vein: '#4b8bff', crossing: '#3ddc84', mix: false },
};

const HEX = /^#[0-9a-f]{6}$/i;
const valid = (v: unknown): v is VesselColours => {
  const c = v as VesselColours;
  return !!c && typeof c === 'object' && ['rrwnet', 'automorph', 'custom'].includes(c.preset)
    && [c.vessel, c.artery, c.vein, c.crossing].every((x) => typeof x === 'string' && HEX.test(x)) && typeof c.mix === 'boolean';
};

let current: VesselColours = recall('vessel-colours', PRESETS.rrwnet, valid);
let version = 0;
const listeners = new Set<() => void>();

export const vesselColours = () => current;
/** changes whenever the colours do: part of every cache of drawn layers */
export const coloursVersion = () => version;

export function setVesselColours(c: VesselColours) {
  current = c;
  version++;
  remember('vessel-colours', c);
  listeners.forEach((l) => l());
}

export function useVesselColours(): VesselColours {
  return useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => current);
}

export function rgb(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

const toHex = (c: number[]) => `#${c.map((x) => Math.min(255, x).toString(16).padStart(2, '0')).join('')}`;

/** What a class looks like where it lies on the vessels: with Mix, its light added to the vessels'. */
export function seen(kind: 'vessel' | 'artery' | 'vein' | 'crossing', c: VesselColours = current): string {
  if (!c.mix || kind === 'vessel') return c[kind];
  const a = rgb(c[kind]), v = rgb(c.vessel);
  return toHex(a.map((x, i) => x + v[i]));
}
