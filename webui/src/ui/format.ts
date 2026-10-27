import type { NodeState } from '../engine/executor';

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

export function ms(v: number | undefined): string {
  if (v === undefined) return '–';
  return v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(v < 10_000 ? 2 : 1)} s`;
}

export const mb = (bytes: number) => `${(bytes / 2 ** 20).toFixed(bytes < 10 * 2 ** 20 ? 1 : 0)} MB`;

export const int = (v: number) => v.toLocaleString('en-US');

export function statusText(state: NodeState): string {
  switch (state.status) {
    case 'idle':
      return 'Ready';
    case 'blocked':
      return state.message ?? 'Waiting';
    case 'running':
      return state.message ?? 'Running';
    case 'done':
      return state.cached ? `Unchanged · ${ms(state.ms)}` : `Done · ${ms(state.ms)}`;
    case 'error':
      return state.message ?? 'Failed';
  }
}
