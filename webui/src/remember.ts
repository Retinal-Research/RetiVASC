// Small per-browser preferences: folded sidebar sections, panel sizes, the GPU to use. Kept in
// localStorage; where that is blocked (some private windows) they last until the page closes.

import { useCallback, useState } from 'react';

const PREFIX = 'retivasc-studio-';

export const isBool = (v: unknown): v is boolean => typeof v === 'boolean';
export const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

export function recall<T>(name: string, fallback: T, valid: (v: unknown) => v is T): T {
  try {
    const raw = localStorage.getItem(PREFIX + name);
    if (raw === null) return fallback;
    const v: unknown = JSON.parse(raw);
    return valid(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

export function remember(name: string, value: unknown) {
  try {
    if (value === undefined) localStorage.removeItem(PREFIX + name);
    else localStorage.setItem(PREFIX + name, JSON.stringify(value));
  } catch {
    // storage blocked: the choice just won't be remembered
  }
}

export function useRemembered<T>(name: string, fallback: T, valid: (v: unknown) => v is T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => recall(name, fallback, valid));
  const set = useCallback((v: T) => { setValue(v); remember(name, v); }, [name]);
  return [value, set];
}
