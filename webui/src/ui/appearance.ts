// Day/night theme and accent colour. Kept in localStorage per browser; the
// inline script in index.html applies it before the first paint.

import { useEffect, useState } from 'react';

export type ThemeMode = 'system' | 'light' | 'dark';
export type Theme = 'light' | 'dark';

export interface Appearance {
  mode: ThemeMode;
  accent: string;
}

export const DEFAULT_APPEARANCE: Appearance = { mode: 'system', accent: '#2dd4bf' };

export const ACCENTS: Array<{ name: string; value: string }> = [
  { name: 'Teal', value: '#2dd4bf' },
  { name: 'Sky', value: '#38bdf8' },
  { name: 'Blue', value: '#3b82f6' },
  { name: 'Indigo', value: '#6366f1' },
  { name: 'Violet', value: '#a78bfa' },
  { name: 'Pink', value: '#ec4899' },
  { name: 'Red', value: '#ef4444' },
  { name: 'Orange', value: '#f97316' },
  { name: 'Amber', value: '#f59e0b' },
  { name: 'Green', value: '#22c55e' },
];

const KEY = 'retivasc-studio-appearance';
const HEX = /^#[0-9a-f]{6}$/i;
const DARK_INK = '#0b1210';

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Text colour for a filled accent: whichever of white / near-black contrasts more. */
export function inkFor(accent: string): string {
  const l = luminance(accent);
  const onWhite = 1.05 / (l + 0.05);
  const onDark = (l + 0.05) / (luminance(DARK_INK) + 0.05);
  return onDark >= onWhite ? DARK_INK : '#ffffff';
}

export function loadAppearance(): Appearance {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Appearance>;
    const mode = raw.mode === 'light' || raw.mode === 'dark' || raw.mode === 'system' ? raw.mode : DEFAULT_APPEARANCE.mode;
    const accent = typeof raw.accent === 'string' && HEX.test(raw.accent) ? raw.accent : DEFAULT_APPEARANCE.accent;
    return { mode, accent };
  } catch {
    return DEFAULT_APPEARANCE;
  }
}

function save(a: Appearance) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...a, ink: inkFor(a.accent) }));
  } catch {
    // private windows or blocked storage: the choice just won't be remembered
  }
}

const systemDark = () => window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true;

export function resolveTheme(mode: ThemeMode): Theme {
  return mode === 'system' ? (systemDark() ? 'dark' : 'light') : mode;
}

export function isHex(value: string): boolean {
  return HEX.test(value);
}

/** Current appearance, a setter that applies and remembers it, and the resolved theme. */
export function useAppearance(): [Appearance, (a: Appearance) => void, Theme] {
  const [appearance, setAppearance] = useState(loadAppearance);
  const [theme, setTheme] = useState<Theme>(() => resolveTheme(appearance.mode));

  useEffect(() => {
    const update = () => setTheme(resolveTheme(appearance.mode));
    update();
    if (appearance.mode !== 'system') return;
    const query = window.matchMedia('(prefers-color-scheme: dark)');
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, [appearance.mode]);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.style.setProperty('--accent', appearance.accent);
    root.style.setProperty('--accent-ink', inkFor(appearance.accent));
  }, [theme, appearance.accent]);

  const update = (a: Appearance) => {
    setAppearance(a);
    save(a);
  };
  return [appearance, update, theme];
}
