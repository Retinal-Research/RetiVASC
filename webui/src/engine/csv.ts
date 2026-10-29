// The results as a CSV file: one row per photograph, for a folder or for the one photo on the canvas.
// Columns: each M1 step's grade, each M3 step's 72 features in AutoMorph's order, the simple measures of
// each segmentation, landmark and RetiFlow step (named after their step), then time and errors.

import { pyRepr } from '../automorph/m3/pyfloat';
import { FEATURE_ORDER } from '../ui/FeaturesPanel';
import type { PhotoRecord } from './folder';

export function csvText(rows: PhotoRecord[]): string {
  const m1 = [...new Set(rows.flatMap((r) => Object.keys(r.quality)))];
  const m3 = [...new Set(rows.flatMap((r) => Object.keys(r.features)))];
  // segmentation measures, named after their step: vessel_Vessel_share, av_Artery_share, fovea_Fovea_x, ...
  const extra = [...new Set(rows.flatMap((r) => Object.entries(r.measures ?? {}).flatMap(([id, m]) => Object.keys(m).map((k) => `${id}_${k}`))))];
  const extraValue = (r: PhotoRecord, col: string) => {
    for (const [id, m] of Object.entries(r.measures ?? {})) {
      if (col.startsWith(`${id}_`) && col.slice(id.length + 1) in m) return pyRepr(m[col.slice(id.length + 1)]);
    }
    return '';
  };
  const tag = (ids: string[], id: string, col: string) => (ids.length > 1 ? `${id}_${col}` : col);
  const head = [
    'Name',
    ...m1.flatMap((id) => ['Quality', 'P_good', 'P_usable', 'P_reject', 'Gradable'].map((c) => tag(m1, id, c))),
    ...m3.flatMap((id) => ['Centred_on', ...FEATURE_ORDER].map((c) => tag(m3, id, c))),
    ...extra,
    'Seconds', 'Error',
  ];
  const cell = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const line = (r: PhotoRecord) => [
    r.path,
    ...m1.flatMap((id) => { const q = r.quality[id]; return q ? [q.prediction, ...q.probs.map((p) => pyRepr(p)), q.gradable ? 'yes' : 'no'] : ['', '', '', '', '']; }),
    ...m3.flatMap((id) => {
      const f = r.features[id];
      return f ? [f.centredOn, ...FEATURE_ORDER.map((k) => { const v = f.features[k]; return v == null ? '' : pyRepr(v); })] : new Array(1 + FEATURE_ORDER.length).fill('');
    }),
    ...extra.map((col) => extraValue(r, col)),
    (r.ms / 1000).toFixed(1), r.error ?? '',
  ].map(String).map(cell).join(',');
  return [head.join(','), ...rows.map(line)].join('\n') + '\n';
}

/** "<photo or folder>_<workflow>_<date>.csv", with only letters, digits, - and _. */
export function csvName(name: string, workflow: string): string {
  const safe = (s: string) => s.replace(/\.[a-z0-9]{2,5}$/i, '').replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '');
  const date = new Date().toISOString().slice(0, 10);
  return `${safe(name) || 'photo'}_${safe(workflow) || 'RetiVASC'}_${date}.csv`;
}

export function downloadCsv(rows: PhotoRecord[], name: string, workflow: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csvText(rows)], { type: 'text/csv' }));
  a.download = csvName(name, workflow);
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
