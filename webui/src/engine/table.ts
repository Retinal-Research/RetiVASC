// Results as tables: the Results table step (view.table) and the Photo folder's CSV. One row per
// photograph: each M1 step's grade, each M3 step's 72 features in AutoMorph's order, the simple measures
// of each segmentation, landmark and RetiFlow step (named after their step). An AI report connected to
// the table step writes a table of its own (engine/report.ts), shown next to these.

import { pyRepr } from '../automorph/m3/pyfloat';
import { FEATURE_ORDER } from '../ui/FeaturesPanel';
import type { PhotoRecord } from './folder';
import type { RunContext } from './specs';
import { measures } from './summary';
import type { NodeValue, Table, TableValue } from './values';

/** One row per photo; with timing, the seconds each took and its error, as a folder run has them. */
export function recordsTable(records: PhotoRecord[], timing: boolean, title = 'Results'): Table {
  const m1 = [...new Set(records.flatMap((r) => Object.keys(r.quality)))];
  const m3 = [...new Set(records.flatMap((r) => Object.keys(r.features)))];
  // segmentation measures, named after their step: vessel_Vessel_share, av_Artery_share, fovea_Fovea_x, ...
  const extra = [...new Set(records.flatMap((r) => Object.entries(r.measures ?? {}).flatMap(([id, m]) => Object.keys(m).map((k) => `${id}_${k}`))))];
  const extraValue = (r: PhotoRecord, col: string) => {
    for (const [id, m] of Object.entries(r.measures ?? {})) {
      if (col.startsWith(`${id}_`) && col.slice(id.length + 1) in m) return pyRepr(m[col.slice(id.length + 1)]);
    }
    return '';
  };
  const tag = (ids: string[], id: string, col: string) => (ids.length > 1 ? `${id}_${col}` : col);
  const columns = [
    'Name',
    ...m1.flatMap((id) => ['Quality', 'P_good', 'P_usable', 'P_reject', 'Gradable'].map((c) => tag(m1, id, c))),
    ...m3.flatMap((id) => ['Centred_on', ...FEATURE_ORDER].map((c) => tag(m3, id, c))),
    ...extra,
    ...(timing ? ['Seconds', 'Error'] : []),
  ];
  const rows = records.map((r) => [
    r.path,
    ...m1.flatMap((id) => { const q = r.quality[id]; return q ? [q.prediction, ...q.probs.map((p) => pyRepr(p)), q.gradable ? 'yes' : 'no'] : ['', '', '', '', '']; }),
    ...m3.flatMap((id) => {
      const f = r.features[id];
      return f ? [f.centredOn, ...FEATURE_ORDER.map((k) => { const v = f.features[k]; return v == null ? '' : pyRepr(v); })] : new Array(1 + FEATURE_ORDER.length).fill('');
    }),
    ...extra.map((col) => extraValue(r, col)),
    ...(timing ? [(r.ms / 1000).toFixed(1), r.error ?? ''] : []),
  ].map(String));
  return { title, columns, rows };
}

/** The numbers of one photo's connected steps, kept as a folder run keeps them. */
function recordOf(name: string, sources: Array<{ from: string; value: NodeValue }>): PhotoRecord {
  const record: PhotoRecord = { path: name, status: 'done', ms: 0, quality: {}, features: {}, measures: {} };
  for (const { from, value: v } of sources) {
    if (v.kind === 'quality') record.quality[from] = v.result;
    if (v.kind === 'features') record.features[from] = { features: v.result.features, centredOn: v.result.centredOn, gradable: v.gradable };
    const m = measures(v);
    if (m) record.measures![from] = m;
  }
  return record;
}

const hasNumbers = (t: Table) => t.columns.length > 1 && t.rows.length > 0;

/** The Results table step: the numbers of what is connected to it, for one photo or, after a folder, all. */
export async function resultsTable(ctx: RunContext): Promise<Record<string, NodeValue>> {
  const tables: Table[] = [];
  let name = 'photo';
  if (ctx.cohort) {
    const ids = new Set(ctx.cohort.sources.map((s) => s.id));
    const pick = <T,>(o: Record<string, T>) => Object.fromEntries(Object.entries(o).filter(([id]) => ids.has(id)));
    const records = ctx.cohort.records.map((r) => ({ ...r, quality: pick(r.quality), features: pick(r.features), measures: pick(r.measures ?? {}) }));
    name = 'folder';
    tables.push({ ...recordsTable(records, true, 'Results, one row per photograph'), note: `${records.length} of ${ctx.cohort.total} photographs` });
  } else {
    const sources = ctx.many?.results ?? [];
    const base = sources.map(({ value: v }) => ('base' in v ? v.base : v.kind === 'm0' ? v : undefined)).find(Boolean);
    const photo = sources.find(({ value: v }) => v.kind === 'fundus')?.value;
    name = photo?.kind === 'fundus' ? photo.name : base?.source ?? 'photo';
    tables.push(recordsTable([recordOf(name, sources)], false));
  }
  // tables an AI report wrote for this step
  for (const { value: v } of ctx.many?.results ?? []) {
    if (v.kind === 'report' && v.table) tables.push(v.table);
  }
  // it runs whatever came before it failed: with nothing to show it says so, and the steps after it go on
  return { table: { kind: 'table', name, tables: tables.filter(hasNumbers) } satisfies TableValue };
}

// ---------------------------------------------------------------- CSV

export function csvText(t: Table): string {
  const cell = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  return [t.columns, ...t.rows].map((r) => r.map(cell).join(',')).join('\n') + '\n';
}

/** "<photo or folder>_<workflow>[_<table>]_<date>.csv", with only letters, digits, - and _. */
export function csvName(name: string, workflow: string, table = ''): string {
  const safe = (s: string) => s.replace(/\.[a-z0-9]{2,5}$/i, '').replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '');
  const date = new Date().toISOString().slice(0, 10);
  return [safe(name) || 'photo', safe(workflow) || 'RetiVASC', safe(table), date].filter(Boolean).join('_') + '.csv';
}

export function downloadCsv(t: Table, name: string, workflow: string, table = '') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csvText(t)], { type: 'text/csv' }));
  a.download = csvName(name, workflow, table);
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
