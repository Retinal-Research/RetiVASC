// The batch node: choose a folder, start, stop, resume, download the CSV. The run itself is in
// src/engine/batch.ts; this file only shows it.

import { useRef, useSyncExternalStore } from 'react';
import { pyRepr } from '../automorph/m3/pyfloat';
import { batch, BATCH_LIMIT, type BatchState } from '../engine/batch';
import { useStudio } from './context';
import { FEATURE_ORDER } from './FeaturesPanel';
import { ms, pct } from './format';

export const useBatch = (): BatchState => useSyncExternalStore(batch.subscribe, batch.get);

function counts(s: BatchState) {
  const done = s.photos.filter((p) => p.status === 'done').length;
  const errors = s.photos.filter((p) => p.status === 'error').length;
  return { done, errors, left: s.photos.length - done - errors, total: s.photos.length };
}

/** One row per finished photo: grade, AutoMorph's 72 feature columns, then how it ran. */
function downloadCsv(folder: string) {
  const head = ['Name', 'Quality', 'P_good', 'P_usable', 'P_reject', 'Gradable', 'Centred_on', ...FEATURE_ORDER, 'Seconds', 'Ran_on'];
  const cell = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const lines = batch.results().map((r) => [
    r.path, r.quality.prediction, ...r.quality.probs.map((p) => pyRepr(p)), r.quality.gradable ? 'yes' : 'no', r.centredOn,
    ...FEATURE_ORDER.map((k) => { const v = r.features[k]; return v == null ? '' : pyRepr(v); }),
    (r.ms / 1000).toFixed(1), r.backend,
  ].map(String).map(cell).join(','));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([[head.join(','), ...lines].join('\n') + '\n'], { type: 'text/csv' }));
  a.download = `${folder}_RetiVASC_features.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function BatchBody() {
  const s = useBatch();
  const studio = useStudio();
  const input = useRef<HTMLInputElement>(null);
  const c = counts(s);
  const perPhoto = s.sessionDone ? s.sessionMs / s.sessionDone : undefined;

  return (
    <div className="study batch">
      <div className="row">
        <button className="btn small" disabled={s.running} onClick={() => input.current?.click()}>Choose a photo folder…</button>
        {/* webkitdirectory: the browser keeps each file's path inside the folder (subject folders matter) */}
        <input ref={input} type="file" hidden multiple {...({ webkitdirectory: '' } as object)}
               onChange={(e) => { void batch.open(Array.from(e.target.files ?? [])); e.target.value = ''; }} />
      </div>
      {s.refused ? (
        <div className="hint bad">{s.folder}: {s.refused}</div>
      ) : s.folder ? (
        <div className="meta"><b>{s.folder}</b> · {c.total} photos{s.resumed > 0 && !s.running && ` · ${s.resumed} done earlier, skipped`}</div>
      ) : (
        <div className="hint">Up to {BATCH_LIMIT} photos (tif, jpg, png, bmp), subfolders included. They stay on this computer.</div>
      )}
      <label className="scheck" title="AutoMorph measures vessels on gradable photos only">
        <input type="checkbox" checked={s.measureAll} disabled={s.running} onChange={(e) => void batch.setMeasureAll(e.target.checked)} />
        Measure anyway when not gradable
      </label>

      {c.total > 0 && (
        <>
          <div className="bprog">
            <span className="bbar"><i style={{ width: pct(c.done / c.total, 1) }} />{c.errors > 0 && <i className="err" style={{ width: pct(c.errors / c.total, 1) }} />}</span>
            <span className="meta">
              {c.done} of {c.total} done{c.errors > 0 && <span className="bad"> · {c.errors} failed</span>}
              {perPhoto && <> · {ms(perPhoto)} per photo{s.running && c.left > 0 && <> · about {ms(perPhoto * c.left)} left</>}</>}
            </span>
          </div>
          {s.current && <div className="hint" title={s.current.path}>{s.current.stage} · {s.current.path}</div>}
          <div className="row">
            {s.running ? (
              <button className="btn small" disabled={s.stopping} onClick={() => batch.stop()}>{s.stopping ? 'Stopping…' : 'Stop'}</button>
            ) : (
              <button className="btn small primary" disabled={c.left === 0 || studio.running} title={studio.running ? 'Wait for the canvas run to finish' : undefined} onClick={() => void batch.start()}>
                {c.done > 0 && c.left > 0 ? `Resume (${c.left} left)` : c.left === 0 ? 'All done' : 'Start'}
              </button>
            )}
            <button className="btn small" disabled={c.done === 0} onClick={() => downloadCsv(s.folder ?? 'batch')}>Download CSV</button>
            <button className="btn small" disabled={s.running || (c.done === 0 && c.errors === 0)} title="Forget this folder's results and start again"
                    onClick={() => { if (window.confirm('Forget the results of this folder?')) void batch.clear(); }}>Clear</button>
          </div>
          <div className="hint">Stop any time. Finished photos are kept in this browser: choose the same folder again to continue.</div>
        </>
      )}
    </div>
  );
}

/** Status line under the node: the batch, not the canvas Run. */
export function batchStatus(s: BatchState): string {
  const c = counts(s);
  if (!c.total) return 'Choose a folder';
  if (s.running) return s.stopping ? 'Stopping after this step' : `Running · ${c.done} of ${c.total}`;
  return c.left === 0 ? `Done · ${c.done} photos${c.errors ? `, ${c.errors} failed` : ''}` : `${c.done} of ${c.total} done`;
}

/** The inspector: every photo with its state, grade and time. */
export function BatchInspector() {
  const s = useBatch();
  if (!s.photos.length) return <p className="faint">Choose a folder on the batch step to see its photos here.</p>;
  return (
    <table className="btable">
      <thead><tr><th>Photo</th><th>Quality</th><th>Time</th></tr></thead>
      <tbody>
        {s.photos.map((p) => (
          <tr key={p.path} className={`b-${p.status}`} title={p.error ?? p.path}>
            <td className="bname">{p.path}</td>
            <td>{p.status === 'error' ? 'failed' : p.status === 'running' ? 'running…' : p.grade ?? ''}</td>
            <td>{p.ms !== undefined ? ms(p.ms) : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
