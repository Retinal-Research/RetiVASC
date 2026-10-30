// The Photo folder node: choose a folder, follow the run, stop it, download one CSV for the folder.
// The run itself is the canvas Run (src/engine/executor.ts); this file only shows it.

import { useEffect, useRef, useSyncExternalStore } from 'react';
import type { Executor, FolderRun } from '../engine/executor';
import { addFolder, FOLDER_LIMIT, getFolder, sampleFolder } from '../engine/folder';
import { useStudio } from './context';
import { downloadCsv, recordsTable } from '../engine/table';
import { ms, pct } from './format';

export const useFolderRun = (executor: Executor): FolderRun | undefined => useSyncExternalStore(executor.subscribe, executor.folder);

/** The run of this node's current folder, if there is one. */
function runOf(run: FolderRun | undefined, nodeId: string, folderId: unknown): FolderRun | undefined {
  const folder = getFolder(folderId);
  return run && folder && run.nodeId === nodeId && run.key.startsWith(`${folder.signature}-`) ? run : undefined;
}

function counts(run: FolderRun | undefined, total: number) {
  const all = run ? [...run.records.values()] : [];
  const done = all.filter((r) => r.status === 'done').length;
  const errors = all.filter((r) => r.status === 'error').length;
  return { done, errors, left: total - done, total };
}

export function FolderBody({ nodeId, params }: { nodeId: string; params: Record<string, unknown> }) {
  const studio = useStudio();
  // until a folder is chosen, the sample folder that comes with the site
  useEffect(() => {
    if (params.folderId === undefined) void sampleFolder().then((folderId) => studio.setParams(nodeId, { folderId }), () => {});
  }, [params.folderId, nodeId, studio]);
  const input = useRef<HTMLInputElement>(null);
  const folder = getFolder(params.folderId);
  const run = runOf(useFolderRun(studio.executor), nodeId, params.folderId);
  const c = counts(run, folder?.photos.length ?? 0);
  const perPhoto = run?.sessionDone ? run.sessionMs / run.sessionDone : undefined;

  return (
    <div className="study folder">
      <div className="row">
        <button className="btn small" disabled={studio.running} onClick={() => input.current?.click()}>Choose a photo folder…</button>
        {/* webkitdirectory: the browser keeps each file's path inside the folder (subject folders matter) */}
        <input ref={input} type="file" hidden multiple {...({ webkitdirectory: '' } as object)}
               onChange={(e) => { studio.setParams(nodeId, { folderId: addFolder(Array.from(e.target.files ?? [])) }); e.target.value = ''; }} />
      </div>
      {!folder ? (
        <div className="hint">Up to {FOLDER_LIMIT} photos (tif, jpg, png, bmp), subfolders included. They stay on this computer.</div>
      ) : folder.refused ? (
        <div className="hint bad">{folder.name}: {folder.refused}</div>
      ) : (
        <div className="meta">
          <b>{folder.name}</b> · {c.total} photos{run && run.resumed > 0 && ` · ${run.resumed} done earlier, skipped`}
        </div>
      )}

      {run && (
        <div className="bprog">
          <span className="bbar"><i style={{ width: pct(c.done / c.total, 1) }} />{c.errors > 0 && <i className="err" style={{ width: pct(c.errors / c.total, 1) }} />}</span>
          <span className="meta">
            {c.done} of {c.total} done{c.errors > 0 && <span className="bad"> · {c.errors} failed</span>}
            {perPhoto && <> · {ms(perPhoto)} per photo{run.running && c.left > 0 && <> · about {ms(perPhoto * (c.left - c.errors))} left</>}</>}
          </span>
          {run.current && <span className="hint" title={run.current.path}>Now: {run.current.path}</span>}
        </div>
      )}
      {folder && !folder.refused && (
        <div className="row">
          {run?.running ? (
            <button className="btn small" disabled={run.stopping} onClick={() => studio.executor.stop()}>{run.stopping ? 'Stopping…' : 'Stop'}</button>
          ) : (
            <span className="hint">{c.left === 0 && run ? 'All photos done.' : run && c.done ? 'Press Run to continue.' : 'Press Run to start.'}</span>
          )}
          <button className="btn small" disabled={!run || c.done + c.errors === 0} onClick={() => run && downloadCsv(recordsTable(run.photos.flatMap((p) => (run.records.has(p.path) ? [run.records.get(p.path)!] : [])), true), folder.name, studio.workflow)}>Download CSV</button>
        </div>
      )}
      {folder && !folder.refused && <div className="hint">Stop any time. Finished photos are kept in this browser: choose the same folder and press Run to continue.</div>}
    </div>
  );
}

/** Status line under the folder node: the photos, not the last step's time. */
export function FolderFoot({ nodeId, params }: { nodeId: string; params: Record<string, unknown> }) {
  const studio = useStudio();
  const folder = getFolder(params.folderId);
  const run = runOf(useFolderRun(studio.executor), nodeId, params.folderId);
  const c = counts(run, folder?.photos.length ?? 0);
  const text = !folder || folder.refused ? 'Choose a folder'
    : run?.running ? (run.stopping ? 'Stopping after this step' : `Photo ${(run.current?.index ?? 0) + 1} of ${c.total} · ${c.done} done`)
      : run ? `${c.done} of ${c.total} done${c.errors ? `, ${c.errors} failed` : ''}` : `${c.total} photos · press Run`;
  return (
    <footer className={`node-foot ${run?.running ? 'f-run' : ''}`}>
      <span className="sdot" />
      <span className="stext">{text}</span>
    </footer>
  );
}

/** The inspector: every photo of the folder with its state, grade and time. */
export function FolderInspector({ nodeId, params }: { nodeId: string; params: Record<string, unknown> }) {
  const studio = useStudio();
  const folder = getFolder(params.folderId);
  const run = runOf(useFolderRun(studio.executor), nodeId, params.folderId);
  if (!folder?.photos.length) return <p className="faint">Choose a folder on the step to see its photos here.</p>;
  return (
    <table className="btable">
      <thead><tr><th>Photo</th><th>Quality</th><th>Time</th></tr></thead>
      <tbody>
        {folder.photos.map((p) => {
          const r = run?.records.get(p.path);
          const status = run?.current?.path === p.path ? 'running' : r?.status ?? 'pending';
          const grade = r && Object.values(r.quality)[0]?.prediction;
          return (
            <tr key={p.path} className={`b-${status}`} title={r?.error ?? p.path}>
              <td className="bname">{p.path}</td>
              <td>{status === 'error' ? 'failed' : status === 'running' ? 'running…' : grade ?? ''}</td>
              <td>{r ? ms(r.ms) : ''}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
