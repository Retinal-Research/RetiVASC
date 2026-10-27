// Bodies of the study-sheet nodes: choosing the photos (upload a folder, a server folder,
// or an earlier run) and a sheet template. Picked files stay in this page until the
// sheet step sends them to the RetiVASC server.

import { useEffect, useRef, useState } from 'react';
import { addPicked, api, getPicked, type JobView, type ServerInfo } from '../server/api';

type SetParams = (patch: Record<string, unknown>) => void;
const PHOTO = /\.(tiff?|jpe?g|png|bmp)$/i;

function useServer() {
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    let live = true;
    Promise.all([api.server(), api.jobs()])
      .then(([i, j]) => { if (live) { setInfo(i); setJobs(j); } })
      .catch((e: Error) => live && setError(e.message));
    return () => { live = false; };
  }, []);
  return { info, jobs, error };
}

export function StudyPhotosBody({ params, setParams, photos }: { params: Record<string, unknown>; setParams: SetParams; photos?: number }) {
  const { info, jobs, error } = useServer();
  const folderInput = useRef<HTMLInputElement>(null);
  const source = String(params.source ?? 'upload');
  const picked = getPicked(String(params.pickId ?? ''));
  const finished = jobs.filter((j) => j.kind === 'sheet' && j.status === 'done');

  const pickFolder = (list: FileList | null) => {
    const files = Array.from(list ?? []);
    const top = files[0]?.webkitRelativePath.split('/')[0] ?? '';
    const n = files.filter((f) => PHOTO.test(f.name)).length;
    setParams({ pickId: addPicked(files), pickLabel: `${top} · ${n} photos` });
  };

  return (
    <div className="study">
      {error && <div className="hint">Needs the RetiVASC server: {error}</div>}
      <div className="seg">
        {(['upload', 'server', 'reuse'] as const).map((s) => (
          <button key={s} className={source === s ? 'on' : ''} onClick={() => setParams({ source: s })}>
            {s === 'upload' ? 'Upload a folder' : s === 'server' ? 'Folder on the server' : 'Earlier run'}
          </button>
        ))}
      </div>
      {source === 'upload' && (
        <>
          <button className="btn small" onClick={() => folderInput.current?.click()}>Choose the photo folder…</button>
          {/* webkitdirectory: the browser keeps each file's path inside the folder (subject folders matter) */}
          <input ref={folderInput} type="file" hidden multiple {...({ webkitdirectory: '' } as object)} onChange={(e) => pickFolder(e.target.files)} />
          <div className="meta">{picked ? String(params.pickLabel ?? '') : 'No folder chosen'}</div>
          <div className="hint">Photos go to the RetiVASC server only; the subject folders inside are kept.</div>
        </>
      )}
      {source === 'server' && (
        <label className="sfield">
          <span>Folder on the server</span>
          <input className="select" list="allowed-folders" value={String(params.folder ?? '')} placeholder={info?.allowed_folders[0] ?? 'no folders allowed yet'}
                 onChange={(e) => setParams({ folder: e.target.value })} />
          <datalist id="allowed-folders">{info?.allowed_folders.map((f) => <option key={f} value={f} />)}</datalist>
          <span className="hint">Only folders the administrator allowed: {info?.allowed_folders.join('; ') || 'none'}</span>
        </label>
      )}
      {source === 'reuse' && (
        <label className="sfield">
          <span>Reuse the AutoMorph run of an earlier sheet (no new run)</span>
          <select className="select" value={String(params.reuseJob ?? '')}
                  onChange={(e) => setParams({ reuseJob: e.target.value, reuseLabel: e.target.selectedOptions[0]?.text ?? '' })}>
            <option value="">Choose…</option>
            {finished.map((j) => <option key={j.id} value={j.id}>{j.title} · {new Date(j.created).toLocaleString()}</option>)}
          </select>
        </label>
      )}
      {photos !== undefined && photos > 0 && <div className="meta">{photos} files ready</div>}
    </div>
  );
}

export function FilePick({ label, id, onPick, accept = '.xlsx' }: { label: string; id?: string; onPick: (id: string, name: string) => void; accept?: string }) {
  const input = useRef<HTMLInputElement>(null);
  const file = getPicked(id)?.[0];
  return (
    <label className="sfield">
      <span>{label}</span>
      <div className="row">
        <button className="btn small" onClick={(e) => { e.preventDefault(); input.current?.click(); }}>Choose…</button>
        <span className="meta">{file?.name ?? 'none'}</span>
      </div>
      <input ref={input} type="file" hidden accept={accept}
             onChange={(e) => { const f = e.target.files?.[0]; if (f) onPick(addPicked([f]), f.name); }} />
    </label>
  );
}
