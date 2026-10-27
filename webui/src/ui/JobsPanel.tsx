// The signed-in user's jobs on the server: progress while they run, downloads when done.
// Jobs keep running when the page is closed; this list is where to find them again.

import { useEffect, useState } from 'react';
import { api, type JobDetail, type JobView } from '../server/api';

export function JobsPanel() {
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [detail, setDetail] = useState<JobDetail | null>(null);
  useEffect(() => {
    let live = true;
    const tick = () => api.jobs().then((j) => live && setJobs(j)).catch(() => undefined);
    tick();
    const t = setInterval(tick, 4000);
    return () => { live = false; clearInterval(t); };
  }, []);
  useEffect(() => {
    if (!open) { setDetail(null); return; }
    api.job(open).then(setDetail).catch(() => setDetail(null));
  }, [open, jobs]);
  if (!jobs.length) return null;
  const remove = (id: string) => void api.remove(id).then(() => { setOpen(null); setJobs((js) => js.filter((x) => x.id !== id)); });
  return (
    <section className="jobs">
      <h3>My jobs on the server</h3>
      {jobs.slice(0, 12).map((j) => (
        <div key={j.id} className={`job st-${j.status}`}>
          <button className="job-head" onClick={() => setOpen(open === j.id ? null : j.id)}>
            <span className="job-title">{j.kind === 'sheet' ? 'Sheet' : 'Analysis'} · {j.title}</span>
            <span className="job-status">{j.status === 'running' || j.status === 'queued' ? j.progress.step : j.status}</span>
          </button>
          {(j.status === 'running' || j.status === 'queued') && (
            <span className="job-bar"><i style={{ width: `${Math.round((j.progress.fraction ?? 0) * 100)}%` }} /></span>
          )}
          {open === j.id && detail && (
            <div className="job-body">
              <div className="faint small">{new Date(j.created).toLocaleString()}</div>
              {j.error && <div className="error small">{j.error}</div>}
              {(detail.result.files ?? []).map((f) => (
                <a key={f} href={api.fileUrl(j.id, f)} download className="small">{f}</a>
              ))}
              {j.status !== 'running' && <button className="btn small" onClick={() => remove(j.id)}>Delete from the server</button>}
            </div>
          )}
        </div>
      ))}
    </section>
  );
}
