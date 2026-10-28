// Client for the RetiVASC server (python -m retivasc_server serve). In development the
// Vite dev server proxies /api to it; in production the server serves this web UI itself.
// Without a server the canvas still works: browser steps run locally, server steps say so.

export interface Account { name: string; admin: boolean; mode?: 'accounts' | 'open' }
export interface ServerInfo { version: string; mode?: 'accounts' | 'open'; queued: number; running: boolean; allowed_folders: string[]; retention_days: number; device: string; devices?: ServerDevice[]; features?: { sheet: boolean } }
/** An inference device on the server; GPUs come first, the CPU last. */
export interface ServerDevice { id: string; name: string; memory_gb?: number; free_gb?: number }
/** One second of the server machine's load; GPU memory is in bytes and may be unknown. */
export interface StatGpu { id: string; name: string; util: number; mem_used: number | null; mem_total: number | null }
export interface StatSample { t: number; cpu: number; mem_used: number; mem_total: number; gpus: StatGpu[] }
export interface ServerStats { host: string; queued: number; running: boolean; last_run: { seconds: number; photos: number } | null; samples: StatSample[] }
export interface JobProgress { step: string; fraction: number | null }
export interface JobView {
  id: string; owner: string; kind: 'analyze' | 'sheet'; title: string; created: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  started: string | null; finished: string | null; error: string | null; progress: JobProgress;
}
export interface PhotoResult {
  name: string; stem: string;
  quality: { good: number; usable: number; reject: number; grade: 'Good' | 'Usable' | 'Reject'; gradable: boolean } | null;
  centred_on: 'macula' | 'optic disc' | null;
  features: Record<string, number | null> | null;
  images: Partial<Record<'photo' | 'vessels' | 'artery_vein' | 'disc', string>>;
}
export interface JobDetail extends JobView { result: { photos?: PhotoResult[]; files?: string[]; deliverable?: boolean; summary?: unknown; photos_in_run?: number } }

export class ServerError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`/api${path}`, { credentials: 'same-origin', ...init });
  } catch {
    throw new ServerError(0, 'the RetiVASC server is not reachable');
  }
  if (r.status === 502 || r.status === 503 || r.status === 504) throw new ServerError(0, 'the RetiVASC server is not running');
  if (!r.ok) {
    let detail = `server error ${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ServerError(r.status, detail);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const api = {
  me: () => call<Account>('/me'),
  login: (name: string, password: string) => call<Account>('/login', json({ name, password })),
  logout: () => call<{ ok: boolean }>('/logout', { method: 'POST' }),
  server: () => call<ServerInfo>('/server'),
  stats: () => call<ServerStats>('/stats'),
  jobs: () => call<JobView[]>('/jobs'),
  job: (id: string) => call<JobDetail>(`/jobs/${id}`),
  remove: (id: string) => call<{ ok: boolean }>(`/jobs/${id}`, { method: 'DELETE' }),
  imageUrl: (id: string, path: string, width = 0) =>
    `/api/jobs/${id}/image/${path.split('/').map(encodeURIComponent).join('/')}${width ? `?w=${width}` : ''}`,
  fileUrl: (id: string, name: string) => `/api/jobs/${id}/file/${encodeURIComponent(name)}`,
};

/** POST multipart with upload progress (fetch cannot report it). */
export function upload(path: string, form: FormData, onProgress: (fraction: number) => void): Promise<JobView> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api${path}`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body: { detail?: string } & Partial<JobView> = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* empty */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as JobView);
      else reject(new ServerError(xhr.status, body.detail ?? `server error ${xhr.status}`));
    };
    xhr.onerror = () => reject(new ServerError(0, 'the RetiVASC server is not reachable'));
    xhr.send(form);
  });
}

/** Poll a job until it ends; returns the finished job or throws with its error. */
export async function waitJob(id: string, onProgress: (p: JobProgress) => void): Promise<JobDetail> {
  for (;;) {
    const j = await api.job(id);
    if (j.status === 'done') return j;
    if (j.status === 'failed' || j.status === 'cancelled') throw new Error(j.error ?? `job ${j.status}`);
    onProgress(j.progress);
    await new Promise((r) => setTimeout(r, 1500));
  }
}

// Files the user picked for a server step (a photo folder, a sheet template). They stay in
// this page's memory and are sent only to the RetiVASC server when the step runs.
const picked = new Map<string, File[]>();
let nextPick = 1;
export function addPicked(files: File[]): string {
  const id = `picked:${nextPick++}`;
  picked.set(id, files);
  return id;
}
export const getPicked = (id: string | undefined) => (id ? picked.get(id) : undefined);
