// Batch mode: the full AutoMorph v2 pipeline on a folder of photos, one photo at a time, in this
// browser. Each photo goes through the same engine operations as the canvas steps (decode, M0, M1,
// the three M2 models, M3), so a photo gives the same values in both. With the session budget at
// 2 GB every ensemble member stays loaded, so models load once for the whole folder.
//
// Each finished photo is stored in IndexedDB at once. A run can be stopped at any time and resumed
// later, also after the page was closed: choosing the same folder again (same files, same sizes and
// dates, same settings) skips the photos that are already done.

import type { QualityResult } from '../automorph/types';
import type { M3Result } from '../automorph/m3/m3';
import { engine } from '../runtime/engine';
import type { ImageHandle, LabelOutput, M0Output, M3Output, QualityOutput, VesselOutput } from '../runtime/protocol';

export const BATCH_LIMIT = 1000;
const PHOTO = /\.(tiff?|jpe?g|png|bmp)$/i;

export type PhotoStatus = 'pending' | 'running' | 'done' | 'error';

/** One photo's results, as stored and exported. */
export interface BatchRow {
  path: string;
  quality: QualityResult;
  centredOn: M3Result['centredOn'];
  features: Record<string, number | null>;
  ms: number;
  backend: string;
}

export interface BatchPhoto {
  path: string;
  file: File;
  status: PhotoStatus;
  ms?: number;
  error?: string;
  grade?: string;
}

export interface BatchState {
  folder?: string;
  photos: BatchPhoto[];
  /** more photos than BATCH_LIMIT, or none: why the folder cannot run */
  refused?: string;
  /** measure vessels on photos that are not gradable, as the canvas M3 option does */
  measureAll: boolean;
  running: boolean;
  stopping: boolean;
  current?: { path: string; stage: string };
  /** this session's processing time, for the speed and the time left */
  sessionMs: number;
  sessionDone: number;
  /** done before this page opened the folder (restored from IndexedDB) */
  resumed: number;
}

// ---------------------------------------------------------------- IndexedDB

const DB_NAME = 'retivasc-batch';
const STORE = 'results';

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, work: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const req = work(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

/** Every stored row of one batch (key prefix), by path. */
async function loadRows(prefix: string): Promise<Map<string, BatchRow>> {
  const range = IDBKeyRange.bound(prefix, `${prefix}￿`);
  const rows = await tx('readonly', (s) => s.getAll(range) as IDBRequest<BatchRow[]>);
  return new Map(rows.map((r) => [r.path, r]));
}

// ---------------------------------------------------------------- the runner

/** Same folder (paths, sizes, dates) and same settings give the same key, so a rerun resumes. */
function batchKey(photos: BatchPhoto[], measureAll: boolean, models: string): string {
  const text = [models, measureAll ? 'all' : 'gradable', ...photos.map((p) => `${p.path}:${p.file.size}:${p.file.lastModified}`)].join('|');
  let h = 0x811c9dc5;  // FNV-1a
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
  return `${(h >>> 0).toString(16).padStart(8, '0')}:`;
}

type Listener = () => void;

class BatchRunner {
  private state: BatchState = { photos: [], measureAll: false, running: false, stopping: false, sessionMs: 0, sessionDone: 0, resumed: 0 };
  private listeners = new Set<Listener>();
  private rows = new Map<string, BatchRow>();
  private key = '';

  get = () => this.state;

  subscribe = (fn: Listener) => {
    this.listeners.add(fn);
    return () => { this.listeners.delete(fn); };
  };

  private set(patch: Partial<BatchState>) {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn();
  }

  private patchPhoto(path: string, patch: Partial<BatchPhoto>) {
    this.set({ photos: this.state.photos.map((p) => (p.path === path ? { ...p, ...patch } : p)) });
  }

  /** Rows of the finished photos, in folder order. */
  results(): BatchRow[] {
    return this.state.photos.flatMap((p) => (p.status === 'done' && this.rows.has(p.path) ? [this.rows.get(p.path)!] : []));
  }

  /** A folder picked with <input webkitdirectory>: its photos, sorted by path. */
  async open(files: File[]) {
    if (this.state.running) return;
    const photos = files
      .filter((f) => PHOTO.test(f.name))
      .map((file) => ({ path: file.webkitRelativePath || file.name, file, status: 'pending' as PhotoStatus }))
      .sort((a, b) => a.path.localeCompare(b.path));
    const folder = files[0]?.webkitRelativePath.split('/')[0] || 'folder';
    const refused = !photos.length
      ? 'No photos (tif, jpg, png, bmp) in this folder'
      : photos.length > BATCH_LIMIT
        ? `${photos.length} photos: the limit is ${BATCH_LIMIT}. Split the folder and run each part.`
        : undefined;
    this.rows = new Map();
    this.set({ folder, photos: refused ? [] : photos, refused, current: undefined, sessionMs: 0, sessionDone: 0, resumed: 0 });
    if (!refused) await this.restore();
  }

  async setMeasureAll(measureAll: boolean) {
    if (this.state.running || measureAll === this.state.measureAll) return;
    this.set({ measureAll, photos: this.state.photos.map((p) => ({ ...p, status: 'pending', ms: undefined, error: undefined, grade: undefined })) });
    await this.restore();
  }

  /** Mark the photos this folder and these settings already finished, from IndexedDB. */
  private async restore() {
    const info = await engine.ready;
    this.key = batchKey(this.state.photos, this.state.measureAll, `${info.manifest.version}`);
    try {
      this.rows = await loadRows(this.key);
    } catch {
      this.rows = new Map();  // private windows may refuse IndexedDB: the run still works, without resume
    }
    const photos = this.state.photos.map((p) => {
      const r = this.rows.get(p.path);
      return r ? { ...p, status: 'done' as PhotoStatus, ms: r.ms, grade: r.quality.prediction } : p;
    });
    this.set({ photos, resumed: photos.filter((p) => p.status === 'done').length });
  }

  stop() {
    if (this.state.running) this.set({ stopping: true });
  }

  /** Forget this folder's stored results, so the next start does every photo again. */
  async clear() {
    if (this.state.running) return;
    try {
      await tx('readwrite', (s) => s.delete(IDBKeyRange.bound(this.key, `${this.key}￿`)));
    } catch { /* nothing stored */ }
    this.rows = new Map();
    this.set({ photos: this.state.photos.map((p) => ({ ...p, status: 'pending', ms: undefined, error: undefined, grade: undefined })), resumed: 0, sessionMs: 0, sessionDone: 0 });
  }

  async start() {
    if (this.state.running || !this.state.photos.length) return;
    this.set({ running: true, stopping: false });
    try {
      for (const photo of this.state.photos) {
        if (this.state.stopping) break;
        const p = this.state.photos.find((x) => x.path === photo.path)!;
        if (p.status === 'done') continue;
        this.patchPhoto(p.path, { status: 'running', error: undefined });
        const t = performance.now();
        try {
          const row = await this.one(p);
          if (!row) { this.patchPhoto(p.path, { status: 'pending' }); break; }  // stopped part-way
          const ms = performance.now() - t;
          row.ms = ms;
          this.rows.set(p.path, row);
          try { await tx('readwrite', (s) => s.put(row, this.key + p.path)); } catch { /* kept in memory */ }
          this.patchPhoto(p.path, { status: 'done', ms, grade: row.quality.prediction });
          this.set({ sessionMs: this.state.sessionMs + ms, sessionDone: this.state.sessionDone + 1 });
        } catch (error) {
          this.patchPhoto(p.path, { status: 'error', ms: performance.now() - t, error: error instanceof Error ? error.message : String(error) });
        }
      }
    } finally {
      this.set({ running: false, stopping: false, current: undefined });
    }
  }

  /** One photo through M0–M3; undefined when a stop was asked for part-way. */
  private async one(photo: BatchPhoto): Promise<BatchRow | undefined> {
    const stage = (s: string) => this.set({ current: { path: photo.path, stage: s } });
    const held: string[] = [];
    try {
      stage('Decoding');
      const bytes = await photo.file.arrayBuffer();
      const fundus = await engine.call<ImageHandle>({ op: 'decode', bytes, name: photo.file.name }, undefined, [bytes]);
      held.push(fundus.imageId);
      fundus.preview.close();
      stage('M0 · preprocess');
      const m0 = await engine.call<M0Output>({ op: 'm0', imageId: fundus.imageId });
      held.push(m0.imageId);
      m0.preview.close();
      void engine.call({ op: 'release', ids: [held.shift()!] });  // the original is not needed any more
      if (this.state.stopping) return undefined;
      stage('M1 · image quality');
      const quality = await engine.call<QualityOutput>({ op: 'quality', imageId: m0.imageId });
      if (this.state.stopping) return undefined;
      stage('M2 · vessels');
      const vessel = await engine.call<VesselOutput>({ op: 'vessel', imageId: m0.imageId });
      if (this.state.stopping) return undefined;
      stage('M2 · artery / vein');
      const av = await engine.call<LabelOutput>({ op: 'artery_vein', imageId: m0.imageId });
      if (this.state.stopping) return undefined;
      stage('M2 · optic disc');
      const disc = await engine.call<LabelOutput>({ op: 'disc_cup', imageId: m0.imageId });
      if (this.state.stopping) return undefined;
      stage('M3 · 72 features');
      const m3 = await engine.call<M3Output>({ op: 'm3', vessel: vessel.mask, arteryVein: av.map.labels, discCup: disc.map.labels, radius: m0.geometry.radius });
      if (!quality.result.gradable && !this.state.measureAll) {
        // as the canvas M3 step: AutoMorph only measures vessels on gradable photos
        for (const k of Object.keys(m3.features)) if (!/^(Disc|Cup|CDR)_/.test(k)) m3.features[k] = null;
      }
      return { path: photo.path, quality: quality.result, centredOn: m3.centredOn, features: m3.features, ms: 0, backend: quality.timing.backend };
    } finally {
      if (held.length) void engine.call({ op: 'release', ids: held });
    }
  }
}

export const batch = new BatchRunner();
