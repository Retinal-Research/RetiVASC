// A folder of photos for the Photo folder node: the files the user picked (they stay in this page),
// and the results of each finished photo in IndexedDB, so a run that was stopped, or a page that
// was closed, resumes where it left off when the same folder is chosen again.

import type { QualityResult } from '../automorph/types';
import type { M3Result } from '../automorph/m3/m3';

export const FOLDER_LIMIT = 1000;
const PHOTO = /\.(tiff?|jpe?g|png|bmp)$/i;

export interface FolderPhoto { path: string; file: File }

export interface PickedFolder {
  name: string;
  photos: FolderPhoto[];
  /** why the folder cannot run: no photos, or more than FOLDER_LIMIT */
  refused?: string;
  /** paths, sizes and dates: the same folder picked again gives the same signature */
  signature: string;
}

const folders = new Map<string, PickedFolder>();
let nextFolder = 1;

/** Files from <input webkitdirectory>: the photos among them, sorted by path. */
export function addFolder(files: File[]): string {
  return addPhotos(files[0]?.webkitRelativePath.split('/')[0] || 'folder', files.map((file) => ({ path: file.webkitRelativePath || file.name, file })));
}

let samples: Promise<string> | undefined;

/** The sample folder that comes with the site (samples.json entries with a folder), as a picked folder. */
export function sampleFolder(): Promise<string> {
  samples ??= (async () => {
    const list = (await (await fetch(`${import.meta.env.BASE_URL}samples/samples.json`)).json()) as Array<{ file: string; folder?: string }>;
    const items = list.filter((s) => s.folder);
    const name = items[0]?.folder ?? 'samples';
    const photos = await Promise.all(items.map(async (s) => {
      const bytes = await (await fetch(`${import.meta.env.BASE_URL}${s.file}`)).blob();
      const base = s.file.split('/').pop()!;
      return { path: `${name}/${base}`, file: new File([bytes], base, { type: bytes.type, lastModified: 0 }) };
    }));
    return addPhotos(name, photos);
  })();
  return samples;
}

function addPhotos(name: string, items: FolderPhoto[]): string {
  const photos = items.filter((p) => PHOTO.test(p.file.name)).sort((a, b) => a.path.localeCompare(b.path));
  const refused = !photos.length
    ? 'no photos (tif, jpg, png, bmp) in this folder'
    : photos.length > FOLDER_LIMIT
      ? `${photos.length} photos: the limit is ${FOLDER_LIMIT}. Split the folder and run each part.`
      : undefined;
  const signature = hash(photos.map((p) => `${p.path}:${p.file.size}:${p.file.lastModified}`).join('|'));
  const id = `folder:${nextFolder++}`;
  folders.set(id, { name, photos: refused ? [] : photos, refused, signature });
  return id;
}

export const getFolder = (id: unknown) => (typeof id === 'string' ? folders.get(id) : undefined);

export function hash(text: string): string {
  let h = 0x811c9dc5;  // FNV-1a
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(16).padStart(8, '0');
}

// ---------------------------------------------------------------- per-photo results

/** What a run keeps of one photo: small values only (the images are released after each photo). */
export interface PhotoRecord {
  path: string;
  status: 'done' | 'error';
  ms: number;
  error?: string;
  /** by node id: each M1 step's grade */
  quality: Record<string, QualityResult>;
  /** by node id: each M3 step's values */
  features: Record<string, { features: Record<string, number | null>; centredOn: M3Result['centredOn']; gradable?: boolean }>;
  /** by node id: simple measures of each segmentation and the fovea (engine/summary.ts) */
  measures?: Record<string, Record<string, number>>;
}

const DB_NAME = 'retivasc-folders';
const STORE = 'photos';

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

const range = (run: string) => IDBKeyRange.bound(`${run}:`, `${run}:￿`);

/** The finished photos of a run (folder + workflow), by path; empty where IndexedDB is refused. */
export async function loadRecords(run: string): Promise<Map<string, PhotoRecord>> {
  try {
    const rows = await tx('readonly', (s) => s.getAll(range(run)) as IDBRequest<PhotoRecord[]>);
    return new Map(rows.filter((r) => r.status === 'done').map((r) => [r.path, r]));
  } catch {
    return new Map();  // private windows may refuse IndexedDB: the run still works, without resume
  }
}

export async function saveRecord(run: string, record: PhotoRecord): Promise<void> {
  try { await tx('readwrite', (s) => s.put(record, `${run}:${record.path}`)); } catch { /* kept in memory only */ }
}

export async function clearRecords(run: string): Promise<void> {
  try { await tx('readwrite', (s) => s.delete(range(run))); } catch { /* nothing stored */ }
}
