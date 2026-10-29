// Runs a canvas graph in dependency order. A node is skipped (and reuses its
// last result) when its type, parameters and upstream results are unchanged,
// so after a tweak only the affected part of the graph runs again.
//
// With a Photo folder node the steps downstream of it run once per photo, the whole downstream
// graph for one photo before the next, so memory holds one photo at a time (each photo's images
// replace the previous one's). Each finished photo's grades and features are kept, also in
// IndexedDB, and a stopped run resumes with the photos that are not done yet.

import { engine } from '../runtime/engine';
import { getFolder, hash, loadRecords, saveRecord, type FolderPhoto, type PhotoRecord } from './folder';
import { SPECS } from './specs';
import { measures } from './summary';
import type { NodeValue } from './values';

export interface GraphNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export type NodeStatus = 'idle' | 'blocked' | 'running' | 'done' | 'error';

export interface NodeState {
  status: NodeStatus;
  message?: string;
  fraction?: number;
  ms?: number;
  cached?: boolean;
  version: number;
}

export interface RunSummary {
  ms: number;
  errors: number;
  /** steps that actually ran (the rest were unchanged) */
  ran: number;
  /** steps left waiting for a connection */
  blocked: number;
  /** photos this run finished: those of a folder run, else 1 */
  photos: number;
}

/** The photo a folder run is on, for the Photo folder node's run(). */
export interface BatchItem {
  photo: FolderPhoto;
  index: number;
  total: number;
}

/** A folder run, for the Photo folder node and the inspector. */
export interface FolderRun {
  nodeId: string;
  /** IndexedDB key of this folder with this workflow */
  key: string;
  photos: FolderPhoto[];
  records: Map<string, PhotoRecord>;
  running: boolean;
  stopping: boolean;
  current?: { path: string; index: number };
  /** this session's processing time, for the speed and the time left */
  sessionMs: number;
  sessionDone: number;
  /** finished before this run started (from IndexedDB) */
  resumed: number;
}

interface NodeRecord {
  key: string;
  outputs: Record<string, NodeValue>;
  inputs: Record<string, NodeValue | undefined>;
  ms: number;
}

type StepResult = { status: 'done' | 'cached' | 'blocked' } | { status: 'error'; message: string };

function stable(value: unknown): string {
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
    return `{${entries.map(([k, v]) => `${k}:${stable(v)}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function order(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[] {
  const indegree = new Map(nodes.map((n) => [n.id, 0]));
  for (const e of edges) if (indegree.has(e.target) && indegree.has(e.source)) indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const queue = nodes.filter((n) => indegree.get(n.id) === 0).map((n) => n.id);
  const out: GraphNode[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    out.push(byId.get(id)!);
    for (const e of edges) {
      if (e.source !== id || !indegree.has(e.target)) continue;
      const d = (indegree.get(e.target) ?? 0) - 1;
      indegree.set(e.target, d);
      if (d === 0) queue.push(e.target);
    }
  }
  if (out.length !== nodes.length) throw new Error('the workflow has a loop; remove one connection');
  return out;
}

function downstreamOf(edges: GraphEdge[], start: string): Set<string> {
  const seen = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const id = queue.shift()!;
    for (const e of edges) if (e.source === id && !seen.has(e.target)) { seen.add(e.target); queue.push(e.target); }
  }
  return seen;
}

function imageIds(record: NodeRecord | undefined): string[] {
  if (!record) return [];
  return Object.values(record.outputs).flatMap((v) => (v.kind === 'fundus' || v.kind === 'm0' ? [v.imageId] : []));
}

export class Executor {
  private records = new Map<string, NodeRecord>();
  private version = 0;
  private stopRequested = false;
  private folderRun: FolderRun | undefined;
  private listeners = new Set<() => void>();
  running = false;

  outputs(id: string): Record<string, NodeValue> | undefined {
    return this.records.get(id)?.outputs;
  }

  inputs(id: string): Record<string, NodeValue | undefined> | undefined {
    return this.records.get(id)?.inputs;
  }

  // -- the folder run, for useSyncExternalStore
  folder = (): FolderRun | undefined => this.folderRun;

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => { this.listeners.delete(fn); };
  };

  private setFolder(patch: Partial<FolderRun>) {
    if (!this.folderRun) return;
    this.folderRun = { ...this.folderRun, ...patch };
    for (const fn of this.listeners) fn();
  }

  /** Stop after the step that is running now; a folder run keeps the photos it finished. */
  stop() {
    if (!this.running) return;
    this.stopRequested = true;
    this.setFolder({ stopping: true });
  }

  async run(nodes: GraphNode[], edges: GraphEdge[], onState: (id: string, state: NodeState) => void): Promise<RunSummary> {
    if (this.running) throw new Error('a run is already in progress');
    this.running = true;
    this.stopRequested = false;
    const started = performance.now();
    const tally = { errors: 0, ran: 0, blocked: 0 };
    let photos = 1;
    try {
      // forget nodes that were deleted from the canvas
      const live = new Set(nodes.map((n) => n.id));
      const stale: string[] = [];
      for (const [id, record] of this.records) {
        if (!live.has(id)) { stale.push(...imageIds(record)); this.records.delete(id); }
      }
      if (stale.length) void engine.call({ op: 'release', ids: stale });

      const sorted = order(nodes, edges);
      const folderNode = sorted.find((n) => n.type === 'input.folder');
      const perPhoto = folderNode ? downstreamOf(edges, folderNode.id) : new Set<string>();
      const finished = new Map<string, string>();
      for (const node of sorted) {
        if (perPhoto.has(node.id)) continue;
        if (this.stopRequested) break;
        this.count(tally, await this.step(node, edges, finished, onState));
      }
      if (folderNode && !this.stopRequested) {
        const steps = sorted.filter((n) => perPhoto.has(n.id));
        photos = await this.runFolder(folderNode, steps, edges, finished, onState, tally);
      }
    } finally {
      this.running = false;
      this.stopRequested = false;
    }
    return { ms: performance.now() - started, ...tally, photos };
  }

  private count(tally: { errors: number; ran: number; blocked: number }, r: StepResult) {
    if (r.status === 'error') tally.errors++;
    else if (r.status === 'blocked') tally.blocked++;
    else if (r.status === 'done') tally.ran++;
  }

  /** One node: wait for its inputs, reuse an unchanged result, or run it. */
  private async step(node: GraphNode, edges: GraphEdge[], finished: Map<string, string>, onState: (id: string, state: NodeState) => void,
                     batch?: BatchItem): Promise<StepResult> {
    const spec = SPECS[node.type];
    if (!spec) {
      const message = `unknown node type ${node.type}`;
      onState(node.id, { status: 'error', message, version: this.version });
      return { status: 'error', message };
    }
    const inputs: Record<string, NodeValue | undefined> = {};
    const parts: string[] = [];
    let waiting: string | undefined;
    for (const port of spec.inputs) {
      const edge = edges.find((e) => e.target === node.id && e.targetHandle === port.id);
      if (!edge) {
        if (!port.optional) waiting ??= `Connect “${port.label}”`;
        continue;
      }
      const upstream = finished.get(edge.source);
      if (upstream === undefined) { waiting ??= 'An earlier step did not finish'; continue; }
      inputs[port.id] = this.records.get(edge.source)?.outputs[edge.sourceHandle];
      parts.push(`${port.id}<${edge.source}.${edge.sourceHandle}#${upstream}`);
    }
    if (waiting) { onState(node.id, { status: 'blocked', message: waiting, version: this.version }); return { status: 'blocked' }; }

    // in a folder run every photo is new: the key names the photo, so nothing is reused across photos
    const key = `${node.type}|${stable(node.params)}|${parts.join('|')}${batch ? `|photo:${batch.photo.path}` : ''}`;
    const previous = this.records.get(node.id);
    if (previous && previous.key === key) {
      finished.set(node.id, key);
      onState(node.id, { status: 'done', ms: previous.ms, cached: true, version: this.version });
      return { status: 'cached' };
    }
    const prefix = batch ? `Photo ${batch.index + 1} of ${batch.total} · ` : '';
    onState(node.id, { status: 'running', message: `${prefix}Starting`, version: this.version });
    try {
      const t = performance.now();
      const outputs = await spec.run({
        inputs,
        params: node.params,
        batch,
        progress: (p) => onState(node.id, { status: 'running', message: prefix + p.message, fraction: p.fraction, version: this.version }),
      });
      const ms = performance.now() - t;
      const replaced = imageIds(previous);
      this.records.set(node.id, { key, outputs, inputs, ms });
      if (replaced.length) void engine.call({ op: 'release', ids: replaced });
      finished.set(node.id, key);
      onState(node.id, { status: 'done', ms, version: ++this.version });
      return { status: 'done' };
    } catch (error) {
      // an earlier result of this node (or of a node with the same id in another workflow) is not this run's
      if (previous) {
        this.records.delete(node.id);
        const stale = imageIds(previous);
        if (stale.length) void engine.call({ op: 'release', ids: stale });
      }
      const message = error instanceof Error ? error.message : String(error);
      onState(node.id, { status: 'error', message, version: ++this.version });
      return { status: 'error', message };
    }
  }

  /** The folder node and everything downstream of it, once per photo that is not done yet. */
  private async runFolder(folderNode: GraphNode, steps: GraphNode[], edges: GraphEdge[], finished: Map<string, string>,
                          onState: (id: string, state: NodeState) => void, tally: { errors: number; ran: number; blocked: number }): Promise<number> {
    const folder = getFolder(folderNode.params.folderId);
    if (!folder || folder.refused) {
      const message = folder?.refused ?? 'Choose a photo folder';
      for (const n of steps) onState(n.id, { status: 'blocked', message: n.id === folderNode.id ? message : 'An earlier step did not finish', version: this.version });
      tally.blocked += steps.length;
      return 0;
    }
    // the same folder with the same steps and settings resumes; any change starts afresh
    // (the folder node's own params only point at the picked files, which differ every time the page opens)
    const workflow = steps.map((n) => `${n.id}|${n.type}|${n.id === folderNode.id ? '' : stable(n.params)}`).join(';')
      + edges.filter((e) => steps.some((n) => n.id === e.target)).map((e) => `${e.source}.${e.sourceHandle}>${e.target}.${e.targetHandle}`).sort().join(';');
    const key = `${folder.signature}-${hash(workflow)}`;
    const records = await loadRecords(key);
    this.folderRun = { nodeId: folderNode.id, key, photos: folder.photos, records, running: true, stopping: false, sessionMs: 0, sessionDone: 0, resumed: records.size };
    this.setFolder({});
    const total = folder.photos.length;
    try {
      for (const [index, photo] of folder.photos.entries()) {
        if (this.stopRequested) break;
        if (this.folderRun.records.get(photo.path)?.status === 'done') continue;
        this.setFolder({ current: { path: photo.path, index } });
        const t = performance.now();
        const batch: BatchItem = { photo, index, total };
        const photoFinished = new Map(finished);
        let failed: string | undefined;
        let stopped = false;
        for (const node of steps) {
          if (this.stopRequested) { stopped = true; break; }
          const r = await this.step(node, edges, photoFinished, onState, batch);
          this.count(tally, r);
          if (r.status === 'error') { failed = `${SPECS[node.type]?.title ?? node.type}: ${r.message}`; break; }
        }
        if (stopped) break;
        const record = this.collect(photo.path, steps, performance.now() - t, failed);
        const next = new Map(this.folderRun.records).set(photo.path, record);
        if (!failed) void saveRecord(key, record);
        this.setFolder({ records: next, sessionMs: this.folderRun.sessionMs + record.ms, sessionDone: this.folderRun.sessionDone + 1 });
      }
    } finally {
      this.setFolder({ running: false, stopping: false, current: undefined });
    }
    return this.folderRun.sessionDone;
  }

  /** The small results of one photo: every M1 grade and every M3 feature set downstream of the folder. */
  private collect(path: string, steps: GraphNode[], ms: number, error?: string): PhotoRecord {
    const record: PhotoRecord = { path, status: error ? 'error' : 'done', ms, error, quality: {}, features: {}, measures: {} };
    if (error) return record;
    for (const n of steps) {
      for (const v of Object.values(this.records.get(n.id)?.outputs ?? {})) {
        if (v.kind === 'quality') record.quality[n.id] = v.result;
        if (v.kind === 'features') record.features[n.id] = { features: v.result.features, centredOn: v.result.centredOn, gradable: v.gradable };
        const m = measures(v);
        if (m) record.measures![n.id] = m;
      }
    }
    return record;
  }
}
