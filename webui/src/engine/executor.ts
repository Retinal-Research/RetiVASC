// Runs a canvas graph in dependency order. A node is skipped (and reuses its
// last result) when its type, parameters and upstream results are unchanged,
// so after a tweak only the affected part of the graph runs again.

import { engine } from '../runtime/engine';
import { SPECS } from './specs';
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
}

interface NodeRecord {
  key: string;
  outputs: Record<string, NodeValue>;
  inputs: Record<string, NodeValue | undefined>;
  ms: number;
}

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

function imageIds(record: NodeRecord | undefined): string[] {
  if (!record) return [];
  return Object.values(record.outputs).flatMap((v) => (v.kind === 'fundus' || v.kind === 'm0' ? [v.imageId] : []));
}

export class Executor {
  private records = new Map<string, NodeRecord>();
  private version = 0;
  running = false;

  outputs(id: string): Record<string, NodeValue> | undefined {
    return this.records.get(id)?.outputs;
  }

  inputs(id: string): Record<string, NodeValue | undefined> | undefined {
    return this.records.get(id)?.inputs;
  }

  async run(nodes: GraphNode[], edges: GraphEdge[], onState: (id: string, state: NodeState) => void): Promise<RunSummary> {
    if (this.running) throw new Error('a run is already in progress');
    this.running = true;
    const started = performance.now();
    let errors = 0;
    let ran = 0;
    let blocked = 0;
    try {
      // forget nodes that were deleted from the canvas
      const live = new Set(nodes.map((n) => n.id));
      const stale: string[] = [];
      for (const [id, record] of this.records) {
        if (!live.has(id)) { stale.push(...imageIds(record)); this.records.delete(id); }
      }
      if (stale.length) void engine.call({ op: 'release', ids: stale });

      const finished = new Map<string, string>();
      for (const node of order(nodes, edges)) {
        const spec = SPECS[node.type];
        if (!spec) { onState(node.id, { status: 'error', message: `unknown node type ${node.type}`, version: this.version }); errors++; continue; }
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
        if (waiting) { blocked++; onState(node.id, { status: 'blocked', message: waiting, version: this.version }); continue; }

        const key = `${node.type}|${stable(node.params)}|${parts.join('|')}`;
        const previous = this.records.get(node.id);
        if (previous && previous.key === key) {
          finished.set(node.id, key);
          onState(node.id, { status: 'done', ms: previous.ms, cached: true, version: this.version });
          continue;
        }
        onState(node.id, { status: 'running', message: 'Starting', version: this.version });
        ran++;
        try {
          const t = performance.now();
          const outputs = await spec.run({
            inputs,
            params: node.params,
            progress: (p) => onState(node.id, { status: 'running', message: p.message, fraction: p.fraction, version: this.version }),
          });
          const ms = performance.now() - t;
          const replaced = imageIds(previous);
          this.records.set(node.id, { key, outputs, inputs, ms });
          if (replaced.length) void engine.call({ op: 'release', ids: replaced });
          finished.set(node.id, key);
          onState(node.id, { status: 'done', ms, version: ++this.version });
        } catch (error) {
          errors++;
          onState(node.id, { status: 'error', message: error instanceof Error ? error.message : String(error), version: this.version });
        }
      }
    } finally {
      this.running = false;
    }
    return { ms: performance.now() - started, errors, ran, blocked };
  }
}
