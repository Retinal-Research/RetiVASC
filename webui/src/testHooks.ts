// window.__studio: a small handle for the automated browser checks in
// tests/browser.mjs. It only reads results and drives the same actions as the UI.

import type { Executor, RunSummary } from './engine/executor';
import type { WorkflowFile } from './engine/workflows';
import { WORKFLOWS } from './engine/workflows';
import type { EngineInfo } from './runtime/protocol';
import type { StudioNode } from './ui/FlowNode';
import { llm } from './llm/llm';

interface Hooks {
  executor: Executor;
  run: () => Promise<RunSummary | undefined>;
  load: (wf: WorkflowFile, autorun?: boolean) => void;
  setParams: (id: string, params: Record<string, unknown>) => void;
  nodes: () => StudioNode[];
  engine: () => EngineInfo | undefined;
}

function base64(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function exposeTestHooks(h: Hooks): void {
  (window as unknown as { __studio: unknown }).__studio = {
    workflows: () => WORKFLOWS.map((w) => w.name),
    llm,
    // a library entry by position or by id ('retivasc', 'automorph', 'vascx', 'custom')
    load: (which: number | string) => h.load(typeof which === 'string' ? WORKFLOWS.find((w) => w.id === which) ?? WORKFLOWS[0] : WORKFLOWS[which], false),
    // a library entry as data, to change and load (e.g. with one more connection)
    workflow: (id: string) => structuredClone(WORKFLOWS.find((w) => w.id === id)),
    loadFile: (wf: (typeof WORKFLOWS)[number]) => h.load(wf, false),
    setSource: (id: string, source: string) => h.setParams(id, { source }),
    setParams: (id: string, params: Record<string, unknown>) => h.setParams(id, params),
    run: () => h.run(),
    engine: () => h.engine(),
    states: () => Object.fromEntries(h.nodes().map((n) => [n.id, n.data.state])),
    result: (id: string) => {
      const out = h.executor.outputs(id);
      if (!out) return null;
      const v = Object.values(out)[0];
      if (!v) return null;
      switch (v.kind) {
        case 'fundus':
          return { kind: v.kind, width: v.width, height: v.height };
        case 'm0':
          return { kind: v.kind, side: v.side, geometry: v.geometry, ms: v.ms };
        case 'quality':
          return { kind: v.kind, result: v.result, timing: v.timing };
        case 'vessel':
          return { kind: v.kind, size: v.size, labels: base64(v.mask), timing: v.timing };
        case 'av':
        case 'disc':
          return { kind: v.kind, size: v.map.width, labels: base64(v.map.labels), timing: v.timing };
        case 'fovea':
          return { kind: v.kind, x: v.x, y: v.y, size: v.size, timing: v.timing };
        case 'features':
          return { kind: v.kind, features: v.result.features, centredOn: v.result.centredOn, discValid: v.result.discValid, ms: v.result.ms, gradable: v.gradable };
        case 'bifurcations':
          return { kind: v.kind, size: v.size, root: v.root, ms: v.ms, gradable: v.gradable, warning: v.warning, failed: v.failed,
                   artery: v.artery.points.map((p) => p.angle), vein: v.vein.points.map((p) => p.angle) };
        case 'table':
          return { kind: v.kind, name: v.name, tables: v.tables };
        case 'report':
          return { kind: v.kind, text: v.text, writer: v.writer, model: v.model, ms: v.ms, unchecked: v.unchecked, judged: v.judged, scope: v.facts.scope, facts: v.facts.groups, qa: v.qa, result: v.result, table: v.table, tableError: v.tableError };
      }
    },
  };
}
