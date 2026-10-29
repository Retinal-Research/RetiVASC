// Node types on the canvas. Each AutoMorph node is a thin call into the engine
// worker; the canvas only wires them together.

import { engine } from '../runtime/engine';
import type { EnhanceOutput, FoveaOutput, ImageHandle, LabelOutput, M0Output, M3Output, ModelKey, ModelNeed, Progress, QualityOutput, RetiFlowOutput, RRWNetOutput, VascxPrepOutput, VesselOutput } from '../runtime/protocol';
import type { BatchItem } from './executor';
import { readSource } from './sources';
import { fovShare } from './summary';
import type { M0Value, NodeValue, PortType } from './values';
import { serverAnalyze, studyPhotos, studySheet } from './study';
import type { Cohort } from './facts';
import { aiReport } from './report';
import { resultsTable } from './table';

export type NodeGroup = 'input' | 'm0' | 'retivasc' | 'm1' | 'm2' | 'm3' | 'vascx' | 'view' | 'ai' | 'server' | 'study';

export interface PortSpec {
  id: string;
  label: string;
  /** 'any': an input that takes every kind of result */
  type: PortType | 'any';
  /** other value kinds this input also takes */
  accepts?: PortType[];
  optional?: boolean;
  /** an input that takes several connections (RunContext.many) */
  multi?: boolean;
}

/** A vessel map covering less or more of the field of view than this is flagged (RetiFlow's fallback). */
const VESSEL_SHARE: [number, number] = [0.02, 0.3];

export const acceptsType = (port: PortSpec, type: PortType) => port.type === 'any' || port.type === type || !!port.accepts?.includes(type);

export interface RunContext {
  inputs: Record<string, NodeValue | undefined>;
  params: Record<string, unknown>;
  progress: (p: Progress) => void;
  /** in a folder run: the photo this pass is for */
  batch?: BatchItem;
  /** every value on an input that takes several connections, by port, in canvas order */
  many?: Record<string, Array<{ from: string; type: string; value: NodeValue }>>;
  /** after a folder run, for a step that reports on all of its photos (NodeSpec.cohort) */
  cohort?: Cohort;
  /** the types of the steps this one's outputs are connected to, so a step can give them what they need */
  downstream?: string[];
}

export interface NodeSpec {
  type: string;
  title: string;
  subtitle: string;
  group: NodeGroup;
  inputs: PortSpec[];
  outputs: PortSpec[];
  model?: ModelKey;
  /** writes with the language model (src/llm) */
  llm?: boolean;
  /** downstream of a Photo folder: runs once, after the last photo, on the results of all of them */
  cohort?: boolean;
  /** shown in the library but not ready: it never runs */
  soon?: boolean;
  defaults?: Record<string, unknown>;
  run: (ctx: RunContext) => Promise<Record<string, NodeValue>>;
}

export const GROUP_LABEL: Record<NodeGroup, string> = {
  input: 'Input',
  m0: 'AutoMorph M0',
  retivasc: 'RetiVASC',
  m1: 'AutoMorph M1',
  m2: 'AutoMorph M2',
  m3: 'AutoMorph M3',
  vascx: 'VascX',
  view: 'View',
  ai: 'AI',
  server: 'Server · full model',
  study: 'Study sheet',
};

export const DEFAULT_SOURCE = 'sample:samples/cc0_left_3744.jpg';

function m0Input(ctx: RunContext): M0Value {
  const v = ctx.inputs.image;
  if (!v || v.kind !== 'm0') throw new Error('expects a preprocessed (M0) photo');
  return v;
}

const specs: NodeSpec[] = [
  {
    type: 'input.fundus',
    title: 'Fundus photo',
    subtitle: 'A sample, or a file from this computer',
    group: 'input',
    inputs: [],
    outputs: [{ id: 'image', label: 'photo', type: 'fundus' }],
    defaults: { source: DEFAULT_SOURCE },
    async run(ctx) {
      const { name, bytes } = await readSource(String(ctx.params.source ?? DEFAULT_SOURCE));
      const h = await engine.call<ImageHandle>({ op: 'decode', bytes, name }, ctx.progress, [bytes]);
      return { image: { kind: 'fundus', name, source: String(ctx.params.source ?? DEFAULT_SOURCE), ...h } };
    },
  },
  {
    // the steps connected after it run once per photo (src/engine/executor.ts)
    type: 'input.folder',
    title: 'Photo folder',
    subtitle: 'Up to 1000 photos; the steps after it run on each one',
    group: 'input',
    inputs: [],
    outputs: [{ id: 'image', label: 'photo', type: 'fundus' }],
    async run(ctx) {
      if (!ctx.batch) throw new Error('choose a photo folder');
      const { file, path } = ctx.batch.photo;
      const bytes = await file.arrayBuffer();
      const h = await engine.call<ImageHandle>({ op: 'decode', bytes, name: file.name }, ctx.progress, [bytes]);
      return { image: { kind: 'fundus', name: path, source: `folder:${path}`, ...h } };
    },
  },
  {
    type: 'automorph.m0',
    title: 'M0 · Preprocess',
    subtitle: 'Find the field of view, crop to a square',
    group: 'm0',
    inputs: [{ id: 'image', label: 'photo', type: 'fundus' }],
    outputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    async run(ctx) {
      const src = ctx.inputs.image;
      if (!src || src.kind !== 'fundus') throw new Error('expects a fundus photo');
      const out = await engine.call<M0Output>({ op: 'm0', imageId: src.imageId }, ctx.progress);
      return {
        image: { kind: 'm0', imageId: out.imageId, side: out.width, geometry: out.geometry, preview: out.preview, ms: out.ms, source: src.name, made: 'm0' },
      };
    },
  },
  {
    type: 'retivasc.enhance',
    title: 'Enhance · TPOT',
    subtitle: "The lab's TPOT enhancement, restored to full resolution",
    group: 'retivasc',
    model: 'tpot',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'image', label: 'enhanced', type: 'm0' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<EnhanceOutput>({ op: 'enhance', imageId: base.imageId, quick: !!ctx.params.quick }, ctx.progress);
      // the same square and field of view as the input, so every later step works as without enhancement
      return { image: { ...base, kind: 'm0', imageId: out.imageId, preview: out.preview, ms: out.ms, timing: out.timing, made: 'enhanced', before: base.preview, hires: undefined } };
    },
  },
  {
    type: 'retivasc.rrwnet',
    title: 'Arteries, veins and vessels · RRWNet',
    subtitle: 'Multi-task segmentation, refined five times',
    group: 'retivasc',
    model: 'rrwnet',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [
      { id: 'av', label: 'artery/vein', type: 'av' },
      { id: 'vessel', label: 'vessels', type: 'vessel' },
      { id: 'maps', label: 'probability maps', type: 'avprob' },
    ],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<RRWNetOutput>({ op: 'rrwnet', imageId: base.imageId }, ctx.progress);
      return {
        av: { kind: 'av', map: { width: 720, height: 720, labels: out.labels }, timing: out.timing, base },
        vessel: { kind: 'vessel', size: 912, mask: out.vessel, timing: out.timing, base },
        maps: { kind: 'avprob', size: out.size, data: out.av3, timing: out.timing, base },
      };
    },
  },
  {
    type: 'retivasc.retiflow',
    title: 'Branching angles · RetiFlow',
    subtitle: 'Bifurcations of the arteries and the veins',
    group: 'retivasc',
    inputs: [
      { id: 'maps', label: 'probability maps', type: 'avprob' },
      { id: 'quality', label: 'grade', type: 'quality', optional: true },
    ],
    outputs: [{ id: 'angles', label: 'branching', type: 'bifurcations' }],
    // Every photo is measured, gradable or not (Hao, 10-29), with fallbacks: a failure gives no branching
    // points and says why instead of stopping the run, and an implausible vessel map is flagged.
    async run(ctx) {
      const maps = ctx.inputs.maps;
      if (maps?.kind !== 'avprob') throw new Error('connect the probability maps of RRWNet');
      const quality = ctx.inputs.quality;
      const gradable = quality?.kind === 'quality' ? quality.result.gradable : undefined;
      let vessels = 0;
      for (let i = 2; i < maps.data.length; i += 3) if (maps.data[i] > 127) vessels++;
      const share = fovShare(vessels, maps.size, maps.base);
      const warning = share < VESSEL_SHARE[0] || share > VESSEL_SHARE[1]
        ? `the vessels cover ${(share * 100).toFixed(1)}% of the field of view, outside the usual ${VESSEL_SHARE[0] * 100}–${VESSEL_SHARE[1] * 100}%: check the photo and the segmentation`
        : undefined;
      const empty = { points: [], centerline: new Uint8Array(maps.size * maps.size) };
      try {
        const out = await engine.call<RetiFlowOutput>({ op: 'retiflow', size: maps.size, av3: maps.data.slice() }, ctx.progress);
        return { angles: { kind: 'bifurcations', size: maps.size, root: out.root, artery: out.artery, vein: out.vein, ms: out.ms, base: maps.base, gradable, warning } };
      } catch (e) {
        const failed = e instanceof Error ? e.message : String(e);
        return { angles: { kind: 'bifurcations', size: maps.size, root: [maps.size / 2, maps.size / 2], artery: empty, vein: empty, ms: 0, base: maps.base, gradable, warning, failed } };
      }
    },
  },
  {
    type: 'retivasc.gpt',
    title: 'RetinalGPT',
    subtitle: 'Ask about the photograph',
    group: 'retivasc',
    soon: true,
    inputs: [{ id: 'image', label: 'photo', type: 'fundus', accepts: ['m0'] }],
    outputs: [],
    async run() {
      return {};
    },
  },
  {
    type: 'retivasc.retify',
    title: 'Retify',
    subtitle: 'A RetiVASC module in development',
    group: 'retivasc',
    soon: true,
    inputs: [{ id: 'image', label: 'photo', type: 'fundus', accepts: ['m0'] }],
    outputs: [],
    async run() {
      return {};
    },
  },
  {
    type: 'automorph.m1',
    title: 'M1 · Image quality',
    subtitle: 'Good / Usable / Reject',
    group: 'm1',
    model: 'quality',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'quality', label: 'grade', type: 'quality' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<QualityOutput>({ op: 'quality', imageId: base.imageId, quick: !!ctx.params.quick }, ctx.progress);
      return { quality: { kind: 'quality', result: out.result, timing: out.timing, base } };
    },
  },
  {
    type: 'automorph.vessel',
    title: 'M2 · Vessels',
    subtitle: 'Binary vessel segmentation',
    group: 'm2',
    model: 'vessel',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'vessel', label: 'vessels', type: 'vessel' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<VesselOutput>({ op: 'vessel', imageId: base.imageId, quick: !!ctx.params.quick }, ctx.progress);
      return { vessel: { kind: 'vessel', size: out.size, mask: out.mask, timing: out.timing, base } };
    },
  },
  {
    type: 'automorph.av',
    title: 'M2 · Artery / vein',
    subtitle: 'Arteries, veins and crossings',
    group: 'm2',
    model: 'artery_vein',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'av', label: 'artery/vein', type: 'av' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<LabelOutput>({ op: 'artery_vein', imageId: base.imageId, quick: !!ctx.params.quick }, ctx.progress);
      return { av: { kind: 'av', map: out.map, timing: out.timing, base } };
    },
  },
  {
    type: 'automorph.disc',
    title: 'M2 · Optic disc & cup',
    subtitle: 'Disc and cup segmentation',
    group: 'm2',
    model: 'disc_cup',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'disc', label: 'disc/cup', type: 'disc' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<LabelOutput>({ op: 'disc_cup', imageId: base.imageId }, ctx.progress);
      return { disc: { kind: 'disc', map: out.map, timing: out.timing, base } };
    },
  },
  {
    type: 'automorph.m3',
    title: 'M3 · Vascular features',
    subtitle: '72 measurements: disc, whole image, Zone B and C',
    group: 'm3',
    inputs: [
      { id: 'image', label: 'cropped', type: 'm0' },
      { id: 'vessel', label: 'vessels', type: 'vessel' },
      { id: 'av', label: 'artery/vein', type: 'av' },
      { id: 'disc', label: 'disc/cup', type: 'disc' },
      { id: 'quality', label: 'grade', type: 'quality', optional: true },
    ],
    outputs: [{ id: 'features', label: 'features', type: 'features' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const { vessel, av, disc, quality } = ctx.inputs;
      if (vessel?.kind !== 'vessel' || av?.kind !== 'av' || disc?.kind !== 'disc') throw new Error('connect the vessel, artery/vein and disc/cup steps');
      if (vessel.size !== 912 || av.map.width !== 720 || disc.map.width !== 512) throw new Error('M3 needs the M2 maps of the full-size photo');
      const result = await engine.call<M3Output>(
        { op: 'm3', vessel: vessel.mask.slice(), arteryVein: av.map.labels.slice(), discCup: disc.map.labels.slice(), radius: base.geometry.radius },
        ctx.progress,
      );
      const gradable = quality?.kind === 'quality' ? quality.result.gradable : undefined;
      const forced = !!ctx.params.measureAll;
      if (gradable === false && !forced) {
        // AutoMorph only measures vessels on gradable photos; its tables keep just the disc values
        for (const k of Object.keys(result.features)) if (!/^(Disc|Cup|CDR)_/.test(k)) result.features[k] = null;
      }
      return { features: { kind: 'features', result, base, gradable, forced } };
    },
  },
  {
    type: 'vascx.prep',
    title: 'VascX · Preprocess',
    subtitle: 'Find the field of view, crop it to 1024 × 1024',
    group: 'vascx',
    inputs: [{ id: 'image', label: 'photo', type: 'fundus' }],
    outputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    async run(ctx) {
      const src = ctx.inputs.image;
      if (!src || src.kind !== 'fundus') throw new Error('expects a fundus photo');
      const out = await engine.call<VascxPrepOutput>({ op: 'vx_prep', imageId: src.imageId }, ctx.progress);
      return {
        image: { kind: 'm0', imageId: out.imageId, side: out.width, geometry: out.geometry, preview: out.preview, ms: out.ms, source: src.name, fovRadius: out.width / 2, made: 'vascx' },
      };
    },
  },
  {
    type: 'vascx.vessel',
    title: 'VascX · Vessels',
    subtitle: 'Binary vessel segmentation, 4 flips averaged',
    group: 'vascx',
    model: 'vx_vessel',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'vessel', label: 'vessels', type: 'vessel' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<VesselOutput>({ op: 'vx_vessel', imageId: base.imageId }, ctx.progress);
      return { vessel: { kind: 'vessel', size: out.size, mask: out.mask, timing: out.timing, base } };
    },
  },
  {
    type: 'vascx.av',
    title: 'VascX · Artery / vein',
    subtitle: '9 windows of 512, 4 flips averaged',
    group: 'vascx',
    model: 'vx_av',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'av', label: 'artery/vein', type: 'av' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<LabelOutput>({ op: 'vx_av', imageId: base.imageId }, ctx.progress);
      return { av: { kind: 'av', map: out.map, timing: out.timing, base } };
    },
  },
  {
    type: 'vascx.disc',
    title: 'VascX · Optic disc',
    subtitle: 'Optic disc (no cup), the largest region',
    group: 'vascx',
    model: 'vx_disc',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'disc', label: 'disc', type: 'disc' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<LabelOutput>({ op: 'vx_disc', imageId: base.imageId }, ctx.progress);
      return { disc: { kind: 'disc', map: out.map, timing: out.timing, base } };
    },
  },
  {
    type: 'vascx.fovea',
    title: 'VascX · Fovea',
    subtitle: 'Centre of the macula, from a heatmap',
    group: 'vascx',
    model: 'vx_fovea',
    inputs: [{ id: 'image', label: 'cropped', type: 'm0' }],
    outputs: [{ id: 'fovea', label: 'fovea', type: 'fovea' }],
    async run(ctx) {
      const base = m0Input(ctx);
      const out = await engine.call<FoveaOutput>({ op: 'vx_fovea', imageId: base.imageId }, ctx.progress);
      // in the base image's pixels
      const k = base.side / out.size;
      return { fovea: { kind: 'fovea', x: out.x * k, y: out.y * k, size: base.side, timing: out.timing, base } };
    },
  },
  {
    type: 'ai.report',
    title: 'AI report',
    subtitle: 'The lab’s language model writes up the results',
    group: 'ai',
    llm: true,
    cohort: true,
    inputs: [{ id: 'results', label: 'results of any steps', type: 'any', multi: true }],
    outputs: [{ id: 'report', label: 'report', type: 'report' }],
    defaults: { prompt: '', effort: 'medium' },
    run: aiReport,
  },
  {
    type: 'server.analyze',
    title: 'AutoMorph v2 · full analysis',
    subtitle: 'All ensemble members, 72 features, on the server',
    group: 'server',
    inputs: [{ id: 'image', label: 'photo', type: 'fundus' }],
    outputs: [{ id: 'analysis', label: 'results', type: 'analysis' }],
    run: serverAnalyze,
  },
  {
    type: 'study.photos',
    title: 'Study photos',
    subtitle: 'A photo folder: upload it, or use one on the server',
    group: 'study',
    inputs: [],
    outputs: [{ id: 'run', label: 'photos', type: 'run' }],
    defaults: { source: 'upload' },
    run: studyPhotos,
  },
  {
    type: 'study.sheet',
    title: 'Excel · results sheet',
    subtitle: 'AutoMorph v2 on the server, fill the sheet, audit it',
    group: 'study',
    inputs: [{ id: 'run', label: 'photos', type: 'run' }],
    outputs: [{ id: 'sheet', label: 'sheet', type: 'sheet' }],
    defaults: { outName: 'automorph_results_v1.xlsx', strict: false },
    run: studySheet,
  },
  {
    type: 'view.table',
    title: 'Results table',
    subtitle: 'The numbers of the steps connected to it; download as CSV',
    group: 'view',
    // after a folder: one row per photo; an AI report connected to it also writes a table for it
    cohort: true,
    inputs: [{ id: 'results', label: 'results of any steps', type: 'any', multi: true }],
    outputs: [{ id: 'table', label: 'table', type: 'table' }],
    run: resultsTable,
  },
  {
    type: 'view.overlay',
    title: 'Overlay view',
    subtitle: 'Results layered on the photo',
    group: 'view',
    inputs: [
      { id: 'image', label: 'cropped', type: 'm0' },
      { id: 'quality', label: 'grade', type: 'quality', optional: true },
      { id: 'vessel', label: 'vessels', type: 'vessel', optional: true },
      { id: 'av', label: 'artery/vein', type: 'av', optional: true },
      { id: 'disc', label: 'disc/cup', type: 'disc', optional: true },
      { id: 'fovea', label: 'fovea', type: 'fovea', optional: true },
    ],
    outputs: [],
    async run() {
      return {};
    },
  },
];

export const SPECS: Record<string, NodeSpec> = Object.fromEntries(specs.map((s) => [s.type, s]));
export const SPEC_LIST = specs;

/** The engine models a step runs with these settings: "quick" uses the first ensemble member only,
 *  and the quick enhancement skips the fusion head. */
export function modelsUsed(type: string, params: Record<string, unknown>): ModelNeed[] {
  const model = SPECS[type]?.model;
  if (!model) return [];
  const quick = !!params.quick && type !== 'automorph.disc';  // the disc step always runs every member
  if (model === 'tpot') return quick ? [{ key: 'tpot', parts: 1 }] : [{ key: 'tpot', parts: 1 }, { key: 'tpot_fusion', parts: 1 }];
  return [{ key: model, parts: quick ? 1 : Number.POSITIVE_INFINITY }];
}
