// Node types on the canvas. Each AutoMorph node is a thin call into the engine
// worker; the canvas only wires them together.

import { engine } from '../runtime/engine';
import type { ImageHandle, LabelOutput, M0Output, M3Output, ModelKey, Progress, QualityOutput, VesselOutput } from '../runtime/protocol';
import { readSource } from './sources';
import type { M0Value, NodeValue, PortType } from './values';
import { serverAnalyze, studyPhotos, studySheet } from './study';

export type NodeGroup = 'input' | 'm0' | 'm1' | 'm2' | 'm3' | 'view' | 'server' | 'study';

export interface PortSpec {
  id: string;
  label: string;
  type: PortType;
  /** other value kinds this input also takes */
  accepts?: PortType[];
  optional?: boolean;
}

export const acceptsType = (port: PortSpec, type: PortType) => port.type === type || !!port.accepts?.includes(type);

export interface RunContext {
  inputs: Record<string, NodeValue | undefined>;
  params: Record<string, unknown>;
  progress: (p: Progress) => void;
}

export interface NodeSpec {
  type: string;
  title: string;
  subtitle: string;
  group: NodeGroup;
  inputs: PortSpec[];
  outputs: PortSpec[];
  model?: ModelKey;
  defaults?: Record<string, unknown>;
  run: (ctx: RunContext) => Promise<Record<string, NodeValue>>;
}

export const GROUP_LABEL: Record<NodeGroup, string> = {
  input: 'Input',
  m0: 'AutoMorph M0',
  m1: 'AutoMorph M1',
  m2: 'AutoMorph M2',
  m3: 'AutoMorph M3',
  view: 'View',
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
        image: { kind: 'm0', imageId: out.imageId, side: out.width, geometry: out.geometry, preview: out.preview, ms: out.ms, source: src.name },
      };
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
    ],
    outputs: [],
    async run() {
      return {};
    },
  },
];

export const SPECS: Record<string, NodeSpec> = Object.fromEntries(specs.map((s) => [s.type, s]));
export const SPEC_LIST = specs;
