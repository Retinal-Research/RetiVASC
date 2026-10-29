// The pipeline library: ready-made workflows. Each is plain data, the same shape that
// "Save" writes, so a saved file and a template load the same way. The pipelines are what the
// home page offers (RetiVASC, AutoMorph, VascX, Custom); the single steps and the server
// workflows are listed after them in the sidebar.

import { DEFAULT_SOURCE } from './specs';

export interface WorkflowFile {
  format: 'retivasc-studio/1';
  name: string;
  description?: string;
  /** library entries only: a stable id (the home page links to it) and where the sidebar lists it */
  id?: string;
  section?: 'pipeline' | 'single' | 'server';
  nodes: Array<{ id: string; type: string; position: { x: number; y: number }; params?: Record<string, unknown> }>;
  edges: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>;
}

const photo = (y: number) => ({ id: 'photo', type: 'input.fundus', position: { x: 0, y }, params: { source: DEFAULT_SOURCE } });
const m0 = (y: number) => ({ id: 'm0', type: 'automorph.m0', position: { x: 340, y } });
const link = (source: string, sourceHandle: string, target: string, targetHandle: string) => ({ source, sourceHandle, target, targetHandle });

const studySheet: WorkflowFile = {
  format: 'retivasc-studio/1',
  section: 'server',
  name: 'Study · AutoMorph → Excel sheet',
  description: 'Server: run AutoMorph v2 on a folder of photos (or reuse an earlier run), then fill and audit a results sheet.',
  nodes: [
    { id: 'run', type: 'study.photos', position: { x: 0, y: 0 } },
    { id: 'sheet', type: 'study.sheet', position: { x: 440, y: 0 } },
  ],
  edges: [link('run', 'run', 'sheet', 'run')],
};

const serverFull: WorkflowFile = {
  format: 'retivasc-studio/1',
  section: 'server',
  name: 'Full analysis on the server',
  description: 'Server: the complete AutoMorph v2 (every ensemble member) with all 72 vascular features.',
  nodes: [photo(0), { id: 'full', type: 'server.analyze', position: { x: 340, y: 0 } }],
  edges: [link('photo', 'image', 'full', 'image')],
};

const automorph: WorkflowFile = {
  format: 'retivasc-studio/1',
  id: 'automorph',
  section: 'pipeline',
  name: 'AutoMorph',
  description: 'AutoMorph v2: grade the quality, segment vessels, arteries/veins and the optic disc, then measure 72 vascular features.',
  nodes: [
    photo(500),
    m0(500),
    { id: 'm1', type: 'automorph.m1', position: { x: 690, y: 0 } },
    { id: 'vessel', type: 'automorph.vessel', position: { x: 690, y: 285 } },
    { id: 'av', type: 'automorph.av', position: { x: 690, y: 665 } },
    { id: 'disc', type: 'automorph.disc', position: { x: 690, y: 1045 } },
    { id: 'view', type: 'view.overlay', position: { x: 1060, y: 460 } },
    { id: 'm3', type: 'automorph.m3', position: { x: 1060, y: -40 } },
    { id: 'table', type: 'view.table', position: { x: 1430, y: -40 } },
  ],
  edges: [
    link('photo', 'image', 'm0', 'image'),
    link('m0', 'image', 'm1', 'image'),
    link('m0', 'image', 'vessel', 'image'),
    link('m0', 'image', 'av', 'image'),
    link('m0', 'image', 'disc', 'image'),
    link('m0', 'image', 'view', 'image'),
    link('m1', 'quality', 'view', 'quality'),
    link('av', 'av', 'view', 'av'),
    link('disc', 'disc', 'view', 'disc'),
    link('m0', 'image', 'm3', 'image'),
    link('vessel', 'vessel', 'm3', 'vessel'),
    link('av', 'av', 'm3', 'av'),
    link('disc', 'disc', 'm3', 'disc'),
    link('m1', 'quality', 'm3', 'quality'),
    link('m1', 'quality', 'table', 'results'),
    link('m3', 'features', 'table', 'results'),
  ],
};

/** Ours: the lab's TPOT enhancement first, and every later step on the enhanced photo: the quality grade,
 *  RRWNet's arteries, veins and vessels (in place of AutoMorph's segmentation), AutoMorph's optic disc and
 *  72 vascular features, RetiFlow's branching angles on RRWNet's probability maps, and the AI report. */
const retivasc: WorkflowFile = {
  format: 'retivasc-studio/1',
  id: 'retivasc',
  section: 'pipeline',
  name: 'RetiVASC',
  description: "Our pipeline: the lab's TPOT enhancement, then on the enhanced photo the quality grade, RRWNet's arteries, veins and vessels, the optic disc, 72 vascular features, RetiFlow's branching angles, an AI report and a results table.",
  // Fallbacks, so one step's failure does not stop the rest (a port with several connections takes the first
  // that has a result; dashed on the canvas): without the enhancement, the steps use the cropped photo; if
  // RRWNet fails (a weak graphics card, say), VascX's segmentation stands in, and VascX's disc for AutoMorph's.
  // The VascX steps are on standby: they run only when needed. The AI report and the table take whatever finished.
  nodes: [
    photo(500),
    m0(500),
    { id: 'enhance', type: 'retivasc.enhance', position: { x: 690, y: 500 } },
    { id: 'm1', type: 'automorph.m1', position: { x: 1040, y: 0 } },
    { id: 'rrwnet', type: 'retivasc.rrwnet', position: { x: 1040, y: 300 } },
    { id: 'disc', type: 'automorph.disc', position: { x: 1040, y: 740 } },
    { id: 'vx_vessel', type: 'vascx.vessel', position: { x: 690, y: 1100 }, params: { standby: true } },
    { id: 'vx_av', type: 'vascx.av', position: { x: 1040, y: 1100 }, params: { standby: true } },
    { id: 'vx_disc', type: 'vascx.disc', position: { x: 1040, y: 1460 }, params: { standby: true } },
    { id: 'm3', type: 'automorph.m3', position: { x: 1410, y: -20 } },
    { id: 'retiflow', type: 'retivasc.retiflow', position: { x: 1410, y: 480 } },
    { id: 'view', type: 'view.overlay', position: { x: 1410, y: 900 } },
    { id: 'report', type: 'ai.report', position: { x: 1790, y: -20 } },
    { id: 'table', type: 'view.table', position: { x: 1790, y: 520 } },
  ],
  edges: [
    link('photo', 'image', 'm0', 'image'),
    link('m0', 'image', 'enhance', 'image'),
    // every step on the enhanced photo, or on the cropped one if the enhancement failed
    ...['m1', 'rrwnet', 'disc', 'm3', 'view', 'vx_vessel', 'vx_av', 'vx_disc'].flatMap((id) => [link('enhance', 'image', id, 'image'), link('m0', 'image', id, 'image')]),
    link('rrwnet', 'vessel', 'm3', 'vessel'),
    link('vx_vessel', 'vessel', 'm3', 'vessel'),
    link('rrwnet', 'av', 'm3', 'av'),
    link('vx_av', 'av', 'm3', 'av'),
    link('disc', 'disc', 'm3', 'disc'),
    link('vx_disc', 'disc', 'm3', 'disc'),
    link('m1', 'quality', 'm3', 'quality'),
    link('rrwnet', 'maps', 'retiflow', 'maps'),
    link('vx_av', 'av', 'retiflow', 'maps'),
    link('m1', 'quality', 'retiflow', 'quality'),
    link('m1', 'quality', 'view', 'quality'),
    link('rrwnet', 'vessel', 'view', 'vessel'),
    link('vx_vessel', 'vessel', 'view', 'vessel'),
    link('rrwnet', 'av', 'view', 'av'),
    link('vx_av', 'av', 'view', 'av'),
    link('disc', 'disc', 'view', 'disc'),
    link('vx_disc', 'disc', 'view', 'disc'),
    link('m1', 'quality', 'report', 'context'),
    link('m3', 'features', 'report', 'context'),
    link('retiflow', 'angles', 'report', 'context'),
    link('m1', 'quality', 'table', 'results'),
    link('m3', 'features', 'table', 'results'),
    link('retiflow', 'angles', 'table', 'results'),
  ],
};

/** VascX (Eyened Reading Center, Rotterdam): its own field-of-view crop, then its four models. */
const vascx: WorkflowFile = {
  format: 'retivasc-studio/1',
  id: 'vascx',
  section: 'pipeline',
  name: 'VascX',
  description: 'VascX (Eyened, Rotterdam): its own field-of-view crop, then vessels, arteries/veins, the optic disc and the fovea.',
  nodes: [
    photo(560),
    { id: 'prep', type: 'vascx.prep', position: { x: 340, y: 560 } },
    { id: 'vessel', type: 'vascx.vessel', position: { x: 690, y: 0 } },
    { id: 'av', type: 'vascx.av', position: { x: 690, y: 380 } },
    { id: 'disc', type: 'vascx.disc', position: { x: 690, y: 760 } },
    { id: 'fovea', type: 'vascx.fovea', position: { x: 690, y: 1140 } },
    { id: 'view', type: 'view.overlay', position: { x: 1060, y: 520 } },
    { id: 'table', type: 'view.table', position: { x: 1060, y: 0 } },
  ],
  edges: [
    link('photo', 'image', 'prep', 'image'),
    link('prep', 'image', 'vessel', 'image'),
    link('prep', 'image', 'av', 'image'),
    link('prep', 'image', 'disc', 'image'),
    link('prep', 'image', 'fovea', 'image'),
    link('prep', 'image', 'view', 'image'),
    link('av', 'av', 'view', 'av'),
    link('disc', 'disc', 'view', 'disc'),
    link('fovea', 'fovea', 'view', 'fovea'),
    link('vessel', 'vessel', 'table', 'results'),
    link('av', 'av', 'table', 'results'),
    link('disc', 'disc', 'table', 'results'),
    link('fovea', 'fovea', 'table', 'results'),
  ],
};

const custom: WorkflowFile = {
  format: 'retivasc-studio/1',
  id: 'custom',
  section: 'pipeline',
  name: 'Custom',
  description: 'Start from a photo and connect skills from the library yourself.',
  nodes: [photo(0)],
  edges: [],
};

export const WORKFLOWS: WorkflowFile[] = [
  retivasc,
  automorph,
  vascx,
  custom,
  {
    format: 'retivasc-studio/1',
    section: 'single',
    name: 'Image quality check',
    description: 'Is the photo good enough for AutoMorph to measure?',
    nodes: [photo(0), m0(0), { id: 'm1', type: 'automorph.m1', position: { x: 690, y: 0 } }],
    edges: [link('photo', 'image', 'm0', 'image'), link('m0', 'image', 'm1', 'image')],
  },
  {
    format: 'retivasc-studio/1',
    section: 'single',
    name: 'Vessel segmentation',
    description: 'Binary vessel map on top of the photo.',
    nodes: [
      photo(0),
      m0(0),
      { id: 'vessel', type: 'automorph.vessel', position: { x: 690, y: 0 } },
      { id: 'view', type: 'view.overlay', position: { x: 1060, y: 0 } },
    ],
    edges: [
      link('photo', 'image', 'm0', 'image'),
      link('m0', 'image', 'vessel', 'image'),
      link('m0', 'image', 'view', 'image'),
      link('vessel', 'vessel', 'view', 'vessel'),
    ],
  },
  {
    format: 'retivasc-studio/1',
    section: 'single',
    name: 'Artery / vein and optic disc',
    description: 'Arteries in red, veins in blue, the optic disc and cup outlined.',
    nodes: [
      photo(180),
      m0(180),
      { id: 'av', type: 'automorph.av', position: { x: 690, y: 0 } },
      { id: 'disc', type: 'automorph.disc', position: { x: 690, y: 370 } },
      { id: 'view', type: 'view.overlay', position: { x: 1060, y: 110 } },
    ],
    edges: [
      link('photo', 'image', 'm0', 'image'),
      link('m0', 'image', 'av', 'image'),
      link('m0', 'image', 'disc', 'image'),
      link('m0', 'image', 'view', 'image'),
      link('av', 'av', 'view', 'av'),
      link('disc', 'disc', 'view', 'disc'),
    ],
  },
  serverFull,
  studySheet,
];
