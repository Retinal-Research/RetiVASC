import { createContext, useContext } from 'react';
import type { Executor, NodeState } from '../engine/executor';
import type { Sample } from '../engine/sources';
import type { EngineInfo } from '../runtime/protocol';
import type { LayerKind } from './render';

export interface ViewSettings {
  layers: Record<LayerKind, boolean>;
  opacity: number;
}

export const DEFAULT_VIEW: ViewSettings = { layers: { vessel: true, av: true, disc: true, fovea: true }, opacity: 0.85 };

export interface StudioNodeData extends Record<string, unknown> {
  type: string;
  params: Record<string, unknown>;
  state: NodeState;
  view?: ViewSettings;
}

export interface Studio {
  executor: Executor;
  /** the workflow on the canvas, e.g. for naming downloads */
  workflow: string;
  engine?: EngineInfo;
  samples: Sample[];
  running: boolean;
  setParams: (nodeId: string, params: Record<string, unknown>) => void;
  /** swap a node for another type with the same ports (one photo / a folder), keeping its connections */
  setType: (nodeId: string, type: string) => void;
  setView: (nodeId: string, view: ViewSettings) => void;
}

export const StudioContext = createContext<Studio | null>(null);

export function useStudio(): Studio {
  const s = useContext(StudioContext);
  if (!s) throw new Error('StudioContext missing');
  return s;
}
