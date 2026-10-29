import {
  addEdge,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type IsValidConnection,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent } from 'react';
import { Executor, type NodeState, type RunSummary } from './engine/executor';
import { loadSamples, type Sample } from './engine/sources';
import { acceptsType, modelsUsed, SPECS, type PortSpec } from './engine/specs';
import type { PortType } from './engine/values';
import { WORKFLOWS, type WorkflowFile } from './engine/workflows';
import { engine } from './runtime/engine';
import type { EngineInfo, GpuChoice, ModelKey, ModelNeed, ModelStatus, Progress } from './runtime/protocol';
import { StudioContext, type Studio, type StudioNodeData, type ViewSettings } from './ui/context';
import { FlowNode, type StudioNode } from './ui/FlowNode';
import { Home } from './ui/Home';
import { Inspector } from './ui/Inspector';
import { DRAG_TYPE, Sidebar } from './ui/Sidebar';
import { TopBar, type PreloadState } from './ui/TopBar';
import { mb } from './ui/format';
import { useAppearance } from './ui/appearance';
import { Login, useAccount } from './ui/Login';
import { PerfBar } from './ui/PerfBar';
import { useFolderRun } from './ui/FolderPanel';
import { api } from './server/api';
import { exposeTestHooks } from './testHooks';
import { llm } from './llm/llm';
import { onlineStatus, stopWriting } from './engine/report';
import type { LlmManifest, LlmStatus } from './llm/protocol';
import { Grip } from './ui/Grip';
import { isNum, useRemembered } from './remember';

const nodeTypes = { studio: FlowNode };

/** Edge colour per port type; the values live in styles.css so they follow the theme. */
const portColor = (type: PortType | undefined) => (type ? `var(--port-${type})` : 'var(--faint)');

const IDLE: NodeState = { status: 'idle', version: 0 };

/** The inspector's width; it can be dragged between these, leaving the canvas some room. */
const INSPECTOR_W = 392;
const inspectorMax = () => window.innerWidth - 248 - 360;

/** The pipeline the canvas opens with: ours when it is there. */
const FIRST = WORKFLOWS.find((w) => w.id === 'retivasc') ?? WORKFLOWS[0];
const PIPELINES = WORKFLOWS.filter((w) => w.section === 'pipeline');

/** The home page, or the canvas (#studio, so a reload stays on the canvas). */
type View = 'home' | 'studio';
const viewOf = (): View => (window.location.hash === '#studio' ? 'studio' : 'home');

function port(nodes: StudioNode[], nodeId: string, handle: string | null | undefined, side: 'source' | 'target'): PortSpec | undefined {
  const node = nodes.find((n) => n.id === nodeId);
  const spec = node && SPECS[node.data.type];
  const ports = side === 'source' ? spec?.outputs : spec?.inputs;
  return ports?.find((p) => p.id === handle);
}

const portType = (nodes: StudioNode[], nodeId: string, handle: string | null | undefined, side: 'source' | 'target'): PortType | undefined => {
  const type = port(nodes, nodeId, handle, side)?.type;
  return type === 'any' ? undefined : type;  // only inputs take any kind
};

function styledEdge(nodes: StudioNode[], e: { source: string; sourceHandle: string; target: string; targetHandle: string }): Edge {
  const type = portType(nodes, e.source, e.sourceHandle, 'source');
  return {
    id: `${e.source}.${e.sourceHandle}->${e.target}.${e.targetHandle}`,
    ...e,
    style: { stroke: portColor(type), strokeWidth: 2 },
  };
}

function toNodes(wf: WorkflowFile): StudioNode[] {
  return wf.nodes.map((n) => ({
    id: n.id,
    type: 'studio',
    position: n.position,
    data: { type: n.type, params: { ...(SPECS[n.type]?.defaults ?? {}), ...(n.params ?? {}) }, state: IDLE },
  }));
}

function downstream(edges: Edge[], start: string): Set<string> {
  const seen = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const id = queue.shift()!;
    for (const e of edges) if (e.source === id && !seen.has(e.target)) { seen.add(e.target); queue.push(e.target); }
  }
  return seen;
}

export function App() {
  const executor = useRef(new Executor()).current;
  const [appearance, setAppearance, theme] = useAppearance();
  const [account, setAccount] = useAccount();
  const flow = useReactFlow<StudioNode, Edge>();
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selected, setSelected] = useState<string | undefined>();
  const [engineInfo, setEngineInfo] = useState<EngineInfo>();
  const [engineError, setEngineError] = useState<string>();
  const [gpu, setGpu] = useState<GpuChoice>(engine.gpu);
  const [modelStatus, setModelStatus] = useState<ModelStatus>();
  const [preloading, setPreloading] = useState<Progress>();
  // the language model of the AI report step: null when this site has none
  const [llmManifest, setLlmManifest] = useState<LlmManifest | null>();
  const [llmStatus, setLlmStatus] = useState<LlmStatus>();
  // the lab's API, through the RetiVASC server, writes reports when this page can reach it
  const [online, setOnline] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [lastRun, setLastRun] = useState<RunSummary>();
  const [name, setName] = useState(FIRST.name);
  const [view, setViewMode] = useState<View>(viewOf);
  const [inspectorW, setInspectorW] = useRemembered('inspector-width', INSPECTOR_W, isNum);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  const counter = useRef(1);
  const fitPending = useRef(false);
  const nodesReady = useNodesInitialized();
  const folderRun = useFolderRun(executor);

  // closing the page stops a folder run; finished photos are kept, but say so
  useEffect(() => {
    if (!folderRun?.running) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [folderRun?.running]);

  // frame a freshly loaded workflow once React Flow has measured its nodes
  useEffect(() => {
    if (!nodesReady || !fitPending.current) return;
    fitPending.current = false;
    void flow.fitView({ padding: 0.06, duration: 250, maxZoom: 1 });
  }, [nodesReady, flow]);

  const patchNode = useCallback(
    (id: string, patch: Partial<StudioNodeData>) =>
      setNodes((ns) => ns.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n))),
    [setNodes],
  );

  const refreshModels = useCallback(
    () => engine.call<ModelStatus>({ op: 'status' }).then(setModelStatus, () => setModelStatus(undefined)),
    [],
  );

  const busy = useRef(false);
  const again = useRef(false);  // a run asked for while another runs (e.g. a workflow opened mid-run)
  const run = useCallback(async (): Promise<RunSummary | undefined> => {
    if (busy.current) {
      again.current = true;
      return;
    }
    busy.current = true;
    setRunning(true);
    try {
      await engine.ready;
      const graphNodes = nodesRef.current.map((n) => ({ id: n.id, type: n.data.type, params: n.data.params }));
      const graphEdges = edgesRef.current.map((e) => ({ source: e.source, sourceHandle: e.sourceHandle ?? '', target: e.target, targetHandle: e.targetHandle ?? '' }));
      const summary = await executor.run(graphNodes, graphEdges, (id, state) => patchNode(id, { state }));
      setLastRun(summary);
      void refreshModels();
      return summary;
    } catch (error) {
      setLastRun({ ms: 0, errors: 1, ran: 0, blocked: 0, photos: 0 });
      window.alert(error instanceof Error ? error.message : String(error));
    } finally {
      busy.current = false;
      setRunning(false);
      setStopping(false);
      if (again.current) {
        again.current = false;
        setTimeout(() => void runRef.current?.(), 0);
      }
    }
  }, [executor, patchNode, refreshModels]);
  const runRef = useRef(run);
  runRef.current = run;

  const load = useCallback(
    (wf: WorkflowFile, autorun = false) => {
      const ns = toNodes(wf);
      setNodes(ns);
      setEdges(wf.edges.map((e) => styledEdge(ns, e)));
      setName(wf.name);
      setSelected(undefined);
      setLastRun(undefined);
      nodesRef.current = ns;
      edgesRef.current = wf.edges.map((e) => styledEdge(ns, e));
      fitPending.current = true;
      if (autorun) void run();
    },
    [flow, run, setEdges, setNodes],
  );

  useEffect(() => {
    engine.ready.then((info) => { setEngineInfo(info); void refreshModels(); }, (e: Error) => setEngineError(e.message));
    loadSamples().then(setSamples, () => setSamples([]));
    void llm.manifest.then((m) => setLlmManifest(m ?? null));
    void onlineStatus().then((st) => setOnline(st.enabled));
    load(FIRST);
  }, []);

  // nothing runs by itself: the canvas waits for Run, so opening the page downloads no models
  const go = useCallback((v: View) => {
    if (v === 'studio') fitPending.current = true;  // the canvas mounts again and has to be framed
    setViewMode(v);
    const hash = v === 'studio' ? '#studio' : '';
    if (window.location.hash !== hash) window.history.pushState(null, '', hash || window.location.pathname + window.location.search);
  }, []);
  useEffect(() => {
    const back = () => { const v = viewOf(); if (v === 'studio') fitPending.current = true; setViewMode(v); };
    window.addEventListener('popstate', back);
    return () => window.removeEventListener('popstate', back);
  }, []);

  const setParams = useCallback(
    (id: string, params: Record<string, unknown>) => {
      const affected = downstream(edgesRef.current, id);
      setNodes((ns) =>
        ns.map((n) => {
          if (!affected.has(n.id)) return n;
          const data = { ...n.data, state: { ...n.data.state, status: 'idle' as const, message: 'Changed · press Run' } };
          if (n.id === id) data.params = { ...n.data.params, ...params };
          return { ...n, data };
        }),
      );
    },
    [setNodes],
  );

  const setView = useCallback((id: string, view: ViewSettings) => patchNode(id, { view }), [patchNode]);

  const setType = useCallback(
    (id: string, type: string) => {
      const spec = SPECS[type];
      if (!spec) return;
      setNodes((ns) => ns.map((n) => (n.id === id && n.data.type !== type
        ? { ...n, data: { ...n.data, type, params: { ...(spec.defaults ?? {}) }, state: IDLE } }
        : n)));
      setParams(id, {});  // everything after it has to run again
    },
    [setNodes, setParams],
  );

  const isValidConnection: IsValidConnection<Edge> = useCallback((c) => {
    if (c.source === c.target) return false;
    const ns = nodesRef.current;
    const a = portType(ns, c.source, c.sourceHandle, 'source');
    const b = port(ns, c.target, c.targetHandle, 'target');
    if (!a || !b || !acceptsType(b, a)) return false;
    return !downstream(edgesRef.current, c.target).has(c.source); // no loops
  }, []);

  const onConnect = useCallback(
    (c: Connection) => {
      const multi = !!port(nodesRef.current, c.target, c.targetHandle, 'target')?.multi;
      setEdges((es) => {
        const kept = multi ? es : es.filter((e) => !(e.target === c.target && e.targetHandle === c.targetHandle));
        const edge = styledEdge(nodesRef.current, { source: c.source, sourceHandle: c.sourceHandle ?? '', target: c.target, targetHandle: c.targetHandle ?? '' });
        return addEdge(edge, kept);
      });
      setParams(c.target, {});
    },
    [setEdges, setParams],
  );

  const addNode = useCallback(
    (type: string, position?: { x: number; y: number }) => {
      const spec = SPECS[type];
      if (!spec) return;
      let id = '';
      do id = `${type.split('.').pop()}-${counter.current++}`;
      while (nodesRef.current.some((n) => n.id === id));
      let pos = position ?? flow.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
      if (!position) {
        // a clicked (not dropped) step goes to the first free spot below the centre
        const overlaps = (p: { x: number; y: number }) =>
          nodesRef.current.some((n) => {
            const w = n.measured?.width ?? 280;
            const h = n.measured?.height ?? 320;
            return p.x < n.position.x + w + 24 && p.x + 296 > n.position.x && p.y < n.position.y + h + 24 && p.y + 220 > n.position.y;
          });
        for (let i = 0; i < 80 && overlaps(pos); i++) pos = { x: pos.x, y: pos.y + 40 };
        fitPending.current = true;
      }
      setNodes((ns) => [...ns, { id, type: 'studio', position: pos, data: { type, params: { ...(spec.defaults ?? {}) }, state: IDLE } }]);
      setSelected(id);
    },
    [flow, setNodes],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      const type = e.dataTransfer.getData(DRAG_TYPE);
      if (!type) return;
      e.preventDefault();
      addNode(type, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY }));
    },
    [addNode, flow],
  );

  const save = useCallback(() => {
    const file: WorkflowFile = {
      format: 'retivasc-studio/1',
      name,
      nodes: nodesRef.current.map((n) => {
        const params = { ...n.data.params };
        if (typeof params.source === 'string' && !params.source.startsWith('sample:')) delete params.source; // files are not saved
        delete params.folderId;
        return { id: n.id, type: n.data.type, position: n.position, params };
      }),
      edges: edgesRef.current.map((e) => ({ source: e.source, sourceHandle: e.sourceHandle ?? '', target: e.target, targetHandle: e.targetHandle ?? '' })),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(file, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[^\w-]+/g, '_').replace(/^_+|_+$/g, '') || 'workflow'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [name]);

  const open = useCallback(
    async (f: File) => {
      try {
        const wf = JSON.parse(await f.text()) as WorkflowFile;
        if (wf.format !== 'retivasc-studio/1' || !Array.isArray(wf.nodes) || !Array.isArray(wf.edges)) throw new Error('not a RetiVASC Studio workflow file');
        const unknown = wf.nodes.find((n) => !SPECS[n.type]);
        if (unknown) throw new Error(`unknown step type “${unknown.type}”`);
        load({ ...wf, name: wf.name || f.name }, false);
      } catch (error) {
        window.alert(`Could not open ${f.name}: ${error instanceof Error ? error.message : String(error)}`);
      }
    },
    [load],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); void run(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [run]);

  useEffect(() => exposeTestHooks({ executor, run, load, setParams, nodes: () => nodesRef.current, engine: () => engineInfo }), [executor, run, load, setParams, engineInfo]);

  // another device: a new engine, so every result (and the images behind it) is gone
  const switchGpu = useCallback((next: GpuChoice) => {
    if (busy.current) return;
    setGpu(next);
    setEngineInfo(undefined);
    setEngineError(undefined);
    setModelStatus(undefined);
    setLastRun(undefined);
    executor.reset();
    setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, state: { status: 'idle', message: 'Engine restarted · press Run', version: 0 } } })));
    llm.restart();
    setLlmStatus(undefined);
    engine.restart(next).then((info) => { setEngineInfo(info); void refreshModels(); }, (e: Error) => setEngineError(e.message));
  }, [executor, setNodes, refreshModels]);

  // the models the canvas uses (that this site has), each with the members its settings run
  const needs: ModelNeed[] = useMemo(() => {
    const most = new Map<ModelKey, number>();
    for (const n of nodes) for (const m of modelsUsed(n.data.type, n.data.params)) most.set(m.key, Math.max(most.get(m.key) ?? 0, m.parts));
    return [...most].filter(([key]) => engineInfo?.manifest.models[key]).map(([key, parts]) => ({ key, parts }));
  }, [nodes, engineInfo]);

  // the model on this computer, where a report step would use it
  const local = (params: Record<string, unknown>) => params.engine === 'local' || (params.engine !== 'online' && !online);
  const usesLlm = !!llmManifest && nodes.some((n) => SPECS[n.data.type]?.llm && local(n.data.params));
  useEffect(() => {
    if (usesLlm) llm.status().then(setLlmStatus, () => setLlmStatus(undefined));
  }, [usesLlm, lastRun]);

  const preload = useCallback(async () => {
    setPreloading({ stage: 'download', message: 'Starting', fraction: 0 });
    try {
      if (needs.length) setModelStatus(await engine.call<ModelStatus>({ op: 'preload', models: needs }, setPreloading));
      if (usesLlm) setLlmStatus(await llm.load((p) => setPreloading({ ...p, message: `Language model · ${p.message}` })));
    } catch (error) {
      window.alert(`Could not load the models: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setPreloading(undefined);
    }
  }, [needs, usesLlm]);

  const preloadState = useMemo((): PreloadState => {
    const models = engineInfo?.manifest.models;
    if (!models || (!needs.length && !usesLlm)) return { state: 'none', label: '', title: '' };
    if (preloading) {
      const pct = preloading.fraction === undefined ? '' : ` ${Math.round(preloading.fraction * 100)}%`;
      return { state: 'loading', label: `Loading${pct}`, title: preloading.message };
    }
    let files = 0, ready = 0, fetch = 0;
    for (const { key, parts } of needs) {
      const info = models[key]!;
      const n = Math.min(parts, info.parts.length);
      const st = modelStatus?.[key];
      files += n;
      ready += Math.min(n, st?.loaded ?? 0);
      fetch += info.parts.slice(Math.min(n, st?.cached ?? 0), n).reduce((sum, p) => sum + p.bytes, 0);
    }
    if (usesLlm && llmManifest) {
      files += 1;
      if (llmStatus?.loaded) ready += 1;
      if (!llmStatus?.cached) fetch += llmManifest.files.model.bytes + llmManifest.files.tokenizer.bytes;
    }
    if ((modelStatus || !needs.length) && (llmStatus || !usesLlm) && ready >= files) {
      return { state: 'ready', label: 'Models ready', title: `All ${files} model files this workflow uses are loaded; Run starts at once.` };
    }
    return {
      state: 'needed',
      label: 'Preload',
      title: `Load the ${files} model files this workflow uses now${fetch ? `, downloading ${mb(fetch)}` : ' (already downloaded)'}, so that Run starts at once. `
        + 'Downloaded models stay in this browser, also for use without the internet.',
    };
  }, [engineInfo, needs, modelStatus, preloading, usesLlm, llmManifest, llmStatus]);

  const studio: Studio = useMemo(
    () => ({ executor, workflow: name, engine: engineInfo, samples, running, setParams, setType, setView }),
    [executor, name, engineInfo, samples, running, setParams, setType, setView],
  );
  const selectedNode = nodes.find((n) => n.id === selected);
  const browserNode = nodes.find((n) => n.data.state.status === 'running' && SPECS[n.data.type]?.group !== 'server');

  // a pipeline needs every model its steps use; until the engine has started, assume they are there
  const missing = (wf: WorkflowFile) =>
    (!!engineInfo && wf.nodes.some((n) => { const m = SPECS[n.type]?.model; return !!m && !engineInfo.manifest.models[m]; }))
    || (llmManifest === null && !online && wf.nodes.some((n) => SPECS[n.type]?.llm));

  if (view === 'home') {
    return (
      <Home pipelines={PIPELINES} missing={missing} samples={samples} appearance={appearance} theme={theme} onAppearance={setAppearance}
            onStart={(wf) => { if (!running) load(wf); go('studio'); }} />
    );
  }

  return (
    <StudioContext.Provider value={studio}>
      <div className="app" style={{ '--inspector-w': `${inspectorW}px` } as CSSProperties}>
        <TopBar name={name} engine={engineInfo} engineError={engineError} gpu={gpu} onGpu={switchGpu} preload={preloadState} onPreload={() => void preload()} running={running} stopping={stopping} lastRun={lastRun} onRun={() => void run()} onStop={() => { setStopping(true); executor.stop(); stopWriting(); }} onSave={save} onOpen={(f) => void open(f)} appearance={appearance} theme={theme} onAppearance={setAppearance} account={account} onSignOut={() => void api.logout().finally(() => setAccount({ mode: 'signin' }))} onHome={() => go('home')} />
        <Sidebar current={name} onLoad={(wf) => load(wf)} onAdd={(t) => addNode(t)} server={account.mode === 'ready'} />
        <main className="canvas" onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }} onDrop={onDrop}>
          <ReactFlow<StudioNode, Edge>
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onNodeClick={(_, n) => setSelected(n.id)}
            onPaneClick={() => setSelected(undefined)}
            colorMode={theme}
            minZoom={0.2}
            maxZoom={2}
            deleteKeyCode={['Backspace', 'Delete']}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="var(--dots)" />
            <Controls showInteractive={false} />
            <MiniMap pannable zoomable nodeColor={(n) => `var(--g-${SPECS[(n.data as StudioNodeData).type]?.group ?? 'input'})`} maskColor="var(--mask)" />
          </ReactFlow>
        </main>
        <div className="inspector-cell">
          <Grip axis="x" sign={-1} value={inspectorW} min={300} max={inspectorMax} onChange={setInspectorW}
                onReset={() => setInspectorW(INSPECTOR_W)} label="Drag to change the width of the inspector" />
          <Inspector node={selectedNode} />
        </div>
        {(account.mode === 'ready' || account.mode === 'offline') && (
          <PerfBar engine={engineInfo} browserStep={browserNode && SPECS[browserNode.data.type]?.title} lastRun={lastRun} server={account.mode === 'ready'} />
        )}
        {account.mode === 'signin' && <Login onDone={(a) => setAccount({ mode: 'ready', account: a })} />}
      </div>
    </StudioContext.Provider>
  );
}
