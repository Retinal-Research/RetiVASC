import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { RunSummary } from '../engine/executor';
import type { EngineInfo } from '../runtime/protocol';
import { api, type ServerStats } from '../server/api';
import { mb, ms } from './format';

const POLL_MS = 1000;
const MEMORY_MS = 5000;
const HISTORY = 60;

const gb = (bytes: number) => `${(bytes / 2 ** 30).toFixed(1)} GB`;

// A web page cannot read this computer's CPU, memory or GPU load. The closest it gets, in the same
// columns as the server: the CPU pressure level (Compute Pressure API, Chrome/Edge 125+), the memory
// of this tab (cross-origin isolated Chrome/Edge), and how much of each second the engine computed.
type Pressure = 'nominal' | 'fair' | 'serious' | 'critical';
const PRESSURE_LEVEL: Record<Pressure, number> = { nominal: 10, fair: 40, serious: 75, critical: 100 };
interface PressureRecord { state: Pressure }
interface PressureObserverLike { observe(source: 'cpu', options?: { sampleInterval?: number }): Promise<void>; disconnect(): void }
type PressureObserverCtor = new (callback: (records: PressureRecord[]) => void) => PressureObserverLike;
type MemoryPerformance = Performance & {
  measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
  memory?: { usedJSHeapSize: number };
};

/** The last minute as a filled line, 0–100 %, newest on the right. */
function Spark({ values }: { values: number[] }) {
  const w = 64, h = 16;
  const step = w / (HISTORY - 1);
  const x0 = w - (values.length - 1) * step;
  const pts = values.map((v, i) => `${(x0 + i * step).toFixed(1)},${(h - (Math.min(Math.max(v, 0), 100) / 100) * h).toFixed(1)}`);
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      {pts.length > 1 && <polygon points={`${x0.toFixed(1)},${h} ${pts.join(' ')} ${w},${h}`} className="spark-fill" />}
      {pts.length > 1 && <polyline points={pts.join(' ')} className="spark-line" />}
    </svg>
  );
}

/** One column of a row; without a value it shows a dash, so the columns of both rows still line up. */
function Cell({ label, title, values, text }: { label: string; title: string; values?: number[]; text?: ReactNode }) {
  const now = values?.at(-1) ?? 0;
  return (
    <span className={`meter ${now >= 90 ? 'hot' : now >= 60 ? 'warm' : ''}`} title={title}>
      <span className="meter-label">{label}</span>
      <Spark values={values ?? []} />
      <span className={`meter-value ${text === undefined ? 'none' : ''}`}>{text ?? '—'}</span>
    </span>
  );
}

function Host({ active, title, name, hint }: { active: boolean; title: string; name: string; hint: string }) {
  return (
    <span className={`perf-host ${active ? 'active' : ''}`} title={hint}>
      <span className="sdot" />
      <span className="perf-title">{title}</span>
      <span className="perf-device">{name}</span>
    </span>
  );
}

/** Keeps the last minute of a value, one reading a second. */
function useSeries(read: () => number | undefined): number[] {
  const [values, setValues] = useState<number[]>([]);
  const reader = useRef(read);
  reader.current = read;
  useEffect(() => {
    const id = setInterval(() => {
      const v = reader.current();
      if (v !== undefined) setValues((vs) => [...vs.slice(1 - HISTORY), v]);
    }, POLL_MS);
    return () => clearInterval(id);
  }, []);
  return values;
}

function usePressure(): Pressure | undefined {
  const [state, setState] = useState<Pressure>();
  useEffect(() => {
    const Ctor = (globalThis as { PressureObserver?: PressureObserverCtor }).PressureObserver;
    if (!Ctor) return;
    const observer = new Ctor((records) => { const r = records.at(-1); if (r) setState(r.state); });
    observer.observe('cpu', { sampleInterval: POLL_MS }).catch(() => setState(undefined));
    return () => observer.disconnect();
  }, []);
  return state;
}

/** Memory of this tab and its workers; the browser may take a few seconds to answer. */
function useTabMemory(): number | undefined {
  const [bytes, setBytes] = useState<number>();
  useEffect(() => {
    const perf = performance as MemoryPerformance;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let alive = true;
    let precise = crossOriginIsolated && !!perf.measureUserAgentSpecificMemory;
    const tick = async () => {
      let b: number | undefined;
      if (precise) {
        try { b = (await perf.measureUserAgentSpecificMemory!()).bytes; } catch { precise = false; }  // not offered in every build
      }
      b ??= perf.memory?.usedJSHeapSize;  // JavaScript heap only, the closest left
      if (alive) {
        if (b !== undefined) setBytes(b);
        timer = setTimeout(() => void tick(), MEMORY_MS);
      }
    };
    void tick();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  return bytes;
}

/** Share of each second, in %, that a browser step was running. */
function useBusy(running: boolean): number[] {
  const acc = useRef({ busy: 0, since: undefined as number | undefined, last: performance.now() });
  useEffect(() => {
    const a = acc.current;
    const now = performance.now();
    if (running && a.since === undefined) a.since = now;
    if (!running && a.since !== undefined) { a.busy += now - a.since; a.since = undefined; }
  }, [running]);
  return useSeries(() => {
    const a = acc.current;
    const now = performance.now();
    const busy = a.busy + (a.since !== undefined ? now - a.since : 0);
    const share = Math.min(100, (busy / Math.max(1, now - a.last)) * 100);
    a.busy = 0;
    a.last = now;
    if (a.since !== undefined) a.since = now;
    return share;
  });
}

function useServerStats(enabled: boolean): [ServerStats | undefined, boolean] {
  const [stats, setStats] = useState<ServerStats>();
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!enabled) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let alive = true;
    const tick = async () => {
      if (document.visibilityState === 'visible') {
        try {
          const s = await api.stats();
          if (alive) { setStats(s); setError(false); }
        } catch {
          if (alive) setError(true);
        }
      }
      if (alive) timer = setTimeout(() => void tick(), POLL_MS);
    };
    void tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [enabled]);
  return [stats, error];
}

interface Props {
  engine?: EngineInfo;
  /** the browser step running now, if any */
  browserStep?: string;
  lastRun?: RunSummary;
  /** a RetiVASC server is behind this page */
  server: boolean;
}

/** Live load, one row per machine, in the same columns: this computer (browser steps) and the server (server steps). */
export function PerfBar({ engine, browserStep, lastRun, server }: Props) {
  const pressure = usePressure();
  const pressureSeries = useSeries(() => (pressure ? PRESSURE_LEVEL[pressure] : undefined));
  const tabMemory = useTabMemory();
  const busy = useBusy(!!browserStep);
  const [stats, statsError] = useServerStats(server);
  const webgpu = engine?.backend === 'webgpu';
  const device = !engine ? 'starting…' : webgpu ? `WebGPU · ${engine.adapter}` : `CPU · ${engine.threads} thread${engine.threads === 1 ? '' : 's'}`;
  const browserRun = lastRun && lastRun.ran > 0 && !lastRun.errors ? lastRun : undefined;
  const [browserTime, setBrowserTime] = useState<{ ms: number; photos: number }>();
  useEffect(() => { if (browserRun) setBrowserTime({ ms: browserRun.ms, photos: browserRun.photos }); }, [browserRun]);

  const samples = stats?.samples ?? [];
  const last = samples.at(-1);

  return (
    <footer className="perfbar">
      <div className="perf-row">
        <Host active={!!browserStep} title="This computer" name={device} hint="Browser steps (M0–M3, overlay) run in this tab, on the computer you are looking at" />
        <Cell
          label="CPU"
          title="CPU pressure of this computer as the browser reports it (nominal, fair, serious, critical). Browsers give no percentage; Chrome or Edge 125+ only"
          values={pressureSeries}
          text={pressure}
        />
        <Cell
          label="RAM"
          title="Memory used by this tab and its engine. Browsers do not report the whole computer's memory; Chrome or Edge only"
          text={tabMemory === undefined ? undefined : `${tabMemory < 2 ** 30 ? mb(tabMemory) : gb(tabMemory)} · tab`}
        />
        <Cell
          label="GPU"
          title="Share of each second the in-browser engine was computing on this computer's GPU (WebGPU), last 60 s"
          values={webgpu ? busy : undefined}
          text={webgpu ? `${(busy.at(-1) ?? 0).toFixed(0).padStart(3)}%` : undefined}
        />
        <Cell label="Last run" title="How long the last run of the browser workflow took" text={browserTime && `${ms(browserTime.ms)} · ${browserTime.photos} photo${browserTime.photos === 1 ? '' : 's'}`} />
        <span className={`perf-state ${browserStep ? 'on' : ''}`}>{browserStep ? `Running ${browserStep}` : 'Idle'}</span>
      </div>

      {server && (
        <div className="perf-row">
          <Host
            active={!!stats?.running}
            title="Server"
            name={stats ? `${stats.host} · AutoMorph v2` : statsError ? 'not reachable' : '…'}
            hint="Server steps run on the RetiVASC server, with the complete AutoMorph v2"
          />
          <Cell label="CPU" title="Processor load of the server, last 60 s" values={samples.map((s) => s.cpu)} text={last && `${last.cpu.toFixed(0).padStart(3)}%`} />
          <Cell
            label="RAM"
            title="Memory in use on the server, last 60 s"
            values={samples.map((s) => (s.mem_used / s.mem_total) * 100)}
            text={last && `${gb(last.mem_used)} / ${gb(last.mem_total)}`}
          />
          {last && last.gpus.length > 0 ? (
            <span className="perf-gpus">
              {last.gpus.map((g) => (
                <Cell
                  key={g.id}
                  label={last.gpus.length > 1 ? `GPU ${g.id.replace(/^cuda:/, '')}` : 'GPU'}
                  title={`${g.name} (${g.id}) on the server: utilisation, last 60 s`}
                  values={samples.map((s) => s.gpus.find((x) => x.id === g.id)?.util ?? 0)}
                  text={`${g.util.toFixed(0).padStart(3)}%${g.mem_used != null ? ` · ${gb(g.mem_used)}` : ''}`}
                />
              ))}
            </span>
          ) : (
            <Cell label="GPU" title="No GPU the server can read" />
          )}
          <Cell
            label="Last run"
            title="How long the latest finished analysis on the server took"
            text={stats?.last_run ? `${ms(stats.last_run.seconds * 1000)} · ${stats.last_run.photos} photo${stats.last_run.photos === 1 ? '' : 's'}` : undefined}
          />
          <span className={`perf-state ${stats?.running ? 'on' : ''}`}>
            {stats?.running ? 'Analysing' : 'Idle'}
            {!!stats?.queued && ` · ${stats.queued} waiting`}
          </span>
        </div>
      )}
    </footer>
  );
}
