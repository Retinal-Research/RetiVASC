import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import { useMemo, useRef, type DragEvent } from 'react';
import { DISC } from '../automorph/postprocess';
import { QUALITY_CLASSES } from '../automorph/types';
import { GROUP_LABEL, SPECS } from '../engine/specs';
import { addUpload, sourceName } from '../engine/sources';
import type { DiscCupValue, M0Value, NodeValue, QualityValue } from '../engine/values';
import { DEFAULT_VIEW, useStudio, type StudioNodeData } from './context';
import { int, mb, pct, statusText } from './format';
import { countLabels, isLayer, LAYER_LABEL, layerCanvas, type LayerKind } from './render';
import { AnalysisView } from './AnalysisPanel';
import { FeaturesView } from './FeaturesPanel';
import { ANALYSIS_STEPS, SheetProgress, SheetSummaryView } from './SheetPanel';
import { FilePick, StudyPhotosBody } from './StudyNodes';
import { Thumb } from './Thumb';

export type StudioNode = Node<StudioNodeData, 'studio'>;

const THUMB_W = 248;
const THUMB_H = 168;

/** Share of the circular field of view (at a model's resolution) that a pixel count covers. */
export function fovShare(count: number, size: number, base: M0Value): number {
  const r = (base.geometry.radius * size) / base.side;
  return count / (Math.PI * r * r);
}

/** Bounding box of the disc in base-preview pixels, padded, or undefined. */
export function discCrop(v: DiscCupValue): { x: number; y: number; w: number; h: number } | undefined {
  const { width, height, labels } = v.map;
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (labels[y * width + x] === DISC.background) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return undefined;
  const s = v.base.preview.width / width;
  const side = Math.max(x1 - x0, y1 - y0) * 2.2 + 8;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  return { x: (cx - side / 2) * s, y: (cy - side / 2) * s, w: side * s, h: side * s };
}

function QualityBars({ value }: { value: QualityValue }) {
  const { probs, prediction, gradable } = value.result;
  return (
    <div className="quality">
      {QUALITY_CLASSES.map((c, i) => (
        <div key={c} className={`qrow ${prediction === c ? 'top' : ''}`}>
          <span className="qlabel">{c}</span>
          <span className="qbar"><span className={`qfill q${i}`} style={{ width: pct(probs[i]) }} /></span>
          <span className="qval">{pct(probs[i], 1)}</span>
        </div>
      ))}
      <div className={`verdict ${gradable ? 'ok' : 'bad'}`}>
        {gradable ? 'Gradable — AutoMorph keeps it' : 'Not gradable — AutoMorph drops it'}
      </div>
    </div>
  );
}

export function FlowNode({ id, data, selected }: NodeProps<StudioNode>) {
  const studio = useStudio();
  const spec = SPECS[data.type];
  const fileInput = useRef<HTMLInputElement>(null);
  const outputs = studio.executor.outputs(id);
  const inputs = studio.executor.inputs(id);
  const version = data.state.version;
  const model = spec.model ? studio.engine?.manifest.models[spec.model] : undefined;

  const openFile = async (file: File | undefined) => {
    if (!file) return;
    const source = addUpload(file.name, await file.arrayBuffer());
    studio.setParams(id, { source });
  };
  const onDrop = (e: DragEvent) => {
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    e.preventDefault();
    e.stopPropagation();
    void openFile(file);
  };

  const body = useMemo(() => {
    const out = outputs ? Object.values(outputs)[0] : undefined;
    switch (data.type) {
      case 'input.fundus': {
        const source = String(data.params.source ?? '');
        const sample = studio.samples.find((s) => `sample:${s.file}` === source);
        const v = out?.kind === 'fundus' ? out : undefined;
        return (
          <div className="fundus" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
            <div className="row">
              <select className="select" value={source} onChange={(e) => studio.setParams(id, { source: e.target.value })}>
                {studio.samples.map((s) => (
                  <option key={s.file} value={`sample:${s.file}`}>{s.label}</option>
                ))}
                {!sample && <option value={source}>{sourceName(source)}</option>}
              </select>
              <button className="btn small" onClick={() => fileInput.current?.click()}>Open…</button>
              <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => void openFile(e.target.files?.[0])} />
            </div>
            <Thumb base={v?.preview} width={THUMB_W} height={150} />
            <div className="meta">
              {v ? `${v.width} × ${v.height} px` : sample ? `${sample.width} × ${sample.height} px` : sourceName(source)}
              {sample && <span className="faint"> · {sample.license}</span>}
            </div>
            <div className="hint">Drop a photo here. It stays on this computer.</div>
          </div>
        );
      }
      case 'automorph.m0': {
        const v = out?.kind === 'm0' ? out : undefined;
        return (
          <>
            <Thumb base={v?.preview} width={THUMB_W} height={THUMB_H} />
            {v && <div className="meta">field radius {int(v.geometry.radius)} px · square {int(v.side)} px</div>}
          </>
        );
      }
      case 'automorph.m1':
        return out?.kind === 'quality' ? <QualityBars value={out} /> : <div className="empty">Run to grade the photo</div>;
      case 'automorph.vessel': {
        const v = out?.kind === 'vessel' ? out : undefined;
        const count = v ? v.mask.reduce((a, b) => a + b, 0) : 0;
        return (
          <>
            <Thumb base={v?.base.preview} layers={v ? [layerCanvas(v)] : []} width={THUMB_W} height={THUMB_H} />
            {v && <div className="meta">vessels cover {pct(fovShare(count, v.size, v.base))} of the field</div>}
          </>
        );
      }
      case 'automorph.av': {
        const v = out?.kind === 'av' ? out : undefined;
        const c = v ? countLabels(v.map.labels, 4) : [];
        return (
          <>
            <Thumb base={v?.base.preview} layers={v ? [layerCanvas(v)] : []} width={THUMB_W} height={THUMB_H} />
            {v && (
              <div className="meta">
                <span className="dot" style={{ background: '#ff4b4b' }} /> artery {pct(fovShare(c[1], v.map.width, v.base))}
                <span className="dot" style={{ background: '#4b8bff' }} /> vein {pct(fovShare(c[2], v.map.width, v.base))}
              </div>
            )}
          </>
        );
      }
      case 'automorph.disc': {
        const v = out?.kind === 'disc' ? out : undefined;
        const c = v ? countLabels(v.map.labels, 3) : [];
        const crop = v ? discCrop(v) : undefined;
        return (
          <>
            <Thumb base={v?.base.preview} layers={v ? [layerCanvas(v)] : []} crop={crop} width={THUMB_W} height={150} />
            {v && (
              <div className="meta">
                {crop ? `disc ${int(c[1] + c[2])} px · cup ${int(c[2])} px (at 512²)` : 'no optic disc found'}
              </div>
            )}
          </>
        );
      }
      case 'view.overlay': {
        const base = inputs?.image?.kind === 'm0' ? inputs.image : undefined;
        const view = data.view ?? DEFAULT_VIEW;
        const present = (['vessel', 'av', 'disc'] as LayerKind[]).filter((k) => isLayer(inputs?.[k]));
        const layers = present.filter((k) => view.layers[k]).map((k) => layerCanvas(inputs![k] as Parameters<typeof layerCanvas>[0]));
        const quality = inputs?.quality?.kind === 'quality' ? inputs.quality.result : undefined;
        return (
          <>
            <div className="view-frame">
              <Thumb base={base?.preview} layers={layers} opacity={view.opacity} width={300} height={300} />
              {quality && (
                <span className={`badge ${quality.gradable ? 'ok' : 'bad'}`}>
                  {quality.prediction} · {quality.gradable ? 'gradable' : 'not gradable'}
                </span>
              )}
            </div>
            {present.length > 0 && (
              <div className="row wrap">
                {present.map((k) => (
                  <button
                    key={k}
                    className={`chip ${view.layers[k] ? 'on' : ''}`}
                    onClick={() => studio.setView(id, { ...view, layers: { ...view.layers, [k]: !view.layers[k] } })}
                  >
                    {LAYER_LABEL[k]}
                  </button>
                ))}
                <input
                  className="slider"
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={view.opacity}
                  onChange={(e) => studio.setView(id, { ...view, opacity: Number(e.target.value) })}
                  title="Layer opacity"
                />
              </div>
            )}
            {!base && <div className="empty">Connect a cropped photo and run</div>}
          </>
        );
      }
      case 'automorph.m3': {
        const v = out?.kind === 'features' ? out : undefined;
        return (
          <>
            <label className="scheck" title="AutoMorph measures vessels on gradable photos only">
              <input type="checkbox" checked={!!data.params.measureAll} onChange={(e) => studio.setParams(id, { measureAll: e.target.checked })} />
              Measure anyway when not gradable
            </label>
            {v ? <FeaturesView value={v} /> : <div className="empty">Connect M0 and the three M2 steps, then run</div>}
          </>
        );
      }
      case 'server.analyze': {
        const v = out?.kind === 'analysis' ? out : undefined;
        return data.state.status === 'running' ? (
          <SheetProgress message={data.state.message} fraction={data.state.fraction} steps={ANALYSIS_STEPS} />
        ) : v ? (
          <AnalysisView value={v} />
        ) : (
          <div className="empty">Connect a photo and press Run. It is sent to the RetiVASC server, which runs the complete AutoMorph v2.</div>
        );
      }
      case 'study.photos': {
        const v = out?.kind === 'run' ? out : undefined;
        return <StudyPhotosBody params={data.params} setParams={(patch) => studio.setParams(id, patch)} photos={v?.photos} />;
      }
      case 'study.sheet': {
        const v = out?.kind === 'sheet' ? out : undefined;
        return (
          <div className="study">
            <FilePick label="Sheet template" id={String(data.params.templateId ?? '')} onPick={(templateId) => studio.setParams(id, { templateId })} />
            <FilePick label="Previous version (for the change log, optional)" id={String(data.params.previousId ?? '')} onPick={(previousId) => studio.setParams(id, { previousId })} />
            <label className="sfield">
              <span>Output file name</span>
              <input className="select" value={String(data.params.outName ?? '')} onChange={(e) => studio.setParams(id, { outName: e.target.value })} />
            </label>
            <label className="scheck">
              <input type="checkbox" checked={!!data.params.strict} onChange={(e) => studio.setParams(id, { strict: e.target.checked })} />
              Strict: never pick among several photos
            </label>
            {data.state.status === 'running' ? (
              <SheetProgress message={data.state.message} fraction={data.state.fraction} />
            ) : v ? (
              <SheetSummaryView value={v} />
            ) : (
              <div className="empty">Choose the photos and the sheet template, then press Run</div>
            )}
          </div>
        );
      }
      default:
        return null;
    }
  }, [data.type, data.params, data.view, data.state, version, outputs, inputs, studio.samples]);

  const s = data.state;
  return (
    <div className={`node g-${spec.group} st-${s.status}${selected ? ' selected' : ''}`}>
      <header className="node-head">
        <div className="node-kicker">
          <span className="kdot" />
          {GROUP_LABEL[spec.group]}
          {model && <span className="kmodel">{mb(model.bytes)}</span>}
        </div>
        <div className="node-title">{spec.title}</div>
        <div className="node-sub">{spec.subtitle}</div>
      </header>
      {(spec.inputs.length > 0 || spec.outputs.length > 0) && (
        <div className="ports">
          <div className="ports-col">
            {spec.inputs.map((p) => (
              <div className="port in" key={p.id}>
                <Handle type="target" position={Position.Left} id={p.id} className={`h t-${p.type}`} />
                <span>{p.label}{p.optional && <i> · optional</i>}</span>
              </div>
            ))}
          </div>
          <div className="ports-col right">
            {spec.outputs.map((p) => (
              <div className="port out" key={p.id}>
                <span>{p.label}</span>
                <Handle type="source" position={Position.Right} id={p.id} className={`h t-${p.type}`} />
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="node-body nodrag nowheel">
        {model && model.parts.length > 1 && (
          <label className="scheck" title="The complete ensemble is what AutoMorph v2 runs; one model is faster on slow computers">
            <input type="checkbox" checked={!!data.params.quick} onChange={(e) => studio.setParams(id, { quick: e.target.checked })} />
            Quick: 1 of {model.parts.length} models
          </label>
        )}
        {body}
      </div>
      <footer className="node-foot">
        <span className="sdot" />
        <span className="stext" title={statusText(s)}>{statusText(s)}</span>
        {s.status === 'running' && s.fraction !== undefined && (
          <span className="sbar"><span style={{ width: pct(s.fraction, 0) }} /></span>
        )}
      </footer>
    </div>
  );
}

export function firstOutput(outputs: Record<string, NodeValue> | undefined): NodeValue | undefined {
  return outputs ? Object.values(outputs)[0] : undefined;
}
