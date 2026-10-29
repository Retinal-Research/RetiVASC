import { useState, type ReactNode } from 'react';
import { QUALITY_CLASSES } from '../automorph/types';
import { GROUP_LABEL, SPECS } from '../engine/specs';
import { sourceName } from '../engine/sources';
import type { NodeValue } from '../engine/values';
import type { Timing } from '../runtime/protocol';
import { DEFAULT_VIEW, useStudio } from './context';
import { firstOutput, fovShare, type StudioNode } from './FlowNode';
import { int, mb, ms, pct } from './format';
import { ImageView } from './ImageView';
import { countLabels, isLayer, LAYER_LABEL, layerCanvas, LEGEND, type LayerKind, type LayerValue } from './render';
import { AnalysisView } from './AnalysisPanel';
import { FolderInspector } from './FolderPanel';
import { FeaturesView } from './FeaturesPanel';
import { ANALYSIS_STEPS, SheetProgress, SheetSummaryView } from './SheetPanel';

function Rows({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <dl className="rows">
      {rows.map(([k, v]) => (
        <div key={k} className="kv">
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="isec">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function timingRows(t: Timing, adapter: string): Array<[string, ReactNode]> {
  const rows: Array<[string, ReactNode]> = [];
  if (t.loadMs !== undefined) rows.push(['Load model (first use)', ms(t.loadMs)]);
  rows.push(['Resize + normalise', ms(t.preprocessMs)], ['Model', ms(t.inferenceMs)], ['Decode', ms(t.postprocessMs)]);
  rows.push(['Ran on', t.backend === 'webgpu' ? `GPU · ${adapter}` : 'CPU (WebAssembly)']);
  return rows;
}

export function Inspector({ node }: { node?: StudioNode }) {
  const studio = useStudio();
  const [opacity, setOpacity] = useState(DEFAULT_VIEW.opacity);
  const [hidden, setHidden] = useState<Partial<Record<LayerKind, boolean>>>({});
  const [before, setBefore] = useState(false);

  if (!node) {
    return (
      <aside className="inspector">
        <div className="inspector-empty">
          <h2>Select a step</h2>
          <p>Click any box on the canvas to see its result at full size, the model it used and how long it took.</p>
          <p className="faint">Every step runs inside this browser. Photos are not uploaded anywhere.</p>
        </div>
      </aside>
    );
  }

  const spec = SPECS[node.data.type];
  const outputs = studio.executor.outputs(node.id);
  const inputs = studio.executor.inputs(node.id);
  const out = firstOutput(outputs);
  const adapter = studio.engine?.adapter ?? '';
  const model = spec.model ? studio.engine?.manifest.models[spec.model] : undefined;

  let base: ImageBitmap | undefined;
  let layerValues: Array<[LayerKind, LayerValue]> = [];
  if (out?.kind === 'fundus' || out?.kind === 'm0') base = before && out.kind === 'm0' && out.before ? out.before : out.preview;
  else if (out && 'base' in out) base = out.base.preview;
  if (isLayer(out)) layerValues = [[out.kind, out]];
  if (node.data.type === 'view.overlay' && inputs) {
    if (inputs.image?.kind === 'm0') base = inputs.image.preview;
    layerValues = (['vessel', 'av', 'disc', 'fovea'] as LayerKind[]).flatMap((k) => {
      const v: NodeValue | undefined = inputs[k];
      return isLayer(v) ? [[k, v] as [LayerKind, LayerValue]] : [];
    });
  }
  const shown = layerValues.filter(([k]) => !hidden[k]).map(([, v]) => layerCanvas(v));

  const details: Array<[string, ReactNode]> = [];
  let footnote: string | undefined;
  if (out?.kind === 'fundus') {
    const sample = studio.samples.find((s) => `sample:${s.file}` === node.data.params.source);
    details.push(['File', out.name], ['Size', `${out.width} × ${out.height} px`]);
    if (sample) {
      details.push(['Image', sample.label], ['Licence', sample.license], ['Credit', <a href={sample.source} target="_blank" rel="noreferrer">{sample.author}</a>]);
    } else {
      details.push(['Origin', 'Opened from this computer; not uploaded']);
    }
  } else if (out?.kind === 'm0' && out.made === 'enhanced') {
    details.push(['From', out.source], ['Square', `${int(out.side)} × ${int(out.side)} px, as before enhancement`],
      ['Enhancement', node.data.params.quick ? 'TPOT at 256 × 256, resized back' : 'TPOT at 256 × 256, then the fusion head at 2048 × 2048'], ['Time', ms(out.ms)]);
    footnote = 'TPOT (Retinal-Research; ISBI 2025) enhances at 256 × 256. TPOT-SR\'s fusion head then puts the detail of the photograph back at 2048 × 2048, in tiles, and the result is resized to the size of the input, so every later step sees the same field of view as without enhancement. Use “Before” to compare.';
  } else if (out?.kind === 'm0' && out.made === 'vascx') {
    const g = out.geometry;
    details.push(['From', out.source], ['Field centre (x, y)', `${g.centreCol.toFixed(1)}, ${g.centreRow.toFixed(1)} px`], ['Field radius', `${g.radius.toFixed(1)} px`],
      ['Output square', `${int(out.side)} × ${int(out.side)} px`], ['Time', ms(out.ms)]);
    footnote = 'Follows retinalysis-fundusprep 1.3.0: the field of view is found on the red channel at 256 × 256 (polar edge path and a RANSAC circle) and cropped to 1024 × 1024 with OpenCV\'s bilinear warp.';
  } else if (out?.kind === 'm0' && node.data.type !== 'automorph.m0') {
    details.push(['From', out.source], ['Square', `${int(out.side)} × ${int(out.side)} px`], ['Time', ms(out.ms)]);
  } else if (out?.kind === 'm0') {
    const g = out.geometry;
    details.push(
      ['From', out.source],
      ['Field centre (row, col)', `${g.centreRow.toFixed(1)}, ${g.centreCol.toFixed(1)}`],
      ['Field radius', `${int(g.radius)} px`],
      ['Crop box (top, left, h, w)', g.bbox.map(int).join(', ')],
      ['Output square', `${int(out.side)} × ${int(out.side)} px`],
      ['Time', ms(out.ms)],
    );
  } else if (out?.kind === 'quality') {
    const r = out.result;
    QUALITY_CLASSES.forEach((c, i) => details.push([`P(${c})`, pct(r.probs[i], 2)]));
    details.push(['Grade', r.prediction], ['AutoMorph keeps it?', r.gradable ? 'Yes' : 'No']);
    footnote = 'AutoMorph keeps a photo graded Good, or Usable with P(Reject) under 25%.';
  } else if (out?.kind === 'vessel') {
    const count = out.mask.reduce((a, b) => a + b, 0);
    details.push([`Vessel pixels (${out.size}²)`, int(count)], ['Share of the field', pct(fovShare(count, out.size, out.base))]);
  } else if (out?.kind === 'av') {
    const c = countLabels(out.map.labels, 4);
    const share = (n: number) => pct(fovShare(n, out.map.width, out.base));
    details.push(['Artery', share(c[1])], ['Vein', share(c[2])], ['Crossing', share(c[3])]);
  } else if (out?.kind === 'disc') {
    const c = countLabels(out.map.labels, 3), n = out.map.width;
    details.push([`Optic disc (${n}²)`, `${int(c[1] + c[2])} px`]);
    if (spec.group !== 'vascx') details.push([`Cup (${n}²)`, `${int(c[2])} px`]);
  } else if (out?.kind === 'fovea') {
    details.push(['Fovea (x, y)', `${out.x.toFixed(1)}, ${out.y.toFixed(1)} px of ${int(out.size)}`]);
  }
  if (spec.group === 'vascx' && spec.model) {
    footnote = (footnote ? `${footnote} ` : '') + 'VascX (Eyened Reading Center, Rotterdam), CC BY-NC 4.0: non-commercial research use. Run as retinalysis-inference does: the same windows, Gaussian blending and flips.';
  }
  if (spec.model === 'quality' || spec.model === 'vessel' || spec.model === 'artery_vein') {
    footnote = (footnote ? `${footnote} ` : '') + 'The browser runs one member of the AutoMorph ensemble; the official pipeline averages all of them, so results can differ slightly.';
  }

  const head = (
    <header className="ihead">
      <div className={`ikicker g-${spec.group}`}>{GROUP_LABEL[spec.group]}</div>
      <h2>{spec.title}</h2>
      <p>{spec.subtitle}</p>
    </header>
  );
  if (node.data.type === 'input.folder') {
    return (
      <aside className="inspector">
        {head}
        <div className="iscroll">
          <FolderInspector nodeId={node.id} params={node.data.params} />
        </div>
      </aside>
    );
  }
  if (out?.kind === 'features') {
    return (
      <aside className="inspector">
        {head}
        <div className="iscroll">
          <FeaturesView value={out} full />
          <p className="footnote">AutoMorph v2 M3, run in this browser: the same steps and arithmetic as the Python module. On the official M2 maps it reproduces every one of the 72 values bit for bit, except the fractal dimension, which agrees to about 15 significant digits.</p>
        </div>
      </aside>
    );
  }
  if (out?.kind === 'analysis' || node.data.type === 'server.analyze') {
    const st = node.data.state;
    return (
      <aside className="inspector">
        {head}
        <div className="iscroll">
          {st.status === 'running' ? (
            <SheetProgress message={st.message} fraction={st.fraction} steps={ANALYSIS_STEPS} />
          ) : out?.kind === 'analysis' ? (
            <AnalysisView value={out} full />
          ) : (
            <p className="footnote">The photo is sent to the RetiVASC server, which runs the complete AutoMorph v2: every ensemble member and all 72 vascular features. The browser steps use one member each, for a quick preview.</p>
          )}
          {st.status === 'error' && <p className="error">{st.message}</p>}
        </div>
      </aside>
    );
  }
  if (out?.kind === 'sheet' || node.data.type === 'study.sheet') {
    const st = node.data.state;
    return (
      <aside className="inspector">
        {head}
        <div className="iscroll">
          {st.status === 'running' ? (
            <SheetProgress message={st.message} fraction={st.fraction} />
          ) : out?.kind === 'sheet' ? (
            <SheetSummaryView value={out} full />
          ) : (
            <p className="footnote">Choose the photos and the sheet template, then press Run. The server runs AutoMorph v2, fills the sheet and audits it against the 33 checks of the manual.</p>
          )}
          {st.status === 'error' && <p className="error">{st.message}</p>}
        </div>
      </aside>
    );
  }
  return (
    <aside className="inspector">
      {head}
      <ImageView base={base} layers={shown} opacity={opacity} empty={node.data.state.status === 'blocked' ? node.data.state.message : undefined} />
      {out?.kind === 'm0' && out.before && (
        <div className="legend">
          <button className={`chip ${before ? '' : 'on'}`} onClick={() => setBefore(false)}>After</button>
          <button className={`chip ${before ? 'on' : ''}`} onClick={() => setBefore(true)}>Before</button>
        </div>
      )}
      {layerValues.length > 0 && (
        <div className="legend">
          {layerValues.map(([k]) => (
            <button key={k} className={`chip ${hidden[k] ? '' : 'on'}`} onClick={() => setHidden({ ...hidden, [k]: !hidden[k] })}>
              {LAYER_LABEL[k]}
            </button>
          ))}
          <input className="slider" type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} title="Layer opacity" />
          <div className="keys">
            {layerValues.flatMap(([k]) => LEGEND[k]).map(([label, color]) => (
              <span key={label}><span className="dot" style={{ background: color }} />{label}</span>
            ))}
          </div>
        </div>
      )}
      <div className="iscroll">
        {details.length > 0 && <Section title="Result"><Rows rows={details} /></Section>}
        {model && (
          <Section title="Model">
            <Rows
              rows={[
                ['Network', model.label],
                ['Ensemble', model.members],
                ['Input', `${model.input[2]} × ${model.input[3]} px`],
                ['Download', `${model.parts.length} file${model.parts.length > 1 ? 's' : ''} · ${mb(model.bytes)} (${model.float32 ? 'float32' : 'float16'} weights)`],
                ['Members', model.parts.map((p) => p.member).join(', ')],
                ['SHA-256', <code>{model.parts[0].sha256.slice(0, 16)}…{model.parts.length > 1 ? ` (+${model.parts.length - 1})` : ''}</code>],
              ]}
            />
          </Section>
        )}
        {out && 'timing' in out && out.timing && <Section title="Timing"><Rows rows={timingRows(out.timing, adapter)} /></Section>}
        {node.data.type === 'input.fundus' && !out && (
          <Section title="Source"><Rows rows={[['Selected', sourceName(String(node.data.params.source ?? ''))]]} /></Section>
        )}
        {footnote && <p className="footnote">{footnote}</p>}
        {node.data.state.status === 'error' && <p className="error">{node.data.state.message}</p>}
      </div>
    </aside>
  );
}
