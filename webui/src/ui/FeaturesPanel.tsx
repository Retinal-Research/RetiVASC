// AutoMorph M3 results of one photo: the measurement zones on the photo, the key values, and all
// 72 features in AutoMorph's column order (downloadable as the same CSV row).

import { useEffect, useRef } from 'react';
import { pyRepr } from '../automorph/m3/pyfloat';
import type { FeaturesValue } from '../engine/values';

const BASE = ['Fractal_dimension', 'Vessel_density', 'Average_width', 'Distance_tortuosity', 'Squared_curvature_tortuosity', 'Tortuosity_density'];
const zone = (s: string) => [
  ...BASE.map((b) => b + s), ...BASE.map((b) => `Artery_${b}${s}`), `CRAE_Hubbard${s}`, `CRAE_Knudtson${s}`,
  ...BASE.map((b) => `Vein_${b}${s}`), `CRVE_Hubbard${s}`, `CRVE_Knudtson${s}`, `AVR_Hubbard${s}`, `AVR_Knudtson${s}`,
];
/** AutoMorph's 72 feature columns, in its CSV order. */
export const FEATURE_ORDER = [
  'Disc_height', 'Disc_width', 'Cup_height', 'Cup_width', 'CDR_vertical', 'CDR_horizontal',
  ...BASE, ...BASE.map((b) => `Artery_${b}`), ...BASE.map((b) => `Vein_${b}`), ...zone('_zone_b'), ...zone('_zone_c'),
];
const GROUPS: Array<[string, string[]]> = [
  ['Optic disc and cup', FEATURE_ORDER.slice(0, 6)],
  ['Whole image', FEATURE_ORDER.slice(6, 24)],
  ['Zone B (2–3 disc radii)', FEATURE_ORDER.slice(24, 48)],
  ['Zone C (2–5 disc radii)', FEATURE_ORDER.slice(48, 72)],
];
const KEY: Array<[string, string, string]> = [
  ['CDR_vertical', 'Cup-to-disc ratio (vertical)', ''],
  ['Fractal_dimension', 'Fractal dimension', ''],
  ['Vessel_density', 'Vessel density', ''],
  ['Average_width', 'Average vessel width', 'µm'],
  ['CRAE_Knudtson_zone_b', 'CRAE · Zone B', 'µm'],
  ['CRVE_Knudtson_zone_b', 'CRVE · Zone B', 'µm'],
  ['AVR_Knudtson_zone_b', 'AVR · Zone B', ''],
];
const UNIT = (k: string) => (/^(Disc|Cup)_(height|width)|Average_width|CRAE|CRVE/.test(k) ? ' µm' : '');
const fmt = (v: number | null) => (v == null ? '—' : Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3));
const nice = (k: string) => k.replace(/_zone_[bc]$/, '').replace(/_/g, ' ');

function ZoneView({ value }: { value: FeaturesValue }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const { base, result } = value;
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const size = 320;
    c.width = size;
    c.height = size;
    const g = c.getContext('2d')!;
    g.drawImage(base.hires ?? base.preview, 0, 0, size, size);
    const z = result.zone;
    if (!z) return;
    const k = size / 912;
    const cx = z.centreCol * k, cy = z.centreRow * k;
    g.lineWidth = 2;
    // Zone C ring (2r–5r), Zone B ring (2r–3r), disc circle (r)
    g.fillStyle = 'rgba(96, 165, 250, 0.16)';
    g.beginPath(); g.arc(cx, cy, 5 * z.radius * k, 0, 2 * Math.PI); g.arc(cx, cy, 2 * z.radius * k, 0, 2 * Math.PI, true); g.fill();
    g.fillStyle = 'rgba(251, 191, 36, 0.28)';
    g.beginPath(); g.arc(cx, cy, 3 * z.radius * k, 0, 2 * Math.PI); g.arc(cx, cy, 2 * z.radius * k, 0, 2 * Math.PI, true); g.fill();
    g.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    g.beginPath(); g.arc(cx, cy, z.radius * k, 0, 2 * Math.PI); g.stroke();
  }, [base, result]);
  return (
    <figure className="fz">
      <canvas ref={canvas} />
      <figcaption>
        {result.zone ? <><i className="fz-b" />Zone B <i className="fz-c" />Zone C <i className="fz-d" />disc</> : 'No zones: the disc/cup map did not pass AutoMorph’s check'}
      </figcaption>
    </figure>
  );
}

function downloadCsv(value: FeaturesValue) {
  const row = FEATURE_ORDER.map((k) => { const v = value.result.features[k]; return v == null ? '' : pyRepr(v); });
  const name = value.base.source.split('/').pop()?.replace(/\.[^.]+$/, '') ?? 'photo';
  const csv = `Name,${FEATURE_ORDER.join(',')}\n${name},${row.join(',')}\n`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `${name}_${value.result.centredOn === 'optic disc' ? 'Disc' : 'Macular'}_Features.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function FeaturesView({ value, full = false }: { value: FeaturesValue; full?: boolean }) {
  const r = value.result;
  const notMeasured = value.gradable === false && !value.forced;
  return (
    <div className={`feat${full ? ' full' : ''}`}>
      <div className="feat-head">
        <b>Centred on the {r.centredOn}</b>
        <span className="faint">{(r.resolution).toFixed(2)} µm per pixel · {Math.round(r.ms)} ms</span>
      </div>
      {notMeasured && <div className="hint">Not gradable: AutoMorph keeps only the disc values for this photo. Tick “measure anyway” to see the rest.</div>}
      {value.forced && value.gradable === false && <div className="hint">Measured although not gradable; AutoMorph itself would leave these empty.</div>}
      {r.failed.length > 0 && <div className="hint">AutoMorph could not measure: {r.failed.join(', ')} (left empty, as in its tables)</div>}
      {full && <ZoneView value={value} />}
      <div className="ana-keys">
        {KEY.map(([k, label, unit]) => (
          <div key={k}><span>{label}</span><b>{fmt(r.features[k])}{r.features[k] != null && unit ? ` ${unit}` : ''}</b></div>
        ))}
      </div>
      {full && GROUPS.map(([title, keys]) => (
        <details key={title} open={title === 'Whole image'} className="ana-group">
          <summary>{title}</summary>
          <table className="stable">
            <tbody>
              {keys.map((k) => <tr key={k}><td>{nice(k)}</td><td className="num">{r.features[k] == null ? '—' : `${r.features[k]}${UNIT(k)}`}</td></tr>)}
            </tbody>
          </table>
        </details>
      ))}
      {full && (
        <button className="sfile" onClick={() => downloadCsv(value)}>
          <span className="sfk">Features</span><span className="sfn">72 values, AutoMorph’s CSV layout</span><span className="sfd">↓</span>
        </button>
      )}
    </div>
  );
}
