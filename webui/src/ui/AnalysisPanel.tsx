// Results of the full AutoMorph v2 run on the server for one photo: grade, the four result
// images and the 72 features. Compact in the canvas node, complete in the inspector.

import { useState } from 'react';
import { api } from '../server/api';
import type { AnalysisValue } from '../engine/values';

const IMAGE_LABEL: Record<string, string> = { photo: 'Photo', vessels: 'Vessels', artery_vein: 'Artery / vein', disc: 'Optic disc & cup' };

const KEY_FEATURES: Array<[string, string]> = [
  ['CDR_vertical', 'Cup-to-disc ratio (vertical)'],
  ['Fractal_dimension', 'Fractal dimension'],
  ['Vessel_density', 'Vessel density'],
  ['CRAE_Knudtson_zone_b', 'CRAE · zone B'],
  ['CRVE_Knudtson_zone_b', 'CRVE · zone B'],
  ['AVR_Knudtson_zone_b', 'AVR · zone B'],
];

const GROUPS: Array<[string, (k: string) => boolean]> = [
  ['Optic disc and cup', (k) => /^(Disc|Cup|CDR)_/.test(k)],
  ['Whole image', (k) => !/zone|^(Disc|Cup|CDR)_/.test(k)],
  ['Zone B', (k) => k.endsWith('_zone_b')],
  ['Zone C', (k) => k.endsWith('_zone_c')],
];

const fmt = (v: number | null | undefined) => (v == null ? '—' : Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3));
const label = (k: string) => k.replace(/_zone_[bc]$/, '').replace(/_/g, ' ');

export function AnalysisView({ value, full = false }: { value: AnalysisValue; full?: boolean }) {
  const { photo, jobId } = value;
  const q = photo.quality;
  const [big, setBig] = useState<string | null>(null);
  const images = Object.entries(photo.images);
  return (
    <div className={`ana${full ? ' full' : ''}`}>
      {q && (
        <div className={`ana-grade g-${q.grade}`}>
          <b>{q.grade}</b>
          <span>{q.gradable ? 'gradable' : 'not gradable'} · reject {Math.round(q.reject * 100)}%</span>
          {photo.centred_on && <span className="faint">centred on the {photo.centred_on}</span>}
        </div>
      )}
      {q && (
        <div className="ana-bars" title="Probability of each quality class (all ensemble members)">
          <span className="b-good" style={{ flexGrow: q.good }} />
          <span className="b-usable" style={{ flexGrow: q.usable }} />
          <span className="b-reject" style={{ flexGrow: q.reject }} />
        </div>
      )}
      <div className="ana-imgs">
        {images.map(([k, path]) => (
          <figure key={k} onClick={() => full && setBig(big === k ? null : k)} className={big === k ? 'big' : ''}>
            <img src={api.imageUrl(jobId, path!, big === k ? 1600 : full ? 640 : 360)} alt={IMAGE_LABEL[k] ?? k} loading="lazy" />
            <figcaption>{IMAGE_LABEL[k] ?? k}</figcaption>
          </figure>
        ))}
      </div>
      {photo.features && (
        <div className="ana-keys">
          {KEY_FEATURES.map(([k, l]) => (
            <div key={k}><span>{l}</span><b>{fmt(photo.features![k])}</b></div>
          ))}
        </div>
      )}
      {full && photo.features && GROUPS.map(([title, test]) => (
        <details key={title} open={title === 'Whole image'} className="ana-group">
          <summary>{title}</summary>
          <table className="stable">
            <tbody>
              {Object.entries(photo.features!).filter(([k]) => test(k)).map(([k, v]) => (
                <tr key={k}><td>{label(k)}</td><td className="num">{v == null ? '—' : String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
      {full && (
        <a className="sfile" href={api.fileUrl(jobId, 'features.csv')} download>
          <span className="sfk">Features</span><span className="sfn">features.csv (all 72, exact values)</span><span className="sfd">↓</span>
        </a>
      )}
    </div>
  );
}
