// AutoMorph v2 M3 in the browser: from the M2 maps to the 72 vascular features, reproducing
// v2's post-processing (M2 artery_vein/vessel/disc_cup) and measurement (M3) step by step.
//
//   vessel 912² (0/1)  ─ binary_process ─ skeleton ─┐
//   artery/vein 720²   ─ nearest 912 ─ artery / vein process ─ skeletons ─┤
//   disc/cup 512²      ─ nearest 912 ─ disc & cup extent, centring, Zone B/C rings ─┤
//                                                    9 measurements (whole, B, C × vessel, artery, vein)
//                                                    → the 72 columns of Macular_/Disc_Features.csv

import { removeSmallObjects } from '../postprocess';
import { circleFill, resizeNearest } from './cv';
import { evaluate, PyError, SENTINEL, type MetricResult, type VesselType } from './metrics';
import { csvRoundTrip } from './pyfloat';
import { skeletonize } from './skeleton';

const S = 912;

export interface M3Input {
  /** vessel map after M2 (threshold 0.5, specks under 30 px removed), 912 x 912, 0/1 */
  vessel: Uint8Array;
  /** artery/vein labels after M2, 720 x 720: 0 background, 1 artery, 2 vein, 3 crossing */
  arteryVein: Uint8Array;
  /** disc/cup labels after M2, 512 x 512: 0 background, 1 disc, 2 cup */
  discCup: Uint8Array;
  /** field-of-view radius from M0, in pixels of the original photo */
  radius: number;
  /** pixel size of the original photo in mm (AutoMorph's default 0.008) */
  pixelMm?: number;
}

export interface Zone { centreRow: number; centreCol: number; radius: number }

export interface M3Result {
  /** the 72 features, AutoMorph's column names; null where AutoMorph leaves the cell empty */
  features: Record<string, number | null>;
  /** which AutoMorph table the photo lands in */
  centredOn: 'macula' | 'optic disc';
  /** false when the disc/cup map fails AutoMorph's plausibility test (no zones, no disc values) */
  discValid: boolean;
  /** Zone B is the ring 2r–3r around the disc centre, Zone C 2r–5r (912 x 912 pixels) */
  zone?: Zone;
  /** µm per pixel at 912 x 912 */
  resolution: number;
  /** which of the nine measurements failed (AutoMorph writes -1, shown as empty) */
  failed: string[];
  ms: number;
}

const to255 = (m: Uint8Array) => { const o = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) o[i] = m[i] ? 255 : 0; return o; };

/** skimage remove_small_objects(min_size=30, connectivity=5 → 8-connected), then 0/255. */
function process(mask01: Uint8Array): Uint8Array {
  return to255(removeSmallObjects(mask01.slice(), S, S, 30, 8));
}

/** disc_cup.remove_skeleton_intersections: zero a radius-1 OpenCV circle at every interior pixel with > 2 neighbours. */
function removeIntersections(skel: Uint8Array): Uint8Array {
  const out = skel.slice();
  for (let r = 1; r < S - 1; r++) {
    for (let c = 1; c < S - 1; c++) {
      const i = r * S + c;
      if (!skel[i]) continue;
      const n = (skel[i - S - 1] > 0 ? 1 : 0) + (skel[i - S] > 0 ? 1 : 0) + (skel[i - S + 1] > 0 ? 1 : 0) + (skel[i - 1] > 0 ? 1 : 0)
        + (skel[i + 1] > 0 ? 1 : 0) + (skel[i + S - 1] > 0 ? 1 : 0) + (skel[i + S] > 0 ? 1 : 0) + (skel[i + S + 1] > 0 ? 1 : 0);
      if (n > 2) circleFill(out, S, S, c, r, 1, 0);
    }
  }
  return out;
}

interface DiscAnalysis {
  valid: boolean;
  centre: 'disc' | 'macular';
  values?: { discHeight: number; discWidth: number; cupHeight: number; cupWidth: number; cdrVertical: number; cdrHorizontal: number };
  zone?: Zone;
}

/** disc_cup.optic_disc_centre for one photo. */
function analyseDisc(discCup512: Uint8Array, resolution: number): DiscAnalysis {
  // the M2 disc/cup image: red = disc (label 1), blue = cup (label 2); M3 reads it with OpenCV (BGR)
  const labels = resizeNearest(discCup512, 512, S);
  let dMinR = Infinity, dMaxR = -Infinity, dMinC = Infinity, dMaxC = -Infinity;
  let cMinR = Infinity, cMaxR = -Infinity, cMinC = Infinity, cMaxC = -Infinity;
  let cupRowSum = 0, cupColSum = 0, cupN = 0, allRowSum = 0, allColSum = 0, allN = 0;
  for (let r = 0; r < S; r++) {
    for (let c = 0; c < S; c++) {
      const l = labels[r * S + c];
      if (l === 1) {
        if (r < dMinR) dMinR = r; if (r > dMaxR) dMaxR = r; if (c < dMinC) dMinC = c; if (c > dMaxC) dMaxC = c;
      } else if (l === 2) {
        if (r < cMinR) cMinR = r; if (r > cMaxR) cMaxR = r; if (c < cMinC) cMinC = c; if (c > cMaxC) cMaxC = c;
        cupRowSum += r; cupColSum += c; cupN++;
      }
      if (l === 1 || l === 2) { allRowSum += r; allColSum += c; allN++; }
    }
  }
  // np.max of an empty selection raises: no disc or no cup means an invalid detection
  if (!Number.isFinite(dMaxR) || cupN === 0) return { valid: false, centre: 'macular' };
  const discW = dMaxC - dMinC, discH = dMaxR - dMinR;
  const cupW = cMaxC - cMinC, cupH = cMaxR - cMinR;
  const cupColMean = cupColSum / cupN, cupRowMean = cupRowSum / cupN;
  const valid = discW < S / 3 && discH < S / 3 && cupColMean <= dMaxC && cupColMean >= dMinC
    && cupRowMean <= dMaxR && cupRowMean >= dMinR && cupH < discH && cupW < discW;
  if (!valid) return { valid: false, centre: 'macular' };
  const meanRow = allRowSum / allN, meanCol = allColSum / allN;
  const h = Math.abs(meanRow - S / 2), v = Math.abs(meanCol - S / 2);
  const distance = Math.sqrt(h * h + v * v);
  const zone: Zone = { centreCol: Math.trunc(meanCol), centreRow: Math.trunc(meanRow), radius: Math.max(Math.trunc(discW / 2), Math.trunc(discH / 2)) };
  return {
    valid: true,
    centre: distance / S < 0.1 ? 'disc' : 'macular',
    zone,
    values: { discHeight: discH * resolution, discWidth: discW * resolution, cupHeight: cupH * resolution, cupWidth: cupW * resolution,
      cdrVertical: cupH / discH, cdrHorizontal: cupW / discW },
  };
}

function ring(z: Zone, outer: number): Uint8Array {
  const m = new Uint8Array(S * S);
  circleFill(m, S, S, z.centreCol, z.centreRow, outer * z.radius, 1);
  circleFill(m, S, S, z.centreCol, z.centreRow, 2 * z.radius, 0);
  return m;
}

const masked = (img: Uint8Array, mask: Uint8Array) => { const o = new Uint8Array(img.length); for (let i = 0; i < img.length; i++) o[i] = mask[i] ? img[i] : 0; return o; };

const BASE = ['Fractal_dimension', 'Vessel_density', 'Average_width', 'Distance_tortuosity', 'Squared_curvature_tortuosity', 'Tortuosity_density'] as const;
const PREFIX: Record<VesselType, string> = { binary: '', artery: 'Artery_', vein: 'Vein_' };
const VALUES = (m: MetricResult) => [m.fractalDimension, m.vesselDensity, m.averageWidth, m.distanceTortuosity, m.squaredCurvatureTortuosity, m.tortuosityDensity];

/** The published value of a cell: written and read back by pandas, -1 shown as empty. */
const cell = (v: number, roundTrips = 1): number | null => {
  let x = v;
  for (let i = 0; i < roundTrips; i++) x = csvRoundTrip(x);
  return Number.isNaN(x) || x === -1 ? null : x;
};

export function runM3(input: M3Input): M3Result {
  const started = performance.now();
  const pixelMm = input.pixelMm ?? 0.008;
  // M0: Scale = radius*2/912, Scale_resolution = resolution*Scale*1000, read back from crop_info.csv by pandas
  const resolution = csvRoundTrip(pixelMm * (input.radius * 2.0 / 912.0) * 1000.0);

  // M2 post-processing
  const binaryProcess = to255(input.vessel);
  const avLabels = resizeNearest(input.arteryVein, 720, S);
  const artery01 = new Uint8Array(S * S), vein01 = new Uint8Array(S * S);
  for (let i = 0; i < avLabels.length; i++) {
    const l = avLabels[i];
    artery01[i] = l === 1 || l === 3 ? 1 : 0;
    vein01[i] = l === 2 || l === 3 ? 1 : 0;
  }
  const proc: Record<VesselType, Uint8Array> = { binary: binaryProcess, artery: process(artery01), vein: process(vein01) };
  const skel: Record<VesselType, Uint8Array> = {
    binary: to255(skeletonize(proc.binary, S, S)), artery: to255(skeletonize(proc.artery, S, S)), vein: to255(skeletonize(proc.vein, S, S)),
  };
  const disc = analyseDisc(input.discCup, resolution);

  // M3 measurements: the whole image uses the M2 masks as they are; zones use skeletons with
  // their junctions removed, cut to the ring
  const failed: string[] = [];
  const measure = (region: 'whole' | 'b' | 'c', type: VesselType, skeleton: Uint8Array, vessel: Uint8Array): MetricResult => {
    try {
      return evaluate(skeleton, vessel, S, S, resolution, region !== 'whole', type);
    } catch (error) {
      if (!(error instanceof PyError)) throw error;
      failed.push(`${region}:${type}`);
      return SENTINEL;
    }
  };
  const features: Record<string, number | null> = {};
  const types: VesselType[] = ['binary', 'artery', 'vein'];
  for (const t of types) {
    const m = measure('whole', t, skel[t], proc[t]);
    VALUES(m).forEach((v, k) => { features[PREFIX[t] + BASE[k]] = cell(v); });
  }
  const dv = disc.values;
  const discCell = (v: number | undefined) => (v === undefined ? null : cell(v, 2));  // Disc_cup_results.csv, then the group table
  features.Disc_height = discCell(dv?.discHeight);
  features.Disc_width = discCell(dv?.discWidth);
  features.Cup_height = discCell(dv?.cupHeight);
  features.Cup_width = discCell(dv?.cupWidth);
  features.CDR_vertical = discCell(dv?.cdrVertical);
  features.CDR_horizontal = discCell(dv?.cdrHorizontal);
  // zones use the skeletons with their junctions removed (once per vessel type)
  const zoneSkel = disc.zone ? { binary: removeIntersections(skel.binary), artery: removeIntersections(skel.artery), vein: removeIntersections(skel.vein) } : undefined;
  for (const [region, outer] of [['b', 3], ['c', 5]] as const) {
    const suffix = `_zone_${region}`;
    if (!disc.zone) {
      for (const t of types) for (const b of BASE) features[PREFIX[t] + b + suffix] = null;
      for (const k of ['CRAE_Hubbard', 'CRAE_Knudtson', 'CRVE_Hubbard', 'CRVE_Knudtson', 'AVR_Hubbard', 'AVR_Knudtson']) features[k + suffix] = null;
      continue;
    }
    const zoneMask = ring(disc.zone, outer);
    const results = {} as Record<VesselType, MetricResult>;
    for (const t of types) {
      results[t] = measure(region, t, masked(zoneSkel![t], zoneMask), masked(proc[t], zoneMask));
      VALUES(results[t]).forEach((v, k) => { features[PREFIX[t] + BASE[k] + suffix] = cell(v); });
    }
    const a = results.artery, v = results.vein;
    features[`CRAE_Hubbard${suffix}`] = cell(a.craeHubbard);
    features[`CRAE_Knudtson${suffix}`] = cell(a.craeKnudtson);
    features[`CRVE_Hubbard${suffix}`] = cell(v.crveHubbard);
    features[`CRVE_Knudtson${suffix}`] = cell(v.crveKnudtson);
    // AVR is divided before the table is written, so -1 / -1 gives 1, as in AutoMorph
    features[`AVR_Hubbard${suffix}`] = cell(a.craeHubbard / v.crveHubbard);
    features[`AVR_Knudtson${suffix}`] = cell(a.craeKnudtson / v.crveKnudtson);
  }
  return {
    features, centredOn: disc.centre === 'disc' ? 'optic disc' : 'macula', discValid: disc.valid, zone: disc.zone,
    resolution, failed, ms: performance.now() - started,
  };
}
