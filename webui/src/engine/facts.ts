// What the AI report is told: the results of the steps connected to it, as plain numbers with names
// and units, in the report's language. Every number is computed here, not by the language model: a
// small model copies numbers well but compares and averages them badly, and it translates medical
// terms badly, so the Chinese names are given too. The same sheet is shown next to the report, so a
// reader can check what the text is based on.

import { QUALITY_CLASSES } from '../automorph/types';
import type { PhotoRecord } from './folder';
import { GROUP_LABEL, SPECS } from './specs';
import PROMPTS from './prompts.json';
import { measures } from './summary';
import type { M0Value, NodeValue } from './values';

export type Language = 'en' | 'zh';
type Text = Record<Language, string>;
const T = (en: string, zh: string): Text => ({ en, zh });

export interface FactGroup {
  title: string;
  /** what the rows are measured on, and caveats */
  note?: string;
  rows: Array<[string, string]>;
}

export interface FactSheet {
  scope: 'photo' | 'folder';
  language: Language;
  groups: FactGroup[];
  /** definitions the model needs for these rows (keys of TERMS) */
  terms: Array<keyof typeof TERMS>;
}

/** A connected step's result. */
export interface Source { from: string; type: string; value: NodeValue }

/** A folder run, for the report written after its last photo. */
export interface Cohort {
  total: number;
  records: PhotoRecord[];
  /** the steps connected to the report, in the order connected */
  sources: Array<{ id: string; type: string }>;
}

/** Definitions the model is given for the rows it gets (shared with the server: prompts.json). */
export const TERMS: Record<keyof typeof PROMPTS.terms, Text> = PROMPTS.terms;

/** The AutoMorph M3 columns a report is about: [column, name, unit, term]. A small model given all
 *  72 values lists them instead of writing about them; all 72 stay on the M3 step and in the CSV. */
const FEATURES: Array<[string, Text, string, keyof typeof TERMS]> = [
  ['CDR_vertical', T('Cup-to-disc ratio, vertical', '杯盘比（垂直）'), '', 'cdr'],
  ['Fractal_dimension', T('Fractal dimension, all vessels', '分形维数（全部血管）'), '', 'fractal'],
  ['Vessel_density', T('Vessel density, all vessels', '血管密度（全部血管）'), '', 'density'],
  ['Artery_Average_width', T('Average width, arteries', '平均管径（动脉）'), 'µm', 'width'],
  ['Vein_Average_width', T('Average width, veins', '平均管径（静脉）'), 'µm', 'width'],
  ['Distance_tortuosity', T('Distance tortuosity, all vessels', '距离迂曲度（全部血管）'), '', 'tortuosity'],
  ['Artery_Distance_tortuosity', T('Distance tortuosity, arteries', '距离迂曲度（动脉）'), '', 'tortuosity'],
  ['Vein_Distance_tortuosity', T('Distance tortuosity, veins', '距离迂曲度（静脉）'), '', 'tortuosity'],
  ['CRAE_Knudtson_zone_b', T('CRAE, Zone B', 'CRAE（B 区）'), 'µm', 'caliber'],
  ['CRVE_Knudtson_zone_b', T('CRVE, Zone B', 'CRVE（B 区）'), 'µm', 'caliber'],
  ['AVR_Knudtson_zone_b', T('AVR, Zone B', 'AVR（B 区）'), '', 'avr'],
  ['AVR_Knudtson_zone_c', T('AVR, Zone C', 'AVR（C 区）'), '', 'avr'],
];

/** The simple measures (engine/summary.ts), as percentages or pixels. */
const MEASURES: Record<string, [Text, 'share' | 'px', keyof typeof TERMS]> = {
  Vessel_share: [T('Vessels, share of the field of view', '血管占视野比例'), 'share', 'share'],
  Artery_share: [T('Arteries, share of the field of view', '动脉占视野比例'), 'share', 'share'],
  Vein_share: [T('Veins, share of the field of view', '静脉占视野比例'), 'share', 'share'],
  Disc_share: [T('Optic disc, share of the field of view', '视盘占视野比例'), 'share', 'share'],
  Fovea_x: [T('Fovea, x', '黄斑中心凹 x'), 'px', 'fovea'],
  Fovea_y: [T('Fovea, y', '黄斑中心凹 y'), 'px', 'fovea'],
};

const GRADE: Record<string, Text> = { Good: T('Good', '好（Good）'), Usable: T('Usable', '可用（Usable）'), Reject: T('Reject', '不合格（Reject）') };
const TITLE_ZH: Record<string, string> = {
  'Fundus photo': '眼底照片', 'Photo folder': '照片文件夹', Preprocess: '预处理', 'Enhance · TPOT': 'TPOT 增强', 'Image quality': '图像质量',
  Vessels: '血管', 'Artery / vein': '动脉/静脉', 'Optic disc & cup': '视盘和视杯', 'Optic disc': '视盘', 'Vascular features': '血管特征', Fovea: '黄斑中心凹',
};
const LABEL_ZH: Record<string, string> = { Input: '输入', View: '视图', 'Server · full model': '服务器 · 完整模型' };

/** Four significant digits at most, no exponent: what the model should copy. */
export function num(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  const digits = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : a >= 0.1 ? 3 : a >= 0.01 ? 4 : 5;
  return v.toFixed(digits).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

const pct = (share: number) => `${num(share * 100)}%`;
const unit = (v: string, u: string) => (u ? `${v} ${u}` : v);

/** "AutoMorph M2 · Vessels" style name of the step that made a value. */
function stepName(type: string, l: Language): string {
  const spec = SPECS[type];
  if (!spec) return type;
  const title = spec.title.replace(/^(M\d|VascX) · /, '');
  const group = GROUP_LABEL[spec.group];
  return l === 'zh' ? `${LABEL_ZH[group] ?? group} · ${TITLE_ZH[title] ?? title}` : `${group} · ${title}`;
}

const baseOf = (v: NodeValue): M0Value | undefined => ('base' in v ? v.base : v.kind === 'm0' ? v : undefined);
function madeOn(base: M0Value | undefined, l: Language): string {
  if (base?.made === 'enhanced') return l === 'zh' ? '在增强后的照片上测量' : 'measured on the enhanced photograph';
  if (base?.made === 'vascx') return l === 'zh' ? '在 VascX 裁剪的照片上测量' : 'measured on VascX’s crop of the photograph';
  return l === 'zh' ? '在原始照片上测量' : 'measured on the original photograph';
}
const centred = (on: string, l: Language) => (l === 'zh' ? `照片以${on === 'macula' ? '黄斑' : '视盘'}为中心` : `photograph centred on the ${on}`);
const yes = (b: boolean, l: Language) => (l === 'zh' ? (b ? '是' : '否') : b ? 'yes' : 'no');
const gradableName = (l: Language) => (l === 'zh' ? '按 AutoMorph 的规则可分析' : 'Gradable by AutoMorph’s rule');
const probability = (c: string, l: Language) => (l === 'zh' ? `${GRADE[c].zh}的概率` : `Probability of ${c}`);

function featureRows(get: (col: string) => number | null | undefined, l: Language, terms: Set<keyof typeof TERMS>): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  for (const [col, name, u, term] of FEATURES) {
    const x = get(col);
    if (x === null || x === undefined) continue;
    rows.push([name[l], unit(num(x), u)]);
    terms.add(term);
    if (term === 'caliber' || term === 'avr') terms.add('zones');
    if (u) terms.add('microns');
  }
  return rows;
}

// ---------------------------------------------------------------- one photograph

export function photoFacts(sources: Source[], l: Language): FactSheet {
  const groups: FactGroup[] = [];
  const terms = new Set<keyof typeof TERMS>();
  const enhanced = sources.some((s) => baseOf(s.value)?.made === 'enhanced');
  const where = (v: NodeValue) => (enhanced ? madeOn(baseOf(v), l) : undefined);
  if (enhanced) terms.add('enhanced');

  for (const { type, value: v } of sources) {
    const title = stepName(type, l);
    switch (v.kind) {
      case 'fundus':
        groups.push({ title: l === 'zh' ? '照片' : 'Photograph', rows: [[l === 'zh' ? '尺寸' : 'Size', `${v.width} × ${v.height} px`]] });
        break;
      case 'm0':
        groups.push({ title, note: v.made === 'enhanced' ? (l === 'zh' ? 'TPOT 增强后的照片' : 'the photograph after TPOT enhancement') : undefined,
          rows: [[l === 'zh' ? '视野半径' : 'Field of view radius', `${num(v.geometry.radius)} px`], [l === 'zh' ? '方形图像' : 'Square image', `${v.side} × ${v.side} px`]] });
        break;
      case 'quality': {
        const r = v.result;
        terms.add('quality');
        groups.push({ title, note: where(v), rows: [
          [l === 'zh' ? '等级' : 'Grade', GRADE[r.prediction][l]],
          ...QUALITY_CLASSES.map((c, i): [string, string] => [probability(c, l), num(r.probs[i])]),
          [gradableName(l), yes(r.gradable, l)],
        ] });
        break;
      }
      case 'vessel': case 'av': case 'disc': case 'fovea': {
        const rows = Object.entries(measures(v) ?? {}).map(([k, x]): [string, string] => {
          const [name, kind, term] = MEASURES[k];
          terms.add(term);
          return [name[l], kind === 'share' ? pct(x) : `${num(x)} px`];
        });
        groups.push({ title, note: where(v), rows });
        break;
      }
      case 'features': {
        const notes = [where(v), centred(v.result.centredOn, l)];
        if (v.gradable === false && !v.forced) notes.push(l === 'zh' ? '不可分析，AutoMorph 只保留视盘数值' : 'not gradable, so AutoMorph keeps only the optic disc values');
        if (!v.result.discValid) notes.push(l === 'zh' ? '没有可靠地找到视盘，所以没有视盘和分区数值' : 'the optic disc was not found reliably, so there are no disc or zone values');
        groups.push({ title, note: notes.filter(Boolean).join(l === 'zh' ? '；' : '; '), rows: featureRows((c) => v.result.features[c], l, terms) });
        break;
      }
      case 'analysis': {
        const p = v.photo;
        const rows: Array<[string, string]> = [];
        if (p.quality) {
          terms.add('quality');
          const probs = [p.quality.good, p.quality.usable, p.quality.reject];
          rows.push([l === 'zh' ? '等级' : 'Grade', GRADE[p.quality.grade][l]], ...QUALITY_CLASSES.map((c, i): [string, string] => [probability(c, l), num(probs[i])]),
            [gradableName(l), yes(p.quality.gradable, l)]);
        }
        rows.push(...featureRows((c) => p.features?.[c], l, terms));
        groups.push({ title: l === 'zh' ? 'AutoMorph v2 完整分析（RetiVASC 服务器）' : 'AutoMorph v2, full analysis on the RetiVASC server',
          note: p.centred_on ? centred(p.centred_on, l) : undefined, rows });
        break;
      }
      default:
        break;  // images, study runs and other reports carry no numbers to report
    }
  }
  return { scope: 'photo', language: l, groups: groups.filter((g) => g.rows.length), terms: [...terms] };
}

// ---------------------------------------------------------------- a folder

interface Stats { n: number; mean: number; sd: number; median: number; min: number; max: number }

function stats(values: number[]): Stats | undefined {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return undefined;
  const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  const median = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
  return { n, mean, sd, median, min: v[0], max: v[n - 1] };
}

function statText(s: Stats, show: (x: number) => string, l: Language): string {
  return l === 'zh'
    ? `n ${s.n}，均值 ${show(s.mean)}，标准差 ${show(s.sd)}，中位数 ${show(s.median)}，范围 ${show(s.min)} 至 ${show(s.max)}`
    : `n ${s.n}, mean ${show(s.mean)}, SD ${show(s.sd)}, median ${show(s.median)}, range ${show(s.min)} to ${show(s.max)}`;
}

const ofTotal = (k: number, n: number, l: Language) => (l === 'zh' ? `${n} 张中 ${k} 张` : `${k} of ${n}`);

export function folderFacts(c: Cohort, l: Language): FactSheet {
  const groups: FactGroup[] = [];
  const terms = new Set<keyof typeof TERMS>(['statistics']);
  const done = c.records.filter((r) => r.status === 'done');
  const zh = l === 'zh';
  groups.push({ title: zh ? '文件夹' : 'Folder', rows: [
    [zh ? '文件夹中的照片' : 'Photos in the folder', String(c.total)], [zh ? '已分析' : 'Analysed', String(done.length)],
    [zh ? '失败' : 'Failed', String(c.records.length - done.length)], [zh ? '尚未分析（运行已停止）' : 'Not yet analysed (run stopped)', String(c.total - c.records.length)],
  ] });

  // each measure, by the step that made it, for comparisons between steps
  const byMeasure = new Map<string, Array<{ type: string; values: Map<string, number> }>>();
  for (const { id, type } of c.sources) {
    const title = stepName(type, l);
    const qualities = done.map((r) => r.quality[id]).filter(Boolean);
    if (qualities.length) {
      terms.add('quality');
      const count = (g: string) => qualities.filter((q) => q.prediction === g).length;
      groups.push({ title, rows: [
        ...QUALITY_CLASSES.map((g): [string, string] => [zh ? `等级为${GRADE[g].zh}` : `Graded ${g}`, ofTotal(count(g), qualities.length, l)]),
        [gradableName(l), ofTotal(qualities.filter((q) => q.gradable).length, qualities.length, l)],
      ] });
    }
    const feats = done.map((r) => r.features[id]).filter(Boolean);
    if (feats.length) {
      const rows: Array<[string, string]> = [];
      for (const [col, name, u, term] of FEATURES) {
        const s = stats(feats.map((f) => f.features[col]).filter((x): x is number => x !== null && x !== undefined));
        if (!s) continue;
        rows.push([name[l], statText(s, (x) => unit(num(x), u), l)]);
        terms.add(term);
        if (term === 'caliber' || term === 'avr') terms.add('zones');
        if (u) terms.add('microns');
      }
      const macula = feats.filter((f) => f.centredOn === 'macula').length;
      groups.push({ title, rows, note: zh
        ? `${feats.length} 张照片中 ${macula} 张以黄斑为中心，其余以视盘为中心；血管数值只来自可分析的照片`
        : `${macula} of ${feats.length} photographs centred on the macula, the rest on the optic disc; vessel values only for gradable photos` });
    }
    const keys = new Set(done.flatMap((r) => Object.keys(r.measures?.[id] ?? {})));
    if (keys.size) {
      const rows: Array<[string, string]> = [];
      for (const k of keys) {
        const values = new Map(done.flatMap((r) => (r.measures?.[id]?.[k] !== undefined ? [[r.path, r.measures[id][k]] as [string, number]] : [])));
        const s = stats([...values.values()]);
        if (!s || !MEASURES[k]) continue;
        const [name, kind, term] = MEASURES[k];
        terms.add(term);
        rows.push([name[l], statText(s, kind === 'share' ? pct : (x) => `${num(x)} px`, l)]);
        byMeasure.set(k, [...(byMeasure.get(k) ?? []), { type, values }]);
      }
      groups.push({ title, rows });
    }
  }

  // the same measure from two steps (e.g. AutoMorph and VascX vessels), photo by photo
  const rows: Array<[string, string]> = [];
  for (const [k, list] of byMeasure) {
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const a = list[i], b = list[j];
      const pairs = [...a.values].flatMap(([path, x]) => (b.values.has(path) ? [[x, b.values.get(path)!]] : []));
      if (pairs.length < 2) continue;
      const [name, kind] = MEASURES[k];
      const show = kind === 'share' ? pct : (x: number) => `${num(x)} px`;
      const d = stats(pairs.map(([x, y]) => y - x))!;
      const mx = pairs.reduce((s, [x]) => s + x, 0) / pairs.length, my = pairs.reduce((s, [, y]) => s + y, 0) / pairs.length;
      const cov = pairs.reduce((s, [x, y]) => s + (x - mx) * (y - my), 0);
      const r = cov / Math.sqrt(pairs.reduce((s, [x]) => s + (x - mx) ** 2, 0) * pairs.reduce((s, [, y]) => s + (y - my) ** 2, 0));
      rows.push(zh
        ? [`${name.zh}：${stepName(b.type, l)} 减 ${stepName(a.type, l)}`, `n ${d.n}，平均差 ${show(d.mean)}，标准差 ${show(d.sd)}，范围 ${show(d.min)} 至 ${show(d.max)}，相关系数 r ${num(r)}`]
        : [`${name.en}: ${stepName(b.type, l)} minus ${stepName(a.type, l)}`, `n ${d.n}, mean difference ${show(d.mean)}, SD ${show(d.sd)}, range ${show(d.min)} to ${show(d.max)}, correlation r ${num(r)}`]);
    }
  }
  if (rows.length) groups.push({ title: zh ? '步骤之间的比较' : 'Comparison between steps', rows });
  return { scope: 'folder', language: l, groups: groups.filter((g) => g.rows.length), terms: [...terms] };
}
