// Server steps: they send photos to the RetiVASC server, which runs the full AutoMorph v2
// (every ensemble member, M0–M3) on its GPU and keeps the results there. Each step is a job
// in the server's queue; the canvas shows its progress and results. Without a server these
// steps say so; the browser steps keep working.

import { api, getPicked, upload, waitJob } from '../server/api';
import { readSource } from './sources';
import type { RunContext } from './specs';
import type { NodeValue, SheetSummary } from './values';

const text = (v: unknown) => String(v ?? '').trim();

function report(ctx: RunContext, step: string, fraction?: number | null) {
  ctx.progress({ stage: 'server', message: step, fraction: fraction ?? undefined });
}

/** One photo -> full AutoMorph v2 on the server. */
export async function serverAnalyze(ctx: RunContext): Promise<Record<string, NodeValue>> {
  const src = ctx.inputs.image;
  if (src?.kind !== 'fundus') throw new Error('connect a fundus photo');
  const { name, bytes } = await readSource(src.source);
  const form = new FormData();
  form.append('files', new File([bytes], name));
  form.append('device', text(ctx.params.device));
  report(ctx, 'Sending the photo', 0);
  const job = await upload('/jobs/analyze', form, (f) => report(ctx, 'Sending the photo', f * 0.05));
  const done = await waitJob(job.id, (p) => report(ctx, p.step, p.fraction));
  const photo = done.result.photos?.[0];
  if (!photo) throw new Error('the server returned no result for this photo');
  return { analysis: { kind: 'analysis', jobId: job.id, photo } };
}

/** Which photos a study sheet uses. Nothing is sent yet; the sheet step sends them. */
export async function studyPhotos(ctx: RunContext): Promise<Record<string, NodeValue>> {
  const source = text(ctx.params.source) || 'upload';
  if (source === 'upload') {
    const files = getPicked(text(ctx.params.pickId));
    if (!files?.length) throw new Error('choose the photo folder');
    return { run: { kind: 'run', source, pickId: text(ctx.params.pickId), photos: files.length, label: text(ctx.params.pickLabel) } };
  }
  if (source === 'server') {
    const folder = text(ctx.params.folder);
    if (!folder) throw new Error('choose a folder on the server');
    return { run: { kind: 'run', source, folder, photos: 0, label: folder } };
  }
  const reuseJob = text(ctx.params.reuseJob);
  if (!reuseJob) throw new Error('choose an earlier run');
  return { run: { kind: 'run', source: 'reuse', reuseJob, photos: 0, label: text(ctx.params.reuseLabel) || reuseJob } };
}

/** Photos + a sheet template -> AutoMorph v2 on the server -> filled sheet, audited. */
export async function studySheet(ctx: RunContext): Promise<Record<string, NodeValue>> {
  const run = ctx.inputs.run;
  if (run?.kind !== 'run') throw new Error('connect the study photos');
  const template = getPicked(text(ctx.params.templateId))?.[0];
  if (!template) throw new Error("choose the sheet template (.xlsx)");
  const outName = text(ctx.params.outName);
  if (!outName.endsWith('.xlsx')) throw new Error('the output file name must end in .xlsx');
  const form = new FormData();
  form.append('template', template);
  const previous = getPicked(text(ctx.params.previousId))?.[0];
  if (previous) form.append('previous', previous);
  form.append('out_name', outName);
  form.append('strict', ctx.params.strict ? 'true' : 'false');
  form.append('device', text(ctx.params.device));
  if (run.source === 'upload') {
    for (const f of getPicked(run.pickId) ?? []) form.append('files', f, f.webkitRelativePath || f.name);
  } else if (run.source === 'server') {
    form.append('server_folder', run.folder ?? '');
  } else {
    form.append('reuse_job', run.reuseJob ?? '');
  }
  report(ctx, 'Sending photos and sheet', 0);
  const job = await upload('/jobs/sheet', form, (f) => report(ctx, `Sending photos and sheet (${Math.round(f * 100)}%)`, f * 0.1));
  const done = await waitJob(job.id, (p) => report(ctx, p.step, p.fraction));
  const r = done.result;
  return {
    sheet: {
      kind: 'sheet', jobId: job.id, result: r.deliverable ? 'deliverable' : 'NOT deliverable (see the audit report)',
      files: (r.files ?? []).map((name) => ({ name, url: api.fileUrl(job.id, name) })),
      summary: r.summary as SheetSummary | undefined,
    },
  };
}
