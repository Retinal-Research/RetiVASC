// Visual summary of a built + audited results sheet, and the live progress while it builds.
// Compact form sits in the canvas node; the full form fills the inspector.

import type { CheckResult, SheetValue } from '../engine/values';

// what the server reports while a job runs, in order (matched by prefix)
export const SHEET_STEPS = ['Sending', 'Waiting in the queue', 'Preparing photos', 'Grading image quality', 'Segmenting', 'Measuring', 'Filling and auditing'];
export const ANALYSIS_STEPS = SHEET_STEPS.slice(0, 6);
const STEP_LABEL: Record<string, string> = {
  Sending: 'Sending to the server', Segmenting: 'Segmenting vessels, arteries/veins and disc', Measuring: 'Measuring 72 vascular features',
  'Filling and auditing': 'Filling and auditing the sheet',
};
const SEVERITIES: Array<{ key: CheckResult; label: string; hint: string }> = [
  { key: 'FAIL', label: 'Fail', hint: 'blocks delivery' },
  { key: 'BLOCKED', label: 'Waiting', hint: "needs a reviewer's answer" },
  { key: 'WARN', label: 'Note', hint: 'listed for the reviewer' },
  { key: 'DEFERRED', label: 'Deferred', hint: 'declared for later' },
];

export function SheetProgress({ message, fraction, steps = SHEET_STEPS }: { message?: string; fraction?: number; steps?: string[] }) {
  const step = Math.max(0, steps.findIndex((s) => message?.startsWith(s)));
  const f = Math.max(0.02, Math.min(1, fraction ?? 0.02));
  return (
    <div className="sprog">
      <div className="sprog-head">
        <span>{message ?? 'Starting'}</span>
        <b>{Math.round(f * 100)}%</b>
      </div>
      <div className="sprog-bar"><span style={{ width: `${f * 100}%` }} /></div>
      <ol className="sprog-steps">
        {steps.map((s, i) => (
          <li key={s} className={i < step ? 'done' : i === step ? 'now' : ''}>{STEP_LABEL[s] ?? s}</li>
        ))}
      </ol>
    </div>
  );
}

function Stack({ parts, total }: { parts: Array<{ n: number; label: string; cls: string }>; total: number }) {
  return (
    <>
      <div className="sstack">
        {parts.filter((p) => p.n > 0).map((p) => (
          <span key={p.label} className={p.cls} style={{ flexGrow: p.n }} title={`${p.label}: ${p.n}`} />
        ))}
      </div>
      <div className="skeys">
        {parts.map((p) => (
          <span key={p.label}><i className={p.cls} />{p.label} <b>{p.n.toLocaleString()}</b>{total ? <em> · {Math.round((p.n / total) * 100)}%</em> : null}</span>
        ))}
      </div>
    </>
  );
}

export function SheetSummaryView({ value, full = false }: { value: SheetValue; full?: boolean }) {
  const s = value.summary;
  const ok = value.result === 'deliverable';
  const b = s?.built ?? {};
  const slots = (b.file ?? 0) + (b.none ?? 0) + (b.blocked ?? 0);
  const fails = s?.checks.filter((c) => c.result === 'FAIL').length ?? 0;
  return (
    <div className={`ssum${full ? ' full' : ''}`}>
      <div className={`sverdict ${ok ? 'ok' : 'bad'}`}>
        <span className="sicon">{ok ? '✓' : '!'}</span>
        <div>
          <div>{ok ? 'Audit passed · deliverable' : `Audit: ${value.result}`}</div>
          {s && <small>{s.checks.length - fails} of {s.checks.length} checks without a failure</small>}
        </div>
      </div>
      {s && (
        <>
          <div className="schips">
            {SEVERITIES.map((v) => (
              <div key={v.key} className={`schip sev-${v.key}`} title={v.hint}>
                <b>{(s.counts[v.key] ?? 0).toLocaleString()}</b>
                <span>{v.label}</span>
              </div>
            ))}
          </div>
          <div className="ssec">
            <div className="slabel">Photo cells · {slots.toLocaleString()} ({slots / 4} subjects × 4 views)</div>
            <Stack total={slots} parts={[
              { n: b.file ?? 0, label: 'Filled', cls: 'c-file' },
              { n: b.none ?? 0, label: 'Not imaged', cls: 'c-none' },
              { n: b.blocked ?? 0, label: 'Waiting for reviewer', cls: 'c-blocked' },
            ]} />
          </div>
          <div className="ssec">
            <div className="slabel">Of the filled photos</div>
            <Stack total={b.file ?? 0} parts={[
              { n: b.gradable ?? 0, label: 'Gradable · 72 results', cls: 'c-grad' },
              { n: (b.file ?? 0) - (b.gradable ?? 0), label: 'Not gradable', cls: 'c-nograd' },
            ]} />
            {(b.picked ?? 0) > 0 && <div className="snote">{b.picked} picked among several photos (listed for the reviewer)</div>}
          </div>
          <div className="ssec">
            <div className="slabel">{s.checks.length} audit checks{full ? '' : ' · select the box for details'}</div>
            <div className="sgrid">
              {s.checks.map((c) => (
                <span key={c.id} className={`scell sev-${c.result}`} title={`${c.id} · ${c.result}${c.n ? ` (${c.n})` : ''}\n${c.rule}`}>{c.id}</span>
              ))}
            </div>
          </div>
          {full && Object.keys(s.changes).length > 0 && (
            <div className="ssec">
              <div className="slabel">Changes against the previous version</div>
              <div className="schanges">
                {Object.entries(s.changes).sort((x, y) => y[1] - x[1]).map(([k, n]) => {
                  const max = Math.max(...Object.values(s.changes));
                  return (
                    <div key={k} className="schg">
                      <span>{k}</span>
                      <span className="sbarh"><i style={{ width: `${(n / max) * 100}%` }} /></span>
                      <b>{n}</b>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {full && (
            <div className="ssec">
              <div className="slabel">Checks with findings</div>
              <table className="stable">
                <tbody>
                  {s.checks.filter((c) => c.n > 0).map((c) => (
                    <tr key={c.id}>
                      <td><span className={`scell sev-${c.result}`}>{c.id}</span></td>
                      <td>{c.n}</td>
                      <td>{c.rule}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      <div className="sfiles">
        {value.files.map(({ name, url }) => {
          const kind = name.endsWith('_for_doctor.xlsx') ? 'For the reviewer' : name.endsWith('.xlsx') ? 'Sheet'
            : name.endsWith('.csv') ? 'Change log' : name.endsWith('_audit.md') ? 'Audit report' : name.endsWith('_audit.json') ? 'Audit data' : 'Provenance';
          return (
            <a key={name} href={url} download className="sfile" title={name}>
              <span className="sfk">{kind}</span>
              <span className="sfn">{name}</span>
              <span className="sfd">↓</span>
            </a>
          );
        })}
      </div>
    </div>
  );
}
