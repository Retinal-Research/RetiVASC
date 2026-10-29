// The AI report in the inspector: the text, follow-up questions, the numbers it was given, and the
// model that wrote it; while it is written, the steps so far; and the visitor's quota for today.

import { useEffect, useState } from 'react';
import { factsText } from '../engine/prompt';
import { askReport, quota, stopWriting, type Quota, type ReportValue } from '../engine/report';
import { ms } from './format';
import { Markdown } from './Markdown';

const EFFORT_LABEL: Record<string, string> = { low: 'Low', medium: 'Medium', high: 'High' };
const seconds = (t: number) => `${(Math.max(0, t) / 1000).toFixed(1)} s`;

/** While the lab's model works: the steps so far and the time each took. The API sends only the answer,
 *  not the model's reasoning, so these are RetiVASC's own steps. */
export function ReportProgress({ since, text, effort, compact }: { since?: number; text?: string; effort: string; compact?: boolean }) {
  const [start] = useState(() => since ?? Date.now());
  const [now, setNow] = useState(() => Date.now());
  const [wrote, setWrote] = useState<number>();
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (text && wrote === undefined) setWrote(Date.now());
  }, [text, wrote]);
  const thinking = !text;
  if (compact) {
    return (
      <div className="ai-thinking">
        <span className="ai-orb" aria-hidden="true" />
        <span className="shimmer">{thinking ? 'Thinking' : 'Writing'}</span>
        <span className="faint">{seconds(now - (thinking ? start : wrote ?? now))}</span>
      </div>
    );
  }
  return (
    <ol className="ai-steps" aria-live="polite">
      <li className="done">Results collected · only numbers are sent, never the photographs</li>
      <li className={thinking ? 'now' : 'done'}>
        {thinking ? <span className="shimmer">The lab’s model is thinking</span> : 'Thought'}
        <span className="faint"> · {EFFORT_LABEL[effort] ?? effort} reasoning · {seconds((wrote ?? now) - start)}</span>
      </li>
      <li className={thinking ? '' : 'now'}>
        {thinking ? 'Write the report' : <span className="shimmer">Writing the report</span>}
        {wrote !== undefined && <span className="faint"> · {seconds(now - wrote)}</span>}
      </li>
      <li>Check every number against the results</li>
    </ol>
  );
}

/** The visitor's quota for today, from the report service; again after every report or answer. */
export function QuotaBar({ refresh }: { refresh?: unknown }) {
  const [q, setQ] = useState<Quota>();
  useEffect(() => {
    let live = true;
    void quota().then((x) => live && setQ(x));
    return () => { live = false; };
  }, [refresh]);
  if (!q || !q.limit) return null;
  const share = Math.min(1, q.used / q.limit);
  // the service counts by UTC day
  const d = new Date(), reset = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1));
  const at = reset.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  return (
    <div className="quota" title={`$${q.used.toFixed(4)} of $${q.limit.toFixed(2)} of the lab’s API today, counted per network address. A report costs about $0.001.`}>
      <div className="quota-text">
        <span>Your AI quota today</span>
        <span>{!q.open ? 'the service is out of today’s budget' : share >= 1 ? 'used up' : `${share < 0.01 && share > 0 ? '<1' : Math.round(share * 100)}% used`}</span>
      </div>
      <div className="quota-bar"><span style={{ width: `${Math.max(share * 100, share > 0 ? 2 : 0)}%` }} /></div>
      <div className="quota-reset faint">Resets at {at}</div>
    </div>
  );
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** What the automatic check found: numbers the results do not contain, and judging words. */
function Checks({ unchecked, judged }: { unchecked: string[]; judged: string[] }) {
  if (!unchecked.length && !judged.length) return <p className="ai-ok">Every number in this text is in the results.</p>;
  return (
    <div className="ai-check">
      {unchecked.length > 0 && <div>Not in the results, check: {unchecked.join(', ')}</div>}
      {judged.length > 0 && <div>Judging words without a reference range: {judged.join(', ')}</div>}
    </div>
  );
}

export function ReportView({ value, workflow }: { value: ReportValue; workflow: string }) {
  const [, redraw] = useState(0);
  const [question, setQuestion] = useState('');
  const [answering, setAnswering] = useState<string>();
  const [error, setError] = useState<string>();
  const r = value.result;

  const ask = async () => {
    const q = question.trim();
    if (!q || answering !== undefined) return;
    setAnswering('');
    setError(undefined);
    try {
      value.qa.push({ q, ...(await askReport(value, q, setAnswering)) });
      setQuestion('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAnswering(undefined);
      redraw((n) => n + 1);
    }
  };

  const markdown = () => [
    `# AI report · ${workflow}`,
    '',
    value.text,
    ...value.qa.flatMap(({ q, a }) => ['', `**Q:** ${q}`, '', a]),
    '',
    `## The numbers the report is based on`,
    '',
    factsText(value.facts),
    '',
    '---',
    `Written by ${value.model}${value.writer === 'online' ? ' through the lab’s API' : ' on this computer'} (RetiVASC), ${new Date().toISOString().slice(0, 10)}. A language model can be wrong: check the text against the numbers. Research use only; not a diagnosis.`,
  ].join('\n');

  return (
    <div className="report">
      <div className="report-tools">
        <button className="btn small" onClick={() => void navigator.clipboard?.writeText(markdown())}>Copy</button>
        <button className="btn small" onClick={() => download(`${workflow.replace(/[^\w-]+/g, '_')}_report.md`, markdown())}>Download .md</button>
        {value.cut && <span className="hint">Stopped before the end</span>}
      </div>
      {value.writer === 'online' && <QuotaBar refresh={`${value.chat}:${value.qa.length}:${answering === undefined}`} />}
      <Markdown text={value.text} />
      <Checks unchecked={value.unchecked} judged={value.judged} />

      <section className="isec">
        <h3>Ask about these results</h3>
        {value.qa.map(({ q, a, unchecked, judged }, i) => (
          <div key={i} className="qa">
            <p className="qa-q">{q}</p>
            <Markdown text={a} />
            <Checks unchecked={unchecked} judged={judged} />
          </div>
        ))}
        {answering !== undefined && (
          <div className="qa">
            <p className="qa-q">{question}</p>
            {answering ? <Markdown text={answering} /> : <p className="faint">Reading…</p>}
          </div>
        )}
        <div className="ask">
          <textarea className="select" rows={2} value={question} placeholder="e.g. How do the arteries compare with the veins?"
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void ask(); } }} />
          {answering === undefined
            ? <button className="btn small" disabled={!question.trim()} onClick={() => void ask()}>Ask</button>
            : <button className="btn small danger" onClick={() => stopWriting()}><span className="stopsq" />Stop</button>}
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="isec">
        <details open>
          <summary>The numbers the report is based on</summary>
          {value.facts.groups.map((g) => (
            <div key={g.title} className="facts">
              <div className="facts-title">{g.title}{g.note && <span> · {g.note}</span>}</div>
              <dl className="rows">
                {g.rows.map(([k, v]) => <div key={k} className="kv"><dt>{k}</dt><dd>{v}</dd></div>)}
              </dl>
            </div>
          ))}
        </details>
      </section>

      <section className="isec">
        <h3>Model</h3>
        <dl className="rows">
          {value.writer === 'online' ? (
            <>
              <div className="kv"><dt>Language model</dt><dd>{value.model}, the lab’s API</dd></div>
              <div className="kv"><dt>Sent</dt><dd>the numbers below, through the RetiVASC server; no photographs</dd></div>
            </>
          ) : r && (
            <>
              <div className="kv"><dt>Language model</dt><dd>{value.model}, on this computer</dd></div>
              <div className="kv"><dt>Ran on</dt><dd>{r.backend === 'webgpu' ? 'GPU (WebGPU)' : 'CPU (WebAssembly)'}</dd></div>
              <div className="kv"><dt>Read</dt><dd>{r.promptTokens} tokens{r.reusedTokens ? ` (${r.reusedTokens} already read)` : ''} in {ms(r.readMs)}</dd></div>
              <div className="kv"><dt>Wrote</dt><dd>{r.newTokens} tokens, {r.tokensPerSecond.toFixed(1)} per second</dd></div>
            </>
          )}
          {value.thinkMs !== undefined && <div className="kv"><dt>Thought</dt><dd>{ms(value.thinkMs)} before its first word ({EFFORT_LABEL[value.effort]} reasoning)</dd></div>}
          <div className="kv"><dt>Time</dt><dd>{ms(value.ms)}</dd></div>
        </dl>
      </section>
      <p className="footnote">
        The numbers come from the steps connected to this one; the language model only writes them up, and every number in its
        text is looked up in them. It can still be wrong: check the text against the numbers. Research use only; not a diagnosis.
        {value.model}{value.license && `, ${value.license}`}.
      </p>
    </div>
  );
}
