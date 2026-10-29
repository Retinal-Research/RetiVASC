// The launch screen between the home page and the canvas. While it shows, the analysis engine starts (its
// worker, WebAssembly and the graphics card) and this computer is checked, so the canvas opens ready and
// people learn what their computer can do: WebGPU or the slower CPU, cores, memory, a phone. A few hints
// pass meanwhile. Nothing is sent anywhere: the checks stay in the browser.

import { useEffect, useState } from 'react';
import type { EngineInfo } from '../runtime/protocol';
import { Logo } from './Logo';
import { isPhone, ShareButtons } from './Share';

const HINTS = [
  'Photographs never leave your computer: every step runs in this browser.',
  'Click any box to see its result at full size on the right.',
  '▶ Run on a box runs just that step and the steps before it.',
  'The Results table step downloads every number as a CSV file.',
  'Connect anything to the AI report and ask it about the results.',
  'A folder of up to 1000 images runs one photograph at a time, and can be stopped and resumed.',
];
const MIN_MS = 1400;

function browserName(): string {
  const ua = navigator.userAgent;
  const m = /(Edg|Chrome|Firefox|Version)\/(\d+)/.exec(ua);
  if (!m) return 'this browser';
  const name = m[1] === 'Edg' ? 'Edge' : m[1] === 'Version' ? 'Safari' : m[1];
  return `${name} ${m[2]}`;
}

type Check = { label: string; value: string; state: 'ok' | 'warn' | 'wait' };

export function Launch({ engine, error, onDone }: { engine?: EngineInfo; error?: string; onDone: () => void }) {
  const [started] = useState(() => performance.now());
  const [hint, setHint] = useState(() => Math.floor(Math.random() * HINTS.length));
  const [now, setNow] = useState(() => performance.now());
  const phone = isPhone();
  const quick = navigator.webdriver;  // automated checks do not wait for the hints
  useEffect(() => {
    const t = setInterval(() => setNow(performance.now()), 200);
    const h = setInterval(() => setHint((i) => (i + 1) % HINTS.length), 2600);
    return () => { clearInterval(t); clearInterval(h); };
  }, []);
  const ready = !!engine || !!error;
  // ready, and shown long enough to read: fade out, then go
  const leaving = ready && !phone && (quick || now - started >= MIN_MS);
  useEffect(() => {
    if (!leaving) return;
    const t = setTimeout(onDone, quick ? 0 : 350);
    return () => clearTimeout(t);
  }, [leaving, quick, onDone]);

  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const checks: Check[] = [
    { label: 'Browser', value: browserName(), state: 'ok' },
    { label: 'Graphics card', value: error ? 'the engine did not start' : !engine ? ('gpu' in navigator ? 'starting…' : 'no WebGPU in this browser') : engine.backend === 'webgpu' ? `WebGPU · ${engine.adapter}` : 'not available: the CPU will be used, slowly', state: error || (engine && engine.backend !== 'webgpu') ? 'warn' : engine ? 'ok' : 'wait' },
    { label: 'Processor', value: `${navigator.hardwareConcurrency || '?'} cores${engine ? ` · ${engine.threads} thread${engine.threads > 1 ? 's' : ''} for the CPU fallback` : ''}`, state: 'ok' },
  ];
  if (memory) checks.push({ label: 'Memory', value: memory >= 8 ? `${memory} GB or more` : `about ${memory} GB: large folders may be slow`, state: memory >= 8 ? 'ok' : 'warn' });
  checks.push({ label: 'Models', value: 'downloaded the first time a step needs them, then kept in this browser', state: 'ok' });
  if (phone) checks.push({ label: 'Device', value: 'a phone: RetiVASC is made for a computer', state: 'warn' });

  const progress = Math.min(1, ready ? 1 : (now - started) / 4000);
  return (
    <div className={`launch${leaving ? ' leaving' : ''}`} role="status" aria-live="polite">
      <div className="launch-card">
        <Logo variant="color" size={64} />
        <h2>{ready ? (error ? 'The engine could not start' : 'Ready') : 'Getting the canvas ready'}</h2>
        <div className="launch-bar"><span style={{ width: `${Math.max(8, progress * 100)}%` }} /></div>
        <ul className="launch-checks">
          {checks.map((c) => (
            <li key={c.label} className={c.state}><span>{c.label}</span><span>{c.value}</span></li>
          ))}
        </ul>
        {error && <p className="error">{error}</p>}
        {phone ? (
          <div className="launch-phone">
            <p>Send yourself the link and open it on a computer, in Chrome or Edge:</p>
            <ShareButtons />
            <button className="linkbtn" onClick={onDone}>Continue on this phone</button>
          </div>
        ) : (
          <p className="launch-hint" key={hint}>{HINTS[hint]}</p>
        )}
        {!phone && !ready && <button className="linkbtn launch-skip" onClick={onDone}>Skip</button>}
      </div>
    </div>
  );
}
