// The device this browser computes on. A web page cannot list the graphics cards: it can ask for the
// faster one or the power-saving one, or leave the card alone and use the CPU.

import { useEffect, useRef, useState } from 'react';
import type { EngineInfo, GpuChoice } from '../runtime/protocol';

const OPTIONS: Array<{ gpu: GpuChoice; title: string; text: string }> = [
  { gpu: 'high-performance', title: 'Faster graphics card', text: 'The dedicated card, when the computer has one. The default.' },
  { gpu: 'low-power', title: 'Power-saving graphics card', text: 'Usually the graphics built into the processor: slower, but cooler and easier on a battery.' },
  { gpu: 'cpu', title: 'CPU only', text: 'Leaves the graphics card alone. Much slower.' },
];

interface Props {
  engine?: EngineInfo;
  engineError?: string;
  /** what this browser asked for (the engine may still be starting) */
  gpu: GpuChoice;
  running: boolean;
  onGpu: (gpu: GpuChoice) => void;
}

export function ComputeMenu({ engine, engineError, gpu, running, onGpu }: Props) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  let state = 'wait', label = 'Starting engine…', title = 'Choose the device to compute on';
  if (engineError) { state = 'bad'; label = 'Engine failed to start'; title = engineError; }
  else if (engine?.backend === 'webgpu') { state = 'ok'; label = `WebGPU · ${engine.adapter}`; }
  else if (engine) {
    state = 'warn';
    label = `CPU only · ${engine.threads} thread${engine.threads === 1 ? '' : 's'}`;
    if (engine.gpu !== 'cpu') title = 'No WebGPU in this browser; models run on the CPU and are much slower';
  }

  return (
    <div className="compute" ref={box}>
      <button className={`engine ${state}`} onClick={() => setOpen((o) => !o)} title={title} aria-expanded={open}>
        <span className="sdot" />
        <span className="engine-label">{label}</span>
        <span className="caret" aria-hidden="true" />
      </button>
      {open && (
        <div className="popover compute-pop" role="dialog" aria-label="Compute device">
          <div className="pop-label">Compute on this computer</div>
          {OPTIONS.map((o) => (
            <button key={o.gpu} className={`choice ${gpu === o.gpu ? 'on' : ''}`} disabled={running}
                    onClick={() => { setOpen(false); if (o.gpu !== gpu) onGpu(o.gpu); }}>
              <span className="radio" aria-hidden="true" />
              <span>
                <span className="choice-title">{o.title}</span>
                <span className="choice-text">{o.text}</span>
              </span>
            </button>
          ))}
          {running && <p className="pop-note warn">Stop the run to change the device.</p>}
          <p className="pop-note">
            Changing the device restarts the engine; run the workflow again afterwards. With two cards of the same kind, the
            operating system decides which one the browser gets (on Windows: Settings → System → Display → Graphics). Steps on the
            RetiVASC server choose their card on the step itself.
          </p>
        </div>
      )}
    </div>
  );
}
