// The ▶ and ⋯ at the top right of every step: run this step alone, or the menu (duplicate,
// disconnect, delete).

import { useEffect, useRef, useState } from 'react';
import { useStudio } from './context';

export function NodeMenu({ id }: { id: string }) {
  const studio = useStudio();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey);
    return () => { window.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey); };
  }, [open]);

  const item = (label: string, act: () => void, danger = false) => (
    <button className={`nmenu-item ${danger ? 'danger' : ''}`} disabled={studio.running} onClick={() => { setOpen(false); act(); }}>{label}</button>
  );
  return (
    <div className="ntools nodrag" ref={box}>
      <button className="ntool" disabled={studio.running} onClick={() => studio.runStep(id)}
              title="Run this step (and what it needs before it); the steps after it wait" aria-label="Run this step">
        <span className="play" />
      </button>
      <button className="ntool" onClick={() => setOpen((o) => !o)} title="More" aria-label="More" aria-expanded={open}>⋯</button>
      {open && (
        <div className="nmenu" role="menu">
          {item('Run this step', () => studio.runStep(id))}
          {item('Duplicate', () => studio.duplicate(id))}
          {item('Disconnect inputs', () => studio.disconnect(id))}
          {item('Delete', () => studio.remove(id), true)}
        </div>
      )}
    </div>
  );
}
