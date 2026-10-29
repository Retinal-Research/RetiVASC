// A drag handle that resizes a panel: the inspector by its left edge, the image by its lower edge.
// Arrow keys work too; double-click puts the default size back.

import { useRef, type PointerEvent } from 'react';

interface Props {
  axis: 'x' | 'y';
  /** 1: dragging right or down makes it bigger; -1: dragging left or up does */
  sign: 1 | -1;
  value: number;
  min: number;
  /** the largest size right now (it depends on the window) */
  max: () => number;
  onChange: (v: number) => void;
  onReset: () => void;
  label: string;
}

export function Grip({ axis, sign, value, min, max, onChange, onReset, label }: Props) {
  const start = useRef<{ pos: number; value: number } | null>(null);
  const pos = (e: PointerEvent) => (axis === 'x' ? e.clientX : e.clientY);
  const set = (v: number) => onChange(Math.round(Math.min(Math.max(v, min), Math.max(min, max()))));
  return (
    <div
      className={`grip grip-${axis}`}
      role="separator"
      tabIndex={0}
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      aria-label={label}
      aria-valuenow={value}
      title={`${label} · double-click to reset`}
      onPointerDown={(e) => {
        start.current = { pos: pos(e), value };
        e.currentTarget.setPointerCapture(e.pointerId);
        e.preventDefault();
      }}
      onPointerMove={(e) => {
        if (start.current) set(start.current.value + sign * (pos(e) - start.current.pos));
      }}
      onPointerUp={() => { start.current = null; }}
      onPointerCancel={() => { start.current = null; }}
      onDoubleClick={onReset}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 64 : 16;
        const bigger = axis === 'x' ? (sign > 0 ? 'ArrowRight' : 'ArrowLeft') : (sign > 0 ? 'ArrowDown' : 'ArrowUp');
        const smaller = axis === 'x' ? (sign > 0 ? 'ArrowLeft' : 'ArrowRight') : (sign > 0 ? 'ArrowUp' : 'ArrowDown');
        if (e.key === bigger) set(value + step);
        else if (e.key === smaller) set(value - step);
        else return;
        e.preventDefault();
      }}
    />
  );
}
