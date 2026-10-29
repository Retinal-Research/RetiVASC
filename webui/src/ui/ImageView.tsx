import { useEffect, useRef, useState } from 'react';
import { Grip } from './Grip';
import { fit, paint } from './render';
import { isNum, useRemembered } from '../remember';

const HEIGHT = 360;

/** A rectangle in image fractions (0..1 of width and height). */
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  base?: ImageBitmap;
  layers: HTMLCanvasElement[];
  opacity: number;
  /** shown instead of the default hint when there is no image */
  empty?: string;
  /** zoom to this part of the image; a new object re-applies it */
  focus?: Rect;
  /** outlines drawn on top, e.g. the selected region */
  boxes?: Array<Rect & { color: string }>;
}

/** Large preview with wheel zoom (around the cursor), drag to pan, double-click to reset. */
export function ImageView({ base, layers, opacity, empty, focus, boxes = [] }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [height, setHeight] = useRemembered('image-height', HEIGHT, isNum);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => setView({ zoom: 1, x: 0, y: 0 }), [base]);

  useEffect(() => {
    if (!focus || !base || !size.w) return;
    const r = fit(base.width, base.height, size.w, size.h);
    const zoom = Math.min(16, Math.max(1, 0.55 * Math.min(size.w / Math.max(focus.w * r.w, 1), size.h / Math.max(focus.h * r.h, 1))));
    const cx = r.x + (focus.x + focus.w / 2) * r.w;
    const cy = r.y + (focus.y + focus.h / 2) * r.h;
    setView({ zoom, x: size.w / 2 - cx * zoom, y: size.h / 2 - cy * zoom });
  }, [focus, base, size]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(size.w * dpr);
    c.height = Math.round(size.h * dpr);
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    if (!base) return;
    const r = fit(base.width, base.height, size.w, size.h);
    const placed = { x: r.x * view.zoom + view.x, y: r.y * view.zoom + view.y, w: r.w * view.zoom, h: r.h * view.zoom };
    paint(ctx, base, layers, placed, opacity);
    for (const b of boxes) {
      const pad = 4;
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(placed.x + b.x * placed.w - pad, placed.y + b.y * placed.h - pad, b.w * placed.w + 2 * pad, b.h * placed.h + 2 * pad);
      ctx.setLineDash([]);
    }
  }, [base, layers, opacity, size, view, boxes]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setView((v) => {
        const zoom = Math.min(16, Math.max(1, v.zoom * Math.exp(-e.deltaY * 0.0015)));
        const k = zoom / v.zoom;
        // keep the point under the cursor fixed; the unzoomed origin is (0, 0)
        return zoom === 1 ? { zoom: 1, x: 0, y: 0 } : { zoom, x: mx - (mx - v.x) * k, y: my - (my - v.y) * k };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  return (
    <>
      <div
        ref={wrap}
        className="imageview"
        style={{ height }}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY };
          (e.target as Element).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const dx = e.clientX - drag.current.x;
          const dy = e.clientY - drag.current.y;
          drag.current = { x: e.clientX, y: e.clientY };
          setView((v) => (v.zoom === 1 ? v : { ...v, x: v.x + dx, y: v.y + dy }));
        }}
        onPointerUp={() => (drag.current = null)}
        onDoubleClick={() => setView({ zoom: 1, x: 0, y: 0 })}
      >
        <canvas ref={canvas} />
        {!base && <div className="imageview-empty">{empty ?? 'Nothing to show yet — run the workflow'}</div>}
        {base && <div className="imageview-hint">{view.zoom > 1 ? `${view.zoom.toFixed(1)}× · double-click to reset` : 'Scroll to zoom'}</div>}
      </div>
      <Grip axis="y" sign={1} value={height} min={160} max={() => Math.round(window.innerHeight * 0.85)} onChange={setHeight}
            onReset={() => setHeight(HEIGHT)} label="Drag to change the height of the image" />
    </>
  );
}
