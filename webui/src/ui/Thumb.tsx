import { useEffect, useRef } from 'react';
import { fit, paint, type Placement } from './render';

interface Props {
  base?: ImageBitmap;
  layers?: HTMLCanvasElement[];
  opacity?: number;
  /** region of the base image to show, in base-image pixels */
  crop?: Placement;
  width: number;
  height: number;
}

/** Small canvas preview: photo plus layers, letterboxed. */
export function Thumb({ base, layers = [], opacity = 1, crop, width, height }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!base) return;
    if (!crop) {
      paint(ctx, base, layers, fit(base.width, base.height, width, height), opacity);
      return;
    }
    // zoomed view: scale so that `crop` fills the box, clip to the box
    const box = fit(crop.w, crop.h, width, height);
    const s = box.w / crop.w;
    const rect = { x: box.x - crop.x * s, y: box.y - crop.y * s, w: base.width * s, h: base.height * s };
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.clip();
    paint(ctx, base, layers, rect, opacity);
    ctx.restore();
  }, [base, layers, opacity, crop, width, height]);
  return <canvas ref={ref} className="thumb" style={{ width, height }} />;
}
