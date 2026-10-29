// Sharing the site: its QR code (for a slide or a poster: phones land on the home page), and links to
// send to oneself. RetiVASC runs on a computer's graphics card, so a phone is told so and offered ways to
// pass the link on, instead of opening the canvas.

import qrcode from 'qrcode-generator';
import { useEffect, useMemo, useState } from 'react';

/** The address of the home page of this site. */
export const siteUrl = () => `${window.location.origin}${import.meta.env.BASE_URL}`;

/** A phone or a small tablet: a coarse pointer on a small screen. */
export function isPhone(): boolean {
  const small = Math.min(window.screen.width, window.screen.height) < 820;
  return small && (window.matchMedia?.('(pointer: coarse)').matches || /Mobi|Android|iPhone/i.test(navigator.userAgent));
}

/** The QR code as an SVG path: crisp at any size, in the page's colours. */
export function QrCode({ text, size = 160 }: { text: string; size?: number }) {
  const { d, n } = useMemo(() => {
    const q = qrcode(0, 'M');
    q.addData(text);
    q.make();
    const n = q.getModuleCount();
    let d = '';
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (q.isDark(y, x)) d += `M${x + 4} ${y + 4}h1v1h-1z`;
    return { d, n };
  }, [text]);
  return (
    <svg className="qr" width={size} height={size} viewBox={`0 0 ${n + 8} ${n + 8}`} role="img" aria-label={`QR code for ${text}`} shapeRendering="crispEdges">
      <rect width={n + 8} height={n + 8} fill="#ffffff" />
      <path d={d} fill="#0b1017" />
    </svg>
  );
}

/** Copy, e-mail to oneself, the phone's own share sheet. */
export function ShareButtons({ compact }: { compact?: boolean }) {
  const url = siteUrl();
  const [copied, setCopied] = useState(false);
  useEffect(() => { if (!copied) return; const t = setTimeout(() => setCopied(false), 1800); return () => clearTimeout(t); }, [copied]);
  const mail = `mailto:?subject=${encodeURIComponent('RetiVASC: open on a computer')}&body=${encodeURIComponent(`RetiVASC, retinal vessel analysis in the browser (Geometry Systems Lab, ASU):\n${url}\n\nOpen it in Chrome or Edge on a computer.`)}`;
  return (
    <div className={`share-buttons${compact ? ' compact' : ''}`}>
      {typeof navigator.share === 'function' && (
        <button className="btn primary" onClick={() => void navigator.share({ title: 'RetiVASC', text: 'Retinal vessel analysis in the browser', url }).catch(() => {})}>Share…</button>
      )}
      <a className="btn" href={mail}>E-mail me the link</a>
      <button className="btn" onClick={() => void navigator.clipboard?.writeText(url).then(() => setCopied(true))}>{copied ? 'Copied' : 'Copy link'}</button>
    </div>
  );
}

/** The QR code at full screen, for a talk or a poster. */
export function QrOverlay({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClose]);
  const url = siteUrl();
  return (
    <div className="qr-overlay" role="dialog" aria-label="QR code of RetiVASC" onClick={onClose}>
      <div className="qr-card" onClick={(e) => e.stopPropagation()}>
        <h2>Scan to open RetiVASC</h2>
        <QrCode text={url} size={Math.min(420, window.innerHeight - 260, window.innerWidth - 80)} />
        <p className="qr-url">{url.replace(/^https?:\/\//, '')}</p>
        <p className="faint">On a phone you can read about it and send yourself the link; the analysis runs on a computer.</p>
        <button className="btn" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
