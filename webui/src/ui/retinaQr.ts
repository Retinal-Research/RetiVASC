// RetiVASC's QR code, drawn as a retina (the design in _todo/retina-qr/final, chosen by Hao): a standard QR
// code with error correction H, round dots fading from the theme's deep rose to a mist grey, three flat
// "eyes" for the finder patterns, and in the middle a pane of frosted glass floating over the code with an
// R drawn as vessels (the optic disc in coral). The pane covers about 8% of the code, which H recovers.
// Checked with WeChat's, ZXing's and OpenCV's readers (see _todo/retina-qr/check.py). Also the plain black
// and white code, for print.

import qrcode from 'qrcode-generator';

const INK = '#9f1239', DOT_FROM = '#8f1538', DOT_TO = '#8391a5', DISC = '#f9735b';

function modules(text: string) {
  const q = qrcode(0, 'H');
  q.addData(text);
  q.make();
  return { q, n: q.getModuleCount() };
}

/** The retina QR code as SVG markup; `id` keeps its gradients apart from another one on the same page. */
export function retinaQrSvg(text: string, id: string): string {
  const { q, n } = modules(text);
  const pad = 4, margin = 3, size = n + 2 * pad, full = size + 2 * margin, mid = n / 2;
  const finder = (x: number, y: number) => (x < 7 && y < 7) || (x >= n - 7 && y < 7) || (x < 7 && y >= n - 7);
  const X = margin + pad, cx = X + mid, cy = X + mid, chip = 5.6, r = chip * 0.62;
  const dots: string[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    if (!q.isDark(y, x) || finder(x, y)) continue;
    dots.push(`M${X + x + 0.5} ${X + y + 0.5}m-0.45 0a0.45 0.45 0 1 0 0.9 0a0.45 0.45 0 1 0 -0.9 0`);
  }
  const P = (x: number, y: number) => `${(cx + x * r).toFixed(2)} ${(cy + y * r).toFixed(2)}`;
  const o: string[] = [];
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${full} ${full}" role="img" aria-label="QR code: ${text}">`);
  o.push(`<defs>
    <filter id="${id}-lift" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="1.2" stdDeviation="1.6" flood-color="#4c0519" flood-opacity="0.14"/></filter>
    <filter id="${id}-soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="0.9"/></filter>
    <filter id="${id}-frost" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="0.6"/></filter>
    <linearGradient id="${id}-dots" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${DOT_FROM}"/><stop offset="1" stop-color="${DOT_TO}"/></linearGradient>
    <linearGradient id="${id}-sheen" x1="0.15" y1="0.1" x2="0.85" y2="0.9"><stop offset="0" stop-color="#ffffff" stop-opacity="0.87"/><stop offset="0.6" stop-color="#ffffff" stop-opacity="0.62"/><stop offset="1" stop-color="#ffe4e6" stop-opacity="0.53"/></linearGradient>
    <linearGradient id="${id}-rim" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#ffffff" stop-opacity="0.3"/></linearGradient>
    <clipPath id="${id}-pane"><circle cx="${cx}" cy="${cy}" r="${chip}"/></clipPath>
  </defs>`);
  o.push(`<rect x="${margin}" y="${margin}" width="${size}" height="${size}" rx="${size * 0.12}" fill="#ffffff" filter="url(#${id}-lift)"/>`);
  o.push(`<path id="${id}-d" d="${dots.join('')}" fill="url(#${id}-dots)"/>`);
  for (const [fx, fy] of [[0, 0], [n - 7, 0], [0, n - 7]]) {
    const ex = X + fx + 3.5, ey = X + fy + 3.5;
    o.push(`<circle cx="${ex}" cy="${ey}" r="3" fill="none" stroke="${INK}" stroke-width="1"/>`);
    o.push(`<circle cx="${ex}" cy="${ey}" r="1.5" fill="${INK}"/><circle cx="${ex - 0.5}" cy="${ey - 0.5}" r="0.28" fill="#ffffff" opacity="0.85"/>`);
  }
  o.push(`<g class="qr-chip">`);
  o.push(`<circle cx="${cx}" cy="${cy + 0.8}" r="${chip - 0.2}" fill="#4c0519" opacity="0.18" filter="url(#${id}-soft)"/>`);
  o.push(`<g clip-path="url(#${id}-pane)"><circle cx="${cx}" cy="${cy}" r="${chip}" fill="#ffffff"/><use href="#${id}-d" filter="url(#${id}-frost)"/></g>`);
  o.push(`<circle cx="${cx}" cy="${cy}" r="${chip}" fill="url(#${id}-sheen)"/>`);
  o.push(`<circle cx="${cx}" cy="${cy}" r="${chip - 0.08}" fill="none" stroke="url(#${id}-rim)" stroke-width="0.16"/>`);
  o.push(`<g fill="none" stroke="${INK}" stroke-linecap="round" stroke-linejoin="round">`);
  o.push(`<path d="M${P(-0.42, 0.82)}C${P(-0.47, 0.3)} ${P(-0.38, -0.3)} ${P(-0.42, -0.78)}" stroke-width="${(r * 0.19).toFixed(2)}"/>`);
  o.push(`<path d="M${P(-0.42, -0.78)}C${P(0.05, -0.86)} ${P(0.52, -0.72)} ${P(0.5, -0.38)}C${P(0.48, -0.06)} ${P(0.1, 0)} ${P(-0.3, -0.02)}" stroke-width="${(r * 0.15).toFixed(2)}"/>`);
  o.push(`<path d="M${P(-0.05, 0)}C${P(0.2, 0.2)} ${P(0.35, 0.5)} ${P(0.55, 0.82)}" stroke-width="${(r * 0.13).toFixed(2)}"/>`);
  o.push(`</g><circle cx="${(cx - 0.42 * r).toFixed(2)}" cy="${(cy - 0.02 * r).toFixed(2)}" r="${(r * 0.16).toFixed(2)}" fill="${DISC}"/>`);
  o.push(`</g></svg>`);
  return o.join('');
}

/** The plain code: black squares on white with a quiet zone of four, for print. */
export function plainQrSvg(text: string): string {
  const { q, n } = modules(text);
  const size = n + 8;
  let d = '';
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (q.isDark(y, x)) d += `M${x + 4} ${y + 4}h1v1h-1z`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" shape-rendering="crispEdges"><rect width="${size}" height="${size}" fill="#fff"/><path d="${d}" fill="#000"/></svg>`;
}
