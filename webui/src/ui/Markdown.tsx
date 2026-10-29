// The little Markdown a language model writes: headings, bullet and numbered lists, tables, bold,
// italics and code. Rendered as React elements (never as HTML), so the text cannot inject markup.

import type { ReactNode } from 'react';

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\*[^*\s][^*]*\*)/g;
  let last = 0, k = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('**') || t.startsWith('__')) out.push(<strong key={k++}>{t.slice(2, -2)}</strong>);
    else if (t.startsWith('`')) out.push(<code key={k++}>{t.slice(1, -1)}</code>);
    else out.push(<em key={k++}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const cells = (line: string) => line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

export function Markdown({ text }: { text: string }) {
  const lines = text.split('\n');
  const blocks: ReactNode[] = [];
  let i = 0, k = 0;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    if (!t) { i++; continue; }
    const h = /^(#{1,6})\s+(.*)$/.exec(t);
    if (h) {
      const level = Math.min(h[1].length + 1, 5);
      const Tag = `h${level}` as 'h2' | 'h3' | 'h4' | 'h5';
      blocks.push(<Tag key={k++}>{inline(h[2].replace(/#+$/, '').trim())}</Tag>);
      i++;
    } else if (/^([-*•]|\d+[.)])\s+/.test(t)) {
      const ordered = /^\d/.test(t);
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
        items.push(<li key={items.length}>{inline(lines[i].trim().replace(/^([-*•]|\d+[.)])\s+/, ''))}</li>);
        i++;
      }
      blocks.push(ordered ? <ol key={k++}>{items}</ol> : <ul key={k++}>{items}</ul>);
    } else if (t.startsWith('|')) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        if (!/^\|?\s*:?-{2,}/.test(lines[i].trim())) rows.push(cells(lines[i]));
        i++;
      }
      const [head, ...body] = rows;
      blocks.push(
        <table key={k++} className="md-table">
          <thead><tr>{head.map((c, j) => <th key={j}>{inline(c)}</th>)}</tr></thead>
          <tbody>{body.map((r, j) => <tr key={j}>{r.map((c, n) => <td key={n}>{inline(c)}</td>)}</tr>)}</tbody>
        </table>,
      );
    } else if (/^(-{3,}|\*{3,})$/.test(t)) {
      blocks.push(<hr key={k++} />);
      i++;
    } else {
      const para: string[] = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|[-*•]\s|\d+[.)]\s|\|)/.test(lines[i].trim())) para.push(lines[i++].trim());
      if (!para.length) para.push(lines[i++].trim());  // never stall on a line no rule takes
      blocks.push(<p key={k++}>{inline(para.join(' '))}</p>);
    }
  }
  return <div className="md">{blocks}</div>;
}
