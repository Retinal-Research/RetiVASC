// The Results table step: its tables in the node (a preview) and in the inspector (all of it), with
// Download CSV and Copy (tab-separated, for pasting into Excel).

import { downloadCsv } from '../engine/table';
import type { Table, TableValue } from '../engine/values';

/** On screen, long numbers to 6 significant digits (the full value in the tooltip and in the CSV). */
function Cell({ v }: { v: string }) {
  const x = Number(v);
  if (v === '' || !Number.isFinite(x) || v.replace(/[^0-9]/g, '').length <= 7) return <>{v}</>;
  return <span title={v}>{String(Number(x.toPrecision(6)))}</span>;
}

const tsv = (t: Table) => [t.columns, ...t.rows].map((r) => r.join('\t')).join('\n');

function Grid({ t, rows, cols }: { t: Table; rows?: number; cols?: number }) {
  // one photo: the columns read better as a list of name and value
  if (t.rows.length === 1 && t.columns.length > 2) {
    const pairs = t.columns.map((c, i) => [c, t.rows[0][i]] as const).filter(([, v]) => v !== '').slice(0, rows);
    return (
      <table className="rtable pairs">
        <tbody>{pairs.map(([c, v]) => <tr key={c}><th>{c}</th><td><Cell v={v} /></td></tr>)}</tbody>
      </table>
    );
  }
  const cs = t.columns.slice(0, cols), rs = t.rows.slice(0, rows);
  return (
    <table className="rtable">
      <thead><tr>{cs.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
      <tbody>{rs.map((r, i) => <tr key={i}>{r.slice(0, cols).map((c, j) => <td key={j}><Cell v={c} /></td>)}</tr>)}</tbody>
    </table>
  );
}

const size = (t: Table) => `${t.rows.length} row${t.rows.length === 1 ? '' : 's'} × ${t.columns.length} columns`;
const label = (t: Table, i: number) => (i ? t.title.replace(/[^\w]+/g, '_').slice(0, 30) : '');

export function TableView({ value, workflow, full }: { value: TableValue; workflow: string; full?: boolean }) {
  if (!value.tables.length) return <div className="empty">No numbers from the steps connected to it: they failed or have none.</div>;
  return (
    <div className="tables">
      {value.tables.map((t, i) => (
        <section key={i} className="tbl">
          <div className="tbl-head">
            <span className="tbl-title">{t.title}</span>
            <span className="faint">{size(t)}{t.note ? ` · ${t.note}` : ''}</span>
          </div>
          <div className={`tbl-body${full ? ' full' : ''}`}>
            <Grid t={t} rows={full ? undefined : 6} cols={full ? undefined : 5} />
          </div>
          <div className="row">
            <button className="btn small" onClick={() => downloadCsv(t, value.name, workflow, label(t, i))}>Download CSV</button>
            {full && <button className="btn small" onClick={() => void navigator.clipboard?.writeText(tsv(t))} title="Tab-separated: paste into Excel">Copy</button>}
          </div>
        </section>
      ))}
    </div>
  );
}

