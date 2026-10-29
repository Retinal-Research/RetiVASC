import { useEffect, useState } from 'react';
import { GROUP_LABEL, SPEC_LIST, type NodeGroup } from '../engine/specs';
import { WORKFLOWS, type WorkflowFile } from '../engine/workflows';
import { api } from '../server/api';
import { JobsPanel } from './JobsPanel';

export const DRAG_TYPE = 'application/x-retivasc-node';

interface Props {
  current: string;
  onLoad: (wf: WorkflowFile) => void;
  onAdd: (type: string) => void;
  /** signed in to a RetiVASC server: show its steps and the user's jobs */
  server: boolean;
}

const GROUPS: NodeGroup[] = ['input', 'm0', 'enhance', 'm1', 'm2', 'm3', 'vascx', 'view', 'server', 'study'];

// The study sheet is an optional server module; its steps are shown only where it is installed.
function useSheet(server: boolean): boolean {
  const [sheet, setSheet] = useState(false);
  useEffect(() => {
    if (server) api.server().then((i) => setSheet(!!i.features?.sheet), () => setSheet(false));
  }, [server]);
  return sheet;
}

export function Sidebar({ current, onLoad, onAdd, server }: Props) {
  const sheet = useSheet(server);
  const groupShown = (g: NodeGroup) => (server || (g !== 'server' && g !== 'study')) && (sheet || g !== 'study');
  const needsServer = (wf: WorkflowFile) => wf.nodes.some((n) => n.type.startsWith('server.') || n.type.startsWith('study.'));
  const shown = (wf: WorkflowFile) =>
    (server || !needsServer(wf)) && wf.nodes.every((n) => groupShown(SPEC_LIST.find((s) => s.type === n.type)?.group ?? 'input'));
  const entry = (wf: WorkflowFile) => (
    <button key={wf.name} className={`wf ${current === wf.name ? 'active' : ''}`} onClick={() => onLoad(wf)}>
      <span className="wf-name">{wf.name}</span>
      <span className="wf-desc">{wf.description}</span>
    </button>
  );
  const listed = WORKFLOWS.filter(shown);
  const singles = listed.filter((wf) => wf.section === 'single');
  const onServer = listed.filter((wf) => wf.section === 'server');
  return (
    <aside className="sidebar">
      <section>
        <h3>Pipeline library</h3>
        {listed.filter((wf) => wf.section === 'pipeline' || !wf.section).map(entry)}
        {singles.length > 0 && <h4 className="wf-sub">Single steps</h4>}
        {singles.map(entry)}
        {onServer.length > 0 && <h4 className="wf-sub">On the RetiVASC server</h4>}
        {onServer.map(entry)}
      </section>
      <section>
        <h3>Skills</h3>
        <p className="faint small">Each skill is one step. Drag it onto the canvas, or click to add.</p>
        {GROUPS.filter(groupShown).map((g) => (
          <div key={g} className="lib-group">
            <div className={`lib-label g-${g}`}>{GROUP_LABEL[g]}</div>
            {SPEC_LIST.filter((s) => s.group === g).map((s) => (
              <button
                key={s.type}
                className={`lib-item g-${g}`}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(DRAG_TYPE, s.type);
                  e.dataTransfer.effectAllowed = 'move';
                }}
                onClick={() => onAdd(s.type)}
              >
                <span className="kdot" />
                <span>
                  <span className="lib-title">{s.title}</span>
                  <span className="lib-sub">{s.subtitle}</span>
                </span>
              </button>
            ))}
          </div>
        ))}
      </section>
      {server && <JobsPanel />}
    </aside>
  );
}
