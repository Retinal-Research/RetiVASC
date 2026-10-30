import { useEffect, useState, type ReactNode } from 'react';
import { SPEC_LIST, type NodeGroup } from '../engine/specs';
import { WORKFLOWS, type WorkflowFile } from '../engine/workflows';
import { api } from '../server/api';
import { JobsPanel } from './JobsPanel';
import { isBool, useRemembered } from '../remember';

export const DRAG_TYPE = 'application/x-retivasc-node';

interface Props {
  current: string;
  onLoad: (wf: WorkflowFile) => void;
  onAdd: (type: string) => void;
  /** signed in to a RetiVASC server: show its steps and the user's jobs */
  server: boolean;
  /** download the workflow on the canvas as a file */
  onSave: () => void;
  /** open a saved workflow file */
  onOpen: (file: File) => void;
  /** the library workflows changed from their default */
  edited: string[];
  /** put the workflow on the canvas back to its default */
  onReset: () => void;
}

/** Skills by section: the lab's own modules, then each pipeline's steps under one heading. */
const SECTIONS: Array<{ id: string; title: string; groups: NodeGroup[] }> = [
  { id: 'input', title: 'Input', groups: ['input'] },
  { id: 'retivasc', title: 'RetiVASC', groups: ['retivasc', 'ai'] },
  { id: 'automorph', title: 'AutoMorph V2', groups: ['m0', 'm1', 'm2', 'm3'] },
  { id: 'vascx', title: 'VascX', groups: ['vascx'] },
  { id: 'view', title: 'View', groups: ['view'] },
  { id: 'server', title: 'Server · full model', groups: ['server'] },
  { id: 'study', title: 'Study sheet', groups: ['study'] },
];

// The study sheet is an optional server module; its steps are shown only where it is installed.
function useSheet(server: boolean): boolean {
  const [sheet, setSheet] = useState(false);
  useEffect(() => {
    if (server) api.server().then((i) => setSheet(!!i.features?.sheet), () => setSheet(false));
  }, [server]);
  return sheet;
}

/** A section that folds away; the browser remembers which ones are open. */
function Fold({ id, title, count, open: initial = false, sub = false, children }:
  { id: string; title: string; count?: number; open?: boolean; sub?: boolean; children: ReactNode }) {
  const [open, setOpen] = useRemembered(`fold-${id}`, initial, isBool);
  return (
    <div className={`fold ${sub ? 'sub' : ''} ${open ? 'open' : ''}`}>
      <button className="fold-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="chev" aria-hidden="true" />
        <span>{title}</span>
        {count !== undefined && <span className="fold-count">{count}</span>}
      </button>
      {open && <div className="fold-body">{children}</div>}
    </div>
  );
}

export function Sidebar({ current, onLoad, onAdd, server, onSave, onOpen, edited, onReset }: Props) {
  const sheet = useSheet(server);
  const groupShown = (g: NodeGroup) => (server || (g !== 'server' && g !== 'study')) && (sheet || g !== 'study');
  const needsServer = (wf: WorkflowFile) => wf.nodes.some((n) => n.type.startsWith('server.') || n.type.startsWith('study.'));
  const shown = (wf: WorkflowFile) =>
    (server || !needsServer(wf)) && wf.nodes.every((n) => groupShown(SPEC_LIST.find((s) => s.type === n.type)?.group ?? 'input'));
  const entry = (wf: WorkflowFile) => {
    const changed = edited.includes(wf.name);
    return (
      <div key={wf.name} className="wf-wrap">
        <button className={`wf ${current === wf.name ? 'active' : ''}`} onClick={() => onLoad(wf)}>
          <span className="wf-name">{wf.name}{changed && <span className="wf-changed" title="Changed from the default">edited</span>}</span>
          <span className="wf-desc">{wf.description}</span>
        </button>
        {current === wf.name && (
          <button className="wf-reset" onClick={onReset} disabled={!changed}
                  title={changed ? 'Put this workflow back as it was: its steps, settings and connections' : 'This workflow is as it was'}>
            <span aria-hidden="true">↺</span> Reset
          </button>
        )}
      </div>
    );
  };
  const listed = WORKFLOWS.filter(shown);
  const singles = listed.filter((wf) => wf.section === 'single');
  const onServer = listed.filter((wf) => wf.section === 'server');
  const sections = SECTIONS.map((sec) => ({ ...sec, groups: sec.groups.filter(groupShown) })).filter((sec) => sec.groups.length);
  return (
    <aside className="sidebar">
      {/* the pipelines are what most people want, so they never fold away */}
      <section>
        <div className="side-head">
          <h3>Workflow</h3>
          <button className="btn side-file" onClick={onSave} title="Download this workflow as a file"><span aria-hidden="true">⤓</span>Save</button>
          <label className="btn side-file" title="Open a saved workflow file">
            <span aria-hidden="true">⤒</span>Open
            <input type="file" accept=".json,application/json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onOpen(f); e.target.value = ''; }} />
          </label>
        </div>
        {listed.filter((wf) => wf.section === 'pipeline' || !wf.section).map(entry)}
      </section>
      {singles.length > 0 && <Fold id="singles" title="Single steps" count={singles.length}>{singles.map(entry)}</Fold>}
      {onServer.length > 0 && <Fold id="server" title="On the RetiVASC server" count={onServer.length}>{onServer.map(entry)}</Fold>}
      <Fold id="skills" title="Skills" count={SPEC_LIST.filter((s) => sections.some((sec) => sec.groups.includes(s.group))).length}>
        <p className="faint small">Each skill is one step. Drag it onto the canvas, or click to add.</p>
        {sections.map((sec) => (
          <Fold key={sec.id} id={`skills-${sec.id}`} title={sec.title} open sub>
            {SPEC_LIST.filter((s) => sec.groups.includes(s.group)).map((s) => (
              <button
                key={s.type}
                className={`lib-item g-${s.group}`}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(DRAG_TYPE, s.type);
                  e.dataTransfer.effectAllowed = 'move';
                }}
                onClick={() => onAdd(s.type)}
              >
                <span className="kdot" />
                <span>
                  <span className="lib-title">{s.title}{s.soon && <span className="soon">Coming soon</span>}</span>
                  <span className="lib-sub">{s.subtitle}</span>
                </span>
              </button>
            ))}
          </Fold>
        ))}
      </Fold>
      {server && <JobsPanel />}
    </aside>
  );
}
