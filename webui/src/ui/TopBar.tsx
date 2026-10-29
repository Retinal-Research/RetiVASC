import type { RunSummary } from '../engine/executor';
import type { Appearance, Theme } from './appearance';
import { AppearanceMenu } from './AppearanceMenu';
import type { EngineInfo } from '../runtime/protocol';
import { ms } from './format';
import { SUBTITLE } from './Home';
import { Logo } from './Logo';
import type { AccountState } from './Login';

interface Props {
  name: string;
  engine?: EngineInfo;
  engineError?: string;
  running: boolean;
  lastRun?: RunSummary;
  onRun: () => void;
  /** stop after the step that is running (a folder run keeps the photos it finished) */
  onStop: () => void;
  onSave: () => void;
  onOpen: (file: File) => void;
  appearance: Appearance;
  theme: Theme;
  onAppearance: (a: Appearance) => void;
  account: AccountState;
  onSignOut: () => void;
  /** back to the home page */
  onHome: () => void;
}

export function TopBar({ name, engine, engineError, running, lastRun, onRun, onStop, onSave, onOpen, appearance, theme, onAppearance, account, onSignOut, onHome }: Props) {
  let chip = <span className="engine wait"><span className="sdot" />Starting engine…</span>;
  if (engineError) chip = <span className="engine bad" title={engineError}><span className="sdot" />Engine failed to start</span>;
  else if (engine?.backend === 'webgpu') chip = <span className="engine ok" title={engine.adapter}><span className="sdot" />WebGPU · {engine.adapter}</span>;
  else if (engine) chip = <span className="engine warn" title="No WebGPU in this browser; models run on the CPU and are much slower"><span className="sdot" />CPU only · {engine.threads} thread{engine.threads === 1 ? '' : 's'}</span>;

  return (
    <header className="topbar">
      <button className="brand" onClick={onHome} title="Home page">
        <Logo size={26} />
        <div>
          <div className="brand-name">RetiVASC</div>
          <div className="brand-sub">{SUBTITLE}</div>
        </div>
      </button>
      <div className="wf-current" title={name}>{name}</div>
      <div className="actions">
        <button className="btn primary" onClick={onRun} disabled={running || !engine} title="Run (Ctrl+Enter)">
          {running ? <span className="spinner" /> : <span className="play" />}
          {running ? 'Running…' : 'Run'}
        </button>
        {running && <button className="btn" onClick={onStop} title="Stop after the step that is running">Stop</button>}
        {lastRun && !running && (
          <span className={`lastrun ${lastRun.errors ? 'bad' : lastRun.blocked ? 'warn' : ''}`}>
            {lastRun.errors
              ? `${lastRun.errors} step${lastRun.errors > 1 ? 's' : ''} failed`
              : lastRun.blocked
                ? `${lastRun.blocked} step${lastRun.blocked > 1 ? 's' : ''} not connected`
                : lastRun.ran
                  ? `Finished in ${ms(lastRun.ms)}`
                  : 'Nothing changed'}
          </span>
        )}
        <button className="btn" onClick={onSave} title="Download this workflow as a file">Save</button>
        <label className="btn" title="Open a saved workflow file">
          Open
          <input type="file" accept=".json,application/json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onOpen(f); e.target.value = ''; }} />
        </label>
      </div>
      <div className="status">
        {chip}
        {account.mode === 'ready' ? (
          <span className="private" title="Browser steps run in this tab; server steps run on the RetiVASC server">
            {account.account.mode === 'open'
              ? <>Public beta · your jobs are private to this browser</>
              : <>{account.account.name}{account.account.admin ? ' (admin)' : ''} · <button className="linkbtn" onClick={onSignOut}>Sign out</button></>}
          </span>
        ) : (
          <span className="private" title="Photos are decoded and analysed inside this browser tab">Runs on this device · nothing is uploaded</span>
        )}
        <AppearanceMenu appearance={appearance} theme={theme} onChange={onAppearance} />
      </div>
    </header>
  );
}
