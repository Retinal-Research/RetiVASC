import type { RunSummary } from '../engine/executor';
import type { Appearance, Theme } from './appearance';
import { AppearanceMenu } from './AppearanceMenu';
import type { EngineInfo, GpuChoice } from '../runtime/protocol';
import { ComputeMenu } from './ComputeMenu';
import { ms } from './format';
import { SUBTITLE } from './Home';
import { Logo } from './Logo';
import type { AccountState } from './Login';
import { Hints } from './Hints';
import { TourButton } from './Tour';

/** The Preload button: the models the workflow uses, loaded before Run so that Run starts at once. */
export interface PreloadState {
  state: 'none' | 'needed' | 'loading' | 'ready';
  label: string;
  title: string;
}

interface Props {
  name: string;
  engine?: EngineInfo;
  engineError?: string;
  gpu: GpuChoice;
  onGpu: (gpu: GpuChoice) => void;
  running: boolean;
  preloading: boolean;
  appearance: Appearance;
  theme: Theme;
  onAppearance: (a: Appearance) => void;
  account: AccountState;
  onSignOut: () => void;
  /** back to the home page */
  onHome: () => void;
  /** open the tutorial */
  onTour: () => void;
}

export function TopBar({ name, engine, engineError, gpu, onGpu, running, preloading, appearance, theme, onAppearance, account, onSignOut, onHome, onTour }: Props) {
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
      <Hints />
      <div className="status">
        <ComputeMenu engine={engine} engineError={engineError} gpu={gpu} running={running || preloading} onGpu={onGpu} />
        {account.mode === 'ready' ? (
          <span className="private" title="Browser steps run in this tab; server steps run on the RetiVASC server">
            {account.account.mode === 'open'
              ? <>Public beta · your jobs are private to this browser</>
              : <>{account.account.name}{account.account.admin ? ' (admin)' : ''} · <button className="linkbtn" onClick={onSignOut}>Sign out</button></>}
          </span>
        ) : (
          <span className="private" title="Photos are decoded and analysed inside this browser tab">Images are stored and processed locally</span>
        )}
        <TourButton onClick={onTour} />
        <AppearanceMenu appearance={appearance} theme={theme} onChange={onAppearance} />
      </div>
    </header>
  );
}

interface RunProps {
  engine?: EngineInfo;
  preload: PreloadState;
  onPreload: () => void;
  running: boolean;
  /** Stop was pressed; the step that is running finishes first */
  stopping: boolean;
  lastRun?: RunSummary;
  onRun: () => void;
  /** stop after the step that is running (a folder run keeps the photos it finished) */
  onStop: () => void;
}

/** Run, Stop, Preload and how the last run went: floating at the top centre of the canvas. */
export function RunBar({ engine, preload, onPreload, running, stopping, lastRun, onRun, onStop }: RunProps) {
  return (
    <div className="runbar">
      <button className="btn primary" onClick={onRun} disabled={running || !engine} title="Run (Ctrl+Enter)">
        {running ? <span className="spinner" /> : <span className="play" />}
        {running ? 'Running…' : 'Run'}
      </button>
      {running && (
        <button className="btn danger" onClick={onStop} disabled={stopping} title="Stop after the step that is running">
          <span className="stopsq" />{stopping ? 'Stopping…' : 'Stop'}
        </button>
      )}
      {!running && preload.state !== 'none' && (
        <button className={`btn preload ${preload.state}`} onClick={onPreload} disabled={preload.state !== 'needed' || !engine} title={preload.title}>
          {preload.state === 'loading' ? <span className="spinner" /> : <span className={preload.state === 'ready' ? 'tick' : 'down'} aria-hidden="true" />}
          {preload.label}
        </button>
      )}
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
    </div>
  );
}
