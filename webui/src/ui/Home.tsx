// The home page: what RetiVASC is, the pipelines it offers, and Start. Nothing runs here;
// Start opens the canvas with the chosen pipeline, and the canvas runs only when asked to.

import { useState, type ReactNode } from 'react';
import type { Sample } from '../engine/sources';
import type { WorkflowFile } from '../engine/workflows';
import type { Appearance, Theme } from './appearance';
import { AppearanceMenu } from './AppearanceMenu';
import { Logo } from './Logo';

const LAB = 'Geometry Systems Lab';
const UNIVERSITY = 'Arizona State University';
export const SUBTITLE = 'Retinal Vascular Phenotype Platform for Brain Health';

/** Small line icons for the pipeline cards (currentColor). */
const ICONS: Record<string, ReactNode> = {
  retivasc: <Logo variant="color" size={34} />,
  automorph: (
    <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
        <path d="M16 5 L28 11 L16 17 L4 11 Z" /><path d="M4 16 L16 22 L28 16" /><path d="M4 21 L16 27 L28 21" />
      </g>
    </svg>
  ),
  vascx: (
    <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <path d="M26 16 C 19 10, 12 8, 4 9" /><path d="M26 16 C 19 22, 12 24, 4 23" /><path d="M15 10 C 12 6, 9 5, 6 4" /><path d="M15 22 C 12 26, 9 27, 6 28" />
      </g>
      <circle cx="26" cy="16" r="3.5" fill="currentColor" />
    </svg>
  ),
  custom: (
    <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="5" width="9" height="7" rx="2" /><rect x="20" y="5" width="9" height="7" rx="2" /><rect x="11.5" y="20" width="9" height="7" rx="2" />
        <path d="M12 8.5 H20 M7.5 12 C 7.5 17, 12 18, 14 20 M24.5 12 C 24.5 17, 20 18, 18 20" />
      </g>
    </svg>
  ),
};

const FEATURES: Array<[string, string]> = [
  ['Cross-platform', 'Runs in an up-to-date browser on Windows, macOS and Linux. There is nothing to install.'],
  ['No compute grid needed', 'Every model runs on the graphics card of your own computer. Images are stored and processed locally.'],
  ['Simple interface', 'Connect skills on a canvas and look at the result of every step. Combined with an LLM for report generation and analysis.'],
  ['Pipeline library', 'RetiVASC, AutoMorph and VascX are ready to use, and the skills combine into new ones.'],
];

interface Props {
  pipelines: WorkflowFile[];
  /** a pipeline whose models this site does not have: shown, but not offered */
  missing: (wf: WorkflowFile) => boolean;
  samples: Sample[];
  onStart: (wf: WorkflowFile) => void;
  appearance: Appearance;
  theme: Theme;
  onAppearance: (a: Appearance) => void;
}

export function Home({ pipelines, missing, samples, onStart, appearance, theme, onAppearance }: Props) {
  const [chosen, setChosen] = useState<string>();
  const ready = pipelines.filter((p) => !missing(p));
  const pick = ready.find((p) => p.id === chosen) ?? ready[0];
  const thumb = (s: Sample) => `${import.meta.env.BASE_URL}home/${s.file.split('/').pop()}`;

  return (
    <div className="home">
      <header className="home-top">
        <span className="home-mark"><Logo variant="mark" size={22} />RetiVASC</span>
        <AppearanceMenu appearance={appearance} theme={theme} onChange={onAppearance} />
      </header>

      <section className="hero">
        <Logo variant="color" size={92} />
        <h1>RetiVASC</h1>
        <p className="hero-sub">{SUBTITLE}</p>
        <p className="hero-lab">{LAB} · {UNIVERSITY}</p>
        <p className="hero-what">
          RetiVASC enhances colour fundus photographs and measures their retinal blood vessels: image quality, vessels, arteries and veins,
          the optic disc and 72 vascular phenotypes. An LLM on your computer turns the measurements into a report. One photograph or a folder of up to 1000.
        </p>
        <div className="strip">
          {samples.slice(0, 6).map((s) => (
            <img key={s.file} src={thumb(s)} alt={s.label} title={`${s.label} · ${s.author} · ${s.license}`} loading="lazy" />
          ))}
        </div>
      </section>

      <section className="pick">
        <h2>Choose a pipeline</h2>
        <div className="cards" role="radiogroup" aria-label="Pipeline">
          {pipelines.map((p) => {
            const off = missing(p);
            return (
              <button key={p.id} role="radio" aria-checked={p.id === pick?.id} disabled={off} className={`card ${p.id === pick?.id ? 'on' : ''}`}
                      onClick={() => setChosen(p.id)} onDoubleClick={() => onStart(p)}>
                <span className="card-icon">{ICONS[p.id ?? ''] ?? ICONS.custom}</span>
                <span className="card-name">{p.name}{p.id === 'retivasc' && <span className="badge-ours">Ours</span>}</span>
                <span className="card-desc">{p.description}</span>
                {off && <span className="card-off">Its models are not installed on this site</span>}
              </button>
            );
          })}
        </div>
        <div className="start">
          <button className="btn primary big" disabled={!pick} onClick={() => pick && onStart(pick)}>
            Start{pick ? ` with ${pick.name}` : ''} <span aria-hidden="true">→</span>
          </button>
          <span className="hint">Opens the canvas. Nothing runs until you press Run.</span>
        </div>
      </section>

      <section className="features">
        {FEATURES.map(([title, text]) => (
          <div key={title} className="feature">
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
        ))}
      </section>

      <footer className="home-foot">
        <span>© {new Date().getFullYear()} {LAB}, {UNIVERSITY}</span>
        <span>
          Models: <a href="https://github.com/rmaphoh/AutoMorph" target="_blank" rel="noreferrer">AutoMorph</a> (Apache-2.0)
          {pipelines.some((p) => p.id === 'vascx') && <> · <a href="https://huggingface.co/Eyened/vascx" target="_blank" rel="noreferrer">VascX</a> (CC BY-NC 4.0)</>}
        </span>
        <span title={samples.map((s) => `${s.label}: ${s.author}, ${s.license}`).join('\n')}>Sample photographs: Wikimedia Commons (hover for credits)</span>
        <span>For research use. Not a medical device.</span>
      </footer>
    </div>
  );
}
