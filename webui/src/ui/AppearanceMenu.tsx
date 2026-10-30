import { useEffect, useRef, useState, type ComponentType } from 'react';
import { ACCENTS, DEFAULT_APPEARANCE, isHex, type Appearance, type Theme, type ThemeMode } from './appearance';
import { PRESETS, seen, setVesselColours, useVesselColours, type VesselColours } from './vesselColours';

interface Props {
  appearance: Appearance;
  theme: Theme;
  onChange: (a: Appearance) => void;
}

function Sun() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function Moon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z" />
    </svg>
  );
}

function Monitor() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M8 20h8M12 16v4" />
    </svg>
  );
}

const MODES: Array<{ mode: ThemeMode; label: string; icon: ComponentType }> = [
  { mode: 'system', label: 'System', icon: Monitor },
  { mode: 'light', label: 'Day', icon: Sun },
  { mode: 'dark', label: 'Night', icon: Moon },
];

/** One-click day/night switch plus a popover for the theme mode and accent colour. */
export function AppearanceMenu({ appearance, theme, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(appearance.accent);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => setDraft(appearance.accent), [appearance.accent]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const setAccent = (accent: string) => onChange({ ...appearance, accent: accent.toLowerCase() });

  return (
    <div className="appearance" ref={box}>
      <button
        className="icon-btn"
        onClick={() => onChange({ ...appearance, mode: theme === 'dark' ? 'light' : 'dark' })}
        title={theme === 'dark' ? 'Switch to day theme' : 'Switch to night theme'}
        aria-label={theme === 'dark' ? 'Switch to day theme' : 'Switch to night theme'}
      >
        {theme === 'dark' ? <Sun /> : <Moon />}
      </button>
      <button className="icon-btn" onClick={() => setOpen((o) => !o)} title="Theme and accent colour" aria-label="Theme and accent colour" aria-expanded={open}>
        <span className="swatch-dot" style={{ background: appearance.accent }} />
      </button>
      {open && (
        <div className="popover" role="dialog" aria-label="Appearance">
          <div className="pop-label">Theme</div>
          <div className="segmented">
            {MODES.map(({ mode, label, icon: Icon }) => (
              <button key={mode} className={appearance.mode === mode ? 'on' : ''} onClick={() => onChange({ ...appearance, mode })}>
                <Icon />
                {label}
              </button>
            ))}
          </div>
          <div className="pop-label">Accent colour</div>
          <div className="swatches">
            {ACCENTS.map((a) => (
              <button
                key={a.value}
                className={`swatch ${appearance.accent === a.value ? 'on' : ''}`}
                style={{ background: a.value }}
                onClick={() => setAccent(a.value)}
                title={a.name}
                aria-label={a.name}
              />
            ))}
          </div>
          <div className="custom-accent">
            <label className="picker" title="Pick any colour">
              <input type="color" value={appearance.accent} onChange={(e) => setAccent(e.target.value)} />
              <span style={{ background: appearance.accent }} />
            </label>
            <input
              className="hex"
              value={draft}
              spellCheck={false}
              maxLength={7}
              onChange={(e) => {
                const v = e.target.value.startsWith('#') ? e.target.value : `#${e.target.value}`;
                setDraft(v);
                if (isHex(v)) setAccent(v);
              }}
              aria-label="Accent colour as hex"
            />
            <button className="btn small" onClick={() => onChange(DEFAULT_APPEARANCE)}>Reset</button>
          </div>
          <VesselColoursPicker />
        </div>
      )}
    </div>
  );
}

const VESSEL_PARTS: Array<[keyof Pick<VesselColours, 'vessel' | 'artery' | 'vein' | 'crossing'>, string]> = [
  ['vessel', 'Vessels'], ['artery', 'Arteries'], ['vein', 'Veins'], ['crossing', 'Crossings'],
];

/** The colours of the vessel layers in every view: RRWNet's (light added: arteries magenta, veins cyan),
 *  AutoMorph's classic red and blue, or one's own. */
function VesselColoursPicker() {
  const c = useVesselColours();
  const set = (patch: Partial<VesselColours>) => setVesselColours({ ...c, ...patch, preset: patch.preset ?? 'custom' });
  return (
    <>
      <div className="pop-label">Vessel colours</div>
      <div className="segmented">
        <button className={c.preset === 'rrwnet' ? 'on' : ''} onClick={() => setVesselColours(PRESETS.rrwnet)}>RRWNet</button>
        <button className={c.preset === 'automorph' ? 'on' : ''} onClick={() => setVesselColours(PRESETS.automorph)}>AutoMorph</button>
        <button className={c.preset === 'custom' ? 'on' : ''} onClick={() => set({})}>Own</button>
      </div>
      <div className="vessel-colours">
        {VESSEL_PARTS.map(([k, label]) => (
          <label key={k} className="vc" title={c.mix && k !== 'vessel' ? `${label}: shown as ${seen(k, c)} on the vessels` : label}>
            <span className="picker">
              <input type="color" value={c[k]} onChange={(e) => set({ [k]: e.target.value })} />
              <span style={{ background: c[k] }} />
            </span>
            <span>{label}</span>
            <span className="vc-seen" style={{ background: seen(k, c) }} />
          </label>
        ))}
      </div>
      <label className="scheck">
        <input type="checkbox" checked={c.mix} onChange={(e) => set({ mix: e.target.checked })} />
        Add the light where layers overlap
      </label>
    </>
  );
}
