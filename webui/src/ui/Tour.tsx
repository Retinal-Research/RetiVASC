// The tutorial: a short guided tour for people new to RetiVASC. It dims the page and lights up one part
// at a time (the real button or box, not a picture of it) with a sentence or two on what it is for. It
// opens by itself on the first visit, can be skipped at any step, and the Tutorial button at the top
// right opens it again. It ends on the home page, so people then start on their own. Steps whose part
// is not on the page (another workflow is open) are left out.

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { recall, remember } from '../remember';

export type TourView = 'home' | 'studio';

export interface TourStep {
  /** the part of the page it lights up; none for a card in the middle */
  target?: string;
  /** a step on the canvas to zoom to first */
  node?: string;
  view: TourView;
  title: string;
  body: ReactNode;
  /** the first card: larger, with the picture */
  welcome?: boolean;
}

const WELCOME: Omit<TourStep, 'view'> = {
  welcome: true,
  title: 'Welcome to RetiVASC',
  body: 'RetiVASC measures the blood vessels in a photograph of the back of the eye. This tour shows you how to use it, in about a minute.',
};

/** On the home page, then on to the canvas. */
const HOME: TourStep[] = [
  { ...WELCOME, view: 'home' },
  { view: 'home', target: '.cards', title: 'Choose what to run',
    body: 'Each card is a pipeline: a ready-made series of steps. RetiVASC is the lab’s own. It sharpens the photograph first, then measures it and writes a report.' },
  { view: 'home', target: '.start .btn.primary', title: 'Open it',
    body: 'Start opens the pipeline on a canvas. Nothing runs until you press Run.' },
];

const STUDIO: TourStep[] = [
  { ...WELCOME, view: 'studio' },
  { view: 'studio', target: '.canvas', title: 'This is your pipeline',
    body: 'Each box is one step. The photograph travels along the lines from left to right: it is enhanced, graded, its vessels are found and measured, and the results are written up.' },
  { view: 'studio', target: '.react-flow__node[data-id="photo"]', node: 'photo', title: 'Start with a photograph',
    body: 'A sample photograph is already chosen. To use your own, drop it on this box, or press its Open… button. It never leaves your computer.' },
  { view: 'studio', target: '.topbar .btn.primary', title: 'Press Run',
    body: 'Every step then runs inside this browser, on your own computer. The first run downloads the models once; after that, Run starts straight away.' },
  { view: 'studio', target: '.inspector-cell', title: 'See each result',
    body: 'Click any box to see its result here at full size, with what it measured and how long it took. Scroll on the picture to zoom in.' },
  { view: 'studio', target: '.react-flow__node[data-id="report"]', node: 'report', title: 'Read the AI report',
    body: 'The lab’s AI writes up the numbers in plain language, and you can ask it questions about them. Only the numbers are sent, never the photograph. It is not a diagnosis.' },
  { view: 'studio', target: '.react-flow__node[data-id="m1"] .node-head', node: 'm1', title: 'Run or change one step',
    body: <>▶ Run runs just this step. ⋯ copies, disconnects or deletes it. To connect two steps, drag from a dot on the right of one box to a dot on the left of another.</> },
  { view: 'studio', target: '.sidebar', title: 'More pipelines and steps',
    body: 'Open AutoMorph or VascX from here, or drag a single step onto the canvas to build a pipeline of your own.' },
  { view: 'studio', target: '.react-flow__node[data-id="photo"] .seg', node: 'photo', title: 'Many photographs at once',
    body: 'Choose “A folder” to measure up to 1000 images. The steps run on each in turn, and the AI report then covers all of them.' },
  { view: 'studio', target: '.tour-btn', title: 'Help is always here',
    body: 'Open this tour again at any time from this button.' },
  // the tour ends where people start: on the home page
  { view: 'home', target: '.start .btn.primary', title: 'Your turn',
    body: 'Choose a pipeline and press Start. RetiVASC is for research; it is not a medical device.' },
];

/** The steps of a tour that starts on this page. */
export const tourFrom = (view: TourView): TourStep[] => (view === 'home' ? [...HOME, ...STUDIO.slice(1)] : STUDIO);

/** Whether the tour should open by itself: the first visit, by a person (not an automated browser check). */
export function firstVisit(): boolean {
  return recall('tour', '', (v): v is string => typeof v === 'string') !== 'done' && !navigator.webdriver;
}

/** The ? Tutorial button at the top right. */
export function TourButton({ onClick }: { onClick: () => void }) {
  return (
    <button className="btn tour-btn" onClick={onClick} title="A one-minute tour of RetiVASC">
      <span className="tour-q" aria-hidden="true">?</span>Tutorial
    </button>
  );
}

/** A fundus photograph whose vessels draw themselves in, for the first card. */
function Picture() {
  const vessels = [
    'M44 60 C 60 44, 82 34, 110 30', 'M44 60 C 64 52, 90 52, 116 58', 'M44 60 C 62 72, 86 84, 112 92',
    'M44 60 C 56 80, 70 96, 92 106', 'M44 60 C 52 40, 62 24, 80 14', 'M78 38 C 88 46, 96 48, 106 44', 'M80 78 C 90 76, 100 80, 108 78',
  ];
  return (
    <svg className="tour-pic" viewBox="0 0 120 120" aria-hidden="true">
      <defs>
        <radialGradient id="tour-fundus" cx="0.45" cy="0.45" r="0.6">
          <stop offset="0" stopColor="#f59e5b" /><stop offset="0.7" stopColor="#d9532c" /><stop offset="1" stopColor="#7c2213" />
        </radialGradient>
      </defs>
      <circle cx="60" cy="60" r="56" fill="url(#tour-fundus)" />
      {vessels.map((d, i) => (
        <path key={d} d={d} className={`tour-vessel ${i % 2 ? 'v' : 'a'}`} style={{ animationDelay: `${0.25 + i * 0.12}s` }} pathLength={1} />
      ))}
      <circle cx="44" cy="60" r="9" fill="#ffe08a" className="tour-disc" />
      <circle cx="84" cy="62" r="5" fill="#8a2a14" opacity="0.6" />
    </svg>
  );
}

interface Box { top: number; left: number; width: number; height: number }
const PAD = 8, GAP = 16, MARGIN = 16;

function measure(selector?: string): Box | undefined {
  if (!selector) return undefined;
  const el = document.querySelector(selector);
  if (!el) return undefined;
  const r = el.getBoundingClientRect();
  if (!r.width && !r.height) return undefined;
  const top = Math.max(r.top - PAD, 4), left = Math.max(r.left - PAD, 4);
  return {
    top, left,
    width: Math.min(r.right + PAD, window.innerWidth - 4) - left,
    height: Math.min(r.bottom + PAD, window.innerHeight - 4) - top,
  };
}

/** Next to the lit part where there is room (right, left, below, above), else at the bottom. */
function place(box: Box | undefined, card: { w: number; h: number }): CSSProperties {
  const vw = window.innerWidth, vh = window.innerHeight;
  if (vw < 640) return { left: MARGIN, right: MARGIN, bottom: MARGIN };
  if (!box) return { left: (vw - card.w) / 2, top: Math.max(MARGIN, (vh - card.h) / 2) };
  const clampTop = (t: number) => Math.min(Math.max(t, MARGIN), vh - card.h - MARGIN);
  const clampLeft = (l: number) => Math.min(Math.max(l, MARGIN), vw - card.w - MARGIN);
  if (vw - (box.left + box.width) >= card.w + GAP + MARGIN) return { left: box.left + box.width + GAP, top: clampTop(box.top) };
  if (box.left >= card.w + GAP + MARGIN) return { left: box.left - GAP - card.w, top: clampTop(box.top) };
  if (vh - (box.top + box.height) >= card.h + GAP + MARGIN) return { left: clampLeft(box.left), top: box.top + box.height + GAP };
  if (box.top >= card.h + GAP + MARGIN) return { left: clampLeft(box.left), top: box.top - GAP - card.h };
  return { left: clampLeft(box.left + box.width / 2 - card.w / 2), top: vh - card.h - MARGIN };
}

interface Props {
  steps: TourStep[];
  view: TourView;
  /** take the page to the step's view (the home page's last step opens the canvas) */
  onView: (v: TourView) => void;
  /** zoom the canvas to a step, or out to the whole pipeline */
  onFocus: (node?: string) => void;
  onClose: () => void;
}

export function Tour({ steps, view, onView, onFocus, onClose }: Props) {
  const [index, setIndex] = useState(0);
  const [box, setBox] = useState<Box>();
  const [cardSize, setCardSize] = useState({ w: 360, h: 220 });
  const card = useRef<HTMLDivElement>(null);
  const next = useRef<HTMLButtonElement>(null);
  const direction = useRef(1);
  const step = steps[index];

  const close = useCallback(() => {
    remember('tour', 'done');
    onClose();
  }, [onClose]);
  const go = useCallback((to: number) => {
    if (to >= steps.length) { close(); return; }
    direction.current = to >= index ? 1 : -1;
    setIndex(Math.max(0, to));
  }, [steps.length, index, close]);

  // on the step's page; a part that is not there (another workflow is open) skips the step
  useEffect(() => {
    if (step.view !== view) { onView(step.view); return; }
    if (view === 'studio') onFocus(step.node);
    if (!step.target) return;
    let tries = 0, timer = 0;
    const look = () => {
      if (document.querySelector(step.target!)) return;
      if (++tries > 15) { setIndex((i) => Math.min(Math.max(0, i + direction.current), steps.length - 1)); return; }
      timer = window.setTimeout(look, 60);
    };
    look();
    return () => window.clearTimeout(timer);
  }, [step, view, onView, onFocus, steps.length]);

  // follow the lit part as the page lays itself out (a canvas fitting itself, a panel sliding)
  useLayoutEffect(() => {
    let frame = 0;
    const tick = () => {
      const b = step.view === view ? measure(step.target) : undefined;
      setBox((old) => (old && b && Math.abs(old.top - b.top) < 0.5 && Math.abs(old.left - b.left) < 0.5
        && Math.abs(old.width - b.width) < 0.5 && Math.abs(old.height - b.height) < 0.5) || (!old && !b) ? old : b);
      const c = card.current;
      if (c) setCardSize((s) => (Math.abs(s.w - c.offsetWidth) < 1 && Math.abs(s.h - c.offsetHeight) < 1 ? s : { w: c.offsetWidth, h: c.offsetHeight }));
      frame = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(frame);
  }, [step, view]);

  useEffect(() => next.current?.focus(), [index]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); close(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); go(index + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); go(index - 1); }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [index, go, close]);

  const last = index === steps.length - 1;
  const hole: CSSProperties = box ?? { top: window.innerHeight / 2, left: window.innerWidth / 2, width: 0, height: 0 };
  return (
    <div className={`tour${box ? '' : ' tour-open'}`} role="dialog" aria-modal="true" aria-labelledby="tour-title">
      <div className={`tour-hole${box ? ' lit' : ''}`} style={hole} />
      <div ref={card} key={index} className={`tour-card${step.welcome ? ' welcome' : ''}`} style={place(box, cardSize)}>
        {step.welcome && <Picture />}
        <div className="tour-count">{step.welcome ? 'A one-minute tour' : `Step ${index} of ${steps.length - 1}`}</div>
        <h2 id="tour-title">{step.title}</h2>
        <p>{step.body}</p>
        <div className="tour-dots" aria-hidden="true">
          {steps.map((_, i) => <span key={i} className={i === index ? 'on' : i < index ? 'past' : ''} />)}
        </div>
        <div className="tour-actions">
          {!last && <button className="linkbtn tour-skip" onClick={close}>Skip the tour</button>}
          <span className="tour-grow" />
          {index > 0 && !step.welcome && steps[index - 1].view === step.view && <button className="btn small" onClick={() => go(index - 1)}>Back</button>}
          <button ref={next} className="btn small primary" onClick={() => go(index + 1)}>
            {step.welcome ? 'Show me around' : last ? 'Done' : 'Next'}{!last && <span aria-hidden="true"> →</span>}
          </button>
        </div>
      </div>
    </div>
  );
}
