// Short tips in the top bar, one at a time, changing every few seconds. Hovering holds the tip; clicking shows the next.

import { useEffect, useState } from 'react';

const HINTS = [
  'Click a box to see its result at full size.',
  '⌘/Ctrl + Enter runs the whole pipeline.',
  '▶ Run on a box runs only that step.',
  'Drop your own photo on the photo box.',
  'Choose “A folder” to measure up to 1000 images.',
  'Drag steps from the left, then connect the dots.',
  'Vessel colours: the appearance menu, top right.',
  'The Results table downloads every number as CSV.',
  'Ask the AI report about the numbers.',
  'Save downloads this workflow; Open loads it.',
  'Preload fetches the models before you Run.',
];
const EVERY = 7000;

export function Hints() {
  const [i, setI] = useState(() => Math.floor(Math.random() * HINTS.length));
  const [held, setHeld] = useState(false);
  useEffect(() => {
    if (held) return;
    const t = setTimeout(() => setI((x) => (x + 1) % HINTS.length), EVERY);
    return () => clearTimeout(t);
  }, [i, held]);
  return (
    <button className="hints" onClick={() => setI((x) => (x + 1) % HINTS.length)} onMouseEnter={() => setHeld(true)} onMouseLeave={() => setHeld(false)}
            title="Click for the next tip" aria-live="polite">
      <span className="hints-tag">Tip</span>
      <span key={i} className="hints-text">{HINTS[i]}</span>
    </button>
  );
}
