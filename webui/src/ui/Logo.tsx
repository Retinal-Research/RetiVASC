// The RetiVASC logo: a fundus with the optic disc and the two vascular arcades.
// "color" for the home page; "mark" (one colour, currentColor) for the top bar.

import { useId } from 'react';

/** Upper and lower arcades leave the disc towards the macula; the short ones run nasally. */
const ARTERIES = [
  'M59 40 C 47 27, 31 25, 15 33',
  'M42 30 C 38 21, 32 17, 25 14',
  'M59 49 C 47 61, 31 64, 16 58',
  'M42 58 C 38 67, 32 72, 26 76',
  'M65 41 C 71 34, 77 31, 84 31',
];
const VEINS = [
  'M57 43 C 45 31, 29 30, 14 38',
  'M38 33 C 33 25, 27 22, 19 21',
  'M57 46 C 45 57, 29 59, 15 53',
  'M38 55 C 33 63, 27 67, 20 69',
  'M65 47 C 71 54, 77 57, 84 58',
];

export function Logo({ variant = 'mark', size = 26 }: { variant?: 'mark' | 'color'; size?: number }) {
  const id = useId().replace(/:/g, '');
  if (variant === 'mark') {
    return (
      <svg width={size} height={size} viewBox="0 0 96 96" aria-hidden="true">
        <circle cx="48" cy="45" r="41" fill="none" stroke="currentColor" strokeWidth="5" opacity="0.9" />
        {VEINS.slice(0, 4).map((d) => <path key={d} d={d} fill="none" stroke="currentColor" strokeWidth="4.5" strokeLinecap="round" opacity="0.85" />)}
        <circle cx="61" cy="44" r="9" fill="currentColor" />
      </svg>
    );
  }
  return (
    <svg width={size} height={size} viewBox="0 0 96 96" role="img" aria-label="RetiVASC">
      <defs>
        <radialGradient id={`f${id}`} cx="42%" cy="44%" r="62%">
          <stop offset="0" stopColor="#f7a15f" />
          <stop offset="0.55" stopColor="#d8592c" />
          <stop offset="0.9" stopColor="#8f2b15" />
          <stop offset="1" stopColor="#5a1a0c" />
        </radialGradient>
        <radialGradient id={`d${id}`} cx="45%" cy="40%" r="60%">
          <stop offset="0" stopColor="#fffbe6" />
          <stop offset="1" stopColor="#f6c768" />
        </radialGradient>
        <radialGradient id={`m${id}`}>
          <stop offset="0" stopColor="#5a1a0c" stopOpacity="0.55" />
          <stop offset="1" stopColor="#5a1a0c" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="48" cy="45" r="43" fill={`url(#f${id})`} />
      <circle cx="35" cy="46" r="11" fill={`url(#m${id})`} />
      {VEINS.map((d) => <path key={d} d={d} fill="none" stroke="#6e1b0e" strokeWidth="2.4" strokeLinecap="round" />)}
      {ARTERIES.map((d) => <path key={d} d={d} fill="none" stroke="#ffc2a1" strokeWidth="1.6" strokeLinecap="round" />)}
      <circle cx="61" cy="44" r="8.5" fill={`url(#d${id})`} />
      <circle cx="48" cy="45" r="43" fill="none" stroke="#3b0f06" strokeOpacity="0.35" strokeWidth="1.5" />
    </svg>
  );
}
