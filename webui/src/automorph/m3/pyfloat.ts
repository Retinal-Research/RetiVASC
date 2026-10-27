// Python / pandas / NumPy number handling that AutoMorph v2 M3's results depend on.
//
// M3 writes intermediate tables with pandas (Python repr of each float) and reads them back with
// pandas' default parser, which is not an exact round trip: it accumulates at most 17 digits
// (leading zeros included) and then scales by a power of ten, so the last bit can change. The
// published feature values have been through these round trips, so the browser repeats them.

/** Python repr() of a float: shortest round-trip digits, fixed notation for 1e-4 <= |x| < 1e16. */
export function pyRepr(x: number): string {
  if (Number.isNaN(x)) return 'nan';
  if (x === Infinity) return 'inf';
  if (x === -Infinity) return '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const sign = x < 0 ? '-' : '';
  // toExponential() without an argument gives the shortest digits that round-trip
  const [mantissa, expText] = Math.abs(x).toExponential().split('e');
  const digits = mantissa.replace('.', '');
  const exp10 = parseInt(expText, 10);
  const decpt = exp10 + 1; // value = 0.<digits> x 10^decpt
  if (decpt > -4 && decpt <= 16) {
    if (decpt <= 0) return `${sign}0.${'0'.repeat(-decpt)}${digits}`;
    if (decpt >= digits.length) return `${sign}${digits}${'0'.repeat(decpt - digits.length)}.0`;
    return `${sign}${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
  }
  const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  return `${sign}${m}e${exp10 < 0 ? '-' : '+'}${String(Math.abs(exp10)).padStart(2, '0')}`;
}

const POW10 = Array.from({ length: 309 }, (_, k) => Number(`1e${k}`));
const isDigit = (c: string | undefined) => c !== undefined && c >= '0' && c <= '9';

/** pandas' default CSV float parser (precise_xstrtod, pandas 2.3), for '.' decimals. */
export function pandasParse(s: string): number {
  let p = 0;
  while (s[p] === ' ' || s[p] === '\t') p++;
  let negative = false;
  if (s[p] === '-') { negative = true; p++; } else if (s[p] === '+') p++;
  let number = 0;
  let exponent = 0;
  let numDigits = 0;
  let numDecimals = 0;
  while (isDigit(s[p])) {
    if (numDigits < 17) { number = number * 10 + (s.charCodeAt(p) - 48); numDigits++; } else exponent++;
    p++;
  }
  if (s[p] === '.') {
    p++;
    while (numDigits < 17 && isDigit(s[p])) { number = number * 10 + (s.charCodeAt(p) - 48); p++; numDigits++; numDecimals++; }
    if (numDigits >= 17) while (isDigit(s[p])) p++;
    exponent -= numDecimals;
  }
  if (numDigits === 0) return NaN;
  if (negative) number = -number;
  if (s[p] === 'e' || s[p] === 'E') {
    p++;
    let expNegative = false;
    if (s[p] === '-') { expNegative = true; p++; } else if (s[p] === '+') p++;
    let n = 0;
    let digits = 0;
    while (digits < 17 && isDigit(s[p])) { n = n * 10 + (s.charCodeAt(p) - 48); digits++; p++; }
    exponent += expNegative ? -n : n;
  }
  if (exponent > 308) return negative ? -Infinity : Infinity;
  if (exponent > 0) return number * POW10[exponent];
  if (exponent < -308) return exponent < -616 ? 0 : number / POW10[-308 - exponent] / POW10[308];
  return number / POW10[-exponent];
}

/** A value written by pandas to_csv and read back by pandas read_csv (NaN is written as empty and stays NaN). */
export function csvRoundTrip(x: number): number {
  if (Number.isNaN(x) || !Number.isFinite(x)) return x;
  return pandasParse(pyRepr(x));
}

/** NumPy's pairwise summation (np.add.reduce on a contiguous float64 array). */
export function pairwiseSum(a: ArrayLike<number>, lo = 0, n = a.length): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[lo + i];
    return res;
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
}
