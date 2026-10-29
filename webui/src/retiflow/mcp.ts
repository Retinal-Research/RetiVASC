// scikit-image's MCP_Geometric (graph/_mcp.pyx) with its FastUpdateBinaryHeap (graph/heap.pyx), ported
// line for line: positions are flattened in Fortran order, the eight offsets come in np.ndenumerate's
// order, the heap keeps its values in its last level and pops the leftmost minimum, and a node's
// predecessor changes only for a strictly lower cost. So costs and traceback paths, ties included,
// come out as RetiFlow gets them in Python. BSD-2-Clause, scikit-image's notice in NOTICE.md.

const INF = Infinity;

/** heap.pyx: BinaryHeap with cross-references (FastUpdateBinaryHeap), initial capacity 128. */
class Heap {
  private levels = 7;
  private readonly minLevels = 7;
  private values = new Float64Array(2 * 128).fill(INF);
  private refs = new Int32Array(128).fill(-1);
  private cross: Int32Array;
  count = 0;
  popped = -1;
  pushed = false;

  constructor(maxReference: number) {
    this.cross = new Int32Array(maxReference + 1).fill(-1);
  }

  private resize(add: number) {
    const levels = this.levels + add, number = 2 ** levels;
    const values = new Float64Array(number * 2).fill(INF), refs = new Int32Array(number).fill(-1);
    if (this.count) {
      const i1 = number - 1, i2 = 2 ** this.levels - 1, n = Math.min(number, 2 ** this.levels);
      for (let i = 0; i < n; i++) values[i1 + i] = this.values[i2 + i];
      for (let i = 0; i < n; i++) refs[i] = this.refs[i];
    }
    this.values = values;
    this.refs = refs;
    this.levels = levels;
    // _update
    for (let level = this.levels; level > 1; level--) {
      const i0 = (1 << level) - 1, n = i0 + 1;
      for (let i = i0; i < i0 + n; i += 2) {
        const ii = Math.floor((i - 1) / 2);
        this.values[ii] = this.values[i] < this.values[i + 1] ? this.values[i] : this.values[i + 1];
      }
    }
  }

  private updateOne(i: number) {
    const values = this.values;
    if (i % 2 === 0) i -= 1;
    for (let level = this.levels; level > 1; level--) {
      const ii = Math.floor((i - 1) / 2);
      values[ii] = values[i] < values[i + 1] ? values[i] : values[i + 1];
      i = ii % 2 ? ii : ii - 1;
    }
  }

  private remove(i1: number) {
    const levels = this.levels, i0 = (1 << levels) - 1;
    let count = this.count;
    const i2 = i0 + count - 1, r1 = i1 - i0, r2 = count - 1;
    this.cross[this.refs[r2]] = r1;
    this.cross[this.refs[r1]] = -1;
    this.values[i1] = this.values[i2];
    this.refs[r1] = this.refs[r2];
    this.values[i2] = INF;
    this.count -= 1;
    count -= 1;
    if (levels > this.minLevels && count < (1 << (levels - 2))) this.resize(-1);
    else { this.updateOne(i1); this.updateOne(i2); }
  }

  private append(value: number, reference: number): number {
    let levels = this.levels;
    const count = this.count;
    if (count >= (1 << levels)) { this.resize(1); levels += 1; }
    const i = ((1 << levels) - 1) + count;
    this.values[i] = value;
    this.refs[count] = reference;
    this.count += 1;
    this.updateOne(i);
    return count;
  }

  /** push_fast of FastUpdateBinaryHeap: update the reference's value, or append it. */
  push(value: number, reference: number) {
    const ir = this.cross[reference];
    if (ir !== -1) {
      const i = (1 << this.levels) - 1 + ir;
      this.values[i] = value;
      this.updateOne(i);
      return;
    }
    this.cross[reference] = this.append(value, reference);
  }

  pushIfLower(value: number, reference: number) {
    const ir = this.cross[reference];
    this.pushed = true;
    if (ir !== -1) {
      const i = (1 << this.levels) - 1 + ir;
      if (this.values[i] > value) { this.values[i] = value; this.updateOne(i); }
      else this.pushed = false;
      return;
    }
    this.cross[reference] = this.append(value, reference);
  }

  pop(): number {
    const values = this.values, levels = this.levels;
    let i = 1;
    for (let level = 1; level < levels; level++) i = values[i] <= values[i + 1] ? i * 2 + 1 : (i + 1) * 2 + 1;
    if (!(values[i] <= values[i + 1])) i += 1;
    const ir = i - ((1 << levels) - 1);
    const value = values[i];
    this.popped = this.refs[ir];
    if (this.count) this.remove(i);
    return value;
  }
}

// np.ndenumerate over the 3 x 3 mask without its centre
const OFFSETS: Array<[number, number]> = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
const LENGTHS = OFFSETS.map(([dy, dx]) => Math.sqrt(dy * dy + dx * dx));

/** MCP_Geometric(costs, fully_connected=True) on an h x w image of float64 costs (row-major input). */
export class MCPGeometric {
  private flatCosts: Float64Array;  // Fortran order
  private cumulative: Float64Array;
  private trace: Int8Array;
  private flatOffsets: number[];

  constructor(costs: Float64Array, private w: number, private h: number) {
    this.flatCosts = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) this.flatCosts[y + x * h] = costs[y * w + x];
    this.cumulative = new Float64Array(w * h);
    this.trace = new Int8Array(w * h);
    this.flatOffsets = OFFSETS.map(([dy, dx]) => dy + dx * h);
  }

  /** find_costs(starts): the cumulative costs from the nearest start, row-major. */
  findCosts(starts: Array<[number, number]>): Float64Array {
    const { w, h } = this, size = w * h, costs = this.flatCosts, cum = this.cumulative, trace = this.trace;
    const heap = new Heap(size - 1 < 127 ? 127 : size - 1);
    trace.fill(-2);
    cum.fill(INF);
    for (const [y, x] of starts) {
      const s = y + x * h;
      trace[s] = -1;
      heap.push(0, s);  // MCP_Geometric: use_start_cost = 0
    }
    for (let iter = 0; iter < size; iter++) {
      if (heap.count === 0) break;
      const cumcost = heap.pop();
      const index = heap.popped;
      cum[index] = cumcost;
      const y = index % h, x = (index - y) / h;
      const atEdge = y === 0 || y === h - 1 || x === 0 || x === w - 1;
      for (let i = 0; i < 8; i++) {
        if (atEdge) {
          const yy = y + OFFSETS[i][0], xx = x + OFFSETS[i][1];
          if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
        }
        const next = index + this.flatOffsets[i];
        if (cum[next] !== INF) continue;
        const cost = costs[index], newCost = costs[next];
        if (newCost < 0 || newCost === INF) continue;
        const newCum = cumcost + LENGTHS[i] * 0.5 * (cost + newCost);
        if (newCum !== INF) {
          heap.pushIfLower(newCum, next);
          if (heap.pushed) trace[next] = i;
        }
      }
    }
    const out = new Float64Array(size);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = cum[y + x * h];
    return out;
  }

  /** traceback(end): the path from its start to `end`, [y, x] pairs. */
  traceback(end: [number, number]): Array<[number, number]> {
    let [y, x] = end;
    let flat = y + x * this.h;
    if (this.cumulative[flat] === INF) throw new Error('no minimum-cost path to that point');
    const path: Array<[number, number]> = [[y, x]];
    for (;;) {
      const o = this.trace[flat];
      if (o === -1) break;
      flat -= this.flatOffsets[o];
      y -= OFFSETS[o][0];
      x -= OFFSETS[o][1];
      path.push([y, x]);
    }
    return path.reverse();
  }
}
