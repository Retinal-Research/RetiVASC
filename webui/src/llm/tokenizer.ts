// Qwen's byte-level BPE tokenizer, read from the model's tokenizer.json (the same file and rules as
// Hugging Face tokenizers: added tokens split out first, NFC, the split regex, byte-level symbols,
// merges by rank). tests/tokenizer.ts compares it with the Python library.

interface TokenizerJson {
  added_tokens: Array<{ id: number; content: string; special: boolean }>;
  model: { type: string; vocab: Record<string, number>; merges: Array<[string, string] | string> };
}

// Qwen2's split pattern; JavaScript has no inline (?i:…) everywhere yet, so the contractions spell out both cases
const SPLIT = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

/** GPT-2's byte to printable-character table (bytes_to_unicode). */
function byteTable(): string[] {
  const keep: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) keep.push(b);
  for (let b = 0xa1; b <= 0xac; b++) keep.push(b);
  for (let b = 0xae; b <= 0xff; b++) keep.push(b);
  const table: string[] = new Array(256);
  let n = 0;
  for (let b = 0; b < 256; b++) table[b] = String.fromCharCode(keep.includes(b) ? b : 256 + n++);
  return table;
}

export class Tokenizer {
  private vocab = new Map<string, number>();
  private tokens: string[] = [];
  private ranks = new Map<string, number>();
  private added: Array<{ id: number; content: string }>;
  private addedSplit: RegExp;
  private specialIds = new Set<number>();
  private byteChar = byteTable();
  private charByte = new Map(this.byteChar.map((c, b) => [c, b]));
  private cache = new Map<string, number[]>();

  constructor(json: TokenizerJson) {
    if (json.model.type !== 'BPE') throw new Error(`unsupported tokenizer ${json.model.type}`);
    for (const [token, id] of Object.entries(json.model.vocab)) { this.vocab.set(token, id); this.tokens[id] = token; }
    json.model.merges.forEach((m, rank) => this.ranks.set(Array.isArray(m) ? `${m[0]} ${m[1]}` : m, rank));
    this.added = json.added_tokens.map(({ id, content }) => ({ id, content }));
    for (const t of json.added_tokens) {
      this.tokens[t.id] = t.content;
      if (t.special) this.specialIds.add(t.id);
    }
    // longest first, so that no added token is split by a shorter one
    const escaped = [...this.added].sort((a, b) => b.content.length - a.content.length).map((t) => t.content.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    this.addedSplit = new RegExp(`(${escaped.join('|')})`);
  }

  id(token: string): number {
    const t = this.added.find((a) => a.content === token);
    if (!t) throw new Error(`no token ${token}`);
    return t.id;
  }

  /** Token ids of text. With control false, text such as "<|im_end|>" is ordinary characters, as it
   *  must be for anything a user typed. */
  encode(text: string, control = true): number[] {
    const out: number[] = [];
    const parts = control ? text.split(this.addedSplit) : [text];
    parts.forEach((part, i) => {
      if (!part) return;
      if (control && i % 2 === 1) { out.push(this.id(part)); return; }
      for (const piece of part.normalize('NFC').match(SPLIT) ?? []) out.push(...this.bpe(piece));
    });
    return out;
  }

  private bpe(piece: string): number[] {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    let symbols = Array.from(new TextEncoder().encode(piece), (b) => this.byteChar[b]);
    for (;;) {
      let best = -1, rank = Infinity;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const r = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (r !== undefined && r < rank) { rank = r; best = i; }
      }
      if (best < 0) break;
      // merge every occurrence of the pair, left to right, as the reference implementation does
      const a = symbols[best], b = symbols[best + 1], merged: string[] = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i + 1 < symbols.length && symbols[i] === a && symbols[i + 1] === b) { merged.push(a + b); i++; }
        else merged.push(symbols[i]);
      }
      symbols = merged;
    }
    const ids = symbols.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`token ${s} is not in the vocabulary`);
      return id;
    });
    if (this.cache.size < 50_000) this.cache.set(piece, ids);
    return ids;
  }

  /** The bytes of tokens, for decoding as UTF-8 (a character can span two tokens). Special tokens
   *  give nothing; other added tokens (e.g. <think>) give their text. */
  bytes(ids: number[]): Uint8Array {
    const out: number[] = [];
    for (const id of ids) {
      if (this.specialIds.has(id)) continue;
      const token = this.tokens[id];
      if (token === undefined) continue;
      if (this.added.some((a) => a.id === id)) out.push(...new TextEncoder().encode(token));
      else for (const ch of token) out.push(this.charByte.get(ch) ?? 0x3f);
    }
    return new Uint8Array(out);
  }

  decode(ids: number[]): string {
    return new TextDecoder().decode(this.bytes(ids));
  }
}
