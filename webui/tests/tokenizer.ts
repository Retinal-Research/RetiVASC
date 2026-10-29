// Our Qwen tokenizer against Hugging Face tokenizers (Python): the same token ids for English, Chinese,
// numbers and units, Markdown, the chat template's control tokens, and a user's text that only looks like them.
// The reference (tokenizer_ref.json) came from tokenizers 0.22.1 with the model's own tokenizer.json.
// Run: npx tsx tests/tokenizer.ts   (needs webui/public/models/llm/tokenizer.json: tools/release.py fetch --llm)
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Tokenizer } from '../src/llm/tokenizer';

const ROOT = join(import.meta.dirname, '..');
const tok = new Tokenizer(JSON.parse(readFileSync(join(ROOT, 'public', 'models', 'llm', 'tokenizer.json'), 'utf8')));
const cases = JSON.parse(readFileSync(join(import.meta.dirname, 'tokenizer_ref.json'), 'utf8')) as Array<{ text: string; ids: number[]; control?: boolean }>;

let failed = 0;
for (const c of cases) {
  const ids = tok.encode(c.text, c.control ?? true);
  const same = ids.length === c.ids.length && ids.every((v, i) => v === c.ids[i]);
  const back = tok.decode(ids);
  // decoding drops control tokens; everything else comes back as written (after NFC)
  const expected = (c.control ?? true) ? c.text.replace(/<\|[a-z_]+\|>/g, '').normalize('NFC') : c.text.normalize('NFC');
  const round = back === expected;
  if (!same || !round) failed++;
  const label = JSON.stringify(c.text.length > 50 ? `${c.text.slice(0, 50)}…` : c.text);
  console.log(`${same && round ? 'PASS' : 'FAIL'}  ${label}: ${ids.length} tokens${same ? '' : ` (expected ${c.ids.length}; first difference at ${ids.findIndex((v, i) => v !== c.ids[i])})`}${round ? '' : ' · decode differs'}`);
}
process.exit(failed ? 1 : 0);
