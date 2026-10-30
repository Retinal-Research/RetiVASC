// Where a Fundus photo node gets its bytes: a bundled sample, or a file the
// user opened. Opened files stay in memory on this device and are never sent.

export interface Sample {
  file: string;
  label: string;
  width: number;
  height: number;
  license: string;
  author: string;
  source: string;
  /** the sample folder it belongs to: the Photo folder step opens it until another folder is chosen */
  folder?: string;
}

const uploads = new Map<string, { name: string; bytes: ArrayBuffer }>();
let nextUpload = 1;

export function addUpload(name: string, bytes: ArrayBuffer): string {
  const id = `upload:${nextUpload++}:${name}`;
  uploads.set(id, { name, bytes });
  return id;
}

export function sourceName(source: string): string {
  if (source.startsWith('sample:')) return source.slice('sample:'.length).split('/').pop() ?? source;
  return uploads.get(source)?.name ?? 'missing file';
}

export async function readSource(source: string): Promise<{ name: string; bytes: ArrayBuffer }> {
  if (source.startsWith('sample:')) {
    const file = source.slice('sample:'.length);
    const response = await fetch(`${import.meta.env.BASE_URL}${file}`);
    if (!response.ok) throw new Error(`could not load sample ${file}`);
    return { name: file.split('/').pop() ?? file, bytes: await response.arrayBuffer() };
  }
  const upload = uploads.get(source);
  if (!upload) throw new Error('the opened file is no longer in memory; open it again');
  // the worker takes ownership of the buffer, so hand it a copy
  return { name: upload.name, bytes: upload.bytes.slice(0) };
}

let samples: Promise<Sample[]> | undefined;
export function loadSamples(): Promise<Sample[]> {
  samples ??= fetch(`${import.meta.env.BASE_URL}samples/samples.json`).then((r) => r.json() as Promise<Sample[]>);
  return samples;
}
