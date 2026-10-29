// A 1.4 GB ONNX model with its weights inside does not fit in onnxruntime-web's WebAssembly memory
// once parsed (std::bad_alloc). Given as "external data" instead, the weights go from JavaScript
// straight to the graphics card. This rewrites the model's protobuf so that every large initializer
// refers to an external file, and hands those files to onnxruntime as views of the original bytes:
// nothing is copied. Fields used (onnx.proto): ModelProto.graph = 7, GraphProto.initializer = 5,
// TensorProto.raw_data = 9, .external_data = 13 (key 1, value 2), .data_location = 14 (1 = external).

export interface SplitModel {
  /** the model without the weights (about a megabyte) */
  graph: Uint8Array;
  /** one file per moved initializer, for InferenceSession.create(graph, { externalData }) */
  files: Array<{ path: string; data: Uint8Array }>;
}

function readVarint(buf: Uint8Array, pos: number): [number, number] {
  let value = 0, scale = 1, b: number;
  do {
    b = buf[pos++];
    value += (b & 0x7f) * scale;  // arithmetic, not shifts: lengths pass 2^31
    scale *= 128;
  } while (b & 0x80);
  return [value, pos];
}

function varint(value: number): Uint8Array {
  const out: number[] = [];
  do {
    let b = value % 128;
    value = Math.floor(value / 128);
    if (value) b |= 0x80;
    out.push(b);
  } while (value);
  return Uint8Array.from(out);
}

/** The fields of a message: [field number, wire type, start of the field, start of the value, end]. */
function* fields(buf: Uint8Array, start: number, end: number): Generator<[number, number, number, number, number]> {
  let pos = start;
  while (pos < end) {
    const from = pos;
    let key: number;
    [key, pos] = readVarint(buf, pos);
    const field = Math.floor(key / 8), wire = key % 8;
    let valueStart = pos;
    switch (wire) {
      case 0: [, pos] = readVarint(buf, pos); break;
      case 1: pos += 8; break;
      case 2: { let n: number; [n, pos] = readVarint(buf, pos); valueStart = pos; pos += n; break; }
      case 5: pos += 4; break;
      default: throw new Error(`unsupported protobuf wire type ${wire}`);
    }
    yield [field, wire, from, valueStart, pos];
  }
}

const utf8 = new TextEncoder();

function lengthDelimited(field: number, parts: Uint8Array[]): Uint8Array[] {
  const n = parts.reduce((s, p) => s + p.length, 0);
  return [varint(field * 8 + 2), varint(n), ...parts];
}

const entry = (key: string, value: string) => lengthDelimited(13, [...lengthDelimited(1, [utf8.encode(key)]), ...lengthDelimited(2, [utf8.encode(value)])]);

function join(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

export function splitWeights(model: Uint8Array, minBytes = 1024): SplitModel {
  const files: SplitModel['files'] = [];
  const tensor = (start: number, end: number): Uint8Array[] => {
    const kept: Uint8Array[] = [];
    let raw: Uint8Array | undefined;
    for (const [field, , from, valueStart, to] of fields(model, start, end)) {
      if (field === 9 && to - valueStart >= minBytes) raw = model.subarray(valueStart, to);
      else if (field !== 14) kept.push(model.subarray(from, to));
    }
    if (!raw) return [model.subarray(start, end)];
    const path = `weights_${files.length}`;
    files.push({ path, data: raw });
    return [...kept, ...entry('location', path), ...entry('length', String(raw.length)), varint(14 * 8), varint(1)];
  };
  const graph = (start: number, end: number): Uint8Array[] => {
    const out: Uint8Array[] = [];
    for (const [field, wire, from, valueStart, to] of fields(model, start, end)) {
      if (field === 5 && wire === 2) out.push(...lengthDelimited(5, tensor(valueStart, to)));
      else out.push(model.subarray(from, to));
    }
    return out;
  };
  const out: Uint8Array[] = [];
  for (const [field, wire, from, valueStart, to] of fields(model, 0, model.length)) {
    if (field === 7 && wire === 2) out.push(...lengthDelimited(7, graph(valueStart, to)));
    else out.push(model.subarray(from, to));
  }
  return { graph: join(out), files };
}
