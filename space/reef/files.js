// Readers for the model files as they are published (no conversion step):
//   * embeddings.npz: a zip (numpy writes zip64 headers, stored or deflated) of .npy arrays.
//   * reef_int8.tflite: just enough FlatBuffers to read the input/output quantisation parameters,
//     which LiteRT.js does not expose but the runtime needs to (de)quantise like the Python one.

// -- npz -------------------------------------------------------------------------------------------

async function inflateRaw(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function u64(dv, off) {
  return Number(dv.getBigUint64(off, true));
}

/** List zip entries from the central directory: [{name, method, compSize, size, localOffset}]. */
function zipEntries(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 65535); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("npz: not a zip file");
  let count = dv.getUint16(eocd + 10, true);
  let cdOffset = dv.getUint32(eocd + 16, true);
  if (count === 0xffff || cdOffset === 0xffffffff) {
    const loc = eocd - 20;
    if (dv.getUint32(loc, true) !== 0x07064b50) throw new Error("npz: missing zip64 locator");
    const z = u64(dv, loc + 8);
    if (dv.getUint32(z, true) !== 0x06064b50) throw new Error("npz: bad zip64 record");
    count = u64(dv, z + 32);
    cdOffset = u64(dv, z + 48);
  }
  const entries = [];
  let p = cdOffset;
  for (let n = 0; n < count; n++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("npz: bad central directory");
    const method = dv.getUint16(p + 10, true);
    let compSize = dv.getUint32(p + 20, true);
    let size = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    let localOffset = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(buf.subarray(p + 46, p + 46 + nameLen));
    // zip64 extra field: the 0xFFFFFFFF fields, in order size, compSize, localOffset
    for (let e = p + 46 + nameLen; e < p + 46 + nameLen + extraLen; ) {
      const id = dv.getUint16(e, true);
      const len = dv.getUint16(e + 2, true);
      if (id === 0x0001) {
        let q = e + 4;
        if (size === 0xffffffff) (size = u64(dv, q)), (q += 8);
        if (compSize === 0xffffffff) (compSize = u64(dv, q)), (q += 8);
        if (localOffset === 0xffffffff) localOffset = u64(dv, q);
      }
      e += 4 + len;
    }
    entries.push({ name, method, compSize, size, localOffset });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

function parseNpy(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes[0] !== 0x93 || new TextDecoder().decode(bytes.subarray(1, 6)) !== "NUMPY") throw new Error("not a .npy array");
  const major = bytes[6];
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  if (fortran) throw new Error("npy: fortran order not supported");
  const n = shape.reduce((a, b) => a * b, 1);
  const raw = bytes.slice(hstart + hlen, hstart + hlen + n * Number(descr.slice(2))); // copy => aligned
  let data;
  if (descr === "<f4") data = new Float32Array(raw.buffer, 0, n);
  else if (descr === "<f2") data = f16ToF32(new Uint16Array(raw.buffer, 0, n));
  else throw new Error(`npy: dtype ${descr} not supported`);
  return { shape, data };
}

/** Exact float16 -> float32 conversion (via a 64K lookup table). */
function f16ToF32(h) {
  const lut = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const s = i & 0x8000 ? -1 : 1;
    const e = (i >> 10) & 0x1f;
    const f = i & 0x3ff;
    if (e === 0) lut[i] = s * f * 2 ** -24;
    else if (e === 31) lut[i] = f ? NaN : s * Infinity;
    else lut[i] = s * (1 + f / 1024) * 2 ** (e - 15);
  }
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = lut[h[i]];
  return out;
}

/** Load all arrays from an .npz buffer: {name: {shape, data: Float32Array}}. */
export async function readNpz(buffer) {
  const buf = new Uint8Array(buffer);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const out = {};
  for (const e of zipEntries(buf)) {
    const p = e.localOffset;
    if (dv.getUint32(p, true) !== 0x04034b50) throw new Error("npz: bad local header");
    const start = p + 30 + dv.getUint16(p + 26, true) + dv.getUint16(p + 28, true);
    const comp = buf.subarray(start, start + e.compSize);
    let bytes;
    if (e.method === 0) bytes = comp;
    else if (e.method === 8) bytes = await inflateRaw(comp);
    else throw new Error(`npz: compression method ${e.method} not supported`);
    out[e.name.replace(/\.npy$/, "")] = parseNpy(bytes);
  }
  return out;
}

// -- tflite ----------------------------------------------------------------------------------------

const TFLITE_TYPES = { 0: "float32", 2: "int32", 3: "uint8", 9: "int8" };

/** Read {dtype, shape, scale, zeroPoint} of subgraph 0's first input and first output. */
export function tfliteIoDetails(bytes) {
  const buf = new Uint8Array(bytes);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const deref = (p) => p + dv.getUint32(p, true);
  // Position of table field `i`, or -1 if absent.
  const field = (t, i) => {
    const vt = t - dv.getInt32(t, true);
    const o = 4 + 2 * i;
    if (o >= dv.getUint16(vt, true)) return -1;
    const off = dv.getUint16(vt + o, true);
    return off ? t + off : -1;
  };
  const vec = (t, i) => {
    const f = field(t, i);
    if (f < 0) return null;
    const v = deref(f);
    return { len: dv.getUint32(v, true), at: v + 4 };
  };
  const tableAt = (v, k) => deref(v.at + 4 * k);

  const model = deref(0);
  const subgraph = tableAt(vec(model, 2), 0);
  const tensors = vec(subgraph, 0);
  const describe = (index) => {
    const t = tableAt(tensors, index);
    const shapeV = vec(t, 0);
    const shape = shapeV ? Array.from({ length: shapeV.len }, (_, k) => dv.getInt32(shapeV.at + 4 * k, true)) : [];
    const tf = field(t, 1);
    const type = tf < 0 ? 0 : dv.getInt8(tf);
    let scale = null;
    let zeroPoint = 0;
    const qf = field(t, 4);
    if (qf >= 0) {
      const q = deref(qf);
      const s = vec(q, 2);
      const z = vec(q, 3);
      if (s?.len) scale = dv.getFloat32(s.at, true);
      if (z?.len) zeroPoint = Number(dv.getBigInt64(z.at, true));
    }
    return { dtype: TFLITE_TYPES[type] ?? `type${type}`, shape, scale, zeroPoint };
  };
  const inputs = vec(subgraph, 1);
  const outputs = vec(subgraph, 2);
  return {
    input: describe(dv.getInt32(inputs.at, true)),
    output: describe(dv.getInt32(outputs.at, true)),
  };
}
