// Port of src/reef/runtime.py for the browser: fetch the model files, gather embeddings on the
// host, quantise to int8 exactly like numpy does, run the encoder with LiteRT.js (WASM / XNNPack),
// and decode the option-marker logits.

import { Tensor, loadAndCompile, loadLiteRt } from "../vendor/litert-core.js";
import { readNpz, tfliteIoDetails } from "./files.js";
import { Packer, argmax, makeAnswer, normalizeQuestions, npSum, planGroups, planRound, prepareState, softmax } from "./packing.js";
import { renderState } from "./text.js";
import { WordPieceTokenizer } from "./tokenizer.js";

export const LITERT_VERSION = "2.5.3"; // must match space/vendor/litert-core.js
export const DEFAULT_WASM_URL = `https://cdn.jsdelivr.net/npm/@litertjs/core@${LITERT_VERSION}/wasm/`;
export const DEFAULT_MODEL_URL = "https://huggingface.co/seraphimserapis/reef/resolve/main/";
export const MODEL_FILES = ["config.json", "tokenizer.json", "embeddings.npz", "reef_int8.tflite"];

const f32 = Math.fround;

/** numpy's np.round (half to even) on a value that is already float32. */
function roundHalfEven(x) {
  const r = Math.round(x); // half up
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

let liteRtReady = null;
function ensureLiteRt(wasmUrl) {
  // loadLiteRt may only be called once per page.
  liteRtReady ??= loadLiteRt(wasmUrl);
  return liteRtReady;
}

/** Fetch with byte progress: onProgress(loadedBytes, totalBytesOrNull). */
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}${res.status === 401 || res.status === 404 ? " (private or missing repo?)" : ""}`);
  const total = Number(res.headers.get("content-length")) || null;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) out.set(c, o), (o += c.length);
  return out;
}

export class Reef {
  /**
   * Load an exported model directory from `baseUrl` (a folder URL ending in "/").
   * Options: wasmUrl (LiteRT.js wasm folder), onProgress({file, loaded, total}),
   * maxStateWindows (overlapping chunks read of a long state; 1 = truncate).
   */
  static async load(baseUrl = DEFAULT_MODEL_URL, { wasmUrl = DEFAULT_WASM_URL, onProgress, maxStateWindows = 8 } = {}) {
    const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
    const runtime = ensureLiteRt(wasmUrl);
    const [config, tokenizer, npz, tflite] = await Promise.all(
      MODEL_FILES.map((f) => fetchBytes(base + f, onProgress && ((loaded, total) => onProgress({ file: f, loaded, total })))),
    );
    await runtime;
    const reef = new Reef();
    reef.maxStateWindows = maxStateWindows;
    const dec = new TextDecoder();
    reef.meta = JSON.parse(dec.decode(config));
    await reef.init(JSON.parse(dec.decode(tokenizer)), await readNpz(npz), tflite);
    reef.downloadBytes = config.length + tokenizer.length + npz.length + tflite.length;
    return reef;
  }

  async init(tokenizerSpec, tables, tflite) {
    const m = this.meta.model;
    this.name = this.meta.name ?? "reef";
    this.stateLen = m.state_len;
    this.schemaLen = m.schema_len;
    this.grid = [(this.stateLen + this.schemaLen) / m.grid_width, m.grid_width];
    this.temperatures = this.meta.temperatures;
    this.tok = tables.tok.data;
    this.role = tables.role.data;
    this.level = tables.level.data;
    this.dim = tables.tok.shape[1];
    this.tokenizer = new WordPieceTokenizer(tokenizerSpec);
    // Lenient runtime (reef.runtime.Reef with strict=False): the limits the model was trained
    // with (old checkpoints: 40/28), truncating longer text; over-long instructions are first moved
    // into the state if the model was trained to read them there.
    const limI = m.max_instruction_tokens ?? 40;
    const limO = m.max_option_tokens ?? 28;
    // v2 encoders: one fixed block per question band, and bare descriptions for opaque option keys.
    this.packer = new Packer(this.tokenizer, this.stateLen, this.schemaLen, limI, limO, {
      bands: Number(m.bands ?? 0) || 0,
      dropOpaqueKeys: pyBool(m.drop_opaque_keys ?? false),
    });
    this.hoist = pyBool(m.hoist_instructions ?? false);
    // an empty state takes the instructions as the state (and each question alone, if several)
    this.hoistEmpty = this.hoist && pyBool(m.hoist_empty_state ?? false);
    this.instructionCap = limI;
    this.maxStateWindows ??= 8;
    this.io = tfliteIoDetails(tflite);
    this.model = await loadAndCompile(tflite, { accelerator: "wasm" });
    this.device = "wasm";
  }

  /** Free the compiled model (wasm memory). */
  dispose() {
    this.model?.delete();
    this.model = null;
  }

  /** Summed token + role + level embeddings, float32 arithmetic like numpy. Shape [L * D]. */
  embed(ids, roles, levels) {
    const D = this.dim;
    const x = new Float32Array(ids.length * D);
    for (let p = 0; p < ids.length; p++) {
      if (roles[p] === 0) continue; // x[roles == 0] = 0
      const t = ids[p] * D;
      const r = roles[p] * D;
      const l = levels[p] * D;
      const o = p * D;
      for (let d = 0; d < D; d++) x[o + d] = f32(f32(this.tok[t + d] + this.role[r + d]) + this.level[l + d]);
    }
    return x;
  }

  /** Model input for one window: int8 via np.clip(np.round(x / scale + zp), -128, 127), or float32. */
  quantize(x) {
    const { dtype, scale, zeroPoint } = this.io.input;
    if (dtype === "float32") return x;
    if (dtype !== "int8") throw new Error(`unsupported model input type ${dtype}`);
    const s = f32(scale);
    const zp = f32(zeroPoint);
    const q = new Int8Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = roundHalfEven(f32(f32(x[i] / s) + zp));
      q[i] = v < -128 ? -128 : v > 127 ? 127 : v;
    }
    return q;
  }

  /** Encoder output for one quantised window, as raw tensor values (Int8Array or Float32Array). */
  async invoke(input) {
    const tensor = new Tensor(input, [1, ...this.grid, this.dim]);
    try {
      const [out] = await this.model.run(tensor);
      try {
        return out.toTypedArray();
      } finally {
        out.delete();
      }
    } finally {
      tensor.delete();
    }
  }

  dequantize(y) {
    const { dtype, scale, zeroPoint } = this.io.output;
    if (dtype === "float32") return Float32Array.from(y);
    const s = f32(scale);
    return Float32Array.from(y, (v) => f32((v - zeroPoint) * s)); // (y.astype(f32) - zp) * scale
  }

  /** Encoder logits for one packed window, shape [seq_len]. */
  async runWindow(ids, roles, levels) {
    return this.dequantize(await this.invoke(this.quantize(this.embed(ids, roles, levels))));
  }

  /**
   * Probabilities for keyed question parts (Reef._score_parts): every schema window runs once per
   * state region and each part's option logits are averaged over regions before the softmax.
   * The sum and the division stay in float32, as numpy does with float32 logits.
   */
  async scoreParts(regions, parts, trace, where) {
    const keys = [...parts.keys()];
    const windows = this.packer.packBlocks(regions[0].ids, regions[0].roles, keys.map((k) => [k, this.packer.block(parts.get(k))]));
    const acc = new Map();
    let tDev = 0;
    for (const w of windows) {
      for (const r of regions) {
        const { ids, roles } = this.packer.withState(w, r.ids, r.roles);
        const x = this.quantize(this.embed(ids, roles, w.levels));
        const t1 = performance.now();
        const y = await this.invoke(x);
        tDev += performance.now() - t1;
        const logits = this.dequantize(y);
        for (const [key, positions] of w.slots) {
          const prev = acc.get(key);
          acc.set(key, positions.map((p, i) => (prev ? f32(prev[i] + logits[p]) : logits[p])));
        }
        if (trace) trace.push({ ...where, ids: Array.from(ids), roles: Array.from(roles), levels: Array.from(w.levels), input: x, output: Array.from(y) });
      }
    }
    const n = regions.length;
    const probs = new Map(
      keys.map((k) => [k, softmax(acc.get(k).map((v) => f32(v / n)), this.temperatures[parts.get(k).type] ?? 1.0)]),
    );
    return { probs, windows: windows.length * n, tDev };
  }

  /**
   * Answer every question about `state`. `state`: string (JSON text or plain text) or parsed value;
   * `questions`: JSON text or PyDict. Returns a SystemOne-style response body.
   * Long states are read in overlapping chunks (up to maxStateWindows) and questions with more
   * options than one window compares are answered as a knockout over parts, as reef.runtime does.
   * With instruction hoisting, a shared prefix moves into the state, and questions whose own
   * instructions are still too long are answered one at a time on their own hoisted state.
   * `debug: true` adds the packed windows and raw outputs (used by the parity test).
   */
  async decide(state, questions, { debug = false } = {}) {
    const t0 = performance.now();
    const asked = normalizeQuestions(questions);
    const groups = planGroups(this.packer, asked, renderState(state), {
      hoist: this.hoist,
      hoistEmpty: this.hoistEmpty,
      instructionCap: this.instructionCap,
    });
    const trace = debug ? [] : null;
    const byIndex = new Map();
    const usage = { state_tokens: 0, state_truncated: false, state_windows: 0, windows: 0, rounds: 0 };
    let tDev = 0;
    for (const [gi, g] of groups.entries()) {
      const r = await this.answerGroup(g.qs, g.stateText, trace, gi);
      g.idx.forEach((qi, k) => byIndex.set(qi, r.answers[k]));
      tDev += r.tDev;
      for (const k of ["state_tokens", "state_windows", "rounds"]) usage[k] = Math.max(usage[k], r.usage[k]);
      usage.windows += r.usage.windows;
      usage.state_truncated ||= r.usage.state_truncated;
      if (debug) g.stateIds = r.stateIds;
    }
    if (groups.length > 1) usage.groups = groups.length;
    const out = {
      model: this.name,
      // Python dicts keep insertion order; a Map does too for numeric-looking ids. Request order.
      answers: new Map(asked.map((q, qi) => [q.id, byIndex.get(qi)])),
      questions: asked, // as asked (before hoisting), for display
      usage: {
        ...usage,
        device: this.device,
        device_ms: Math.round(tDev * 1e3) / 1e3,
        total_ms: Math.round((performance.now() - t0) * 1e3) / 1e3,
      },
    };
    if (debug) Object.assign(out, { trace, groups });
    return out;
  }

  /** Answer already-hoisted questions about one state (Reef._answer): chunked state, knockout rounds. */
  async answerGroup(qs, stateText, trace, group) {
    const { stateIds, regions, gaps } = prepareState(this.packer, stateText, this.maxStateWindows);
    let current = new Map(qs.map((q, qi) => [qi, q.options.map((_, i) => i)]));
    const rounds = qs.map(() => []); // per question: [chunks, chunk probs] per round
    const final = new Array(qs.length);
    let nWindows = 0;
    let nRounds = 0;
    let tDev = 0;
    while (current.size) {
      const { parts, plan } = planRound(this.packer, qs, current);
      const r = await this.scoreParts(regions, parts, trace, { group, round: nRounds });
      nWindows += r.windows;
      nRounds += 1;
      tDev += r.tDev;
      const next = new Map();
      for (const [qi, chunks] of plan) {
        if (chunks === null) final[qi] = [current.get(qi), r.probs.get(`${qi},0`)];
        else {
          const within = chunks.map((_, ci) => r.probs.get(`${qi},${ci}`));
          rounds[qi].push([chunks, within]);
          next.set(qi, chunks.map((c, ci) => c[argmax(within[ci])]));
        }
      }
      current = next;
    }
    const answers = qs.map((q, qi) => {
      const [idx, p] = final[qi];
      let dist = new Map(idx.map((o, i) => [o, p[i]]));
      // Walk back through the rounds: an option knocked out in a part gets its part winner's
      // mass scaled by how it compared with that winner. The final-round winner stays on top.
      for (const [chunks, within] of [...rounds[qi]].reverse()) {
        const d = new Map();
        chunks.forEach((c, ci) => {
          const w = within[ci];
          const m = dist.get(c[argmax(w)]);
          const top = Math.max(Math.max(...w), 1e-12);
          c.forEach((o, k) => d.set(o, (m * w[k]) / top));
        });
        dist = d;
      }
      const probs = q.options.map((_, i) => dist.get(i));
      const total = npSum(probs);
      return makeAnswer(q, probs.map((v) => v / total));
    });
    const usage = { state_tokens: stateIds.length, state_truncated: gaps, state_windows: regions.length, windows: nWindows, rounds: nRounds };
    return { answers, usage, tDev, stateIds };
  }
}

/** Python bool() of a JSON value. */
function pyBool(v) {
  if (Array.isArray(v)) return v.length > 0;
  if (v && typeof v === "object") return Object.keys(v).length > 0;
  return Boolean(v);
}
