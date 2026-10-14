// Port of src/reef/packing.py and src/reef/decode.py: questions -> option lists, token windows,
// and logits -> SystemOne-shaped answers. See packing.py for the window layout.

import { PY_SPACE, PyDict, PyFloat, PyInt, parseJSON, pyRound, pyRoundInt, pyStr, pyStrip, pyTruthy } from "./pyjson.js";
import { humanize } from "./text.js";

// Roles: what kind of position this is.
export const [PAD, BOUNDARY, STATE, Q_NOUL, Q_CHOICE, Q_SCORE, Q_TEXT, OPT, OPT_TEXT] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
export const Q_ROLE = { noul: Q_NOUL, choice: Q_CHOICE, score: Q_SCORE };
const SCORE_BUCKETS = 5;
// Options compared within one window at inference; training never sees more (data.MAX_OPTIONS).
export const MAX_PART_OPTIONS = 8;
const DEFAULT_NOUL = { true: "yes, the statement is true", false: "no, the statement is false" };

function level(i, n) {
  if (n <= 1) return 1;
  return 1 + pyRoundInt((i / (n - 1)) * (SCORE_BUCKETS - 1)); // Python round(): half-even
}

// Python int(str) for score criteria keys like "0", " 2", "+3", "1_0".
function pyInt(s) {
  const t = pyStrip(s);
  if (!/^[+-]?\d+(_\d+)*$/.test(t)) throw new Error(`invalid literal for int(): ${JSON.stringify(s)}`);
  return Number(t.replaceAll("_", ""));
}

function textOf(v, what) {
  if (typeof v !== "string") throw new Error(`${what} must be a string`);
  return v;
}

/** Validate one SystemOne question and turn it into a list of options with readable text. */
export function normalizeQuestion(qid, q) {
  if (!(q instanceof PyDict)) throw new Error(`question ${JSON.stringify(qid)} must be an object`);
  const qtype = q.get("type");
  if (!(typeof qtype === "string" && qtype in Q_ROLE)) {
    throw new Error(`question '${qid}': type must be noul, choice or score, got ${JSON.stringify(pyStr(qtype ?? null))}`);
  }
  const rawInstr = q.get("instructions");
  const instructions = pyStrip(pyTruthy(rawInstr) ? textOf(rawInstr, "instructions") : humanize(qid));
  const criteria = q.get("criteria");
  let opts;
  if (qtype === "noul") {
    const crit = new Map(Object.entries(DEFAULT_NOUL));
    if (criteria instanceof PyDict) for (const [k, v] of criteria) crit.set(k.toLowerCase(), v);
    opts = ["true", "false"].map((k) => ({ id: k, text: `${k}: ${pyStr(crit.get(k))}`, level: 0 }));
  } else if (qtype === "choice") {
    let items;
    if (criteria instanceof PyDict) items = [...criteria].map(([k, v]) => [k, pyStr(v)]);
    else if (Array.isArray(criteria)) items = criteria.map((k) => [pyStr(k), ""]);
    else throw new Error(`choice question '${qid}' needs criteria`);
    if (items.length < 2) throw new Error(`choice question '${qid}' needs at least two options`);
    opts = items.map(([k, v]) => {
      const o = { id: k, text: v ? `${humanize(k)}: ${v}` : humanize(k), level: 0 };
      // The description alone, when the key carries no meaning ("A", "3", "option_12").
      if (v && isOpaqueKey(k)) o.bare = v;
      return o;
    });
  } else {
    let texts;
    if (Array.isArray(criteria)) texts = criteria.map(pyStr);
    else if (criteria instanceof PyDict) {
      // sorted() is stable, like Array.prototype.sort
      texts = [...criteria.keys()].sort((a, b) => pyInt(a) - pyInt(b)).map((k) => pyStr(criteria.get(k)));
    } else throw new Error(`score question '${qid}' needs a criteria list`);
    if (texts.length < 2) throw new Error(`score question '${qid}' needs at least two levels`);
    opts = texts.map((t, i) => ({ id: String(i), text: t, level: level(i, texts.length) }));
  }
  return { id: qid, type: qtype, instructions, options: opts };
}

/** `questions`: JSON text or a parsed PyDict, in request order. */
export function normalizeQuestions(questions) {
  if (typeof questions === "string") questions = parseJSON(questions);
  if (!(questions instanceof PyDict)) throw new Error("questions must be a JSON object");
  if (!questions.size) throw new Error("questions must be a non-empty mapping");
  return [...questions].map(([k, v]) => normalizeQuestion(k, v));
}

// -- packing._OPAQUE_KEY ---------------------------------------------------------------------------
// r"(?i)^(?:[a-z]|\d+|(?:option|opt|choice|answer|candidate|label|class|item)[ _-]?\d+)$" with
// Python re semantics: IGNORECASE also matches İ ı (for i), ſ (for s) and K (Kelvin, for k);
// \d is a Unicode decimal digit (Python 3.12 / Unicode 15 table below); $ also matches before a
// final "\n".
const PY_DIGITS = "30-39 660-669 6f0-6f9 7c0-7c9 966-96f 9e6-9ef a66-a6f ae6-aef b66-b6f be6-bef c66-c6f ce6-cef d66-d6f de6-def e50-e59 ed0-ed9 f20-f29 1040-1049 1090-1099 17e0-17e9 1810-1819 1946-194f 19d0-19d9 1a80-1a89 1a90-1a99 1b50-1b59 1bb0-1bb9 1c40-1c49 1c50-1c59 a620-a629 a8d0-a8d9 a900-a909 a9d0-a9d9 a9f0-a9f9 aa50-aa59 abf0-abf9 ff10-ff19 104a0-104a9 10d30-10d39 11066-1106f 110f0-110f9 11136-1113f 111d0-111d9 112f0-112f9 11450-11459 114d0-114d9 11650-11659 116c0-116c9 11730-11739 118e0-118e9 11950-11959 11c50-11c59 11d50-11d59 11da0-11da9 11f50-11f59 16a60-16a69 16ac0-16ac9 16b50-16b59 1d7ce-1d7ff 1e140-1e149 1e2f0-1e2f9 1e4f0-1e4f9 1e950-1e959 1fbf0-1fbf9"
  .split(" ")
  .map((r) => r.split("-").map((h) => parseInt(h, 16)));
const FOLD_EXTRA = { i: "\u0130\u0131", k: "\u212a", s: "\u017f" };
const OPAQUE_WORDS = ["option", "opt", "choice", "answer", "candidate", "label", "class", "item"];

function pyDigit(ch) {
  const cp = ch.codePointAt(0);
  return PY_DIGITS.some(([a, b]) => cp >= a && cp <= b);
}

function ciLetter(ch, l) {
  return ch === l || ch === l.toUpperCase() || (FOLD_EXTRA[l] ?? "").includes(ch);
}

function allDigits(cps) {
  return cps.length > 0 && cps.every(pyDigit);
}

/** packing._OPAQUE_KEY.match(key): a choice key that carries no meaning ("A", "3", "option_12"). */
export function isOpaqueKey(key) {
  let cps = [...key]; // code points
  if (cps.at(-1) === "\n") cps = cps.slice(0, -1);
  if (cps.length === 1 && (/^[a-zA-Z]$/.test(cps[0]) || "\u0130\u0131\u017f\u212a".includes(cps[0]))) return true;
  if (allDigits(cps)) return true;
  return OPAQUE_WORDS.some((w) => {
    if (cps.length <= w.length || ![...w].every((l, i) => ciLetter(cps[i], l))) return false;
    const rest = cps.slice(w.length);
    if (" _-".includes(rest[0]) && allDigits(rest.slice(1))) return true;
    return allDigits(rest);
  });
}

/** A request Reef cannot read in full (same wording as reef.packing.CapacityError). */
export class CapacityError extends Error {}

export class Packer {
  /**
   * Non-strict only (the browser truncates over-long instructions and options, like training).
   * `bands`: split the schema region into this many fixed bands of schema_len / bands positions,
   * one question block per band (v2 models; 0 = blocks packed back to back).
   * `dropOpaqueKeys`: render options whose keys carry no meaning (`bare`) as their description alone.
   */
  constructor(tokenizer, stateLen = 384, schemaLen = 384, maxInstructionTokens = 40, maxOptionTokens = 28, { bands = 0, dropOpaqueKeys = false } = {}) {
    this.tok = tokenizer;
    this.stateLen = stateLen;
    this.schemaLen = schemaLen;
    this.maxInstructionTokens = maxInstructionTokens;
    this.maxOptionTokens = maxOptionTokens;
    this.bands = bands;
    this.dropOpaqueKeys = dropOpaqueKeys;
    if (bands && schemaLen % bands) throw new Error(`schema_len=${schemaLen} is not divisible into ${bands} bands`);
    this.blockLen = bands ? Math.floor(schemaLen / bands) : schemaLen; // positions one block may use
    this.clsId = tokenizer.tokenToId("[CLS]");
    this.sepId = tokenizer.tokenToId("[SEP]");
    this.markerId = tokenizer.tokenToId("[PAD]") || 0;
    this.cache = new Map(); // question blocks depend only on the schema
  }

  get seqLen() {
    return this.stateLen + this.schemaLen;
  }

  encode(text) {
    return this.tok.encode(text);
  }

  /** The (cached) block for `q`, whether or not it fits the schema region. */
  buildBlock(q) {
    const key = JSON.stringify(q);
    let b = this.cache.get(key);
    if (b) return b;
    const ids = [this.markerId];
    const roles = [Q_ROLE[q.type]];
    const levels = [0];
    const offs = [];
    const instr = this.encode(q.instructions).slice(0, this.maxInstructionTokens);
    for (const t of instr) ids.push(t), roles.push(Q_TEXT), levels.push(0);
    for (const o of q.options) {
      offs.push(ids.length);
      const text = this.dropOpaqueKeys && o.bare ? o.bare : o.text;
      const t = this.encode(text).slice(0, this.maxOptionTokens);
      ids.push(this.markerId, ...t);
      roles.push(OPT, ...t.map(() => OPT_TEXT));
      levels.push(o.level, ...t.map(() => o.level));
    }
    b = { ids, roles, levels, optionOffsets: offs };
    if (this.cache.size > 4096) this.cache.clear();
    this.cache.set(key, b);
    return b;
  }

  block(q) {
    const b = this.buildBlock(q);
    if (b.ids.length > this.blockLen) {
      throw new CapacityError(
        `question '${q.id}' needs ${b.ids.length} positions, over Reef's maximum context length of ${this.blockLen}`,
      );
    }
    return b;
  }

  blockFits(q) {
    return this.buildBlock(q).ids.length <= this.blockLen;
  }

  /**
   * Split the options `idx` of `q` into consecutive chunks of at most `maxOptions` whose blocks
   * (instructions repeated) each fit one question's block. Parts are kept as even as possible
   * (9 options give 5 + 4): the greedy split gives the number of parts; if an even split with that
   * many parts does not fit, the greedy split is used.
   */
  splitOptions(q, idx, maxOptions = MAX_PART_OPTIONS) {
    const chunks = [[]];
    for (const i of idx) {
      const last = chunks.at(-1);
      const cand = [...last, i];
      if (last.length && (cand.length > maxOptions || !this.blockFits(subquestion(q, cand)))) chunks.push([i]);
      else chunks[chunks.length - 1] = cand;
    }
    const k = chunks.length;
    if (k < 2) return chunks;
    const base = Math.floor(idx.length / k);
    const extra = idx.length % k;
    const even = [];
    let start = 0;
    for (let j = 0; j < k; j++) {
      const size = base + (j < extra ? 1 : 0);
      even.push(idx.slice(start, start + size));
      start += size;
    }
    if (even.every((c) => this.blockFits(subquestion(q, c)))) return even;
    return chunks;
  }

  /** `[CLS] state [SEP]` tiled to exactly state_len positions. */
  stateRegion(stateIds) {
    const S = this.stateLen;
    const truncated = stateIds.length > S - 2;
    const unitIds = [this.clsId, ...stateIds.slice(0, S - 2), this.sepId];
    const unitRoles = unitIds.map((_, i) => (i === 0 || i === unitIds.length - 1 ? BOUNDARY : STATE));
    const ids = new Int32Array(S);
    const roles = new Int32Array(S);
    for (let i = 0; i < S; i++) {
      ids[i] = unitIds[i % unitIds.length];
      roles[i] = unitRoles[i % unitIds.length];
    }
    return { ids, roles, truncated };
  }

  /**
   * State regions covering a long state with overlapping chunks (Packer.state_regions). A state
   * that fits gives one region; a longer one is cut into chunks of state_len - 2 tokens that
   * overlap by `overlap`; past `maxWindows` chunks they are spread evenly and `gaps` is true.
   */
  stateRegions(stateIds, maxWindows = 8, overlap = 0.25) {
    const n = stateIds.length;
    const c = this.stateLen - 2;
    if (n <= c || maxWindows <= 1) {
      const { ids, roles, truncated } = this.stateRegion(stateIds);
      return { regions: [{ ids, roles }], gaps: truncated };
    }
    const stride = Math.max(1, Math.trunc(c * (1 - overlap)));
    const set = new Set();
    for (let s = 0; s < n - c + stride; s += stride) set.add(Math.min(s, n - c));
    let starts = [...set].sort((a, b) => a - b);
    const gaps = starts.length > maxWindows;
    if (gaps) {
      // np.linspace(0, n - c, maxWindows): i * step + 0, last element exactly the stop
      const div = maxWindows - 1;
      const step = (n - c) / div;
      const ls = Array.from({ length: maxWindows }, (_, i) => (i === div ? n - c : i * step));
      starts = [...new Set(ls.map((s) => pyRoundInt(s)))].sort((a, b) => a - b);
    }
    const regions = starts.map((s) => {
      const { ids, roles } = this.stateRegion(stateIds.slice(s, s + c));
      return { ids, roles };
    });
    return { regions, gaps };
  }

  /** One state region (truncated if long) and every question in order (training packing). */
  pack(stateIds, questions) {
    const s = this.stateRegion(stateIds);
    return this.packBlocks(s.ids, s.roles, questions.map((q, qi) => [qi, this.block(q)]), s.truncated);
  }

  /**
   * Lay keyed question blocks after a state region, starting a new window when the schema region
   * is full (banded: when every band holds a block). Each window's `slots` hold `[key, option positions]`.
   */
  packBlocks(sIds, sRoles, blocks, truncated = false) {
    const s = { ids: sIds, roles: sRoles, truncated };
    if (this.bands) {
      const out = [];
      for (let i = 0; i < blocks.length; i += this.bands) out.push(this.window(s, blocks.slice(i, i + this.bands)));
      return out;
    }
    const groups = [[]];
    let used = 0;
    blocks.forEach(([, b], i) => {
      if (used + b.ids.length > this.schemaLen && groups.at(-1).length) {
        groups.push([]);
        used = 0;
      }
      groups.at(-1).push(i);
      used += b.ids.length;
    });
    return groups.map((g) => this.window(s, g.map((i) => blocks[i])));
  }

  window(s, blocks) {
    const L = this.seqLen;
    const S = this.stateLen;
    const ids = new Int32Array(L);
    const roles = new Int32Array(L);
    const levels = new Int32Array(L);
    ids.set(s.ids);
    roles.set(s.roles);
    let pos = S;
    const slots = [];
    blocks.forEach(([key, b], j) => {
      if (this.bands) pos = S + j * this.blockLen; // band j starts at a fixed offset; the rest stays PAD
      ids.set(b.ids, pos);
      roles.set(b.roles, pos);
      levels.set(b.levels, pos);
      slots.push([key, b.optionOffsets.map((o) => pos + o)]);
      pos += b.ids.length;
    });
    return { ids, roles, levels, slots, stateTruncated: s.truncated };
  }

  /** `w`'s ids and roles with its state region replaced (the schema region is unchanged). */
  withState(w, sIds, sRoles) {
    const ids = w.ids.slice();
    const roles = w.roles.slice();
    ids.set(sIds);
    roles.set(sRoles);
    return { ids, roles };
  }
}

/** `q` restricted to the options at `idx` (levels are kept from the full rubric). */
export function subquestion(q, idx) {
  return { id: q.id, type: q.type, instructions: q.instructions, options: idx.map((i) => q.options[i]) };
}

// -- hoisting --------------------------------------------------------------------------------------

export const HOIST_STUB = "Answer the question described in the state.";
// re.split(r"\n\s*\n", text): Python's \s on str is exactly str.isspace() (PY_SPACE), not JS's \s.
const PARA_SPLIT = new RegExp(`\\n[${PY_SPACE}]*\\n`);

function paragraphs(text) {
  let parts = text.split(PARA_SPLIT).map(pyStrip).filter((p) => p);
  if (parts.length <= 1) parts = text.split("\n").map(pyStrip).filter((p) => p);
  return parts.length ? parts : [text];
}

function blank(stateText) {
  return ["", "none", ": none", "null"].includes(pyStrip(stateText));
}

/**
 * Port of reef.packing.hoist_instructions: move instruction text that is too long for a question
 * block into the state. One question: its leading paragraphs move and the trailing paragraphs
 * that fit stay (else a stub); several: the paragraphs they all start with move. With `empty`
 * (models trained with hoist_empty_state) and an empty state, the input moves even when it fits.
 * `count` maps text to a token count. Returns `[questions, stateText]`.
 */
export function hoistInstructions(questions, stateText, count, cap, force = false, empty = false) {
  if (empty && questions.length && blank(stateText)) {
    if (questions.length === 1) {
      const q = questions[0];
      return [[{ id: q.id, type: q.type, instructions: HOIST_STUB, options: q.options }], q.instructions];
    }
    force = true;
  }
  if (!force && questions.every((q) => count(q.instructions) <= cap)) return [questions, stateText];
  const paras = questions.map((q) => paragraphs(q.instructions));
  let hoisted;
  let rest;
  if (questions.length === 1) {
    const p = paras[0];
    let keep = [];
    for (let i = p.length - 1; i >= 0; i--) {
      if (count([p[i], ...keep].join("\n\n")) > cap) break;
      keep.unshift(p[i]);
    }
    if (keep.length === p.length) {
      if (!force) return [questions, stateText];
      keep = keep.length > 1 ? keep.slice(1) : [];
    }
    hoisted = p.slice(0, p.length - keep.length);
    rest = [keep.join("\n\n")];
    if (!keep.length) {
      // e.g. a task paragraph followed by a long document: keep the leading task
      const lead = [];
      for (const para of p.slice(0, -1)) {
        if (count([...lead, para].join("\n\n")) > cap) break;
        lead.push(para);
      }
      if (lead.length) {
        hoisted = p.slice(lead.length);
        rest = [lead.join("\n\n")];
      }
    }
  } else {
    let n = 0;
    while (paras.every((pp) => pp.length > n) && new Set(paras.map((pp) => pp[n])).size === 1) n++;
    if (n === 0) return [questions, stateText];
    hoisted = paras[0].slice(0, n);
    rest = paras.map((pp) => pp.slice(n).join("\n\n"));
  }
  let text = hoisted.join("\n\n");
  const s = pyStrip(stateText);
  if (s && s !== "none") text += `\n\n${stateText}`;
  return [questions.map((q, i) => ({ id: q.id, type: q.type, instructions: rest[i] || HOIST_STUB, options: q.options })), text];
}

/**
 * Port of reef.packing.hoist_groups: plan hoisting as groups `{idx, qs, stateText}`.
 * hoistInstructions runs first; if several questions remain whose own instructions are still over
 * `cap` (one long document per question), each becomes its own group, hoisted alone from its
 * original instructions and state; the questions that fit stay together in one (first) group.
 * With `empty` and an empty state that stays empty (several questions share no leading paragraph),
 * every question becomes its own group with its input as the state.
 */
export function hoistGroups(questions, stateText, count, cap, force = false, empty = false) {
  const [hq, hs] = hoistInstructions(questions, stateText, count, cap, force, empty);
  if (empty && hq.length > 1 && blank(stateText) && blank(hs)) {
    return hq.map((_, i) => {
      const [gq, gs] = hoistInstructions([questions[i]], stateText, count, cap, false, true);
      return { idx: [i], qs: gq, stateText: gs };
    });
  }
  const long = [];
  hq.forEach((q, i) => count(q.instructions) > cap && long.push(i));
  if (hq.length < 2 || !long.length) return [{ idx: hq.map((_, i) => i), qs: hq, stateText: hs }];
  const rest = hq.map((_, i) => i).filter((i) => !long.includes(i));
  const groups = long.map((i) => {
    const [gq, gs] = hoistInstructions([questions[i]], stateText, count, cap);
    return { idx: [i], qs: gq, stateText: gs };
  });
  return (rest.length ? [{ idx: rest, qs: rest.map((i) => hq[i]), stateText: hs }] : []).concat(groups);
}

// -- request planning (shared by Reef.decide and the parity test) --------------------------------

/** The groups Reef.decide answers separately: one group unless hoisting splits the request. */
export function planGroups(packer, qs, stateText, { hoist = false, hoistEmpty = false, instructionCap = 40 } = {}) {
  if (!hoist) return [{ idx: qs.map((_, i) => i), qs, stateText }];
  return hoistGroups(qs, stateText, (t) => packer.encode(t).length, instructionCap, false, hoistEmpty);
}

/** Tokenise a group's state and cut it into regions (the start of Reef._answer). */
export function prepareState(packer, stateText, maxStateWindows = 8) {
  const stateIds = packer.encode(stateText);
  const { regions, gaps } = packer.stateRegions(stateIds, maxStateWindows);
  return { stateIds, regions, gaps };
}

/**
 * One knockout round: `current` maps question index -> remaining option indices. Returns `parts`
 * (Map "qi,ci" -> sub-question, in Python's dict order) and `plan` (Map qi -> null when the
 * question is answered in this round, else its option chunks).
 */
export function planRound(packer, qs, current) {
  const parts = new Map();
  const plan = new Map();
  for (const [qi, idx] of current) {
    const sub = subquestion(qs[qi], idx);
    if (idx.length <= MAX_PART_OPTIONS && packer.blockFits(sub)) {
      plan.set(qi, null);
      parts.set(`${qi},0`, sub);
    } else {
      const chunks = packer.splitOptions(qs[qi], idx);
      if (chunks.length >= idx.length) throw new CapacityError(`question '${qs[qi].id}' does not fit Reef's maximum context length`);
      plan.set(qi, chunks);
      chunks.forEach((c, ci) => parts.set(`${qi},${ci}`, subquestion(qs[qi], c)));
    }
  }
  return { parts, plan };
}

/** np.argmax: index of the first maximum. */
export function argmax(p) {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
}

// -- decode --------------------------------------------------------------------------------------

/** numpy's float64 add.reduce of a contiguous array: 0.0 + pairwise_sum (8 accumulators, blocks of 128). */
export function npSum(a, lo = 0, n = a.length) {
  return 0.0 + pairwiseSum(a, lo, n);
}

function pairwiseSum(a, lo, n) {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += a[lo + i];
    return res;
  }
  if (n <= 128) {
    const r = a.slice(lo, lo + 8);
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

/** decode.softmax: float64, divide by the temperature, subtract the max, exp, numpy sum. */
export function softmax(x, t = 1.0) {
  const z = Array.from(x, (v) => v / Math.max(t, 1e-6));
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = npSum(e);
  return e.map((v) => v / s);
}

/** How far the top option sits above uniform, rescaled to [0, 1]. */
export function confidence(p) {
  const k = p.length;
  return k > 1 ? (Math.max(...p) - 1 / k) / (1 - 1 / k) : 1.0;
}

export function questionProbs(logits, positions, qtype, temperatures) {
  return softmax(positions.map((i) => logits[i]), temperatures[qtype] ?? 1.0);
}

/**
 * A SystemOne-shaped answer for one question (same fields and 6-decimal rounding as Python).
 * Unrounded probabilities are kept under the non-enumerable `raw` key for the UI and tests.
 */
export function makeAnswer(q, probs) {
  const ids = q.options.map((o) => o.id);
  const pd = Object.fromEntries(ids.map((id, i) => [id, pyRound(probs[i], 6)]));
  const conf = pyRound(confidence(probs), 6);
  let ans;
  if (q.type === "noul") ans = { type: "noul", noul: pd.true, confidence: conf, probabilities: pd };
  else if (q.type === "choice") {
    ans = { type: "choice", choice: ids[argmax(probs)], confidence: conf, probabilities: pd };
  } else {
    const score = probs.reduce((a, p, i) => a + Number(ids[i]) * p, 0);
    ans = {
      type: "score",
      score: pyRound(score, 6),
      confidence: conf,
      probabilities: pd,
      legend: Object.fromEntries(q.options.map((o) => [o.id, o.text])),
    };
  }
  Object.defineProperty(ans, "raw", { value: probs, enumerable: false });
  return ans;
}

export { PyDict, PyFloat, PyInt };
