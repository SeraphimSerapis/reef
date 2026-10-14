// A minimal, exact port of the HF `tokenizers` pipeline used by Reef's tokenizer.json:
// BertNormalizer (clean text, CJK padding, strip accents, lowercase) -> BertPreTokenizer
// (split on whitespace, isolate punctuation) -> WordPiece (greedy longest match, "##" prefix).
// Added/special tokens ([PAD], [CLS], ...) are matched in the raw text first, like the Rust
// AddedVocabulary does (leftmost-longest). Only what `encode(text, add_special_tokens=False).ids`
// needs is implemented; unsupported tokenizer configurations throw instead of tokenising wrongly.

const RE_CONTROL = /\p{C}/u; // Rust `char::is_other`: Cc | Cf | Cs | Co | Cn
const RE_WHITESPACE = /\p{White_Space}/u; // Rust `char::is_whitespace`
const RE_PUNCT = /\p{P}/u; // Pc Pd Ps Pe Pi Pf Po
const RE_MN = /\p{Mn}/u;
const ASCII_PUNCT = new Set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");

function isChineseChar(cp) {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b920 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

export class WordPieceTokenizer {
  /** @param {object} spec parsed tokenizer.json */
  constructor(spec) {
    const { normalizer: n, pre_tokenizer: pre, model } = spec;
    if (model?.type !== "WordPiece") throw new Error(`unsupported tokenizer model ${model?.type}`);
    if (n && n.type !== "BertNormalizer") throw new Error(`unsupported normalizer ${n.type}`);
    if (pre && pre.type !== "BertPreTokenizer") throw new Error(`unsupported pre-tokenizer ${pre.type}`);
    this.norm = n
      ? {
          cleanText: n.clean_text ?? true,
          chinese: n.handle_chinese_chars ?? true,
          lowercase: n.lowercase ?? true,
          stripAccents: n.strip_accents ?? n.lowercase ?? true, // Rust: strip_accents.unwrap_or(lowercase)
        }
      : null;
    this.preTokenize = !!pre;
    this.vocab = new Map(Object.entries(model.vocab));
    this.unkId = this.vocab.get(model.unk_token);
    this.prefix = model.continuing_subword_prefix ?? "##";
    this.maxChars = model.max_input_chars_per_word ?? 100;
    this.added = (spec.added_tokens ?? []).map((t) => {
      if (t.normalized || t.single_word || t.lstrip || t.rstrip) {
        throw new Error(`unsupported added-token options on ${t.content}`);
      }
      return { content: t.content, id: t.id };
    });
    for (const t of this.added) if (!this.vocab.has(t.content)) this.vocab.set(t.content, t.id);
  }

  tokenToId(token) {
    return this.vocab.get(token);
  }

  /** Token ids for `text`, without [CLS]/[SEP] (= `encode(text, add_special_tokens=False).ids`). */
  encode(text) {
    const ids = [];
    for (const [seg, addedId] of this.splitAdded(text)) {
      if (addedId !== undefined) ids.push(addedId);
      else for (const word of this.preTokenizeText(this.normalize(seg))) this.wordPiece(word, ids);
    }
    return ids;
  }

  // Split on added tokens, leftmost-longest. Yields [segment, undefined] or [content, id].
  *splitAdded(text) {
    if (!this.added.length) {
      yield [text, undefined];
      return;
    }
    let start = 0;
    let i = 0;
    while (i < text.length) {
      let best = null;
      for (const t of this.added) {
        if (text.startsWith(t.content, i) && (!best || t.content.length > best.content.length)) best = t;
      }
      if (best) {
        if (i > start) yield [text.slice(start, i), undefined];
        yield [best.content, best.id];
        i += best.content.length;
        start = i;
      } else {
        i++;
      }
    }
    if (start < text.length) yield [text.slice(start), undefined];
  }

  normalize(text) {
    const n = this.norm;
    if (!n) return text;
    let out = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (n.cleanText) {
        // Rust filters controls first, then maps whitespace, so control characters that are also
        // whitespace (U+0085, \v, \f, \x1c-\x1f) are dropped rather than turned into spaces.
        if (cp === 0 || cp === 0xfffd) continue;
        const tnr = ch === "\t" || ch === "\n" || ch === "\r";
        if (!tnr && RE_CONTROL.test(ch)) continue;
        if (tnr || RE_WHITESPACE.test(ch)) {
          out += " ";
          continue;
        }
      }
      out += n.chinese && isChineseChar(cp) ? ` ${ch} ` : ch;
    }
    if (n.stripAccents) {
      let s = "";
      for (const ch of out.normalize("NFD")) if (!RE_MN.test(ch)) s += ch;
      out = s;
    }
    if (n.lowercase) {
      // Per character, like Rust's char::to_lowercase (no context-dependent final sigma).
      let s = "";
      for (const ch of out) s += ch.toLowerCase();
      out = s;
    }
    return out;
  }

  preTokenizeText(text) {
    if (!this.preTokenize) return text.length ? [text] : [];
    const words = [];
    let cur = "";
    for (const ch of text) {
      if (RE_WHITESPACE.test(ch)) {
        if (cur) words.push(cur);
        cur = "";
      } else if (ASCII_PUNCT.has(ch) || RE_PUNCT.test(ch)) {
        if (cur) words.push(cur);
        words.push(ch);
        cur = "";
      } else {
        cur += ch;
      }
    }
    if (cur) words.push(cur);
    return words;
  }

  wordPiece(word, ids) {
    const chars = Array.from(word); // code points, like Rust chars
    if (chars.length > this.maxChars) {
      ids.push(this.unkId);
      return;
    }
    const pieces = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let found;
      while (start < end) {
        let sub = chars.slice(start, end).join("");
        if (start > 0) sub = this.prefix + sub;
        found = this.vocab.get(sub);
        if (found !== undefined) break;
        end--;
      }
      if (found === undefined) {
        ids.push(this.unkId);
        return;
      }
      pieces.push(found);
      start = end;
    }
    for (const p of pieces) ids.push(p);
  }
}
