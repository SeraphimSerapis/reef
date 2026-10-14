// A JSON parser and Python-style formatting helpers that mirror what the Python runtime sees.
//
// Exact parity with `reef.text.render_state` needs two things plain `JSON.parse` loses:
//   * key order: JS objects list integer-like keys ("2", "10") first, Python dicts keep insertion
//     order. Objects parse to `PyDict` (a Map; duplicate keys keep the first position and the last
//     value, like `json.loads`).
//   * int vs float: Python renders `5` and `5.0` the same, but `100000000000000000000` (int) and
//     `1e20` (float) differently, and ints are exact at any size. Numbers parse to `PyInt` (the
//     literal text) or `PyFloat` (a double).
// Accepted syntax follows Python's `json.loads` (strict mode, plus NaN / Infinity / -Infinity).

export class PyDict extends Map {}

export class PyInt {
  constructor(text) {
    this.text = text === "-0" ? "0" : text; // str(int("-0")) == "0"; JSON ints have no leading zeros
  }
}

export class PyFloat {
  constructor(value) {
    this.value = value;
  }
}

export class PyJSONError extends SyntaxError {}

const WS = new Set([" ", "\t", "\n", "\r"]);
const NUMBER_RE = /(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?/y;
const ESCAPES = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };

/** Parse JSON text into PyDict / Array / PyInt / PyFloat / string / boolean / null. */
export function parseJSON(text) {
  let i = 0;
  const fail = (msg) => {
    throw new PyJSONError(`${msg} at position ${i}`);
  };
  const skip = () => {
    while (i < text.length && WS.has(text[i])) i++;
  };

  function parseString() {
    i++; // opening quote
    let out = "";
    let chunkStart = i;
    for (;;) {
      if (i >= text.length) fail("Unterminated string");
      const c = text.charCodeAt(i);
      if (c === 0x22) {
        out += text.slice(chunkStart, i);
        i++;
        return out;
      }
      if (c < 0x20) fail("Invalid control character");
      if (c !== 0x5c) {
        i++;
        continue;
      }
      out += text.slice(chunkStart, i);
      const e = text[i + 1];
      if (e === "u") {
        const hex = text.slice(i + 2, i + 6);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail("Invalid \\uXXXX escape");
        let cp = parseInt(hex, 16);
        i += 6;
        // Python joins a \uD8xx\uDCxx pair into one code point; lone surrogates are kept.
        if (cp >= 0xd800 && cp <= 0xdbff && text[i] === "\\" && text[i + 1] === "u") {
          const hex2 = text.slice(i + 2, i + 6);
          if (/^[0-9a-fA-F]{4}$/.test(hex2)) {
            const lo = parseInt(hex2, 16);
            if (lo >= 0xdc00 && lo <= 0xdfff) {
              cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
              i += 6;
            }
          }
        }
        out += String.fromCodePoint(cp);
      } else if (e !== undefined && e in ESCAPES) {
        out += ESCAPES[e];
        i += 2;
      } else {
        fail("Invalid \\escape");
      }
      chunkStart = i;
    }
  }

  function parseValue(depth) {
    if (depth > 5000) fail("Too deeply nested");
    skip();
    const c = text[i];
    if (c === '"') return parseString();
    if (c === "{") {
      i++;
      const d = new PyDict();
      skip();
      if (text[i] === "}") {
        i++;
        return d;
      }
      for (;;) {
        skip();
        if (text[i] !== '"') fail("Expecting property name enclosed in double quotes");
        const k = parseString();
        skip();
        if (text[i] !== ":") fail("Expecting ':' delimiter");
        i++;
        d.set(k, parseValue(depth + 1));
        skip();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "}") {
          i++;
          return d;
        }
        fail("Expecting ',' delimiter");
      }
    }
    if (c === "[") {
      i++;
      const a = [];
      skip();
      if (text[i] === "]") {
        i++;
        return a;
      }
      for (;;) {
        a.push(parseValue(depth + 1));
        skip();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "]") {
          i++;
          return a;
        }
        fail("Expecting ',' delimiter");
      }
    }
    for (const [lit, val] of [
      ["null", null],
      ["true", true],
      ["false", false],
      ["NaN", new PyFloat(NaN)],
      ["Infinity", new PyFloat(Infinity)],
      ["-Infinity", new PyFloat(-Infinity)],
    ]) {
      if (text.startsWith(lit, i)) {
        i += lit.length;
        return val;
      }
    }
    NUMBER_RE.lastIndex = i;
    const m = NUMBER_RE.exec(text);
    if (m) {
      i = NUMBER_RE.lastIndex;
      if (m[2] || m[3]) return new PyFloat(Number(m[0]));
      return new PyInt(m[1]);
    }
    fail("Expecting value");
  }

  const v = parseValue(0);
  skip();
  if (i !== text.length) fail("Extra data");
  return v;
}

/** Convert a plain JS value (e.g. from JSON.parse) to the tagged form. Prefer parseJSON on text. */
export function fromJS(v) {
  if (v === null || typeof v === "string" || typeof v === "boolean") return v;
  if (typeof v === "number") return Number.isInteger(v) && Math.abs(v) <= Number.MAX_SAFE_INTEGER ? new PyInt(String(v)) : new PyFloat(v);
  if (v instanceof PyDict || v instanceof PyInt || v instanceof PyFloat) return v;
  if (Array.isArray(v)) return v.map(fromJS);
  if (typeof v === "object") return new PyDict(Object.entries(v).map(([k, x]) => [k, fromJS(x)]));
  throw new TypeError(`unsupported value ${v}`);
}

// -- Python number formatting --------------------------------------------------------------------
// Doubles are exactly m * 2^e, so their decimal expansion is finite. Using that (BigInt) gives the
// same correctly rounded, round-half-even results as CPython's round() and format().

/** Exact decimal of a finite double: |x| = digits * 10^exp (digits: BigInt). */
function exactDecimal(x) {
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, Math.abs(x));
  const hi = buf.getUint32(0);
  const lo = buf.getUint32(4);
  const bexp = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e;
  if (bexp === 0) e = -1074;
  else {
    mant |= 1n << 52n;
    e = bexp - 1075;
  }
  if (e >= 0) return { digits: mant << BigInt(e), exp: 0 };
  return { digits: mant * 5n ** BigInt(-e), exp: e };
}

/** Round digits*10^exp to a multiple of 10^target, half-even. Returns BigInt units of 10^target. */
function roundToExp(digits, exp, target) {
  if (exp >= target) return digits * 10n ** BigInt(exp - target);
  const div = 10n ** BigInt(target - exp);
  let q = digits / div;
  const r = digits % div;
  const twice = 2n * r;
  if (twice > div || (twice === div && q % 2n === 1n)) q += 1n;
  return q;
}

function decimalString(units, scale, negative) {
  // units * 10^-scale as a plain decimal string
  let s = units.toString();
  if (scale > 0) {
    s = s.padStart(scale + 1, "0");
    s = `${s.slice(0, -scale)}.${s.slice(-scale)}`;
  }
  return (negative ? "-" : "") + s;
}

/** Python's round(x, ndigits) for floats. */
export function pyRound(x, ndigits) {
  if (!Number.isFinite(x) || x === 0) return x;
  const { digits, exp } = exactDecimal(x);
  const units = roundToExp(digits, exp, -ndigits);
  const r = Number(decimalString(units, ndigits, x < 0));
  return r === 0 && x < 0 ? -0 : r;
}

/** Python's round(x) (no ndigits): half-even to an integer. */
export function pyRoundInt(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

function pyExp(e) {
  return (e < 0 ? "-" : "+") + String(Math.abs(e)).padStart(2, "0");
}

/** Python repr(float). */
export function pyFloatRepr(x) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  // toExponential() without an argument gives the shortest round-trip digits, as repr does.
  const [m, eStr] = x.toExponential().split("e");
  const e = Number(eStr);
  const neg = m.startsWith("-");
  const digits = m.replace("-", "").replace(".", "");
  if (e >= -4 && e < 16) {
    // CPython's repr uses fixed notation when -4 <= exponent < 16.
    let s;
    if (e < 0) s = `0.${"0".repeat(-e - 1)}${digits}`;
    else if (digits.length <= e + 1) s = `${digits}${"0".repeat(e + 1 - digits.length)}.0`;
    else s = `${digits.slice(0, e + 1)}.${digits.slice(e + 1)}`;
    return (neg ? "-" : "") + s;
  }
  const mant = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  return `${neg ? "-" : ""}${mant}e${pyExp(e)}`;
}

/** Python format(x, '.{p}g'). */
export function pyFormatG(x, p) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const { digits, exp } = exactDecimal(x);
  // decimal exponent of the leading digit
  let lead = digits.toString().length - 1 + exp;
  let units = roundToExp(digits, exp, lead - p + 1);
  if (units.toString().length > p) {
    // rounding carried into a new digit (9.9995 -> 10.00)
    lead += 1;
    units = roundToExp(digits, exp, lead - p + 1);
  }
  const ds = units.toString();
  const neg = x < 0 ? "-" : "";
  if (lead >= -4 && lead < p) {
    let s = decimalString(units, p - 1 - lead, false);
    if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
    return neg + s;
  }
  let mant = `${ds[0]}.${ds.slice(1)}`.replace(/0+$/, "").replace(/\.$/, "");
  return `${neg}${mant}e${pyExp(lead)}`;
}

// -- Python str() / repr() of parsed values ------------------------------------------------------

function pyStrRepr(s) {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (ch === "\\") out += "\\\\";
    else if (ch === quote) out += `\\${quote}`;
    else if (ch === "\t") out += "\\t";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch !== " " && /[\p{C}\p{Z}]/u.test(ch)) {
      // not str.isprintable(): escape like CPython
      if (cp < 0x100) out += `\\x${cp.toString(16).padStart(2, "0")}`;
      else if (cp < 0x10000) out += `\\u${cp.toString(16).padStart(4, "0")}`;
      else out += `\\U${cp.toString(16).padStart(8, "0")}`;
    } else out += ch;
  }
  return out + quote;
}

/** Python repr() of a parsed JSON value. */
export function pyRepr(v) {
  if (typeof v === "string") return pyStrRepr(v);
  if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
  if (v instanceof PyDict) return `{${[...v].map(([k, x]) => `${pyStrRepr(k)}: ${pyRepr(x)}`).join(", ")}}`;
  return pyStr(v);
}

/** Python str() of a parsed JSON value. */
export function pyStr(v) {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "string") return v;
  if (v instanceof PyInt) return v.text;
  if (v instanceof PyFloat) return pyFloatRepr(v.value);
  return pyRepr(v);
}

/** Python truthiness of a parsed JSON value. */
export function pyTruthy(v) {
  if (v === null || v === undefined || v === false) return false;
  if (typeof v === "string") return v.length > 0;
  if (v instanceof PyInt) return v.text !== "0";
  if (v instanceof PyFloat) return v.value !== 0; // NaN is truthy in Python too
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof PyDict) return v.size > 0;
  return true;
}

// Characters for which Python's str.isspace() is true (used by str.strip(); also what \s matches
// in a Python str regex).
export const PY_SPACE =
  "\t\n\u000b\u000c\r\u001c\u001d\u001e\u001f \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";

/** Python str.strip() with no arguments. */
export function pyStrip(s) {
  let a = 0;
  let b = s.length;
  while (a < b && PY_SPACE.includes(s[a])) a++;
  while (b > a && PY_SPACE.includes(s[b - 1])) b--;
  return s.slice(a, b);
}
