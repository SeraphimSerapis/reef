// Port of src/reef/text.py: turning a decision `state` into model text (`path: value ; ...`).
// Values are the tagged form from pyjson.js so the output matches Python character for character.

import { PyDict, PyFloat, PyInt, PyJSONError, parseJSON, pyFloatRepr, pyFormatG, pyRound, pyStr, pyStrip, pyTruthy } from "./pyjson.js";

export const FIELD_SEP = " ; ";

/** `days_until_due` -> `days until due`. */
export function humanize(key) {
  return pyStrip(pyStr(key).replaceAll("_", " ").replaceAll("-", " "));
}

function scalar(v) {
  if (v === null || v === undefined) return "none";
  if (v === true) return "true";
  if (v === false) return "false";
  if (v instanceof PyFloat) {
    const x = v.value;
    if (Number.isInteger(x) && Math.abs(x) < 1e15) return String(Math.trunc(x) || 0); // -0 -> "0"
    const a = Math.abs(x);
    return a < 1e-3 || a >= 1e6 ? pyFormatG(x, 4) : pyFloatRepr(pyRound(x, 3));
  }
  if (v instanceof PyInt) return v.text;
  return pyStrip(pyStr(v));
}

const isContainer = (x) => x instanceof PyDict || Array.isArray(x);
const MESSAGE_KEYS = new Set(["role", "text", "content", "author"]);

function flatten(obj, path, out) {
  if (obj instanceof PyDict) {
    if (obj.size === 0) out.push(path.length ? `${path.join(" ")}: none` : "none");
    for (const [k, v] of obj) flatten(v, [...path, humanize(k)], out);
  } else if (Array.isArray(obj)) {
    if (obj.length === 0) out.push(`${path.join(" ")}: none`);
    else if (obj.every((x) => !isContainer(x))) out.push(`${path.join(" ")}: ${obj.map(scalar).join(", ")}`);
    else {
      obj.forEach((x, idx) => {
        // A list of messages ({"role": ..., "text": ...}) reads better as "customer: ...".
        if (x instanceof PyDict && [...x.keys()].every((k) => MESSAGE_KEYS.has(k)) && (x.has("text") || x.has("content"))) {
          const pick = (k) => (x.has(k) && pyTruthy(x.get(k)) ? x.get(k) : null);
          const who = humanize(pick("role") ?? pick("author") ?? "message");
          out.push(`${who}: ${scalar(x.has("text") ? x.get("text") : x.get("content"))}`);
        } else {
          flatten(x, [...path, String(idx + 1)], out);
        }
      });
    }
  } else {
    out.push(path.length ? `${path.join(" ")}: ${scalar(obj)}` : scalar(obj));
  }
}

/**
 * Render a state to model text. `state` is either a string (what the demo's textarea holds; JSON
 * objects/arrays are parsed, anything else is used verbatim, as in Python) or a parsed value.
 */
export function renderState(state) {
  if (typeof state === "string") {
    const s = pyStrip(state);
    if (s[0] === "{" || s[0] === "[") {
      try {
        state = parseJSON(s);
      } catch (e) {
        if (e instanceof PyJSONError) return s;
        throw e;
      }
    } else {
      return s;
    }
  }
  if (state instanceof PyDict || Array.isArray(state)) {
    const out = [];
    flatten(state, [], out);
    return out.join(FIELD_SEP);
  }
  return scalar(state);
}
