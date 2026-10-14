// Reef demo UI. All inference happens in reef/runtime.js; this file only wires up the page.
//
// URL parameters:
//   ?model=<folder URL>   where config.json, tokenizer.json, embeddings.npz, reef_int8.tflite live
//                         (default: the seraphimserapis/reef model repo on the Hub)
//   ?wasm=<folder URL>    LiteRT.js wasm folder (default: jsDelivr, pinned version)

import { DEFAULT_MODEL_URL, DEFAULT_WASM_URL, Reef } from "./reef/runtime.js";
import { humanize } from "./reef/text.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const wasmUrl = params.get("wasm") || DEFAULT_WASM_URL;
let modelUrl = params.get("model") || DEFAULT_MODEL_URL;
let reef = null;
let examples = [];

function setStatus(text, { error = false } = {}) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

const mb = (n) => `${(n / 1e6).toFixed(1)} MB`;

// -- examples --------------------------------------------------------------------------------------

async function loadExamples() {
  const res = await fetch("examples.json");
  examples = await res.json();
  const select = $("example");
  const groups = new Map();
  examples.forEach((ex, i) => {
    if (!groups.has(ex.pack)) {
      const g = document.createElement("optgroup");
      g.label = ex.pack;
      groups.set(ex.pack, g);
      select.append(g);
    }
    const g = groups.get(ex.pack);
    const opt = new Option(`${ex.pack} #${g.children.length + 1}`, String(i));
    g.append(opt);
  });
  select.addEventListener("change", () => showExample(Number(select.value)));
  showExample(0);
}

function showExample(i) {
  const ex = examples[i];
  if (!ex) return;
  $("example").value = String(i);
  $("example-desc").textContent = ex.description;
  $("state").value = ex.state;
  $("questions").value = ex.questions;
}

// -- model -----------------------------------------------------------------------------------------

async function loadModel(url) {
  reef?.dispose();
  reef = null;
  $("run").disabled = true;
  const progress = new Map();
  setStatus(`Loading model from ${url} …`);
  try {
    const t0 = performance.now();
    reef = await Reef.load(url, {
      wasmUrl,
      onProgress: ({ file, loaded, total }) => {
        progress.set(file, { loaded, total });
        const got = [...progress.values()].reduce((a, p) => a + p.loaded, 0);
        const all = [...progress.values()].every((p) => p.total) ? [...progress.values()].reduce((a, p) => a + p.total, 0) : 0;
        setStatus(`Downloading model… ${mb(got)}${all ? ` / ${mb(all)}` : ""}`);
      },
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Model “${reef.name}” ready (${mb(reef.downloadBytes)} downloaded in ${secs} s, running on WebAssembly). Edit the state or questions and press Run.`);
    $("run").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus(
      `Could not load the model from ${url}: ${e.message}. If the model repo is private or missing, open “Model settings” and point to another copy (or use ?model=…).`,
      { error: true },
    );
  }
}

// -- run -------------------------------------------------------------------------------------------

function pct(p) {
  return `${(p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1)}%`;
}

function renderAnswer(q, a) {
  const card = document.createElement("article");
  card.className = "result";
  const h = document.createElement("h3");
  h.append(q.id);
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = q.type === "noul" ? "yes / no" : q.type;
  h.append(badge);
  const instr = document.createElement("p");
  instr.className = "instructions";
  instr.textContent = q.instructions;

  const probs = a.raw; // unrounded, in option order
  const top = probs.reduce((b, v, i) => (v > probs[b] ? i : b), 0);
  const verdict = document.createElement("p");
  verdict.className = "verdict";
  const small = document.createElement("small");
  if (q.type === "noul") {
    verdict.append(probs[0] >= 0.5 ? "Yes" : "No");
    small.textContent = ` · p(true) ${probs[0].toFixed(3)}`;
  } else if (q.type === "choice") {
    verdict.append(humanize(q.options[top].id));
    small.textContent = ` · ${pct(probs[top])}`;
  } else {
    verdict.append(`${a.score.toFixed(2)} / ${q.options.length - 1}`);
    small.textContent = ` · most likely ${q.options[top].id}: ${q.options[top].text}`;
  }
  small.textContent += ` · confidence ${a.confidence.toFixed(2)}`;
  verdict.append(small);

  const list = document.createElement("ul");
  list.className = "bars";
  q.options.forEach((o, i) => {
    const li = document.createElement("li");
    if (i === top) li.className = "top";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = q.type === "score" ? `${o.id} · ${o.text}` : q.type === "choice" ? humanize(o.id) : o.id;
    name.title = o.text;
    const track = document.createElement("span");
    track.className = "track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.display = "block";
    fill.style.width = `${(probs[i] * 100).toFixed(2)}%`;
    track.append(fill);
    const val = document.createElement("span");
    val.className = "pct";
    val.textContent = pct(probs[i]);
    li.append(name, track, val);
    list.append(li);
  });
  card.append(h, instr, verdict, list);
  return card;
}

async function run() {
  if (!reef) return;
  $("run").disabled = true;
  try {
    const out = await reef.decide($("state").value, $("questions").value);
    const results = $("results");
    results.replaceChildren(...out.questions.map((q) => renderAnswer(q, out.answers.get(q.id))));
    const u = out.usage;
    const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
    $("usage").textContent =
      `${u.total_ms.toFixed(1)} ms total · ${u.device_ms.toFixed(1)} ms in the encoder · ${plural(u.windows, "window")}` +
      `${u.rounds > 1 ? ` in ${u.rounds} knockout rounds` : ""}${u.groups ? ` · ${u.groups} question groups` : ""} · ${u.state_tokens} state tokens` +
      `${u.state_windows > 1 ? ` read in ${u.state_windows} overlapping chunks` : ""}` +
      `${u.state_truncated ? (u.state_windows > 1 ? " (with gaps: too long to read in full)" : " (truncated to fit the window)") : ""}`;
    $("raw").textContent = JSON.stringify({ model: out.model, answers: Object.fromEntries(out.answers), usage: u }, null, 2);
    $("raw-wrap").hidden = false;
  } catch (e) {
    console.error(e);
    $("usage").textContent = "";
    $("results").replaceChildren();
    setStatus(`Error: ${e.message}`, { error: true });
  } finally {
    $("run").disabled = false;
  }
}

// -- wiring ----------------------------------------------------------------------------------------

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  if ($("status").classList.contains("error") && reef) setStatus(`Model “${reef.name}” ready.`);
  run();
});
for (const id of ["state", "questions"]) {
  $(id).addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      $("form").requestSubmit();
    }
  });
}
$("model-url").value = modelUrl;
$("settings").addEventListener("submit", (e) => {
  e.preventDefault();
  modelUrl = $("model-url").value.trim() || DEFAULT_MODEL_URL;
  const u = new URL(location.href);
  u.searchParams.set("model", modelUrl);
  history.replaceState(null, "", u);
  loadModel(modelUrl);
});

loadExamples().catch((e) => setStatus(`Could not load examples: ${e.message}`, { error: true }));
loadModel(modelUrl);
