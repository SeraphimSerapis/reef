// JS <-> Python parity test for the Reef browser port.
//
//   cd space/tools && npm install && node parity_test.mjs [--per-pack 20] [--model ../../artifacts/release-reef]
//       [--legacy ../../artifacts/reef-tiny] [--homelab-per 10] [--skip-ref] [--wasm relaxed,compat]
//       [--no-ui] [--no-legacy] [--no-hoist] [--no-v2] [--kernel-tolerant]
//
// Release-format models: run with --kernel-tolerant (the documented mode). LiteRT.js's wasm XNNPack
// rounds some quantised MULs differently from native XNNPack, so live runs are reported, not gated;
// the runs with Python's encoder outputs replayed still have to be bit-exact.
//
// Four suites:
//   main    the release model (default artifacts/release-reef; space/model/ must be a copy of it, made by
//           prepare.py): data/v2-test, data/homelab-test.jsonl (N per workflow), edge cases, long
//           states (state chunking), many-option questions (knockout), hoisting, v2 cases and 20
//           BANKING77-style 77-option intent cases.
//   legacy  an older v1 model (default artifacts/reef-tiny; temp copy served via tools/.cache/):
//           data, edge, long, many, so pre-v2 exports keep working.
//   hoist   a temporary copy of the legacy model with config.json model.hoist_instructions=true and
//           max_instruction_tokens=128: over-long instructions moved into the state, plus edge cases.
//   v2      a small random encoder in the release format, exported by make_v2_artifact.py into the
//           OS temp dir (bands, gate, leaky MLPs, folded pre-MLP BatchNorm, activation clips,
//           empty-state hoisting, opaque keys; CPU, ~10 s): every case group, run live and with
//           Python's encoder outputs replayed. Node also checks the opaque-key regex against Python.
//
// 1. Python reference (parity_ref.py, the repo's .venv): Reef(model, device="cpu").decide(), with
//    every window it runs (round, ids, roles, levels, int8 tensors) and the final probabilities.
// 2. Node: render_state / hoisting / tokenizer / state regions / first-round packing parity.
// 3. Headless Chrome (Playwright) on a local http.server: the full in-browser pipeline (LiteRT.js
//    WASM) per wasm build, comparing every window (ids, roles, levels, int8 input sha256, int8
//    output), the per-option probabilities and labels, the public answers and usage counters.
// 4. UI smoke test: index.html loads the local model, runs an example, renders result bars.
// Exit code 1 if tokens/windows differ or the max |prob diff| >= 1e-6.

import { spawn, spawnSync } from "node:child_process";
import { copyFileSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { Packer, isOpaqueKey, normalizeQuestions, planGroups, planRound, prepareState } from "../reef/packing.js";
import { renderState } from "../reef/text.js";
import { WordPieceTokenizer } from "../reef/tokenizer.js";

const here = dirname(fileURLToPath(import.meta.url));
const space = resolve(here, "..");
const repo = resolve(space, "..");
const cache = join(here, ".cache");
const { values: args } = parseArgs({
  options: {
    "per-pack": { type: "string", default: "20" },
    model: { type: "string", default: join(repo, "artifacts", "release-reef") },
    legacy: { type: "string", default: join(repo, "artifacts", "reef-tiny") },
    "homelab-per": { type: "string", default: "10" },
    "no-legacy": { type: "boolean", default: false },
    "kernel-tolerant": { type: "boolean", default: false },
    "skip-ref": { type: "boolean", default: false },
    wasm: { type: "string", default: "relaxed,compat" },
    "no-ui": { type: "boolean", default: false },
    "no-hoist": { type: "boolean", default: false },
    "no-v2": { type: "boolean", default: false },
  },
});
const TOL = 1e-6;
const MODEL_FILES = ["config.json", "tokenizer.json", "embeddings.npz", "reef_int8.tflite"];
mkdirSync(cache, { recursive: true });

// -- suites ------------------------------------------------------------------------------------------
// space/model/ (what the page and the main suite load) must be the model Python reads.
for (const f of MODEL_FILES) {
  const a = join(space, "model", f);
  if (!existsSync(a) || !readFileSync(a).equals(readFileSync(join(args.model, f)))) {
    console.error(`space/model/${f} is not a copy of ${args.model}/${f}: run .venv/bin/python space/tools/prepare.py --model ${args.model}`);
    process.exit(1);
  }
}
const ALL_GROUPS = "data,edge,long,many,hoist,v2,homelab,banking";
const suites = [
  { name: "main", model: args.model, url: `/${relative(space, join(space, "model"))}/`, groups: ALL_GROUPS, ref: join(cache, "ref.jsonl"), replay: true },
];
const py = join(repo, ".venv", "bin", "python");
const legacyLink = join(cache, "model-legacy");
const hoistLink = join(cache, "model-hoist");
const v2Link = join(cache, "model-v2");
let legacyTmp = null;
let hoistTmp = null;
let v2Tmp = null;
function cleanupHoist() {
  for (const link of [legacyLink, hoistLink, v2Link]) {
    try {
      if (lstatSync(link, { throwIfNoEntry: false })) unlinkSync(link);
    } catch {}
  }
  for (const dir of [legacyTmp, hoistTmp, v2Tmp]) if (dir) rmSync(dir, { recursive: true, force: true });
  legacyTmp = hoistTmp = v2Tmp = null;
}
cleanupHoist();
if (!args["no-legacy"]) {
  legacyTmp = mkdtempSync(join(tmpdir(), "reef-legacy-"));
  for (const f of MODEL_FILES) copyFileSync(join(args.legacy, f), join(legacyTmp, f));
  symlinkSync(legacyTmp, legacyLink, "dir");
  suites.push({ name: "legacy", model: legacyTmp, url: `/${relative(space, legacyLink)}/`, groups: "data,edge,long,many", ref: join(cache, "ref-legacy.jsonl") });
}
if (!args["no-hoist"]) {
  hoistTmp = mkdtempSync(join(tmpdir(), "reef-hoist-"));
  for (const f of MODEL_FILES) copyFileSync(join(args.legacy, f), join(hoistTmp, f));
  const cfg = JSON.parse(readFileSync(join(hoistTmp, "config.json"), "utf8"));
  cfg.model.hoist_instructions = true;
  cfg.model.max_instruction_tokens = 128;
  writeFileSync(join(hoistTmp, "config.json"), JSON.stringify(cfg, null, 2));
  symlinkSync(hoistTmp, hoistLink, "dir"); // http.server follows symlinks
  suites.push({ name: "hoist", model: hoistTmp, url: `/${relative(space, hoistLink)}/`, groups: "hoist,edge", ref: join(cache, "ref-hoist.jsonl") });
}
if (!args["no-v2"]) {
  // A fresh random v2 model each run (seeded, but the TFLite converter is not guaranteed to be
  // byte-stable), so the v2 reference is always regenerated.
  v2Tmp = mkdtempSync(join(tmpdir(), "reef-v2-"));
  const r = spawnSync(py, [join(here, "make_v2_artifact.py"), join(v2Tmp, "run"), join(v2Tmp, "art")], {
    stdio: ["ignore", "ignore", "pipe"],
    encoding: "utf8",
    env: { ...process.env, HF_HUB_OFFLINE: "1", CUDA_VISIBLE_DEVICES: "" },
  });
  if (r.status !== 0) {
    console.error(r.stderr);
    process.exit(1);
  }
  symlinkSync(join(v2Tmp, "art"), v2Link, "dir");
  suites.push({ name: "v2", model: join(v2Tmp, "art"), url: `/${relative(space, v2Link)}/`, groups: ALL_GROUPS, ref: join(cache, "ref-v2.jsonl"), fresh: true, replay: true });
}
process.on("exit", cleanupHoist);

// -- 1. Python reference ---------------------------------------------------------------------------
for (const s of suites) {
  // --skip-ref reuses a cached reference (the hoist reference does not depend on the temp path).
  if (!args["skip-ref"] || !existsSync(s.ref) || s.fresh) {
    const r = spawnSync(py, [join(here, "parity_ref.py"), "--model", s.model, "--per-pack", args["per-pack"], "--homelab-per", args["homelab-per"], "--groups", s.groups, "--out", s.ref], {
      stdio: ["ignore", "inherit", "pipe"],
      encoding: "utf8",
    });
    if (r.status !== 0) {
      console.error(r.stderr);
      process.exit(1);
    }
  }
  s.cases = readFileSync(s.ref, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const packs = [...new Set(s.cases.map((c) => c.pack))];
  console.log(`[${s.name}] ${s.cases.length} cases (${packs.map((p) => `${p}: ${s.cases.filter((c) => c.pack === p).length}`).join(", ")})`);
}

// -- 2. Text / hoisting / token / packing parity in Node ---------------------------------------------
const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
let tokenFailures = 0;
for (const s of suites) {
  const meta = JSON.parse(readFileSync(join(s.model, "config.json"), "utf8"));
  const m = meta.model;
  const tokenizer = new WordPieceTokenizer(JSON.parse(readFileSync(join(s.model, "tokenizer.json"), "utf8")));
  const packer = new Packer(tokenizer, m.state_len, m.schema_len, m.max_instruction_tokens ?? 40, m.max_option_tokens ?? 28, {
    bands: Number(m.bands ?? 0) || 0,
    dropOpaqueKeys: Boolean(m.drop_opaque_keys),
  });
  const hoist = Boolean(m.hoist_instructions);
  const opts = { hoist, hoistEmpty: hoist && Boolean(m.hoist_empty_state), instructionCap: m.max_instruction_tokens ?? 40, maxStateWindows: 8 };
  let positions = 0;
  let failures = 0;
  for (const c of s.cases) {
    const problems = [];
    const text = renderState(c.state_text);
    if (text !== c.rendered) problems.push(`rendered text differs:\n  js: ${JSON.stringify(text)}\n  py: ${JSON.stringify(c.rendered)}`);
    const groups = planGroups(packer, normalizeQuestions(c.questions_text), text, opts);
    if (groups.length !== c.groups.length) problems.push(`${groups.length} hoist groups vs ${c.groups.length}`);
    groups.forEach((g, gi) => {
      const want = c.groups[gi];
      if (!want) return;
      const p = prepareState(packer, g.stateText, opts.maxStateWindows);
      if (!eq(g.idx, want.idx)) problems.push(`group ${gi} question indices differ`);
      if (g.stateText !== want.state_text) problems.push(`group ${gi} hoisted state text differs`);
      if (!eq(g.qs.map((q) => q.instructions), want.instructions)) problems.push(`group ${gi} hoisted instructions differ`);
      if (!eq(p.stateIds, want.state_ids)) problems.push(`group ${gi} state token ids differ`);
      if (p.regions.length !== want.state_regions || p.gaps !== want.gaps) problems.push(`group ${gi} state regions ${p.regions.length}/${p.gaps} vs ${want.state_regions}/${want.gaps}`);
      // Round 1 does not depend on model outputs: every window x every state region, in run order.
      const { parts } = planRound(packer, g.qs, new Map(g.qs.map((q, qi) => [qi, q.options.map((_, i) => i)])));
      const keys = [...parts.keys()];
      const windows = packer.packBlocks(p.regions[0].ids, p.regions[0].roles, keys.map((k) => [k, packer.block(parts.get(k))]));
      const ref = c.windows.filter((w) => w.group === gi && w.round === 0);
      let i = 0;
      for (const w of windows) {
        for (const r of p.regions) {
          const { ids, roles } = packer.withState(w, r.ids, r.roles);
          const want = ref[i++];
          if (!want) continue;
          positions += ids.length;
          if (!eq(Array.from(ids), want.ids) || !eq(Array.from(roles), want.roles) || !eq(Array.from(w.levels), want.levels)) {
            problems.push(`group ${gi} round 1 window ${i - 1} differs`);
          }
        }
      }
      if (i !== ref.length) problems.push(`group ${gi} round 1: ${i} windows vs ${ref.length}`);
    });
    if (problems.length) {
      failures++;
      console.log(`  ✗ [${s.name}] ${c.id}: ${problems.join("; ")}`);
    }
  }
  tokenFailures += failures;
  console.log(`[${s.name}] text/hoisting/tokens/packing: ${s.cases.length - failures}/${s.cases.length} cases identical (${positions} round-1 positions compared)`);
}

// Opaque option keys (drop_opaque_keys): the JS regex against Python's re (Unicode digits, (?i) folds).
{
  const keys = [
    "A", "z", "Z", "ab", "", " ", "1", "007", "12a", "\u0661\u0662", "\u00b2", "\u2460", "\u0966", "\uff11", "\ud835\udfce", "\u1369",
    "\u212a", "\u017f", "\u0130", "\u0131", "\u00e9", "\u00df", "k", "s", "i", "option_1", "Option-2", "OPTION 3", "option4", "option__5",
    "option-", "option", "opt9", "opt 9", "\u017fhoice1", "choice_\u0663", "answer-10", "candidate 3", "label_7", "class8", "item 99",
    "\u0130tem1", "\u0131tem1", "la\u212abel1", "options_1", "option_1 ", "option_1\n", "A\n", "1\n", "\nA", "A\n\n", "item\t1", "opt-1-2",
    "label\u00a01", "option_\ud835\udfd9", "x1", "q1", "class_\u0e51",
  ];
  const r = spawnSync(py, ["-c", "import json, sys\nfrom reef.packing import _OPAQUE_KEY\nprint(json.dumps([bool(_OPAQUE_KEY.match(k)) for k in json.load(sys.stdin)]))"], {
    input: JSON.stringify(keys),
    encoding: "utf8",
  });
  const want = JSON.parse(r.stdout);
  const bad = keys.filter((k, i) => isOpaqueKey(k) !== want[i]);
  if (bad.length) tokenFailures += bad.length;
  console.log(`opaque option keys: ${keys.length - bad.length}/${keys.length} match Python re${bad.length ? ` (differ: ${JSON.stringify(bad)})` : ""}`);
}

// -- 3. Full pipeline in headless Chrome ------------------------------------------------------------
async function freePort() {
  return new Promise((res) => {
    const srv = createServer();
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => res(port));
    });
  });
}

async function launchBrowser(chromium) {
  const attempts = [
    process.env.CHROME_PATH && { executablePath: process.env.CHROME_PATH },
    { channel: "chrome" },
    {},
  ].filter(Boolean);
  let err;
  for (const opts of attempts) {
    try {
      return await chromium.launch({ headless: true, ...opts });
    } catch (e) {
      err = e;
    }
  }
  throw err;
}

const port = await freePort();
// The repo's venv python: some other python3 builds hang in socket.getfqdn() inside server_bind.
const server = spawn(join(repo, ".venv", "bin", "python"), ["-m", "http.server", String(port), "--bind", "127.0.0.1", "--directory", space], {
  stdio: "ignore",
});
const origin = `http://127.0.0.1:${port}`;
const wasmDir = `/tools/node_modules/@litertjs/core/wasm/`;
const WASM = { relaxed: wasmDir, compat: `${wasmDir}litert_wasm_compat_internal.js` };
let exitCode = tokenFailures ? 1 : 0;
const USAGE_KEYS = ["state_tokens", "state_truncated", "state_windows", "windows", "rounds", "groups"];

function newStats() {
  return { cases: 0, windows: 0, inputMismatch: 0, outputMismatch: 0, maxOutDiff: 0, maxDiff: 0, worst: null, labels: 0, questions: 0, answerMismatch: 0, usageMismatch: 0, deviceMs: 0 };
}

const { chromium } = await import("playwright");
const browser = await launchBrowser(chromium);
try {
  for (let i = 0; i < 50; i++) {
    // wait for http.server to accept connections
    try {
      if ((await fetch(`${origin}/tools/parity.html`)).ok) break;
    } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }

  for (const variant of args.wasm.split(",")) {
    const page = await browser.newPage();
    const logs = [];
    page.on("console", (m) => m.type() === "error" && !/^(INFO|WARNING):/.test(m.text()) && logs.push(m.text()));
    await page.goto(`${origin}/tools/parity.html`);
    console.log(`\n[browser, LiteRT.js wasm "${variant}"]`);
    // Suites with `replay` run twice: live, and with every encoder output replaced by Python's for
    // the same int8 input (by sha256). Both must be bit-exact; if only the live run fails, the
    // encoder kernels differ (LiteRT.js XNNPack wasm vs the native NEON/SSE kernels Python uses),
    // not the JS pipeline. (v2 models before state_avgpool: a quantised MEAN over the state rows
    // rounded differently in a few windows.)
    const runs = suites.flatMap((s) => (s.replay ? [{ s, replay: false }, { s, replay: true }] : [{ s, replay: false }]));
    for (const { s, replay } of runs) {
      const payload = s.cases.map((c) => ({ id: c.id, state: c.state_text, questions: c.questions_text }));
      const replayed = replay ? Object.fromEntries(s.cases.flatMap((c) => c.windows.map((w) => [w.input_sha256, w.output]))) : null;
      const res = await page.evaluate(
        async ({ payload, modelUrl, wasmUrl, replayed }) => {
          const { Reef } = await import("/reef/runtime.js");
          const t0 = performance.now();
          const reef = await Reef.load(modelUrl, { wasmUrl });
          const loadMs = performance.now() - t0;
          const hex = async (a) =>
            Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new Uint8Array(a.buffer, a.byteOffset, a.byteLength))))
              .map((b) => b.toString(16).padStart(2, "0"))
              .join("");
          const kernel = { windows: 0, missing: 0, differ: 0, maxDiff: 0 };
          if (replayed) {
            const invoke = reef.invoke.bind(reef);
            reef.invoke = async (x) => {
              const y = await invoke(x);
              const want = replayed[await hex(x)];
              kernel.windows++;
              if (!want) {
                kernel.missing++;
                return y;
              }
              let d = 0;
              for (let i = 0; i < y.length; i++) d = Math.max(d, Math.abs(y[i] - want[i]));
              if (d) kernel.differ++;
              kernel.maxDiff = Math.max(kernel.maxDiff, d);
              return y.constructor.from(want);
            };
          }
          const out = [];
          for (const c of payload) {
            const r = await reef.decide(c.state, c.questions, { debug: true });
            const answers = {};
            const raw = {};
            for (const [k, a] of r.answers) (answers[k] = a), (raw[k] = a.raw);
            out.push({
              id: c.id,
              answers,
              raw,
              usage: r.usage,
              trace: await Promise.all(
                r.trace.map(async (t) => ({ group: t.group, round: t.round, ids: t.ids, roles: t.roles, levels: t.levels, input_sha256: await hex(t.input), output: t.output })),
              ),
            });
          }
          reef.dispose();
          return { loadMs, out, kernel };
        },
        { payload, modelUrl: s.url, wasmUrl: WASM[variant] ?? variant, replayed },
      );

      const total = newStats();
      const byGroup = new Map();
      const argmax = (p) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0);
      for (const [ci, c] of s.cases.entries()) {
        const js = res.out[ci];
        const g = byGroup.get(c.group) ?? byGroup.set(c.group, newStats()).get(c.group);
        const add = (k, v = 1) => ((g[k] += v), (total[k] += v));
        add("cases");
        if (js.trace.length !== c.windows.length) add("inputMismatch", Math.abs(js.trace.length - c.windows.length));
        c.windows.forEach((w, i) => {
          add("windows");
          const t = js.trace[i];
          if (!t || t.group !== w.group || t.round !== w.round || !eq(t.ids, w.ids) || !eq(t.roles, w.roles) || !eq(t.levels, w.levels) || t.input_sha256 !== w.input_sha256) add("inputMismatch");
          if (t) {
            const d = Math.max(...t.output.map((v, k) => Math.abs(v - w.output[k])));
            for (const x of [g, total]) x.maxOutDiff = Math.max(x.maxOutDiff, d);
            if (d) add("outputMismatch");
          }
        });
        add("deviceMs", js.usage.device_ms);
        if (!USAGE_KEYS.every((k) => js.usage[k] === c.usage[k]) || !eq(Object.keys(js.usage), Object.keys(c.usage))) {
          add("usageMismatch");
          console.log(`  usage differs for ${c.id}: js ${JSON.stringify(js.usage)} py ${JSON.stringify(c.usage)}`);
        }
        for (const [qid, p] of Object.entries(c.probs)) {
          add("questions");
          const q = js.raw[qid];
          const d = q && q.length === p.length ? Math.max(...p.map((v, k) => Math.abs(v - q[k]))) : Infinity;
          for (const x of [g, total]) if (d > x.maxDiff) (x.maxDiff = d), (x.worst = `${c.id}/${qid}`);
          if (q && argmax(p) === argmax(q)) add("labels");
          const a = c.answers[qid];
          const b = js.answers[qid];
          const fieldsOk = ["type", "noul", "choice", "score", "confidence"].every((k) =>
            k in a ? (typeof a[k] === "number" ? Math.abs(a[k] - b[k]) <= 1e-6 : a[k] === b[k]) : !(k in b),
          );
          const probsOk = Object.keys(a.probabilities).every((k) => Math.abs(a.probabilities[k] - b.probabilities[k]) <= 1e-6);
          const legendOk = !a.legend || JSON.stringify(a.legend) === JSON.stringify(Object.fromEntries(Object.keys(a.legend).map((k) => [k, b.legend[k]])));
          if (!(fieldsOk && probsOk && legendOk)) add("answerMismatch");
        }
      }
      const ok = total.maxDiff < TOL && total.inputMismatch === 0 && total.usageMismatch === 0 && total.labels === total.questions && !res.kernel.missing;
      // --kernel-tolerant: live runs of replay suites are reported but not gated (the replayed run
      // still is), for models whose int8 graph XNNPACK's wasm and native kernels round differently.
      const gated = !(args["kernel-tolerant"] && s.replay && !replay);
      if (!ok && gated) exitCode = 1;
      const fmt = (x) =>
        `windows in ${x.windows - x.inputMismatch}/${x.windows}, out ${x.windows - x.outputMismatch}/${x.windows} (max |Δ| ${x.maxOutDiff}), ` +
        `max |Δp| ${x.maxDiff.toExponential(3)}, labels ${x.labels}/${x.questions}, answers ${x.questions - x.answerMismatch}/${x.questions}, ` +
        `usage ${x.cases - x.usageMismatch}/${x.cases}`;
      const label = s.replay ? `${s.name}, ${replay ? "Python encoder outputs replayed" : "live"}` : s.name;
      console.log(`  suite "${label}": model load ${res.loadMs.toFixed(0)} ms, ${total.windows} windows, mean ${(total.deviceMs / total.windows).toFixed(1)} ms/window`);
      if (replay) {
        const k = res.kernel;
        console.log(`    LiteRT.js vs Python on the same int8 inputs: ${k.differ}/${k.windows} windows differ (max |Δ| ${k.maxDiff} int8 steps), ${k.missing} inputs not in the reference`);
      }
      for (const [name, x] of byGroup) console.log(`    ${name.padEnd(6)} ${String(x.cases).padStart(3)} cases: ${fmt(x)}${x.maxDiff ? ` (worst ${x.worst})` : ""}`);
      console.log(`    total  ${String(total.cases).padStart(3)} cases: ${fmt(total)}  ${ok ? "✓" : gated ? "✗" : "– (not gated)"} (target |Δp| < ${TOL})`);
    }
    if (logs.length) console.log(`  console errors: ${logs.slice(0, 3).join(" | ")}`);
    await page.close();
  }

  // -- 4. UI smoke test -------------------------------------------------------------------------------
  if (!args["no-ui"]) {
    const modelUrl = suites[0].url;
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } }); // phone-sized
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${origin}/index.html?model=${encodeURIComponent(modelUrl)}&wasm=${encodeURIComponent(wasmDir)}`);
    await page.waitForSelector("#run:not([disabled])", { timeout: 60000 });
    await page.selectOption("#example", { index: 4 });
    await page.click("#run");
    await page.waitForSelector(".result", { timeout: 30000 });
    const summary = await page.$$eval(".result", (els) => els.map((e) => e.querySelector("h3")?.textContent.trim() + " → " + e.querySelector(".verdict")?.textContent.trim()));
    const status = await page.textContent("#status");
    const usage = await page.textContent("#usage");
    await page.screenshot({ path: join(cache, "ui.png"), fullPage: true });
    console.log(`\n[ui] ${summary.length} result cards: ${summary.join(" | ")}`);
    console.log(`[ui] status: ${status.trim()}`);
    console.log(`[ui] usage: ${usage.trim()}`);
    console.log(`[ui] screenshot: ${join(cache, "ui.png")}`);
    if (errors.length || !summary.length) {
      exitCode = 1;
      console.log(`[ui] page errors: ${errors.join(" | ")}`);
    }
    await page.close();
  }
} finally {
  await browser.close();
  server.kill();
  cleanupHoist();
}
console.log(exitCode ? "\nPARITY: FAIL" : "\nPARITY: OK");
process.exit(exitCode);
