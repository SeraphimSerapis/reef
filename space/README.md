---
title: Reef
emoji: 🪸
colorFrom: red
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: General typed-decision model, running in your browser
models:
  - seraphimserapis/reef
---

# Reef in the browser

A static demo of [Reef](https://github.com/SeraphimSerapis/reef), a general typed-decision model. It is
a small int8 conv encoder (Edge TPU compatible, about 11M parameters in total: 3.8M in the token
embeddings, 7.4M in the encoder). It answers typed questions about a state, JSON or plain text:
`noul` (yes/no), `choice` and `score`. It scores **4.73** on Decision Index 0.2.1 (self-measured with the public harness, not an
official board entry; GLiNER 2.5 small, 73.9M params: 3.82). The public
datasets it learned from carry human labels. The homelab and answer-audit data carry soft labels
from Gemma 4.

The whole pipeline runs **entirely in the browser**: rendering, tokenisation, packing, embedding
lookup, int8 quantisation, the encoder ([LiteRT.js](https://ai.google.dev/edge/litert/web) on
WebAssembly/XNNPack) and decoding. There is no server and no build step. The examples are the
homelab packs (alert triage, camera events, notification gating, …). The state and questions are
editable, so any typed decision can be tried.

## How it works

| File | What it does |
| --- | --- |
| `index.html`, `style.css`, `app.js` | UI: example picker, editable state and questions, per-question probability bars and latency |
| `reef/runtime.js` | Port of `reef.runtime.Reef`: loads the model files, `embed` → int8 → LiteRT.js → `decide()` (state chunks averaged, knockout rounds for many options) |
| `reef/packing.js` | Port of `reef.packing` + `reef.decode`: questions → options, instruction hoisting, state regions, option parts, token windows, softmax with per-type temperatures |
| `tools/make_v2_artifact.py` | Exports a small random v2 encoder (CPU) for the parity test |
| `reef/text.js`, `reef/pyjson.js` | Port of `reef.text.render_state`, with a JSON parser that keeps key order and int/float like Python |
| `reef/tokenizer.js` | Exact port of the BERT WordPiece pipeline in `tokenizer.json` (no dependency) |
| `reef/files.js` | Reads `embeddings.npz` (zip/npy, float16) and the TFLite input/output quantisation directly |
| `vendor/litert-*.js` | `@litertjs/core@2.5.3`, patched to accept int8 tensors (see `tools/vendor_litert.mjs`) |
| `examples.json` | 3 cases per homelab pack from `reef.distill.make_cases(pack, n, seed=123)` |

v2 encoders are supported from their `config.json`: question bands (`bands`, one fixed block per
question, several per window), empty-state hoisting (`hoist_empty_state`: with an empty state the
instructions become the state), bare descriptions for opaque option keys (`drop_opaque_keys`:
`option_3: "refund"` reads as `refund`), and the gate inside the TFLite graph. Knockout parts are
split evenly when that still fits, and knocked-out options keep their mass relative to their part
winner, renormalised, as in `reef.runtime`. The release-candidate format needs no runtime support
beyond that. Leaky MLPs (`max(x, 0.1x)`: scalar MUL + MAXIMUM) are part of the TFLite graph. The
pre-MLP BatchNorm is folded into a conv at export, and the activation clips are folded into int8
ranges. The training-only config fields `mlp_leak`, `mlp_norm` and `act_clip` are ignored.

The app loads the **unmodified exported model folder** (`config.json`, `tokenizer.json`,
`embeddings.npz`, `reef_int8.tflite`), so no conversion is needed when the model changes. The model
comes from `https://huggingface.co/seraphimserapis/reef/resolve/main/` by default. Override it
with `?model=<folder URL>`, or under "Model settings" on the page. The LiteRT.js wasm comes from jsDelivr
(override with `?wasm=<folder URL>`).

A first visit downloads about 18 MB: the model (~15.7 MB: tflite 8.0, embeddings 7.0, tokenizer 0.7)
plus the LiteRT.js wasm (~2.6 MB brotli). After that the browser cache serves them.

> [!NOTE]
> The model repo (`seraphimserapis/reef`) must be **public**, because the browser fetches it
> anonymously. While it is private, test locally with a copy of the model (below).

## Local testing

```bash
.venv/bin/python space/tools/prepare.py            # examples.json + space/model/ from artifacts/release-reef
.venv/bin/python -m http.server 8123 --directory space
# open http://127.0.0.1:8123/?model=model/
```

`space/model/` is gitignored and **must not be uploaded to the Space**. Upload only `README.md`,
`index.html`, `style.css`, `app.js`, `examples.json`, `reef/` and `vendor/`. `tools/` is optional.

## Parity test

`tools/parity_test.mjs` checks the JS port against `Reef(model, device="cpu", strict=False).decide()`
in headless Chrome, for both LiteRT.js wasm builds (relaxed-SIMD and compat). It has four suites:

- **main**: the release model (`--model`, default `artifacts/release-reef`; `space/model/` must be a copy,
  which `prepare.py` makes). It covers:
  - 20 cases per pack from `data/v2-test` and 10 per workflow from `data/homelab-test.jsonl`;
  - edge cases: key order, number formatting, Unicode, control characters, special tokens,
    multi-window schemas;
  - long states read in overlapping chunks, with and without gaps;
  - many-option questions answered as a knockout: 77 and 151 options, 8/9/16 option boundaries;
  - instruction hoisting: shared prefixes, per-question documents, stubs, paragraph separators;
  - v2 cases: empty-state hoisting with and without a shared prefix, tricky opaque keys (Unicode
    digits, `K`/`ſ`/`İ`), even 5+4 knockout splits, banded long states;
  - 20 BANKING77-style 77-option intent cases.
- **legacy**: an older v1 model (`--legacy`, default `artifacts/reef-tiny`), so pre-v2 exports keep
  working.
- **hoist**: the legacy model with `hoist_instructions: true` and `max_instruction_tokens: 128`.
- **v2**: a small random encoder in the release format, exported to the OS temp dir by
  `tools/make_v2_artifact.py` (bands, gate, leaky MLPs, folded BatchNorm, activation clips).

Skip suites with `--no-legacy` / `--no-hoist` / `--no-v2`. Node also checks the opaque-key regex
against Python's `re`. The test ends with a UI smoke test.

The main and v2 suites run twice: live, and with Python's encoder output swapped in for each window
(matched by the sha256 of the int8 input). The swapped-in runs must be bit-exact; they check
everything outside the encoder kernels. For the release format, run the test with
`--kernel-tolerant`: the live runs are then reported without failing the test.

```bash
cd space/tools && npm install && node parity_test.mjs --kernel-tolerant   # --per-pack 0 for all 1500 test cases
```

> [!WARNING]
> **The browser does not reproduce the native runtime exactly.** On the same int8 input, XNNPack's
> WebAssembly kernels (LiteRT.js) round some quantised `MUL`s differently by one int8 step from
> the native kernels Python uses:
>
> - the leaky MLP's `x * 0.1`, in every window;
> - rarely, the gate's `x * tanh(g)`.
>
> The differences carry through the network. On the 217 parity cases (663 questions), the release
> model in the browser gives the same label as `Reef(..., device="cpu")` for 639 of 663 (96%), and
> probabilities differ by at most 6.5 points. The JS pipeline itself (rendering, tokens, packing,
> knockout, decoding) is exact. Rewriting the leak in the export would remove most of the gap, but
> it would stream parameters off-chip on the Edge TPU, so the release keeps the current graph.
> Pre-release models without these ops (`reef-tiny`, and v2 models with the `AVERAGE_POOL_2D`
> state summary but no leaky MLP) match bit for bit.

It compares the rendered and hoisted text (per question group), token ids, and every window
`decide()` runs (group, round, ids, roles, levels, int8 encoder input sha256, int8 encoder output). It also compares the final
per-option probabilities, labels, the public answer fields and the usage counters.

The browser runtime is the lenient (non-strict) one: like `reef serve` without `--strict`, it
truncates over-long text instead of refusing it.
