# Edge TPU notes

What we learned while fitting a text model onto a Coral USB Accelerator (Edge TPU compiler
14.1.317412892, libedgetpu 16.0 / TF 2.19.1, LiteRT 2.3, Apple M-series host, macOS). Most of it
applies to any model you want to map fully onto the Edge TPU.

## The rules Reef follows

| Rule | Why |
|---|---|
| Fully int8 (`TFLITE_BUILTINS_INT8`, int8 input and output) | The Edge TPU only runs int8 ops. Anything float stays on the CPU. |
| Static shapes, batch size 1 | A dynamic batch adds `SHAPE`/`PACK`/`STRIDED_SLICE` ops on the shape tensor, which split the graph into several Edge TPU subgraphs with CPU hops in between. |
| No gathers on the device | Embedding lookups (`GATHER`) are not supported. Reef looks up embeddings on the host (numpy) and sends a dense `[1, H, W, 128]` int8 tensor. |
| No activation × activation `MUL` | Even a plain elementwise multiply of two activations (a mask, gated pooling, GLU) crashed compiler 14.1 with `Internal compiler error. Aborting!`. Multiplying by constants is fine. This rules out attention and masking. |
| No dilated convolutions | `dilation_rate > 1` lowers to `SPACE_TO_BATCH_ND` / `BATCH_TO_SPACE_ND`, which run on the CPU. |
| `MEAN`, broadcast `ADD`, `STRIDED_SLICE` are fine | Reef's global mixing (mean over the state rows, broadcast-added to every position) uses only these. |
| Convert with `TFLiteConverter.from_keras_model` | With Keras 3, going through `tf.saved_model.save` or `from_concrete_functions` left `VAR_HANDLE` / `READ_VARIABLE` ops in the graph, which the compiler rejects. |
| The representative dataset yields `{"emb": x}` dicts | It is keyed by the Keras input name. Bare arrays raised a signature mismatch with Keras 3 models. |

`tests/test_edgetpu_ops.py` checks the op list of a converted model against this allow-list, so a
change that would push ops off the device fails the tests even without a compiler.

## Text as an image

The Edge TPU is a systolic array built for convolutions over images. A sequence laid out as a
width-1 image `[1, L, 1, C]` uses the hardware badly. Folding the same 768 positions into a
`24 × 32` grid (token `t` at row `t // 32`, column `t % 32`) and using `(3, 5)` depthwise kernels
roughly halved latency with the same parameter count:

| Encoder (768 tokens, ~2.7M params, random weights) | Edge TPU | M-series CPU, int8, 1 thread |
|---|---|---|
| `[L, 1, C]` sequence, width 1 | 21 ms | 2.4 ms |
| `24 × 32` grid | 11.6 ms | 2.4 ms |
| grid, 128-wide device input (host projection) | 9.2 ms | 2.4 ms |
| grid, width 256 (4.8M params) | 13.1 ms | 3.6 ms |

A 1x1 convolution on the grid is still a per-token dense layer, so the grid changes nothing about
what those layers compute. Only the depthwise kernels see a 2-D neighbourhood. That neighbourhood
covers tokens `t ± 1, t ± 2` and `t ± 32 (± 1, ± 2)`, so ten blocks reach several hundred tokens.

## Masks without multiplies

A window has a fixed number of state positions, but states vary in length. The usual fix (mask,
then masked mean) needs an activation × activation multiply, which the compiler cannot handle.
Reef repeats `[CLS] state [SEP]` cyclically until it fills its 384 positions. A plain `MEAN`
over those rows then equals the mean over the state, up to the partial last copy, and needs no mask.

Padding in the schema region is handled on the host: padded positions get a zero input vector,
and their output logits are never read.

## Host ↔ device split

```
host:   text → tokens → pack window → gather (tok + role + level) → int8 [1,24,32,128]
device: stem → 10 × polyp block → per-position logit                  (one Edge TPU subgraph)
host:   pick option-marker logits → softmax / temperature → SystemOne answer
```

Folding the 256→128 projection into the host table halves both the bytes sent over USB and the
width of the first device layer. That gave 11.6 → 9.2 ms in the table above.

## USB 2 vs USB 3

On our test machine the accelerator negotiated **480 Mb/s (USB 2)**. Each window sends 96 KiB and
receives 0.75 KiB, so the link is not the main cost. Edge TPU invocations still carry a fixed USB
round-trip overhead of a few milliseconds. `reef doctor` prints the negotiated link speed.

## Is the Edge TPU faster than the CPU?

Not on an Apple M-series Mac. XNNPACK's int8 kernels on a performance core run Reef's 2.7M-param
encoder in about 2–4 ms, against about 9–12 ms on the Edge TPU over USB. The accelerator pays off
when:

* the host is a small ARM board (Raspberry Pi 4/5, Coral Dev Board Micro-class hosts), where the
  same encoder takes tens of milliseconds on the CPU;
* the CPU is busy with other work and you want decisions offloaded at ~2 W;
* models get wider: going from 192 to 256 channels cost the Edge TPU +42% latency, compared with
  +50% on the CPU, and the gap widens as long as the weights fit in the ~8 MB on-chip SRAM.

`reef bench` measures both on your machine.

## macOS specifics

* The accelerator enumerates as `1a6e:089a` (Global Unichip) until libedgetpu uploads firmware.
  It then re-enumerates as `18d1:9302` (Google). macOS (Sequoia and later) may show "Allow
  accessory to connect?" for **each** of these. If you miss the prompt, the device never appears.
* Google's official macOS libedgetpu (2021) does not work with current LiteRT. The community builds
  at [feranick/libedgetpu](https://github.com/feranick/libedgetpu) do, but they link MacPorts'
  `/opt/local/lib/libusb-1.0.0.dylib`. `reef setup-edgetpu` rewrites that path to Homebrew's
  libusb with `install_name_tool` and re-signs the dylib ad hoc (`codesign -s -`). Without the
  re-sign, macOS kills the process on load.
* `edgetpu_compiler` exists only for x86-64 Linux. `docker/compiler.Dockerfile` wraps it, and
  `reef export` uses Docker automatically when the compiler is not on `PATH`. On Apple Silicon it
  runs under Rosetta/QEMU emulation, which takes a few seconds per model.

## v2: what changed, and the new dead ends

The general model (v2) adds question bands, a gate, live MLPs and quantisation-aware training.
Each needed an Edge TPU-friendly form:

| Want | What broke | What works |
|---|---|---|
| A per-band summary × option features (the gate) | A broadcast `MUL` over the 12 state rows or the whole grid stays float ("Model not quantized"); `tanh(b + g)` crashed the compiler | `MUL` of a 6-row band by its broadcast summary vector quantises and maps |
| Leaky MLPs (ReLU MLPs died: 10 of 11 never fired after pre-training) | `LEAKY_RELU` crashes compiler 14.1 (`std::length_error`) | `max(x, 0.1·x)`: scalar `MUL` + `MAXIMUM`, both mapped, no weights |
| BatchNorm in front of each MLP (live MLPs diverged without it) | Left in the graph as per-channel `MUL`/`ADD`, it compiled but the Edge TPU returned a **constant** output | Fold it into the following 1×1 conv at export (`W' = diag(a)·W`, `b' = b + c·W`), which is exact |
| Activation clips for quantisation-aware training | `MINIMUM`/`MAXIMUM` against constant bounds made the compiler store full-size bound tensors (8.6 MiB off-chip); a clip written as ops quantised the tensor at its full range first | Emit `FakeQuant(lo, hi)` at export: the converter uses `[lo, hi]` as the producing tensor's int8 range and adds no op |

Why the quantisation-aware training mattered: the residual stream and the broadcast summaries
grew to values in the thousands (median about 75), so per-tensor int8 left 2–3 levels for typical
values. Float and int8 disagreed on 10–25% of answers. After one epoch with the clips, int8 is
within about 1–2 accuracy points of float.

Browser note: LiteRT.js' wasm kernels round some quantised `MUL`s one int8 step differently from
native XNNPACK, so the Space matches the Python runtime on about 96% of labels, not bit for bit.
