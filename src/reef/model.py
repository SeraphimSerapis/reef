"""The Reef network, written so every op maps onto the Edge TPU.

Layout: text as an image
------------------------
The Edge TPU is a systolic array built for images. Fed a sequence as ``[L, 1, C]`` (width 1) it
runs at a few percent of peak. Reef therefore **folds the window into a 2-D grid**: token ``t``
sits at row ``t // W``, column ``t % W``, giving a ``[H, W, C]`` "image" (24 × 32 for the default
768-token window). This halved Edge TPU latency in our measurements (``docs/edge-tpu-notes.md``).

On the grid, 1x1 convolutions are token-wise dense layers, so their meaning is unchanged. A
``(3, 5)`` depthwise kernel at ``(r, c)`` sees tokens ``t ± 1, t ± 2`` (same row) and
``t ± W, t ± W ± 1, ...`` (neighbouring rows). Stacking ten of them grows the receptive field to
about ±10 rows, or ±320 tokens.

A window is ``state_len`` positions of state followed by ``schema_len`` positions of questions
and options (see :mod:`reef.packing`). Both are whole numbers of rows. The state is repeated
cyclically to fill its rows, so a plain mean over those rows summarises the state with no mask.

What runs where
---------------
* Host (CPU, numpy): tokenisation, packing, and embedding lookup. A lookup is a gather, which the
  Edge TPU cannot run and which is memory-bound anyway.
* Edge TPU: the whole encoder plus the per-token decision head. Everything is CONV_2D,
  DEPTHWISE_CONV_2D, ADD, MEAN, STRIDED_SLICE and RELU.
* Host again: picking out the option-marker logits, softmax and temperature calibration.

Encoder block (a "polyp")
-------------------------
::

    h ─┬─ depthwise conv (3×5) ── BN ───────────┐
       │                                        + ── 1x1 (D→eD) ── ReLU ── 1x1 (eD→D) ──┐
       ├─ mean over state rows ── 1x1 (D→D) ────┘                                       │
       └─────────────────────────────────────────────────────────────────────── + ──────┘

Self-attention would need an activation-by-activation matmul, which the Edge TPU cannot run (in
our tests even an elementwise activation×activation MUL crashes edgetpu_compiler 14.1). The two
mixing paths replace it: depthwise convolutions give each token a wide local view, and the state
summary, broadcast-added to every position, hands each option marker a fresh view of the whole
state in every block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import keras
from keras import layers, ops

from reef.packing import NUM_LEVELS, NUM_ROLES


@dataclass
class ReefConfig:
    """Hyper-parameters. Saved next to every checkpoint as ``config.json``."""

    def packer(self, tokenizer):
        """The :class:`~reef.packing.Packer` this model was trained with."""
        from reef.packing import Packer

        return Packer(tokenizer, self.state_len, self.schema_len, self.max_instruction_tokens,
                      self.max_option_tokens, hoist=self.hoist_instructions, bands=self.bands,
                      hoist_empty=self.hoist_empty_state, drop_opaque_keys=self.drop_opaque_keys)

    state_len: int = 384  # positions reserved for the (cyclically repeated) state
    schema_len: int = 384  # positions for question and option blocks
    grid_width: int = 32  # W: tokens per grid row
    embed_dim: int = 256  # width of the pretrained token table (host)
    input_dim: int = 128  # width of the vectors sent to the device (host projection)
    width: int = 256  # D: channels inside the encoder (5M params with the max path, 5.4 MiB on-chip)
    expansion: int = 3  # e: MLP expansion factor
    kernel: list[int] = field(default_factory=lambda: [3, 5])  # (rows, columns)
    depth: int = 10
    dropout: float = 0.1
    vocab_size: int = 29528
    global_max: bool = True  # add a max-pool state summary next to the mean
    max_instruction_tokens: int = 128  # per question block (longer ones are hoisted or truncated)
    max_option_tokens: int = 48
    hoist_instructions: bool = True  # move over-long instructions into the state (reef.packing)
    # v2: fixed question bands with per-question summaries (0 = v1 layout, state summary only)
    bands: int = 0  # bands in the schema region, one question block per band
    band_max: bool = True  # max-pool next to the mean in each band summary
    schema_to_state: bool = True  # add the mean band summary to the state rows
    mlp_rank: int = 0  # low-rank MLP (D -> r -> eD -> r -> D); 0 = full rank
    # empty-state requests: the input moves from the instructions into the state (reef.packing)
    hoist_empty_state: bool = False
    # multiplicative mixing: features are also multiplied by the state (and band) summaries, so
    # an option can be matched against the input with one product instead of a deep MLP
    gate: bool = False
    # training from scratch: every block's last conv and the output conv start at zero, so each
    # block starts as the identity and the first logits are 0 (with the default init the residual
    # stream and the logits start in the tens, and training collapses to constant outputs)
    zero_init: bool = True
    # options keyed "A" / "3" / "option_12" are rendered as their description alone (reef.packing)
    drop_opaque_keys: bool = True
    # state summary as AVERAGE_POOL_2D instead of MEAN (no weights; same maths, portable rounding)
    state_avgpool: bool = True
    # LeakyReLU slope in the block MLPs (0 = ReLU). With ReLU, v2 training killed most MLPs:
    # every hidden unit went negative for all inputs, so over half the weights did nothing.
    mlp_leak: float = 0.0
    # BatchNorm in front of each block MLP. Without it, live (leaky) MLPs let the residual stream
    # grow without bound and training diverges (run L: loss 28 after 500 steps vs 1.0).
    mlp_norm: bool = False
    # int8-aware training: layer name -> [lo, hi] for QClip after the stem ("stem_q"), each
    # block's mix ("polyp{i}_mixq") and residual ("polyp{i}_resq"). Weightless; see QClip.
    act_clip: dict | None = None

    def __post_init__(self):
        for name in ("state_len", "schema_len"):
            if getattr(self, name) % self.grid_width:
                raise ValueError(f"{name} must be a multiple of grid_width={self.grid_width}")
        if self.gate and not self.bands:
            raise ValueError("gate needs bands")
        if self.bands and (self.schema_len // self.grid_width) % self.bands:
            raise ValueError(f"schema rows ({self.schema_len // self.grid_width}) must split into {self.bands} bands")

    @property
    def seq_len(self) -> int:
        return self.state_len + self.schema_len

    @property
    def grid(self) -> tuple[int, int]:
        return self.seq_len // self.grid_width, self.grid_width

    @property
    def state_rows(self) -> int:
        return self.state_len // self.grid_width

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ReefConfig:
        known = set(cls.__dataclass_fields__)
        # checkpoints from before these fields existed
        d = {"global_max": False, "max_instruction_tokens": 40, "max_option_tokens": 28,
             "hoist_instructions": False, "bands": 0, "mlp_rank": 0,
             "hoist_empty_state": False, "gate": False, "zero_init": False,
             "drop_opaque_keys": False, **d}
        d.setdefault("state_avgpool", bool(d.get("bands")))  # every banded checkpoint uses it
        return cls(**{k: v for k, v in d.items() if k in known})


def _pointwise(units: int, name: str, use_bias: bool = True, zero: bool = False) -> layers.Conv2D:
    return layers.Conv2D(units, 1, use_bias=use_bias, name=name,
                         kernel_initializer="zeros" if zero else "glorot_uniform")


@keras.saving.register_keras_serializable(package="reef")
class QClip(layers.Layer):
    """Clip to ``[lo, hi]``; while training, also round to that range's 256 int8 levels.

    Int8 gives each tensor one range, set by its extremes. In v2 the residual stream grows rare
    huge values (mostly at padding positions; the state max-pool spreads them to every position
    through the summary), so typical values got 2-3 levels. A fixed clip bounds the range on the
    Edge TPU as in training, and the rounding (straight-through gradient) lets the model adapt.
    No weights, so checkpoints load with or without it. Written as ``c + r*clip((x-c)/r, -1, 1)``,
    which lowers to scalar MUL/ADD + RELU_N1_TO_1: MINIMUM/MAXIMUM against constant bounds make
    edgetpu_compiler store full-size bound tensors (8.6 MiB off-chip for 23 clips).
    """

    def __init__(self, lo: float, hi: float, **kw):
        super().__init__(**kw)
        self.lo, self.hi = float(lo), float(hi)

    def call(self, x, training=False):
        if not training and keras.backend.backend() == "tensorflow":
            # Export: a FakeQuant node makes the int8 converter use [lo, hi] as the range of the
            # tensor that feeds it, with no op of its own. (A clip written as ops quantises that
            # tensor at its full calibrated range first and adds two quantisation steps.)
            import tensorflow as tf

            return tf.quantization.fake_quant_with_min_max_args(x, min=self.lo, max=self.hi, num_bits=8)
        c, r = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2
        z = ops.clip(x * (1.0 / r) + (-c / r), -1.0, 1.0)  # MUL + ADD (a SUB would be a new op)
        if training:
            q = ops.round((z + 1.0) * 127.5) / 127.5 - 1.0  # the 256 int8 levels of [lo, hi]
            z = z + ops.stop_gradient(q - z)
        return z * r + c

    def get_config(self):
        return {**super().get_config(), "lo": self.lo, "hi": self.hi}


def _clip(cfg: ReefConfig, name: str, x):
    r = (cfg.act_clip or {}).get(name)
    return QClip(r[0], r[1], name=name)(x) if r else x


@keras.saving.register_keras_serializable(package="reef")
class StateMean(layers.Layer):
    """Mean over the state rows ``[0, state_rows)``. Lowers to STRIDED_SLICE + MEAN."""

    def __init__(self, state_rows: int, **kw):
        super().__init__(**kw)
        self.state_rows = state_rows

    def call(self, x):
        return ops.mean(x[:, : self.state_rows], axis=(1, 2), keepdims=True)

    def get_config(self):
        return {**super().get_config(), "state_rows": self.state_rows}


@keras.saving.register_keras_serializable(package="reef")
class StateAvg(layers.Layer):
    """Mean over the state rows as one AveragePooling2D window (same maths as :class:`StateMean`).

    Lowers to STRIDED_SLICE + AVERAGE_POOL_2D, whose int8 rounding is the same in every TFLite CPU
    build (the int8 MEAN kernel of LiteRT.js' wasm XNNPack rounds differently from native XNNPack
    in rare cases, which breaks browser/Python parity).
    """

    def __init__(self, state_rows: int, grid_width: int, **kw):
        super().__init__(**kw)
        self.state_rows, self.grid_width = state_rows, grid_width
        self.pool = layers.AveragePooling2D(pool_size=(state_rows, grid_width))

    def call(self, x):
        return self.pool(x[:, : self.state_rows])

    def get_config(self):
        return {**super().get_config(), "state_rows": self.state_rows, "grid_width": self.grid_width}


@keras.saving.register_keras_serializable(package="reef")
class StateMax(layers.Layer):
    """Max over the state rows. A single MaxPooling2D window covering the whole state region."""

    def __init__(self, state_rows: int, grid_width: int, **kw):
        super().__init__(**kw)
        self.state_rows, self.grid_width = state_rows, grid_width
        self.pool = layers.MaxPooling2D(pool_size=(state_rows, grid_width))

    def call(self, x):
        return self.pool(x[:, : self.state_rows])

    def get_config(self):
        return {**super().get_config(), "state_rows": self.state_rows, "grid_width": self.grid_width}


@keras.saving.register_keras_serializable(package="reef")
class Rows(layers.Layer):
    """Grid rows ``[start, stop)``. Lowers to STRIDED_SLICE."""

    def __init__(self, start: int, stop: int, **kw):
        super().__init__(**kw)
        self.start, self.stop = start, stop

    def call(self, x):
        return x[:, self.start : self.stop]

    def get_config(self):
        return {**super().get_config(), "start": self.start, "stop": self.stop}


@keras.saving.register_keras_serializable(package="reef")
class BandMean(layers.Layer):
    """Mean of the per-band summaries ``[B, bands, 1, D] -> [B, 1, 1, D]``. Lowers to MEAN."""

    def call(self, x):
        return ops.mean(x, axis=(1, 2), keepdims=True)


def _band_context(cfg: ReefConfig, h, local, p: str, g=None):
    """v2 mixing: each question band gets its own mean(+max) summary, projected and broadcast-added
    over that band only, so every option marker sees its own question in every block. The mean of
    the band summaries is added to the state rows, so the state is read in the light of the
    questions (and the next block's state max-pool can pick question-relevant features).

    With ``cfg.gate``, option rows' local features are also multiplied by ``tanh`` of their band
    summary and, separately, of the state summary ``g``, so one product can match an option
    against the input.
    (The int8 converter leaves a broadcast MUL over the 12 state rows or the whole grid in float,
    a 6-row band quantises; adding ``g`` to a band summary before ``tanh`` crashes
    edgetpu_compiler 14.1.)

    Broadcasting is a per-band slice + broadcast ADD: UpSampling2D would emit TILE, which
    edgetpu_compiler 14.1 rejects.
    """
    H, W = cfg.grid
    S = cfg.state_rows
    band_rows = (H - S) // cfg.bands
    sch = Rows(S, H, name=f"{p}_sch")(h)
    b = layers.AveragePooling2D((band_rows, W), strides=(band_rows, W), name=f"{p}_bmean")(sch)
    if cfg.band_max:
        bx = layers.MaxPooling2D((band_rows, W), strides=(band_rows, W), name=f"{p}_bmax")(sch)
        b = layers.Concatenate(name=f"{p}_bcat")([b, bx])
    b = _pointwise(cfg.width, f"{p}_bctx")(b)  # [B, bands, 1, D]
    st = Rows(0, S, name=f"{p}_stl")(local)
    if cfg.schema_to_state:
        st = layers.Add(name=f"{p}_s2s")([st, BandMean(name=f"{p}_bglob")(b)])
    parts = [st]
    gt = layers.Activation("tanh", name=f"{p}_gt")(g) if cfg.gate else None
    for j in range(cfg.bands):
        lo, hi = S + j * band_rows, S + (j + 1) * band_rows
        lj = Rows(lo, hi, name=f"{p}_l{j}")(local)
        bj = Rows(j, j + 1, name=f"{p}_b{j}")(b)
        terms = [lj, bj]
        if cfg.gate:  # option features gated by their own question's summary and by the state's
            # (the normalised local features, not the residual stream: products of the residual
            # stream compound over blocks and blow up)
            terms.append(layers.Multiply(name=f"{p}_g{j}")([lj, layers.Activation("tanh", name=f"{p}_bt{j}")(bj)]))
            terms.append(layers.Multiply(name=f"{p}_gs{j}")([lj, gt]))
        parts.append(layers.Add(name=f"{p}_band{j}")(terms))
    return layers.Concatenate(axis=1, name=f"{p}_rows")(parts)


def build_encoder(cfg: ReefConfig, batch_size: int | None = None) -> keras.Model:
    """The part that is quantised and compiled for the Edge TPU.

    Input:   ``emb [B, H, W, input_dim]``, the window's embeddings folded into a grid.
    Output:  ``logits [B, H, W, 1]``, one decision score per position. Only option-marker
             positions are read.

    Use ``batch_size=1`` for export: a static batch keeps shape arithmetic out of the graph.
    """
    H, W = cfg.grid
    D = cfg.width
    emb = keras.Input((H, W, cfg.input_dim), batch_size=batch_size, name="emb")

    h = _pointwise(D, "stem")(emb)
    h = layers.BatchNormalization(name="stem_bn")(h)
    h = _clip(cfg, "stem_q", h)

    for i in range(cfg.depth):
        p = f"polyp{i}"
        local = layers.DepthwiseConv2D(tuple(cfg.kernel), padding="same", use_bias=False, name=f"{p}_dw")(h)
        local = layers.BatchNormalization(name=f"{p}_bn")(local)
        if cfg.state_avgpool:
            g = StateAvg(cfg.state_rows, cfg.grid_width, name=f"{p}_pool")(h)
        else:
            g = StateMean(cfg.state_rows, name=f"{p}_pool")(h)
        if cfg.global_max:
            # Max over the state picks up rare but decisive features ("FAILED", "unknown person")
            # that a mean over hundreds of tokens would dilute. Lowers to STRIDED_SLICE + MAX_POOL_2D.
            mx = StateMax(cfg.state_rows, cfg.grid_width, name=f"{p}_maxpool")(h)
            g = layers.Concatenate(name=f"{p}_poolcat")([g, mx])
        g = _pointwise(D, f"{p}_ctx")(g)
        if cfg.bands:
            local = _band_context(cfg, h, local, p, g)
        y = layers.Add(name=f"{p}_mix")([local, g])  # broadcast the state summary everywhere
        y = _clip(cfg, f"{p}_mixq", y)
        if cfg.mlp_norm:
            y = layers.BatchNormalization(name=f"{p}_mlp_bn")(y)
        if cfg.mlp_rank:
            y = _pointwise(cfg.mlp_rank, f"{p}_upr", use_bias=False)(y)
        y = _pointwise(D * cfg.expansion, f"{p}_up")(y)
        if cfg.mlp_leak:
            # max(y, a*y) == LeakyReLU for 0 < a < 1. LEAKY_RELU crashes edgetpu_compiler 14.1;
            # MUL + MAXIMUM map to the Edge TPU and add no weights (ReLU checkpoints still load).
            y = layers.Maximum(name=f"{p}_relu")([y, layers.Rescaling(cfg.mlp_leak, name=f"{p}_leak")(y)])
        else:
            y = layers.ReLU(name=f"{p}_relu")(y)
        if cfg.dropout > 0:
            y = layers.Dropout(cfg.dropout, name=f"{p}_drop")(y)
        if cfg.mlp_rank:
            y = _pointwise(cfg.mlp_rank, f"{p}_downr", use_bias=False)(y)
        y = _pointwise(D, f"{p}_down", zero=cfg.zero_init)(y)
        h = layers.Add(name=f"{p}_res")([h, y])
        h = _clip(cfg, f"{p}_resq", h)

    z = _pointwise(D, "head_hidden")(h)
    z = layers.ReLU(name="head_relu")(z)
    logits = _pointwise(1, "head_out", zero=cfg.zero_init)(z)
    return keras.Model(emb, logits, name="reef_encoder")


class HostEmbedding(layers.Layer):
    """Token, role and level embeddings, projected to ``input_dim``. All linear, so foldable.

    The token table is initialised from a pretrained static embedding model (model2vec / potion)
    and kept frozen: 1.2k training cases are far too few to learn word meanings. A learned linear
    projection maps it down to ``input_dim``, which halves the bytes sent over USB on every call.
    At export the projection is folded into the table (``tok @ P``), so the runtime does one
    lookup and one add per token.
    """

    def __init__(self, cfg: ReefConfig, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        init = keras.initializers.RandomNormal(stddev=0.5)
        self.tok = layers.Embedding(cfg.vocab_size, cfg.embed_dim, trainable=False, name="tok")
        self.proj = layers.Dense(cfg.input_dim, use_bias=False, name="tok_proj")
        self.role = layers.Embedding(NUM_ROLES, cfg.input_dim, embeddings_initializer=init, name="role")
        self.level = layers.Embedding(NUM_LEVELS, cfg.input_dim, embeddings_initializer=init, name="level")

    def call(self, ids, roles, levels):
        x = self.proj(self.tok(ids)) + self.role(roles) + self.level(levels)
        # PAD positions (role 0) stay exactly zero, as they will be on the device.
        return x * ops.cast(ops.expand_dims(roles > 0, -1), x.dtype)  # [B, L, input_dim]


class ReefModel(keras.Model):
    """Embeddings + encoder, used for training. Export splits it back into its two halves."""

    def __init__(self, cfg: ReefConfig, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.embed = HostEmbedding(cfg, name="host_embedding")
        self.encoder = build_encoder(cfg)

    def call(self, inputs, training=False):
        ids, roles, levels = inputs["ids"], inputs["roles"], inputs["levels"]
        x = self.embed(ids, roles, levels)
        if training and self.cfg.dropout > 0:
            # Drop whole token vectors (not channels) so the model learns not to lean on any one word.
            keep = keras.random.uniform(ops.shape(ids)) > self.cfg.dropout
            x = x * ops.cast(ops.expand_dims(keep, -1), x.dtype)
        H, W = self.cfg.grid
        x = ops.reshape(x, (-1, H, W, self.cfg.input_dim))
        logits = self.encoder(x, training=training)
        return ops.reshape(logits, (-1, self.cfg.seq_len))
