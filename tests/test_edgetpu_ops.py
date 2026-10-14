"""Guards the Edge TPU contract: the encoder must lower to int8 ops the Edge TPU can run.

edgetpu_compiler is x86-only and usually not installed, so this checks the op list of the int8
TFLite model instead. Ops outside ALLOWED either fall back to the CPU or (activation x activation
MUL, except a per-band MUL by a broadcast summary vector) crash compiler 14.1, so a new op here
should come with a compile test on real hardware.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="needs the [train] extra")

ALLOWED = {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MEAN", "MAX_POOL_2D", "CONCATENATION", "STRIDED_SLICE", "RELU",
           "QUANTIZE", "DEQUANTIZE", "RESHAPE",
           # v2 band summaries (compiled to one Edge TPU subgraph, 0 off-chip, on edgetpu_compiler 14.1)
           "AVERAGE_POOL_2D"}
# v2 gate: activation x broadcast summary-vector MUL per band (one subgraph, 0 off-chip on 14.1)
GATE_OPS = {"MUL", "TANH"}


def _tiny_int8_model(**shape):
    import tensorflow as tf

    from reef.model import ReefConfig, build_encoder

    cfg = ReefConfig(state_len=64, schema_len=64, grid_width=32, width=32, input_dim=16, depth=2, dropout=0.0, **shape)
    enc = build_encoder(cfg, batch_size=1)
    H, W = cfg.grid
    rng = np.random.default_rng(0)

    def rep():
        for _ in range(8):
            yield {"emb": rng.normal(size=(1, H, W, cfg.input_dim)).astype(np.float32)}

    conv = tf.lite.TFLiteConverter.from_keras_model(enc)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert(), cfg


@pytest.mark.parametrize("shape", [{}, {"bands": 2, "mlp_rank": 8}, {"bands": 2, "gate": True},
                                   {"bands": 2, "gate": True, "mlp_leak": 0.1,
                                    "act_clip": {"stem_q": [-3, 4], "polyp0_mixq": [-50, 40], "polyp1_resq": [-9, 9]}}],
                         ids=["v1", "v2-bands", "v2-gate", "v2-leak-clip"])
def test_encoder_ops_are_edgetpu_friendly(shape):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    blob, cfg = _tiny_int8_model(**shape)
    # Without XNNPACK, which would otherwise swallow the graph into a single DELEGATE node.
    it = Interpreter(model_content=blob,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    details = it._get_ops_details()
    ops = {op["op_name"] for op in details}
    allowed = ALLOWED | (GATE_OPS if shape.get("gate") else set())
    # leaky MLP: MUL by a scalar + MAXIMUM; clips: scalar MUL/ADD (+ RELU_N1_TO_1 unless folded
    # into the int8 range). One subgraph, 0 off-chip on 14.1 for the full-size model.
    if shape.get("mlp_leak") or shape.get("act_clip"):
        allowed |= {"MUL", "MAXIMUM", "RELU_N1_TO_1"}
    assert ops <= allowed, f"ops the Edge TPU may not run: {ops - allowed}"
    if not (shape.get("gate") or shape.get("mlp_leak") or shape.get("act_clip")):
        assert "MUL" not in ops
    # Every op runs in int8 between the input QUANTIZE and the output DEQUANTIZE (a float MUL would
    # make edgetpu_compiler refuse the model as "not quantized").
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    for op in details:
        if op["op_name"] in ("QUANTIZE", "DEQUANTIZE"):
            continue
        dtypes = {tensors[i]["dtype"] for i in [*op["inputs"], *op["outputs"]] if i >= 0}
        assert np.float32 not in dtypes, f"{op['op_name']} runs in float"
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8
    assert tuple(inp["shape"]) == (1, *cfg.grid, cfg.input_dim)
    # No dynamic shapes (they add SHAPE/PACK ops that split the Edge TPU graph).
    assert int(np.prod(out["shape"])) == cfg.seq_len


def test_qclip_clips_at_inference_and_rounds_to_int8_levels_in_training():
    from reef.model import QClip

    q = QClip(-2.0, 6.0)
    step = 8.0 / 255
    x = np.array([[-5.0, -2.0, 0.01, 1.234, 6.0, 9.0]], np.float32)
    # Inference clips (on the TF backend via FakeQuant, which also rounds: the export path).
    assert np.abs(np.asarray(q(x)) - np.clip(x, -2, 6)).max() <= step / 2 + 1e-5
    y = np.asarray(q(x, training=True))
    np.testing.assert_allclose((y + 2.0) / step, np.round((y + 2.0) / step), atol=1e-3)  # on the int8 grid
    assert y.min() >= -2.0 - 1e-6 and y.max() <= 6.0 + 1e-6
    assert np.abs(y - np.clip(x, -2, 6)).max() <= step / 2 + 1e-6
