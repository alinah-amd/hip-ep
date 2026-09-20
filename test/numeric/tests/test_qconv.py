#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the fused quantized 1x1 convolution (hip.qconv).

The fusion is W4A16 over a 1x1 window: per-tensor UINT16 activations, weights
quantized per output channel, and a UINT16 requantize. It matches

    QuantizeLinear(Conv(DequantizeLinear(X), DequantizeLinear(W)))

and lowers to hip_qconv, which dispatches on the number of output positions:
spatial_size <= 8 takes the GEMV kernel (one block per output element, the
decode shape), anything wider takes the tiled kernel (LDS-staged K-tiles, the
prefill shape). Both are covered here, on both sides of that threshold.

Weight storage is the ONNX packed-nibble convention at 4 bits -- two logical
elements per byte, low nibble first over the flattened row-major sequence. A
kernel that reads or vectorizes those bytes has to reproduce it exactly, and
gets it wrong in ways that only some shapes expose:

  * an odd in_channels puts every other output channel's row on an odd
    logical index, so the row no longer starts on a byte boundary;
  * an in_channels that is not a multiple of the load width leaves a tail
    that the vectorized body cannot cover;
  * signed and unsigned nibbles disagree on every value above 7.

Each of those has a case below.

Accumulator width is the other thing worth pinning. |X - z_x| reaches 65535
and |W - z_w| reaches 15 at 4 bits, so one term is up to ~1e6 and int32
overflows after roughly two thousand terms. The wide-K cases push past that.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# hip_qconv dispatches to the GEMV kernel at or below this many output
# positions and to the tiled kernel above it. Mirrors kGemvSpatialThreshold in
# lib/Runtime/Kernels/hip/qconv_kernel.hip.
GEMV_SPATIAL_THRESHOLD = 8

# A decode-shaped projection from a 40-layer model: hidden 5120, FFN 17920.
# Full-width Cout would be 45 MB of initializer, so the wide cases below keep
# Cout small and vary Cin, which is the axis the inner loop reduces over.
HIDDEN = 5120
FFN = 17920


def _make_qconv_model(
    cin: int,
    cout: int,
    spatial: int,
    weight_bits: int = 4,
    weight_signed: bool = True,
    seed: int = 42,
):
    """Build QuantizeLinear(Conv1x1(DQ(X), DQ(W))) with random quantized weights.

    Scales are chosen so the requantized output lands mid-range: the
    accumulator is a sum of ``cin`` random-signed terms, so it grows as
    sqrt(cin), and a fixed weight scale would saturate the wide-K cases and
    flatten the narrow ones. A saturated output compares equal for the wrong
    reason, which is exactly what this suite is meant to catch.
    """
    if weight_bits not in (4, 8):
        raise ValueError(f"weight_bits must be 4 or 8, got {weight_bits}")

    rng = np.random.default_rng(seed)

    if weight_bits == 4:
        w_dtype = TensorProto.INT4 if weight_signed else TensorProto.UINT4
        lo, hi = (-8, 8) if weight_signed else (0, 16)
    else:
        w_dtype = TensorProto.INT8 if weight_signed else TensorProto.UINT8
        lo, hi = (-128, 128) if weight_signed else (0, 256)

    w = rng.integers(lo, hi, [cout, cin, 1, 1], dtype=np.int64)
    w_zp = rng.integers(lo, hi, [cout], dtype=np.int64)

    # Term magnitude is |X - z_x| * |W - z_w| ~ (hi - lo)/4 * 32768, and the
    # sum of cin random-signed terms grows as sqrt(cin), roughly normally.
    # Target one standard deviation at 6000 of the 32768 available either side
    # of the zero point: the rails then sit past 5 sigma, so nothing saturates,
    # while the spread is still four decimal digits wide and an atol of 1 on it
    # is a tight comparison.
    term = (hi - lo) / 4.0 * 32768.0
    w_scale_mag = 6000.0 / (np.sqrt(cin) * term)
    w_scale = rng.uniform(0.5 * w_scale_mag, 1.5 * w_scale_mag, [cout]).astype(
        np.float32
    )

    x = helper.make_tensor_value_info("X", TensorProto.UINT16, [1, cin, 1, spatial])
    y = helper.make_tensor_value_info("Y", TensorProto.UINT16, [1, cout, 1, spatial])

    inits = [
        helper.make_tensor("W", w_dtype, [cout, cin, 1, 1], w.ravel().tolist()),
        helper.make_tensor("Wzp", w_dtype, [cout], w_zp.tolist()),
        numpy_helper.from_array(w_scale, "Wscale"),
        # x_scale / y_scale is the only ratio the kernel sees; keep it at 1 so
        # the weight scale alone controls the output range.
        numpy_helper.from_array(np.array(1.0e-4, np.float32), "Xscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Xzp"),
        numpy_helper.from_array(np.array(1.0e-4, np.float32), "Yscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Yzp"),
    ]

    # No bias: the fusion matches a four-operand hip.conv only, so a biased
    # convolution never reaches hip_qconv. The kernel takes a bias pointer and
    # its epilogue honours one, but nothing can currently pass a non-null value.
    nodes = [
        helper.make_node("DequantizeLinear", ["X", "Xscale", "Xzp"], ["dqx"]),
        helper.make_node("DequantizeLinear", ["W", "Wscale", "Wzp"], ["dqw"], axis=0),
        helper.make_node(
            "Conv",
            ["dqx", "dqw"],
            ["conv"],
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node("QuantizeLinear", ["conv", "Yscale", "Yzp"], ["Y"]),
    ]

    return make_model_from_nodes(nodes, [x], [y], initializers=inits, opset=21)


def _random_activations(cin: int, spatial: int, seed: int = 99):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, [1, cin, 1, spatial], dtype=np.uint16)


def _assert_not_saturated(out: np.ndarray):
    """A clamped output agrees with the reference no matter what the kernel did."""
    pinned = np.count_nonzero((out == 0) | (out == 65535))
    assert pinned < 0.005 * out.size, (
        f"{pinned}/{out.size} outputs saturated -- the scales chose a range the "
        "comparison cannot see through"
    )


def _run(model_runner, cin, cout, spatial, **kwargs):
    model = _make_qconv_model(cin, cout, spatial, **kwargs)
    x = _random_activations(cin, spatial)
    actual, expected = model_runner.run_sample(model, [x])
    _assert_not_saturated(actual[0])
    # Both sides accumulate the same integers exactly; the only divergence is
    # the float requantize at the end, which is well under one output LSB.
    compare_outputs(actual, expected, atol=1, rtol=0, cos_threshold=0.9999)


class TestQConv:
    # ------------------------------------------------------------------
    # Dispatch: both kernels, and the boundary between them.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "spatial",
        [1, GEMV_SPATIAL_THRESHOLD, GEMV_SPATIAL_THRESHOLD + 1, 128],
    )
    def test_qconv_dispatch_boundary(self, model_runner, spatial):
        """Same shape either side of the GEMV/tiled threshold.

        spatial 1 and 8 take the GEMV kernel, 9 and 128 the tiled one. Holding
        everything else fixed means a failure at one spatial and not another
        names the kernel rather than the shape.
        """
        _run(model_runner, 256, 64, spatial)

    # ------------------------------------------------------------------
    # Weight packing.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("spatial", [1, 128])
    def test_qconv_odd_in_channels(self, model_runner, spatial):
        """Odd Cin: odd-numbered weight rows start on a high nibble.

        The logical index of row co is co * cin, so an odd cin alternates the
        parity of the row start. Any weight read that assumes a row begins on a
        byte boundary produces every other output channel wrong.
        """
        _run(model_runner, 63, 32, spatial)

    @pytest.mark.parametrize("spatial", [1, 128])
    @pytest.mark.parametrize("cin", [68, 130, 252])
    def test_qconv_ragged_in_channels(self, model_runner, spatial, cin):
        """Cin that no plausible vector width divides, leaving a tail.

        68, 130 and 252 are each even -- so rows stay byte-aligned -- but leave
        a remainder against 8-, 16- and 32-wide loads respectively. A
        vectorized main loop with a broken tail loses the last few channels of
        the reduction, which shifts the result without breaking it obviously.
        """
        _run(model_runner, cin, 32, spatial)

    @pytest.mark.parametrize("spatial", [1, 128])
    @pytest.mark.parametrize("cout", [1, 3, 5, 33])
    def test_qconv_ragged_out_channels(self, model_runner, spatial, cout):
        """Cout that leaves a partial last group of output channels.

        A kernel that hands several output channels to one block has a last
        group that is not full. Over-running it reads past the weight buffer;
        clamping it to the wrong lane writes one channel's result into
        another's slot, which leaves the tensor the right shape and the wrong
        contents.
        """
        _run(model_runner, 256, cout, spatial)

    @pytest.mark.parametrize("spatial", [1, 128])
    def test_qconv_unsigned_4bit(self, model_runner, spatial):
        """UINT4 weights: nibbles above 7 widen the other way.

        Signed and unsigned agree on 0..7 and disagree on 8..15, so a test that
        only covers one sign convention passes with the widening inverted.
        """
        _run(model_runner, 256, 64, spatial, weight_signed=False)

    @pytest.mark.parametrize("spatial", [1, 128])
    @pytest.mark.parametrize("weight_signed", [True, False])
    def test_qconv_8bit(self, model_runner, spatial, weight_signed):
        """8-bit weights: one element per byte, no nibble unpacking at all.

        The staged difference reaches +-255 here, which is what forces the
        int16 staging width rather than int8.
        """
        _run(model_runner, 256, 64, spatial, weight_bits=8, weight_signed=weight_signed)

    # ------------------------------------------------------------------
    # Accumulator width.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("spatial", [1, 128])
    def test_qconv_wide_k_hidden(self, model_runner, spatial):
        """Cin = 5120, the hidden width of a 40-layer decoder.

        One term reaches ~1e6, so int32 saturates after about two thousand of
        them. This is past that, and the GEMV path reduces the whole of Cin in
        a single block.
        """
        _run(model_runner, HIDDEN, 64, spatial)

    def test_qconv_wide_k_ffn_decode(self, model_runner):
        """Cin = 17920, the FFN width -- the widest reduction in the model.

        Decode only: this is the down_proj shape, and it is the single
        heaviest dispatch in a decode step.
        """
        _run(model_runner, FFN, 32, 1)
