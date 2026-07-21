"""Correctness for vtl's fused LFM2 short-conv DECODE kernel (bcx_conv_gate_quant).

Our kernel is one op; stock is three (`B*x` -> Triton `causal_conv1d_update` -> gate multiply +
per-token fp8 quant). Two oracles, because the two halves fail differently:

  * the CONV half is checked against vLLM's own ``causal_conv1d_update`` at the same shapes --
    both the returned activations and the rotated ``conv_state`` -- so a layout or ring-buffer
    mistake shows up as a state mismatch, not just a numeric one;
  * the QUANT half is checked against a pure-torch reference reproducing Triton's arithmetic
    (dtype-rounded products into an fp32 accumulator, one more rounding on the bf16 store, then
    ``C *`` and the per-token amax/scale), judged on the shared fp8 bar in _fp8_testutil.

    pytest bench/test_bcx_conv_gate_quant.py -q      # skips entirely without vtl._C
    python bench/test_bcx_conv_gate_quant.py         # same checks, no pytest

There is no stock counterpart to A/B here (the op only exists in vtl._C), so the VTL_SKIP_EXT=1
leg of `make test-kernel` skips this module rather than failing.
"""

from __future__ import annotations

import os

import pytest
import torch
from _fp8_testutil import FP8, FP8_MAX, MIN_SCALE, assert_fp8_equivalent, scale_rtol

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")

WIDTH = 3  # LFM2.5 conv_L_cache
STATE_LEN = WIDTH - 1  # what short_conv_state_shape allocates
NULL_BLOCK_ID = 0


def _have_op() -> bool:
    if USE_STOCK:
        return False
    try:
        import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schemas
        import vtl._C  # noqa: F401  -- registers the vllm_cuda:: ops
    except Exception:
        return False
    return hasattr(torch.ops.vllm_cuda, "bcx_conv_gate_quant")


HAVE_OP = _have_op()
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not HAVE_OP, reason="vtl._C bcx_conv_gate_quant unavailable"),
]


def make_state(num_blocks, dim, dtype, device, layout):
    """A conv-state tensor viewed as (blocks, dim, STATE_LEN), in either vLLM layout.

    "SD" is the default: the cache is allocated (blocks, STATE_LEN, dim) and short_conv
    transposes it, so stride_dim == 1 and stride_token == dim. "DS" stores (blocks, dim,
    STATE_LEN) directly. The kernel takes strides, so both must work.

    This also selects which kernel instantiation runs, so parametrizing on it is load-bearing
    coverage, not paranoia: "SD" (stride_dim == 1) takes the vectorised 16-byte-load path,
    "DS" (stride_dim == 2) falls to the scalar one.
    """
    if layout == "SD":
        raw = torch.randn(num_blocks, STATE_LEN, dim, dtype=dtype, device=device)
        return raw.transpose(-1, -2)
    return torch.randn(num_blocks, dim, STATE_LEN, dtype=dtype, device=device)


def reference(bcx, state, weight, bias, state_idx, dtype, scale_ub=None):
    """Pure-torch oracle. Returns (out_fp8, scale, new_state), leaving `state` untouched.

    Mirrors _causal_conv1d_update_kernel at seqlen=1 / KERNEL_WIDTH=3 / state_len=2: the taps
    are multiplied in `dtype` (Triton types bf16*bf16 -> bf16) and summed in fp32, then the
    result is rounded back to `dtype` on the store before the `C *` gate.
    """
    T, dim = bcx.shape[0], weight.shape[0]
    B, C, x = bcx[:, :dim], bcx[:, dim : 2 * dim], bcx[:, 2 * dim :]
    new_state = state.clone()

    live = state_idx[:, 0] != NULL_BLOCK_ID
    coords = state_idx[:, 0].long()

    bx = (B * x).to(dtype)  # dtype multiply
    s0 = state[coords, :, 0]
    s1 = state[coords, :, 1]

    acc = torch.zeros(T, dim, dtype=torch.float32, device=bcx.device)
    if bias is not None:
        acc += bias.float()
    acc += (weight[:, 0] * s0).to(dtype).float()
    acc += (weight[:, 1] * s1).to(dtype).float()
    acc += (weight[:, 2] * bx).to(dtype).float()

    y = (C * acc.to(dtype)).float()  # dtype multiply, widened for the reduction

    amax = y.abs().amax(-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    out = torch.clamp(y / scale, -FP8_MAX, FP8_MAX).to(FP8)

    # Padded rows: state untouched; we define the output as zero at the minimum scale.
    out[~live] = torch.zeros((), dtype=FP8, device=out.device)
    scale[~live] = MIN_SCALE

    # Ring-buffer rotation [s0, s1] -> [s1, bx], live rows only.
    lc = coords[live]
    new_state[lc, :, 0] = s1[live]
    new_state[lc, :, 1] = bx[live]
    return out, scale, new_state


def run(bcx, state, weight, bias, state_idx, scale_ub=None):
    T, dim = bcx.shape[0], weight.shape[0]
    out = torch.empty((T, dim), dtype=FP8, device=bcx.device)
    scale = torch.empty((T, 1), dtype=torch.float32, device=bcx.device)
    torch.ops.vllm_cuda.bcx_conv_gate_quant(
        out, scale, state, bcx, weight, bias, state_idx, NULL_BLOCK_ID, scale_ub
    )
    return out, scale


def build(T, dim, dtype, device="cuda", with_bias=False, layout="SD", nulls=0, seed=0):
    torch.manual_seed(seed)
    num_blocks = T + 4  # block 0 is the null block, so keep spare live slots
    bcx = torch.randn(T, 3 * dim, dtype=dtype, device=device)
    state = make_state(num_blocks, dim, dtype, device, layout)
    weight = torch.randn(dim, WIDTH, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device) if with_bias else None
    # Distinct live blocks (never 0); the last `nulls` rows are cudagraph padding.
    coords = torch.arange(1, T + 1, dtype=torch.int32, device=device)
    if nulls:
        coords[T - nulls :] = NULL_BLOCK_ID
    return bcx, state, weight, bias, coords.unsqueeze(-1).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T", [1, 7, 64, 256])
@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_reference(dtype, T, layout, with_bias):
    dim = 2048  # LFM2.5 conv_dim
    bcx, state, weight, bias, idx = build(
        T, dim, dtype, with_bias=with_bias, layout=layout
    )
    exp_out, exp_scale, exp_state = reference(bcx, state, weight, bias, idx, dtype)

    got_out, got_scale = run(bcx, state, weight, bias, idx)  # mutates `state` in place

    assert_fp8_equivalent(got_out, exp_out, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0.0)
    # The rotated ring buffer must match exactly -- it is fed back in on the next step.
    torch.testing.assert_close(state.float(), exp_state.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_matches_vllm_triton_conv(dtype, layout):
    """The conv half against vLLM's own kernel: same activations AND same state rotation."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    T, dim = 64, 2048
    bcx, state, weight, bias, idx = build(T, dim, dtype, layout=layout, seed=3)
    B, C, x = bcx[:, :dim], bcx[:, dim : 2 * dim], bcx[:, 2 * dim :]

    # Stock: materialise B*x, then the Triton update (which writes into it in place).
    stock_state = state.clone()
    Bx = (B * x).contiguous()
    conv_out = causal_conv1d_update(
        Bx, stock_state, weight, bias, activation=None, conv_state_indices=idx
    )
    stock_y = C * conv_out

    got_out, got_scale = run(bcx, state, weight, bias, idx)

    # State: bit-exact, both kernels write the same two slots.
    torch.testing.assert_close(state.float(), stock_state.float(), rtol=0.0, atol=0.0)
    # Activations: quantize stock's bf16 gate output the same way and compare fp8 codes.
    amax = stock_y.float().abs().amax(-1, keepdim=True)
    exp_scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    exp_out = torch.clamp(stock_y.float() / exp_scale, -FP8_MAX, FP8_MAX).to(FP8)
    assert_fp8_equivalent(got_out, exp_out, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0.0)


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_null_blocks_leave_state_untouched(layout):
    """Cudagraph padding rows must not touch the ring buffer of ANY block."""
    dtype, T, dim, nulls = torch.bfloat16, 32, 2048, 8
    bcx, state, weight, bias, idx = build(
        T, dim, dtype, layout=layout, nulls=nulls, seed=7
    )
    before = state.clone()
    live_coords = idx[: T - nulls, 0].long()

    got_out, got_scale = run(bcx, state, weight, bias, idx)

    untouched = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    untouched[live_coords] = False
    torch.testing.assert_close(
        state[untouched].float(), before[untouched].float(), rtol=0.0, atol=0.0
    )
    # Padded rows get a defined, finite output rather than uninitialised memory.
    assert torch.equal(got_out[T - nulls :].float(), torch.zeros_like(got_out[T - nulls :].float()))
    assert torch.all(got_scale[T - nulls :] == MIN_SCALE)


def test_scale_ub_is_honored():
    dtype, T, dim = torch.bfloat16, 16, 2048
    bcx, state, weight, bias, idx = build(T, dim, dtype, seed=11)
    ub = torch.tensor([0.05], dtype=torch.float32, device=bcx.device)
    exp_out, exp_scale, _ = reference(bcx, state, weight, bias, idx, dtype, scale_ub=ub)
    got_out, got_scale = run(bcx, state, weight, bias, idx, scale_ub=ub)
    assert_fp8_equivalent(got_out, exp_out, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0.0)


def test_supported_predicate():
    """The shape gate short_conv.py queries instead of re-hardcoding kVec / kFastMaxThreads."""
    supported = torch.ops.vllm_cuda.bcx_conv_gate_supported
    assert supported(2048, WIDTH, STATE_LEN)  # LFM2.5
    assert not supported(2048, 4, STATE_LEN)  # wrong conv width
    assert not supported(2048, WIDTH, 3)  # wrong state length
    assert not supported(2050, WIDTH, STATE_LEN)  # not a multiple of the vector width
    assert not supported(1 << 20, WIDTH, STATE_LEN)  # past the one-block reduction cap


if __name__ == "__main__":
    if not HAVE_OP:
        raise SystemExit("vtl._C bcx_conv_gate_quant unavailable; nothing to check")
    for layout in ("SD", "DS"):
        for dtype in (torch.bfloat16, torch.float16):
            for T in (1, 7, 64, 256):
                for with_bias in (False, True):
                    test_matches_reference(dtype, T, layout, with_bias)
            test_matches_vllm_triton_conv(dtype, layout)
        test_null_blocks_leave_state_untouched(layout)
    test_scale_ub_is_honored()
    test_supported_predicate()
    print("bcx_conv_gate_quant checks ok")
