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
TAPS = WIDTH - 1  # slots the kernel READS -- not necessarily the allocated width
# short_conv_state_shape allocates `conv_kernel - 1 + num_spec`, so the shipped config
# (num_speculative_tokens=3) allocates 5. The kernel must be correct over BOTH: on a non-spec
# single-token decode causal_conv1d_update pins state_len to `width - 1` on the host
# (causal_conv1d.py:1181-1184) and never touches the extra slots. Parametrizing on this is the
# regression guard for the `state_len == kStateLen` predicate that used to reject the drafter.
STATE_LENS = [TAPS, TAPS + 3]
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


def make_state(num_blocks, dim, dtype, device, layout, state_len=TAPS):
    """A conv-state tensor viewed as (blocks, dim, state_len), in either vLLM layout.

    "SD" is the default: the cache is allocated (blocks, state_len, dim) and short_conv
    transposes it, so stride_dim == 1 and stride_token == dim. "DS" stores (blocks, dim,
    state_len) directly. The kernel takes strides, so both must work.

    This also selects which kernel instantiation runs, so parametrizing on it is load-bearing
    coverage, not paranoia: "SD" (stride_dim == 1) takes the vectorised 16-byte-load path,
    "DS" (stride_dim == 2) falls to the scalar one.
    """
    if layout == "SD":
        raw = torch.randn(num_blocks, state_len, dim, dtype=dtype, device=device)
        return raw.transpose(-1, -2)
    return torch.randn(num_blocks, dim, state_len, dtype=dtype, device=device)


def reference(bcx, state, weight, bias, state_idx, dtype, scale_ub=None):
    """Pure-torch oracle. Returns (out_fp8, scale, new_state), leaving `state` untouched.

    Mirrors _causal_conv1d_update_kernel at seqlen=1 / KERNEL_WIDTH=3 / state_len=2: the taps
    are multiplied in `dtype` (Triton types bf16*bf16 -> bf16) and summed in fp32, then the
    result is rounded back to `dtype` on the store before the `C *` gate.

    `state` may be allocated wider than 2 slots; only slots 0 and 1 are read or written, and the
    returned `new_state` carries the rest through unchanged.
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


def run(bcx, state, weight, bias, state_idx, scale_ub=None, num_accepted=None, qsl=None):
    T, dim = bcx.shape[0], weight.shape[0]
    out = torch.empty((T, dim), dtype=FP8, device=bcx.device)
    scale = torch.empty((T, 1), dtype=torch.float32, device=bcx.device)
    torch.ops.vllm_cuda.bcx_conv_gate_quant(
        out, scale, state, bcx, weight, bias, state_idx, NULL_BLOCK_ID, scale_ub,
        num_accepted, qsl,
    )
    return out, scale


def build(
    T, dim, dtype, device="cuda", with_bias=False, layout="SD", nulls=0, seed=0,
    state_len=TAPS,
):
    torch.manual_seed(seed)
    num_blocks = T + 4  # block 0 is the null block, so keep spare live slots
    bcx = torch.randn(T, 3 * dim, dtype=dtype, device=device)
    state = make_state(num_blocks, dim, dtype, device, layout, state_len)
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
@pytest.mark.parametrize("state_len", STATE_LENS)
def test_matches_reference(dtype, T, layout, with_bias, state_len):
    dim = 2048  # LFM2.5 conv_dim
    bcx, state, weight, bias, idx = build(
        T, dim, dtype, with_bias=with_bias, layout=layout, state_len=state_len
    )
    exp_out, exp_scale, exp_state = reference(bcx, state, weight, bias, idx, dtype)

    got_out, got_scale = run(bcx, state, weight, bias, idx)  # mutates `state` in place

    assert_fp8_equivalent(got_out, exp_out, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0.0)
    # The rotated ring buffer must match exactly -- it is fed back in on the next step.
    # Includes the spec-widened slots, which must come through untouched.
    torch.testing.assert_close(state.float(), exp_state.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("dim", [2048, 1024])  # target conv_dim, then the LFM2.5-350M drafter
@pytest.mark.parametrize("state_len", STATE_LENS)
def test_matches_vllm_triton_conv(dtype, layout, dim, state_len):
    """The conv half against vLLM's own kernel: same activations AND same state rotation.

    This is the load-bearing case for the spec-widened allocation. `causal_conv1d_update` is
    called exactly the way the drafter calls it -- no num_accepted_tokens, no query_start_loc --
    which is what pins its state_len to `width - 1` on the host regardless of the allocated
    width. If that ever stops being true, the full-state comparison below fails rather than the
    kernel silently diverging on slots 2..4.
    """
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    T = 64
    bcx, state, weight, bias, idx = build(
        T, dim, dtype, layout=layout, seed=3, state_len=state_len
    )
    B, C, x = bcx[:, :dim], bcx[:, dim : 2 * dim], bcx[:, 2 * dim :]

    # Stock: materialise B*x, then the Triton update (which writes into it in place).
    stock_state = state.clone()
    Bx = (B * x).contiguous()
    conv_out = causal_conv1d_update(
        Bx, stock_state, weight, bias, activation=None, conv_state_indices=idx
    )
    stock_y = C * conv_out

    got_out, got_scale = run(bcx, state, weight, bias, idx)

    # State: bit-exact over the WHOLE allocation, including the spec-widened slots.
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


NUM_SPEC = 3  # docker-compose.yaml: num_speculative_tokens


def build_spec(reqs, query_lens, dim, dtype, device="cuda", layout="SD", seed=17,
               num_spec=NUM_SPEC, accepted=None, nulls=0):
    """A chain-spec decode batch: `reqs` requests of `query_lens[i]` tokens each.

    Mirrors what mamba_attn hands ShortConv under spec: state_indices is
    [num_reqs, 1+num_spec] (only column 0 is ever read), query_start_loc_d is [num_reqs+1] over
    the flat token axis, num_accepted_tokens is [num_reqs], and the conv state is allocated
    `conv_kernel - 1 + num_spec` slots wide.
    """
    torch.manual_seed(seed)
    T = int(sum(query_lens))
    state_len = TAPS + num_spec
    bcx = torch.randn(T, 3 * dim, dtype=dtype, device=device)
    state = make_state(reqs + 4, dim, dtype, device, layout, state_len)
    weight = torch.randn(dim, WIDTH, dtype=dtype, device=device)

    coords = torch.arange(1, reqs + 1, dtype=torch.int32, device=device)
    if nulls:
        coords[reqs - nulls :] = NULL_BLOCK_ID
    # Column 0 is the live block; the rest are never read but must exist for the shape contract.
    state_idx = coords.unsqueeze(-1).repeat(1, 1 + num_spec).contiguous()

    qsl = torch.zeros(reqs + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32, device=device).cumsum(0)

    if accepted is None:
        accepted = [1] * reqs
    num_accepted = torch.tensor(accepted, dtype=torch.int32, device=device)
    return bcx, state, weight, state_idx, qsl, num_accepted


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("dim", [2048, 1024])
@pytest.mark.parametrize(
    "query_lens, accepted",
    [
        ([1 + NUM_SPEC] * 4, [1, 1, 1, 1]),                      # everything rejected
        ([1 + NUM_SPEC] * 4, [1 + NUM_SPEC] * 4),                # everything accepted
        ([1 + NUM_SPEC] * 4, [1, 2, 3, 4]),                      # MIXED accept counts
        ([1, 1, 1, 1], [1, 1, 1, 1]),                            # degenerate: single-token rows
        ([1 + NUM_SPEC, 1, 2, 3], [2, 1, 2, 3]),                 # mixed lengths AND counts
    ],
)
def test_spec_matches_vllm_triton_conv(dtype, layout, dim, query_lens, accepted):
    """The rollback path against vLLM's own kernel, on activations AND the whole ring buffer.

    This is the oracle that matters for the target under spec: a wrong tap offset is silent
    recurrent-state corruption that only shows up as a degraded long-context answer.
    """
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    reqs = len(query_lens)
    bcx, state, weight, state_idx, qsl, num_accepted = build_spec(
        reqs, query_lens, dim, dtype, layout=layout, accepted=accepted
    )
    bias = torch.randn(dim, dtype=dtype, device=bcx.device)
    B, C, x = bcx[:, :dim], bcx[:, dim : 2 * dim], bcx[:, 2 * dim :]

    stock_state = state.clone()
    Bx = (B * x).contiguous()
    conv_out = causal_conv1d_update(
        Bx,
        stock_state,
        weight,
        bias,
        activation=None,
        conv_state_indices=state_idx,
        num_accepted_tokens=num_accepted,
        query_start_loc=qsl,
        max_query_len=state_idx.size(-1),
    )
    stock_y = C * conv_out

    got_out, got_scale = run(
        bcx, state, weight, bias, state_idx, num_accepted=num_accepted, qsl=qsl
    )

    torch.testing.assert_close(state.float(), stock_state.float(), rtol=0.0, atol=0.0)
    amax = stock_y.float().abs().amax(-1, keepdim=True)
    exp_scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    exp_out = torch.clamp(stock_y.float() / exp_scale, -FP8_MAX, FP8_MAX).to(FP8)
    assert_fp8_equivalent(got_out, exp_out, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0.0)


def test_spec_zero_accepted_is_clamped():
    """num_accepted == 0 must not read s[-1].

    A DELIBERATE divergence from Triton, which computes `off = num_accepted - 1` and would index
    one slot before the ring. It cannot occur in practice (the bonus token is always accepted, so
    num_accepted >= 1), but the kernel clamps rather than reading out of bounds and the clamp
    needs a test or it is just an untested claim in a comment. Asserting it equals the
    num_accepted == 1 result pins the clamp to `off = 0` specifically.
    """
    dtype, dim, reqs = torch.bfloat16, 2048, 3
    lens = [1 + NUM_SPEC] * reqs

    bcx0, state0, w0, idx0, qsl0, _ = build_spec(reqs, lens, dim, dtype, seed=31)
    na_zero = torch.zeros(reqs, dtype=torch.int32, device=bcx0.device)
    out0, scale0 = run(bcx0, state0, w0, None, idx0, num_accepted=na_zero, qsl=qsl0)

    bcx1, state1, w1, idx1, qsl1, _ = build_spec(reqs, lens, dim, dtype, seed=31)
    na_one = torch.ones(reqs, dtype=torch.int32, device=bcx1.device)
    out1, scale1 = run(bcx1, state1, w1, None, idx1, num_accepted=na_one, qsl=qsl1)

    torch.testing.assert_close(state0.float(), state1.float(), rtol=0.0, atol=0.0)
    assert torch.equal(out0.view(torch.uint8), out1.view(torch.uint8))
    torch.testing.assert_close(scale0, scale1, rtol=0.0, atol=0.0)


def test_spec_null_blocks_leave_state_untouched():
    """Cudagraph padding under spec: NULL_BLOCK_ID rows must not touch any ring buffer.

    The padded requests also carry a degenerate query range (query_start_loc repeats the final
    total past num_reqs), which is what keeps the zero-fill inside the decode slice.
    """
    dtype, dim, reqs, nulls = torch.bfloat16, 2048, 6, 2
    live = reqs - nulls
    query_lens = [1 + NUM_SPEC] * live + [0] * nulls
    bcx, state, weight, state_idx, qsl, num_accepted = build_spec(
        reqs, query_lens, dim, dtype, accepted=[2] * reqs, nulls=nulls, seed=23
    )
    before = state.clone()
    live_coords = state_idx[:live, 0].long()

    run(bcx, state, weight, None, state_idx, num_accepted=num_accepted, qsl=qsl)

    untouched = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    untouched[live_coords] = False
    torch.testing.assert_close(
        state[untouched].float(), before[untouched].float(), rtol=0.0, atol=0.0
    )


def test_spec_requires_both_rollback_args():
    """num_accepted_tokens and query_start_loc are meaningless apart; the op must reject one."""
    dtype, dim, reqs = torch.bfloat16, 2048, 2
    bcx, state, weight, state_idx, qsl, num_accepted = build_spec(
        reqs, [1 + NUM_SPEC] * reqs, dim, dtype, seed=29
    )
    with pytest.raises(Exception):
        run(bcx, state, weight, None, state_idx, num_accepted=num_accepted, qsl=None)
    with pytest.raises(Exception):
        run(bcx, state, weight, None, state_idx, num_accepted=None, qsl=qsl)


def test_supported_predicate():
    """The shape gate short_conv.py queries instead of re-hardcoding kVec / kFastMaxThreads."""
    supported = torch.ops.vllm_cuda.bcx_conv_gate_supported
    assert supported(2048, WIDTH, TAPS)  # LFM2.5, no spec
    assert supported(1024, WIDTH, TAPS)  # LFM2.5-350M drafter
    # Spec decode widens the ALLOCATION; the kernel still reads only the first TAPS slots, so
    # anything at or above that is accepted. An `==` here is what disabled the kernel outright
    # once num_speculative_tokens went non-zero -- on the target as well as the drafter.
    assert supported(2048, WIDTH, TAPS + 3)
    assert supported(1024, WIDTH, TAPS + 3)
    assert not supported(2048, WIDTH, TAPS - 1)  # too narrow to hold the taps
    assert not supported(2048, 4, TAPS)  # wrong conv width
    assert not supported(2050, WIDTH, TAPS)  # not a multiple of the vector width
    assert not supported(1 << 20, WIDTH, TAPS)  # past the one-block reduction cap


if __name__ == "__main__":
    if not HAVE_OP:
        raise SystemExit("vtl._C bcx_conv_gate_quant unavailable; nothing to check")
    for layout in ("SD", "DS"):
        for dtype in (torch.bfloat16, torch.float16):
            for state_len in STATE_LENS:
                for T in (1, 7, 64, 256):
                    for with_bias in (False, True):
                        test_matches_reference(dtype, T, layout, with_bias, state_len)
                for dim in (2048, 1024):
                    test_matches_vllm_triton_conv(dtype, layout, dim, state_len)
        test_null_blocks_leave_state_untouched(layout)
    test_scale_ub_is_honored()
    test_supported_predicate()
    for layout in ("SD", "DS"):
        for dtype in (torch.bfloat16, torch.float16):
            for dim in (2048, 1024):
                for query_lens, accepted in (
                    ([1 + NUM_SPEC] * 4, [1, 1, 1, 1]),
                    ([1 + NUM_SPEC] * 4, [1 + NUM_SPEC] * 4),
                    ([1 + NUM_SPEC] * 4, [1, 2, 3, 4]),
                    ([1, 1, 1, 1], [1, 1, 1, 1]),
                    ([1 + NUM_SPEC, 1, 2, 3], [2, 1, 2, 3]),
                ):
                    test_spec_matches_vllm_triton_conv(
                        dtype, layout, dim, query_lens, accepted
                    )
    test_spec_zero_accepted_is_clamped()
    test_spec_null_blocks_leave_state_untouched()
    print("bcx_conv_gate_quant checks ok")
