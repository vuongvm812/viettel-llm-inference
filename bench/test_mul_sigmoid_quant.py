"""Correctness + micro-benchmark for vtl's fused gate-mul + per-token fp8 quant (o_proj).

Our kernel is one fused op; stock is three ops (sigmoid -> mul -> dynamic_per_token quant). Both
are checked against the same pure-torch oracle, so passing under both proves the fused kernel
matches the stock composition it replaces.

    pytest bench/test_mul_sigmoid_quant.py -q                  # our fused kernel
    VTL_SKIP_EXT=1 pytest bench/test_mul_sigmoid_quant.py -q   # stock three-op path, same oracle
    python bench/test_mul_sigmoid_quant.py                     # benchmark
"""

from __future__ import annotations

import os

import pytest
import torch
from _fp8_testutil import FP8, FP8_MAX, MIN_SCALE, assert_fp8_equivalent, scale_rtol

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")


def _import_ops():
    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schemas
    if not USE_STOCK:
        import vtl._C  # noqa: F401  -- registers the fused vllm_cuda:: op


_import_ops()


def reference(attn, gate, dtype, scale_ub=None):
    """sigmoid(gate) narrowed to dtype, * attn (dtype), then per-token fp8 quant -- exactly the
    stock decomposed o_proj input path."""
    sig = torch.sigmoid(gate.float()).to(dtype)  # sigmoid fp32, narrow to dtype
    g = (attn * sig).float()  # dtype * dtype -> dtype, then widen for the reduction
    amax = g.abs().amax(-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    out = torch.clamp(g / scale, -FP8_MAX, FP8_MAX).to(FP8)
    return out, scale


def run(attn, gate, scale_ub=None):
    T, H = attn.shape
    out = torch.empty((T, H), dtype=FP8, device=attn.device)
    scale = torch.empty((T, 1), dtype=torch.float32, device=attn.device)
    if USE_STOCK:
        g = (attn * torch.sigmoid(gate)).contiguous()
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, g, scale, scale_ub)
    else:
        torch.ops.vllm_cuda.mul_sigmoid_dynamic_per_token_quant(out, scale, attn, gate, scale_ub)
    return out, scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1000, 8192])
@pytest.mark.parametrize(
    "H",
    [
        2048,  # qwen3_5 o_proj input width -- fast path (256 threads, single slice/thread)
        2050,  # not a multiple of the vector width -> generic fallback
    ],
)
def test_matches_reference(dtype, num_tokens, H):
    if num_tokens * H * 2 * 4 > 0.5 * torch.cuda.mem_get_info()[0]:
        pytest.skip("too big for this GPU")
    torch.manual_seed(0)
    attn = torch.randn(num_tokens, H, dtype=dtype, device="cuda")
    gate = torch.randn(num_tokens, H, dtype=dtype, device="cuda")
    got_out, got_scale = run(attn, gate)
    exp_out, exp_scale = reference(attn, gate, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0)
    assert_fp8_equivalent(got_out, exp_out, dtype)


def test_zeros_input_hits_min_scale():
    """amax==0 (attn all-zero -> product all-zero) is the divide-by-scale money path: scale must
    floor at MIN_SCALE (not 0) and the output must be all-zero, not NaN."""
    attn = torch.zeros(8, 2048, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(8, 2048, dtype=torch.bfloat16, device="cuda")
    out, scale = run(attn, gate)
    assert torch.isfinite(scale).all() and (scale == MIN_SCALE).all()
    assert (out.float() == 0).all()


def test_scale_ub_is_honoured():
    torch.manual_seed(0)
    attn = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    ub = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    _, got_scale = run(attn, gate, ub)
    assert (got_scale <= 0.5 / FP8_MAX + 1e-9).all()


@pytest.mark.skipif(USE_STOCK, reason="the fused op is a vtl-only construct")
def test_gate_is_the_sigmoid_operand():
    """`gate` is the sigmoid'd operand, `input` is not. With attn=1 and gate very negative,
    sigmoid(gate)~=0 -> output ~=0. A kernel that sigmoids the wrong arg would produce
    sigmoid(1)*(-30) -> clamped to -448, clearly nonzero."""
    attn = torch.ones(4, 2048, dtype=torch.bfloat16, device="cuda")
    gate = torch.full((4, 2048), -30.0, dtype=torch.bfloat16, device="cuda")
    out, _ = run(attn, gate)
    assert (out.float().abs() < 1.0).all(), "gate is not the sigmoid'd operand"


def _bench():
    import time

    label = "stock (sigmoid+mul+quant)" if USE_STOCK else "vtl fused"
    print(f"impl: {label}   device: {torch.cuda.get_device_name()}")
    print(f"\n{'shape':>16} {'us/call':>10} {'GB/s':>8}")
    for num_tokens in (256, 8192):
        H = 2048
        attn = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")
        for _ in range(20):
            run(attn, gate)
        torch.cuda.synchronize()
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            run(attn, gate)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        # fused reads attn+gate (2*H*2B), writes H (1B). stock also writes+reads the H bf16 product.
        moved = num_tokens * (2 * H * 2 + H * 1)
        print(f"{num_tokens:>7}x{H:<8} {us:>10.1f} {moved / us / 1e3:>8.0f}")


if __name__ == "__main__":
    _bench()
