"""Correctness + micro-benchmark for vtl's fused SiLU-mul + per-token fp8 quant (down_proj).

Our kernel is one fused op; stock is two ops (silu_and_mul -> dynamic_per_token quant). Both
are checked against the same pure-torch oracle, so passing under both proves the fused kernel
matches the stock composition it replaces.

    pytest bench/test_silu_mul_quant.py -q                  # our fused kernel
    VTL_SKIP_EXT=1 pytest bench/test_silu_mul_quant.py -q   # stock two-op path, same oracle
    python bench/test_silu_mul_quant.py                     # benchmark
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


def reference(gu, dtype, scale_ub=None):
    """silu(gate) narrowed to dtype, * up (dtype), then per-token fp8 quant -- exactly the
    stock unfused path."""
    d = gu.shape[-1] // 2
    gate = gu[..., :d].float()
    up = gu[..., d:]
    silu = (gate / (1.0 + torch.exp(-gate))).to(dtype)  # sigmoid fp32, narrow to dtype
    g = (silu * up).float()  # dtype * dtype -> dtype, then widen for the reduction
    amax = g.abs().amax(-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    out = torch.clamp(g / scale, -FP8_MAX, FP8_MAX).to(FP8)
    return out, scale


def run(gu, scale_ub=None):
    T = gu.shape[0]
    I = gu.shape[-1] // 2
    out = torch.empty((T, I), dtype=FP8, device=gu.device)
    scale = torch.empty((T, 1), dtype=torch.float32, device=gu.device)
    if USE_STOCK:
        g = torch.empty((T, I), dtype=gu.dtype, device=gu.device)
        torch.ops._C.silu_and_mul(g, gu)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, g, scale, scale_ub)
    else:
        torch.ops.vllm_cuda.silu_and_mul_dynamic_per_token_quant(out, scale, gu, scale_ub)
    return out, scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1000, 8192])
@pytest.mark.parametrize(
    "inter",
    [
        6144,  # qwen3_5 intermediate -- fast path (768 threads, single slice/thread)
        2048,  # fast path
        2050,  # not a multiple of the vector width -> generic fallback
    ],
)
def test_matches_reference(dtype, num_tokens, inter):
    if num_tokens * inter * 2 * 4 > 0.5 * torch.cuda.mem_get_info()[0]:
        pytest.skip("too big for this GPU")
    torch.manual_seed(0)
    gu = torch.randn(num_tokens, 2 * inter, dtype=dtype, device="cuda")
    got_out, got_scale = run(gu)
    exp_out, exp_scale = reference(gu, dtype)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0)
    assert_fp8_equivalent(got_out, exp_out, dtype)


def test_scale_ub_is_honoured():
    torch.manual_seed(0)
    gu = torch.randn(64, 2 * 6144, dtype=torch.bfloat16, device="cuda")
    ub = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    _, got_scale = run(gu, ub)
    assert (got_scale <= 0.5 / FP8_MAX + 1e-9).all()


@pytest.mark.skipif(USE_STOCK, reason="the fused op is a vtl-only construct")
def test_gate_up_ordering():
    """gate is the FIRST half, up the SECOND (vLLM's SiluAndMul). A swapped-halves kernel
    fails this: with gate=0 -> silu(0)=0 -> all-zero output regardless of up."""
    gu = torch.zeros(4, 2 * 128, dtype=torch.bfloat16, device="cuda")
    gu[:, 128:] = 1.0  # up = 1, gate = 0
    out, _ = run(gu)
    assert (out.float() == 0).all(), "gate half is not the first half"


def _bench():
    import time

    label = "stock (silu+quant)" if USE_STOCK else "vtl fused"
    print(f"impl: {label}   device: {torch.cuda.get_device_name()}")
    print(f"\n{'shape':>16} {'us/call':>10} {'GB/s':>8}")
    for num_tokens in (256, 8192):
        inter = 6144
        gu = torch.randn(num_tokens, 2 * inter, dtype=torch.bfloat16, device="cuda")
        for _ in range(20):
            run(gu)
        torch.cuda.synchronize()
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            run(gu)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        # fused reads 2*inter (2B), writes inter (1B). stock also writes+reads inter bf16.
        moved = num_tokens * (2 * inter * 2 + inter * 1)
        print(f"{num_tokens:>7}x{inter:<8} {us:>10.1f} {moved / us / 1e3:>8.0f}")


if __name__ == "__main__":
    _bench()
