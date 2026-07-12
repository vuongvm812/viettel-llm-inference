"""Correctness + micro-benchmark for vtl's standalone dynamic per-token fp8 quant
(o_proj / down_proj-quant-half), overriding _C::dynamic_per_token_scaled_fp8_quant.

    pytest bench/test_dynamic_per_token_quant.py -q                  # our kernel
    VTL_SKIP_EXT=1 pytest bench/test_dynamic_per_token_quant.py -q   # stock kernel, same oracle
    python bench/test_dynamic_per_token_quant.py                     # benchmark
"""

from __future__ import annotations

import os

import pytest
import torch
from _fp8_testutil import FP8, FP8_MAX, MIN_SCALE, assert_fp8_equivalent, scale_rtol

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")


def _op():
    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schema

    if USE_STOCK:
        return torch.ops._C.dynamic_per_token_scaled_fp8_quant

    import vtl._C  # noqa: F401  -- overrides _C's CUDA kernel; defines vllm_cuda::
    return torch.ops.vllm_cuda.dynamic_per_token_scaled_fp8_quant


OP = _op()


def reference(x_in, scale_ub=None):
    x = x_in.float()
    amax = x.abs().amax(-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    out = torch.clamp(x / scale, -FP8_MAX, FP8_MAX).to(FP8)
    return out, scale


def run(x_in, scale_ub=None):
    out = torch.empty(x_in.shape, dtype=FP8, device=x_in.device)
    scale = torch.empty((x_in.shape[0], 1), dtype=torch.float32, device=x_in.device)
    OP(out, x_in, scale, scale_ub)
    return out, scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1000, 8192])
@pytest.mark.parametrize(
    "hidden",
    [
        2048,  # o_proj -- fast path (256 threads x 8, no loop)
        6144,  # down_proj-quant-half -- 768 threads > 512 cap -> generic fallback
        2050,  # not a multiple of the vector width -> generic fallback
    ],
)
def test_matches_reference(dtype, num_tokens, hidden):
    if num_tokens * hidden * 8 > 0.5 * torch.cuda.mem_get_info()[0]:
        pytest.skip("too big for this GPU")
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    got_out, got_scale = run(x)
    exp_out, exp_scale = reference(x)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0)
    assert_fp8_equivalent(got_out, exp_out, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_row_stride_larger_than_hidden(dtype):
    """Strided input (a view): in_off (strided) must not be confused with row_off (packed)."""
    torch.manual_seed(0)
    x = torch.randn(64, 2056, dtype=dtype, device="cuda")[:, :2048]
    assert x.stride(0) == 2056 and x.stride(1) == 1
    got_out, got_scale = run(x)
    exp_out, exp_scale = reference(x)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0)
    assert_fp8_equivalent(got_out, exp_out, dtype)


@pytest.mark.skipif(
    USE_STOCK,
    reason="the generic fallback is a vtl-only property; stock has no alignment guard",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_misaligned_input_falls_back_to_generic(dtype):
    torch.manual_seed(0)
    flat = torch.randn(4 * 2048 + 1, dtype=dtype, device="cuda")
    x = flat[1:].view(4, 2048)
    assert x.data_ptr() % 16 != 0
    got_out, got_scale = run(x)
    exp_out, exp_scale = reference(x)
    torch.testing.assert_close(got_scale, exp_scale, rtol=scale_rtol(dtype), atol=0)
    assert_fp8_equivalent(got_out, exp_out, dtype)


def test_scale_ub_is_honoured():
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    ub = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    _, got_scale = run(x, ub)
    assert (got_scale <= 0.5 / FP8_MAX + 1e-9).all()


def test_tiny_input_clamps_to_min_scale():
    x = torch.zeros(4, 2048, dtype=torch.bfloat16, device="cuda")
    _, scale = run(x)
    torch.testing.assert_close(scale, torch.full_like(scale, MIN_SCALE), rtol=1e-6, atol=0)


@pytest.mark.skipif(USE_STOCK, reason="asserts the override, not the kernel")
def test_override_is_installed():
    """Our kernel rejects int8 output; stock accepts it. Cheapest probe that _C's CUDA kernel
    is really ours."""
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(4, 2048, dtype=torch.int8, device="cuda")
    scale = torch.empty((4, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="fp8_e4m3 output only"):
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)


def _bench():
    import time

    label = "stock _C" if USE_STOCK else "vtl"
    print(f"impl: {label}   device: {torch.cuda.get_device_name()}")
    print(f"\n{'shape':>16} {'us/call':>10} {'GB/s':>8}")
    for num_tokens in (256, 8192):
        for hidden in (2048, 6144):
            x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(x, dtype=FP8)
            scale = torch.empty((num_tokens, 1), dtype=torch.float32, device="cuda")
            for _ in range(20):
                OP(out, x, scale, None)
            torch.cuda.synchronize()
            iters = 200
            t0 = time.perf_counter()
            for _ in range(iters):
                OP(out, x, scale, None)
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / iters * 1e6
            moved = num_tokens * hidden * 3  # read 2B, write 1B
            print(f"{num_tokens:>7}x{hidden:<8} {us:>10.1f} {moved / us / 1e3:>8.0f}")


if __name__ == "__main__":
    _bench()
