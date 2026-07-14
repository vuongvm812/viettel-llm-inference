"""Correctness + micro-benchmark for vtl's GDN per-head gated RMSNorm and the fused
norm+per-token-fp8-quant op (vllm_cuda::gated_rmsnorm[_dynamic_per_token_quant]).

These are vtl-only ops (no stock _C counterpart), so there is no VTL_SKIP_EXT dual mode -- they
are skipped entirely under stock. The reference encodes vLLM RMSNormGated's ACTUAL config for
qwen3_5: norm_before_gate=True, group_size=None, activation in {silu, sigmoid} =>
`out = rms_norm(x) * act(z)`. `test_matches_vllm_rmsnormgated` cross-checks against vLLM's own
RMSNormGated when importable -- the real gate before enabling gdn_kernels.py (default off).

    pytest bench/test_gdn_gated_rmsnorm.py -q
    python bench/test_gdn_gated_rmsnorm.py       # benchmark
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")
if USE_STOCK:
    pytest.skip("gated_rmsnorm is a vtl-only op (no stock kernel)", allow_module_level=True)

import vtl._C  # noqa: E402,F401  -- registers vllm_cuda::gated_rmsnorm(+_quant)

OP = torch.ops.vllm_cuda.gated_rmsnorm
QOP = torch.ops.vllm_cuda.gated_rmsnorm_dynamic_per_token_quant
EPS = 1e-6
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def _act(zf, is_silu):
    return zf * torch.sigmoid(zf) if is_silu else torch.sigmoid(zf)


def reference(x, gate, weight, is_silu=True, eps=EPS):
    """vLLM RMSNormGated, norm_before_gate=True, group_size=None: rms_norm(x)*weight*act(gate)."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    out = xn * weight * _act(gate.float(), is_silu)
    return out.to(x.dtype)


def run(x, gate, weight, is_silu=True, eps=EPS):
    out = torch.empty_like(x)
    OP(out, x, gate, weight, float(eps), is_silu)
    return out


def reference_quant(x, gate, weight, H, is_silu=True, eps=EPS):
    """Per-head norm, reshape to [N, H*D], per-token dynamic fp8 quant (the out_proj input)."""
    normed = reference(x, gate, weight, is_silu, eps).float()  # [N*H, D]
    N = x.shape[0] // H
    merged = normed.reshape(N, H * weight.numel())
    amax = merged.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    return merged, scale


def run_quant(x, gate, weight, H, is_silu=True, eps=EPS):
    N, D = x.shape[0] // H, weight.numel()
    result = torch.empty((N, H * D), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((N, 1), device="cuda", dtype=torch.float32)
    QOP(result, scale, x, gate, weight, float(eps), H, is_silu, None)
    return result, scale


def _tol(dtype):
    return {torch.bfloat16: (2e-2, 2e-3), torch.float16: (2e-3, 2e-4), torch.float32: (1e-5, 1e-6)}[dtype]


@pytest.mark.parametrize("is_silu", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_rows", [1, 16, 256, 4096])
@pytest.mark.parametrize("D", [128, 256, 64, 100])  # 128 = qwen3_5 head_dim; 100 = odd width
def test_matches_reference(is_silu, dtype, num_rows, D):
    torch.manual_seed(0)
    x = torch.randn(num_rows, D, dtype=dtype, device="cuda")
    gate = torch.randn(num_rows, D, dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, device="cuda")
    got = run(x, gate, weight, is_silu)
    exp = reference(x, gate, weight, is_silu)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(got.float(), exp.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("is_silu", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 7, 256])
@pytest.mark.parametrize("H,D", [(16, 128), (32, 128), (4, 64)])  # qwen3_5: H*D fast path
def test_fused_quant_matches_reference(is_silu, dtype, num_tokens, H, D):
    torch.manual_seed(0)
    x = torch.randn(num_tokens * H, D, dtype=dtype, device="cuda")
    gate = torch.randn(num_tokens * H, D, dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, device="cuda")

    got_q, got_s = run_quant(x, gate, weight, H, is_silu)
    exp_merged, exp_s = reference_quant(x, gate, weight, H, is_silu)

    # Scale = amax(gated-norm output)/448. In bf16 the norm output's max element rounds slightly
    # differently between the kernel's warp reduction and the torch reference, and that rounding is
    # GPU-dependent (Ampere vs Hopper) -- so the scale can drift a few tenths of a percent. That is
    # < 1 ULP of the fp8 scale and harmless (the dequant check below is the real quality gate), so
    # tolerate bf16 reduction drift here instead of asserting fp32-tight equality.
    stol = dict(rtol=6e-3, atol=1e-3) if x.dtype == torch.bfloat16 else dict(rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(got_s, exp_s, **stol)
    # Dequantized product matches the pre-quant values within e4m3 rounding (3 mantissa bits).
    got_deq = got_q.float() * got_s
    torch.testing.assert_close(got_deq, exp_merged, rtol=8e-2, atol=1e-2 * exp_s.mean().item())


def test_matches_vllm_rmsnormgated():
    """The real gate: does our kernel match vLLM's own RMSNormGated (norm_before_gate=True)?
    Skips if the class is not importable / has a different signature -- in which case
    gdn_kernels.py must stay disabled until the wiring is confirmed on the H200."""
    try:
        from vllm.model_executor.layers.layernorm import RMSNormGated
    except Exception:
        pytest.skip("vLLM RMSNormGated not importable in this environment")

    D = 128
    torch.manual_seed(0)
    x = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    try:
        norm = RMSNormGated(
            hidden_size=D, eps=EPS, group_size=None, norm_before_gate=True, activation="silu"
        ).cuda()
        norm.weight.data = torch.randn(D, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            ref = norm.forward_native(x, gate) if hasattr(norm, "forward_native") else norm(x, gate)
    except Exception as exc:
        pytest.skip(f"RMSNormGated signature differs on this vLLM: {exc}")

    got = run(x, gate, norm.weight.data, is_silu=True)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-3)


def _bench():
    import time

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"\n{'op':>18} {'tokens x H x D':>18} {'us/call':>10}")
    H, D = 16, 128
    for tokens in (256, 8192):
        x = torch.randn(tokens * H, D, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(tokens * H, D, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        for name, fn in (("norm", lambda: run(x, gate, weight)),
                         ("norm+quant", lambda: run_quant(x, gate, weight, H))):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            iters = 200
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / iters * 1e6
            print(f"{name:>18} {f'{tokens}x{H}x{D}':>18} {us:>10.1f}")


if __name__ == "__main__":
    _bench()
