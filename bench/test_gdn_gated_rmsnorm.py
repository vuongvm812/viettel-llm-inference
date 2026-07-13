"""Correctness + micro-benchmark for vtl's GDN per-head gated RMSNorm (vllm_cuda::gated_rmsnorm).

Unlike the fp8 kernels this op has NO stock _C counterpart, so there is no VTL_SKIP_EXT dual
mode -- it is skipped entirely under stock. The reference encodes the ASSUMED standard Mamba2
RMSNormGated semantics; `test_matches_vllm_rmsnormgated` additionally cross-checks against
vLLM's own RMSNormGated when that class is importable, which is the real gate before enabling
the kernel in production (vtl/patches/gdn_kernels.py, default off).

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

import vtl._C  # noqa: E402,F401  -- registers vllm_cuda::gated_rmsnorm

OP = torch.ops.vllm_cuda.gated_rmsnorm
EPS = 1e-6


def reference(x, gate, weight, eps=EPS):
    """Assumed standard Mamba2 RMSNormGated: x * silu(gate), RMS-normed over the last dim,
    scaled by an fp32 weight, narrowed to the input dtype."""
    xf = x.float()
    zf = gate.float()
    h = xf * (zf * torch.sigmoid(zf))
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return (h * weight).to(x.dtype)


def run(x, gate, weight, eps=EPS):
    out = torch.empty_like(x)
    OP(out, x, gate, weight, float(eps))
    return out


def _tol(dtype):
    # bf16 output: ~1 ulp is 2**-7; give a few ulps of headroom for the fp32 reduction order.
    return {torch.bfloat16: (2e-2, 2e-3), torch.float16: (2e-3, 2e-4), torch.float32: (1e-5, 1e-6)}[dtype]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_rows", [1, 16, 256, 4096])
@pytest.mark.parametrize("D", [128, 256, 64, 100])  # 128 = qwen3_5 head_dim; 100 = odd width
def test_matches_reference(dtype, num_rows, D):
    torch.manual_seed(0)
    x = torch.randn(num_rows, D, dtype=dtype, device="cuda")
    gate = torch.randn(num_rows, D, dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, device="cuda")
    got = run(x, gate, weight)
    exp = reference(x, gate, weight)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(got.float(), exp.float(), rtol=rtol, atol=atol)


def test_matches_vllm_rmsnormgated():
    """The real gate: does our kernel match vLLM's own RMSNormGated? Skips if the class is not
    importable / has a different signature -- in which case gdn_kernels.py must stay disabled
    until the wiring is confirmed on the H200."""
    try:
        from vllm.model_executor.layers.layernorm import RMSNormGated
    except Exception:
        pytest.skip("vLLM RMSNormGated not importable in this environment")

    D = 128
    torch.manual_seed(0)
    x = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    try:
        norm = RMSNormGated(hidden_size=D, eps=EPS).cuda()
        norm.weight.data = torch.randn(D, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            ref = norm.forward_native(x, gate) if hasattr(norm, "forward_native") else norm(x, gate)
    except Exception as exc:
        pytest.skip(f"RMSNormGated signature differs on this vLLM: {exc}")

    got = run(x, gate, norm.weight.data)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-3)


def _bench():
    import time

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"\n{'rows x D':>16} {'us/call':>10}")
    D = 128
    for num_rows in (256 * 16, 8192 * 16):  # tokens * num_v_heads(16)
        x = torch.randn(num_rows, D, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(num_rows, D, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        for _ in range(20):
            run(x, gate, weight)
        torch.cuda.synchronize()
        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            run(x, gate, weight)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        print(f"{num_rows:>10}x{D:<4} {us:>10.1f}")


if __name__ == "__main__":
    _bench()
