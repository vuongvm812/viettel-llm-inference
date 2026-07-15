"""Correctness + micro-benchmark for vtl's MTP-head fused-projection GEMM
(vllm_cuda::mtp_fc_gemm), the Stage-A cuBLASDx device GEMM behind VTL_ENABLE_MTP_KERNELS.

The op computes a plain Linear: result[M,N] = input[M,K] @ weight[N,K]^T, matching the MTP
head's `fc` (in=2H=4096, out=H=2048). It's a vtl-only op (no stock _C counterpart) and is
built only when vtl._C had cuBLASDx headers, so this test skips under stock / when the op is
absent. The reference casts the SAME bf16/fp16 operands to fp32 and accumulates in fp32 --
matching the kernel (bf16/fp16 smem, fp32 accumulate) -- so tolerances are tight.

    pytest bench/test_mtp_fc_gemm.py -q
    python bench/test_mtp_fc_gemm.py       # benchmark vs torch (cuBLAS)
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")
if USE_STOCK:
    pytest.skip("mtp_fc_gemm is a vtl-only op (no stock kernel)", allow_module_level=True)

import vtl._C  # noqa: E402,F401  -- registers vllm_cuda::mtp_fc_gemm (if built with cuBLASDx)

if not hasattr(torch.ops.vllm_cuda, "mtp_fc_gemm"):
    pytest.skip("vtl._C built without cuBLASDx (no nvidia-mathdx)", allow_module_level=True)

OP = torch.ops.vllm_cuda.mtp_fc_gemm


def reference(x, w):
    """Linear: x @ w^T, in fp32 over the exact (possibly-narrowed) operand values."""
    return (x.float() @ w.float().t()).to(x.dtype)


def run(x, w):
    out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device="cuda")
    OP(out, x, w)
    return out


def _tol(dtype):
    # K=4096 accumulation in fp32; operands narrowed to the storage dtype on both sides.
    return {
        torch.bfloat16: (2e-2, 2e-2),
        torch.float16: (4e-3, 4e-3),
        torch.float32: (1e-4, 1e-4),
    }[dtype]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
# M crosses the 64-row M-tile (48 fits, 96 spills to a 2nd tile); N/K exercise the mask edges.
@pytest.mark.parametrize("M", [1, 16, 48, 64, 96])
@pytest.mark.parametrize("N,K", [(2048, 4096), (128, 256), (100, 72)])  # qwen3_5 fc + masked tiles
def test_matches_reference(dtype, M, N, K):
    torch.manual_seed(0)
    scale = 1.0 / (K**0.5)  # keep the fp32 accumulation in a sane range
    x = (torch.randn(M, K, dtype=dtype, device="cuda") * scale)
    w = (torch.randn(N, K, dtype=dtype, device="cuda") * scale)
    got = run(x, w)
    exp = reference(x, w)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(got.float(), exp.float(), rtol=rtol, atol=atol)


def test_contiguous_and_shape_guards():
    """The host wrapper must reject bad shapes/strides rather than fault the GPU."""
    x = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    # K mismatch
    with pytest.raises(RuntimeError):
        OP(torch.empty(8, 2048, dtype=torch.bfloat16, device="cuda"), x, w[:, :2048])
    # dtype mismatch
    with pytest.raises(RuntimeError):
        OP(
            torch.empty(8, 2048, dtype=torch.bfloat16, device="cuda"),
            x.float(),
            w,
        )


def _bench():
    import time

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"\n{'op':>14} {'M x N x K':>18} {'us/call':>10}")
    N, K = 2048, 4096  # qwen3_5 MTP fc
    for M in (16, 48):
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
        for name, fn in (("vtl", lambda: run(x, w)), ("torch", lambda: x @ w.t())):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            iters = 200
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / iters * 1e6
            print(f"{name:>14} {f'{M}x{N}x{K}':>18} {us:>10.1f}")


if __name__ == "__main__":
    _bench()
