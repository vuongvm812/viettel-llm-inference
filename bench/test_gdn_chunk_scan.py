"""Parity + micro-bench for vtl's GDN prefill scan (vllm_cuda::gdn_chunk_scan) against a pure-torch
oracle of the SAME sequential gated-delta-rule recurrence (see vtl/csrc/gdn/chunk_scan.cu).

The kernel body is a sequential recurrence (correctness-first), NOT the chunk-parallel WY algorithm
-- so this proves numerics, and the on-box A/B vs vLLM's chunk_gated_delta_rule is the definitive
check + the perf gate (it will not beat the chunked incumbent on long prefills). Enable
VTL_GDN_CHUNK_SCAN only where the on-box bench wins. This is a vtl-only op (no stock _C
counterpart), so it is skipped entirely under VTL_SKIP_EXT.

    pytest bench/test_gdn_chunk_scan.py -q
    python bench/test_gdn_chunk_scan.py          # benchmark
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")
if USE_STOCK:
    pytest.skip("gdn_chunk_scan is a vtl-only op (no stock kernel)", allow_module_level=True)

import vllm._C_stable_libtorch  # noqa: E402,F401
import vtl._C  # noqa: E402,F401  -- registers vllm_cuda::gdn_chunk_scan

L2_EPS = 1e-6


def reference(q, k, v, g, beta, qsl, init_state, l2norm):
    """Sequential gated-delta recurrence per sequence, in fp32. Returns (o, final_state)."""
    L, H, D = q.shape
    S_count = qsl.numel() - 1
    o = torch.zeros(L, H, D, dtype=torch.float32, device=q.device)
    final = torch.zeros(S_count, H, D, D, dtype=torch.float32, device=q.device)
    qf, kf, vf = q.float(), k.float(), v.float()
    for s in range(S_count):
        start, end = int(qsl[s]), int(qsl[s + 1])
        St = (
            init_state[s].clone().float()
            if init_state is not None
            else torch.zeros(H, D, D, dtype=torch.float32, device=q.device)
        )
        for t in range(start, end):
            qt, kt, vt = qf[t], kf[t], vf[t]  # [H,D]
            if l2norm:
                qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + L2_EPS)
                kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + L2_EPS)
            St = St * torch.exp(g[t]).unsqueeze(-1).unsqueeze(-1)  # [H,1,1]
            kS = torch.einsum("hi,hij->hj", kt, St)
            vnew = vt - kS
            St = St + beta[t].unsqueeze(-1).unsqueeze(-1) * torch.einsum("hi,hj->hij", kt, vnew)
            o[t] = torch.einsum("hi,hij->hj", qt, St)
        final[s] = St
    return o, final


def run(q, k, v, g, beta, qsl, init_state, l2norm):
    L, H, D = q.shape
    S_count = qsl.numel() - 1
    o = torch.empty(L, H, D, dtype=q.dtype, device=q.device)
    final = torch.empty(S_count, H, D, D, dtype=torch.float32, device=q.device)
    torch.ops.vllm_cuda.gdn_chunk_scan(o, q, k, v, g, beta, qsl, init_state, final, l2norm)
    return o, final


def _make(seqlens, H, D, dtype, with_init):
    torch.manual_seed(0)
    L = sum(seqlens)
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32,
                       device="cuda")
    q = torch.randn(L, H, D, dtype=dtype, device="cuda")
    k = torch.randn(L, H, D, dtype=dtype, device="cuda")
    v = torch.randn(L, H, D, dtype=dtype, device="cuda")
    g = -torch.rand(L, H, dtype=torch.float32, device="cuda")  # log-decay <= 0
    beta = torch.rand(L, H, dtype=torch.float32, device="cuda")
    init = (
        torch.randn(len(seqlens), H, D, D, dtype=torch.float32, device="cuda") * 0.1
        if with_init else None
    )
    return q, k, v, g, beta, qsl, init


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlens", [[5], [1, 16, 7], [64, 64], [0, 8]])  # [0,8]: empty leading seq
@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize("l2norm", [True, False])
def test_matches_reference(dtype, seqlens, with_init, l2norm):
    H, D = 16, 128  # qwen3_5 GDN head count / head_dim
    q, k, v, g, beta, qsl, init = _make(seqlens, H, D, dtype, with_init)
    got_o, got_final = run(q, k, v, g, beta, qsl, init, l2norm)
    exp_o, exp_final = reference(q, k, v, g, beta, qsl, init, l2norm)
    otol = dict(rtol=3e-2, atol=3e-2) if dtype == torch.bfloat16 else dict(rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_o.float(), exp_o.float(), **otol)
    torch.testing.assert_close(got_final, exp_final, rtol=1e-3, atol=1e-3)


def _bench():
    import time

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"\n{'op':>14} {'seqlens':>18} {'us/call':>10}")
    H, D = 16, 128
    # Trace is prefill-bound (p50 ~18.7K prompt); one long seq is the real shape.
    for seqlens in ([256], [2048], [16, 16, 16, 16]):
        q, k, v, g, beta, qsl, init = _make(seqlens, H, D, torch.bfloat16, False)
        fn = lambda: run(q, k, v, g, beta, qsl, init, True)  # noqa: E731
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        print(f"{'gdn_chunk_scan':>14} {str(seqlens):>18} {us:>10.1f}")


if __name__ == "__main__":
    _bench()
