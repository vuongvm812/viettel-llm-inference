"""Parity for vtl's GDN prefill scan (vllm_cuda::gdn_chunk_scan) against a pure-torch oracle of
the SAME sequential gated-delta-rule recurrence (see vtl/csrc/gdn/chunk_gated_delta.cu).

The kernel is a sequential recurrence (correctness-first), NOT the chunk-parallel WY algorithm --
so this proves numerics, and the on-box A/B vs vLLM's chunk_gated_delta_rule is the definitive
check + the perf gate (it will not beat the chunked incumbent on long prefills). Enable
VTL_GDN_CHUNK_SCAN only where the on-box bench wins.

    pytest bench/test_gdn_chunk_scan.py -q
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L2_EPS = 1e-6


def _import_ops():
    import vllm._C_stable_libtorch  # noqa: F401
    import vtl._C  # noqa: F401


_import_ops()


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlens", [[5], [1, 16, 7], [64, 64], [0, 8]])  # [0,8]: empty leading seq
@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize("l2norm", [True, False])
def test_matches_reference(dtype, seqlens, with_init, l2norm):
    torch.manual_seed(0)
    H, D = 16, 128
    L = sum(seqlens)
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    q = torch.randn(L, H, D, dtype=dtype, device="cuda")
    k = torch.randn(L, H, D, dtype=dtype, device="cuda")
    v = torch.randn(L, H, D, dtype=dtype, device="cuda")
    g = -torch.rand(L, H, dtype=torch.float32, device="cuda")
    beta = torch.rand(L, H, dtype=torch.float32, device="cuda")
    init = (
        torch.randn(len(seqlens), H, D, D, dtype=torch.float32, device="cuda") * 0.1
        if with_init
        else None
    )

    got_o, got_final = run(q, k, v, g, beta, qsl, init, l2norm)
    exp_o, exp_final = reference(q, k, v, g, beta, qsl, init, l2norm)

    otol = dict(rtol=3e-2, atol=3e-2) if dtype == torch.bfloat16 else dict(rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_o.float(), exp_o.float(), **otol)
    torch.testing.assert_close(got_final, exp_final, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
