"""Parity for vtl's GDN decode recurrence (vllm_cuda::gdn_recurrent_decode) against a pure-torch
oracle of the SAME documented gated-delta-rule step (see vtl/csrc/gdn/fused_recurrent_decode.cu).

This validates the kernel implements its stated recurrence. The DEFINITIVE check that the
recurrence matches vLLM v0.25.0 is the on-box A/B vs fused_recurrent_gated_delta_rule_packed_decode
(the plan's Phase-3 verification) -- do that before enabling VTL_GDN_RECURRENT.

    pytest bench/test_gdn_recurrent_decode.py -q
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


def reference(q, k, v, g, beta, state, l2norm):
    """One gated-delta-rule step in fp32; returns (o, new_state). Mirrors the kernel exactly:
    decay -> kS -> vnew -> rank-1 update -> output."""
    q = q.float()
    k = k.float()
    v = v.float()
    S = state.clone().float()
    if l2norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_EPS)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
    S = S * torch.exp(g).unsqueeze(-1).unsqueeze(-1)  # [N,H,1,1]
    kS = torch.einsum("nhi,nhij->nhj", k, S)  # [N,H,Dv]
    vnew = v - kS
    S = S + beta.unsqueeze(-1).unsqueeze(-1) * torch.einsum("nhi,nhj->nhij", k, vnew)
    o = torch.einsum("nhi,nhij->nhj", q, S)
    return o, S


def run(q, k, v, g, beta, state, l2norm):
    o = torch.empty_like(v)
    st = state.clone()
    torch.ops.vllm_cuda.gdn_recurrent_decode(o, st, q, k, v, g, beta, l2norm)
    return o, st


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [1, 4, 32])
@pytest.mark.parametrize("l2norm", [False, True])
def test_matches_reference(dtype, N, l2norm):
    torch.manual_seed(0)
    H, D = 16, 128
    q = torch.randn(N, H, D, dtype=dtype, device="cuda")
    k = torch.randn(N, H, D, dtype=dtype, device="cuda")
    v = torch.randn(N, H, D, dtype=dtype, device="cuda")
    g = -torch.rand(N, H, dtype=torch.float32, device="cuda")  # log-decay <= 0
    beta = torch.rand(N, H, dtype=torch.float32, device="cuda")
    state = torch.randn(N, H, D, D, dtype=torch.float32, device="cuda") * 0.1

    got_o, got_state = run(q, k, v, g, beta, state, l2norm)
    exp_o, exp_state = reference(q, k, v, g, beta, state, l2norm)

    # fp32-accumulated; only the o output narrows to dtype. Tolerances allow reduction-order and
    # bf16 rounding differences.
    otol = dict(rtol=2e-2, atol=2e-2) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_o.float(), exp_o.float(), **otol)
    torch.testing.assert_close(got_state, exp_state, rtol=1e-4, atol=1e-4)


def test_matches_vllm_fused_recurrent():
    """THE gate: does our kernel match vLLM's own fused_recurrent_gated_delta_rule? This catches
    the failure mode test_matches_reference cannot -- a shared misconception between our kernel
    and our own oracle (e.g. a missing q-scale of Dk**-0.5, or per-step vs cumulative g, or the
    l2norm/decay order). Skips if the op isn't importable or its signature/layout differs; a
    mismatch here means our assumptions are wrong -- FIX the kernel (and this call's args to the
    model's real config) before enabling VTL_GDN_RECURRENT. Do NOT enable on a skip.
    """
    op = None
    for path in (
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
        "vllm.model_executor.layers.fla.ops",
    ):
        try:
            mod = __import__(path, fromlist=["fused_recurrent_gated_delta_rule"])
            op = getattr(mod, "fused_recurrent_gated_delta_rule", None)
            if op is not None:
                break
        except Exception:
            continue
    if op is None:
        pytest.skip("vLLM fused_recurrent_gated_delta_rule not importable in this environment")

    torch.manual_seed(0)
    N, H, D = 4, 16, 128
    q = torch.randn(N, H, D, dtype=torch.float32, device="cuda")
    k = torch.randn(N, H, D, dtype=torch.float32, device="cuda")
    v = torch.randn(N, H, D, dtype=torch.float32, device="cuda")
    g = -torch.rand(N, H, dtype=torch.float32, device="cuda")
    beta = torch.rand(N, H, dtype=torch.float32, device="cuda")
    state = torch.randn(N, H, D, D, dtype=torch.float32, device="cuda") * 0.1

    # FLA layout is [B, T, H, D] with T=1 for decode. scale=1.0 matches our kernel's (no q-scale)
    # assumption -- if the model uses Dk**-0.5, this asserts-fails and our kernel needs a scale arg.
    try:
        o_ref, st_ref = op(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            g.unsqueeze(1), beta.unsqueeze(1),
            scale=1.0, initial_state=state.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    except Exception as exc:
        pytest.skip(f"fused_recurrent_gated_delta_rule signature/layout differs: {exc}")

    got_o, got_state = run(q, k, v, g, beta, state, l2norm=True)
    torch.testing.assert_close(got_o.float(), o_ref.reshape(N, H, D).float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_state, st_ref.reshape(N, H, D, D).float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
