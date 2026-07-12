"""Parity for vtl's GDN causal conv1d kernels (vllm_cuda::gdn_causal_conv1d_update / _fn) against
a pure-torch oracle of the width-4 causal depthwise conv (see vtl/csrc/gdn/causal_conv1d.cu).

Validates the kernel math + state handling. The definitive check that it matches vLLM v0.25.0's
causal_conv1d is the on-box A/B; enable VTL_GDN_CONV1D only after that + a bench win.

    pytest bench/test_gdn_causal_conv1d.py -q
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

W = 4  # qwen3_5 conv kernel width


def _import_ops():
    import vllm._C_stable_libtorch  # noqa: F401
    import vtl._C  # noqa: F401


_import_ops()


def _act(y, silu):
    return y * torch.sigmoid(y) if silu else y


# ---- decode update ---------------------------------------------------------------------------
def ref_update(x, conv_state, weight, bias, silu):
    """x [B,D] -> y [B,D]; conv_state [B,D,W-1] shifted. fp32."""
    xf = x.float()
    st = conv_state.float()
    wf = weight.float()
    acc = torch.zeros_like(xf)
    if bias is not None:
        acc += bias.float().unsqueeze(0)
    for k in range(W - 1):
        acc += wf[:, k].unsqueeze(0) * st[:, :, k]
    acc += wf[:, W - 1].unsqueeze(0) * xf
    y = _act(acc, silu)
    new_state = torch.stack([st[:, :, 1], st[:, :, 2], xf], dim=-1)
    return y, new_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("silu", [True, False])
def test_update(dtype, B, silu):
    torch.manual_seed(0)
    D = 6144
    x = torch.randn(B, D, dtype=dtype, device="cuda")
    conv_state = torch.randn(B, D, W - 1, dtype=dtype, device="cuda")
    weight = torch.randn(D, W, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")

    x_run, st_run = x.clone(), conv_state.clone()
    torch.ops.vllm_cuda.gdn_causal_conv1d_update(x_run, st_run, weight, bias, silu)
    exp_y, exp_st = ref_update(x, conv_state, weight, bias, silu)

    tol = dict(rtol=2e-2, atol=2e-2) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x_run.float(), exp_y, **tol)
    torch.testing.assert_close(st_run.float(), exp_st, **tol)


# ---- prefill fn ------------------------------------------------------------------------------
def ref_fn(x, weight, bias, qsl, init_state, has_init, silu):
    """x [D,L] -> y [D,L]; final_state [S,D,W-1]. fp32, per-sequence causal conv."""
    D, L = x.shape
    S = qsl.numel() - 1
    xf = x.float()
    wf = weight.float()
    y = torch.zeros_like(xf)
    final = torch.zeros(S, D, W - 1, dtype=torch.float32, device=x.device)
    for s in range(S):
        start, end = int(qsl[s]), int(qsl[s + 1])
        taps = (
            init_state[s].float()
            if (has_init is not None and init_state is not None and bool(has_init[s]))
            else torch.zeros(D, W - 1, dtype=torch.float32, device=x.device)
        )
        if start >= end:
            final[s] = taps  # empty sequence -> conv state passes through unchanged
            continue
        # padded[:, :W-1] = taps (oldest..newest), then the sequence inputs
        seq = xf[:, start:end]  # [D, n]
        padded = torch.cat([taps, seq], dim=1)  # [D, W-1+n]
        acc = bias.float().unsqueeze(1).expand(D, end - start).clone() if bias is not None \
            else torch.zeros(D, end - start, dtype=torch.float32, device=x.device)
        for k in range(W):
            acc += wf[:, k].unsqueeze(1) * padded[:, k : k + (end - start)]
        y[:, start:end] = _act(acc, silu)
        final[s] = padded[:, -(W - 1):]
    return y, final


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seqlens", [[5], [1, 16, 7], [64, 64], [0, 8]])  # [0,8]: empty leading seq
@pytest.mark.parametrize("with_init", [False, True])
@pytest.mark.parametrize("silu", [True, False])
def test_fn(dtype, seqlens, with_init, silu):
    torch.manual_seed(0)
    D = 6144
    L = sum(seqlens)
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    x = torch.randn(D, L, dtype=dtype, device="cuda")
    weight = torch.randn(D, W, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")
    init = torch.randn(len(seqlens), D, W - 1, dtype=dtype, device="cuda") if with_init else None
    has_init = (
        torch.ones(len(seqlens), dtype=torch.bool, device="cuda") if with_init else None
    )

    y = torch.empty_like(x)
    final = torch.empty(len(seqlens), D, W - 1, dtype=dtype, device="cuda")
    torch.ops.vllm_cuda.gdn_causal_conv1d_fn(
        y, x, weight, bias, qsl, init, has_init, final, silu
    )
    exp_y, exp_final = ref_fn(x, weight, bias, qsl, init, has_init, silu)

    tol = dict(rtol=2e-2, atol=2e-2) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y.float(), exp_y, **tol)
    torch.testing.assert_close(final.float(), exp_final, **tol)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
