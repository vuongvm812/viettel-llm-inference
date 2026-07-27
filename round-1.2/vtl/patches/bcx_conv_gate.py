"""Register the fused short-conv decode ops ``vllm_cuda::bcx_conv_gate_quant`` / ``_supported``.

Collapses the whole LFM2 short-conv DECODE block into one launch: ``B*x`` -> depthwise causal
conv1d + ring-buffer rotation -> ``C*`` gate -> dynamic per-token fp8 quant. Stock needs three
kernels there -- the elementwise ``(B_d * x_d).contiguous()``, the Triton ``causal_conv1d_update``,
and the out_proj input quant (which [[mul_quant]] already folded into the gate multiply) -- and
round-trips a bf16 ``[T, dim]`` intermediate through HBM twice on the way. The CUDA kernel lives in
``vtl/csrc/bcx_conv_gate_quant.cu``; the exact Triton semantics it replicates are documented there.

Like mul_quant this is NOT wired by an FX fusion pattern: the whole ShortConv forward runs inside
the opaque ``torch.ops.vllm.short_conv`` custom op, so none of it is visible to the graph passes.
The patched ``ShortConv.forward_cuda`` calls the op DIRECTLY on the decode half of the batch and
writes into the shared fp8 staging buffer that ``out_proj``'s ``cutlass_scaled_mm`` consumes (see
the ``_vtl_conv_gate_*`` helpers in ``short_conv.py``). That call site is gated on
``VTL_ENABLE_BCX_CONV_GATE`` and fails closed to the stock ``causal_conv1d_update`` path.

DEPENDS ON [[mul_quant]]: the staging buffer only exists when ``VTL_ENABLE_MUL_QUANT`` is on, so
this patch is inert without it. Also needs the conv projections in fp8 ([[shortconv_quant]]),
otherwise ``out_proj`` is bf16 and there is nothing to feed pre-quantized.

SCOPE: decode only, conv width 3, reading the first ``width - 1 = 2`` slots of a state that may be
ALLOCATED wider -- i.e. exactly LFM2.5 (``conv_L_cache = 3``). Under spec decode the state is
allocated with ``conv_kernel - 1 + num_spec`` slots, but ``causal_conv1d_update`` pins
``state_len = width - 1`` whenever ``num_accepted_tokens`` is None (causal_conv1d.py:1181-1184), so
the extra slots are inert on this path and the kernel is correct over them unchanged. That is what
lets the DRAFT model use this kernel: its decode passes are single-token with no ``num_accepted``.

The TARGET under chain spec is also served, by the ``kSpec`` instantiation: one block per REQUEST
instead of per token, looping that request's 1..1+num_spec query tokens and reading the conv taps
at ``num_accepted - 1`` so rejected drafts are rolled back rather than committed. That path takes
``num_accepted_tokens``/``query_start_loc`` and is gated separately on ``VTL_BCX_SPEC``.

Prefill keeps ``causal_conv1d_fn`` and the tree-spec staging seam (``_VTL_CONV_STAGE``) keeps the
stock path. ``bcx_conv_gate_supported`` is the C++-side shape predicate the Python gate queries, so
the vector width and block-size caps are never re-hardcoded here.
"""

from __future__ import annotations

import logging

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl")

_OP = "bcx_conv_gate_quant"
_SUPPORTED_OP = "bcx_conv_gate_supported"
_ARGMAX_OP = "draft_argmax"

_fake_registered = False


def _register_fake() -> None:
    """FakeTensor/meta rule so torch.compile can trace an inserted node. The op mutates
    ``y_fp8``/``y_scale``/``conv_state`` in place and returns (); the fake only has to run
    without touching data. Idempotent -- registering a fake twice raises."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake(f"vllm_cuda::{_OP}")
    def _fake(  # noqa: ANN001
        y_fp8,
        y_scale,
        conv_state,
        bcx,
        conv_weight,
        conv_bias,
        state_indices,
        null_block_id,
        scale_ub=None,
        num_accepted_tokens=None,
        query_start_loc=None,
    ):
        return None

    # Same for the fused draft GEMV+argmax (vtl/csrc/draft_argmax.cu). It runs eagerly from
    # `_greedy_sample` today, so this is currently unreachable -- but an op with no meta rule
    # fails at TRACE time, i.e. at boot, if it is ever pulled into a compiled region, and that
    # is a much worse way to find out than a fake that costs nothing.
    @torch.library.register_fake(f"vllm_cuda::{_ARGMAX_OP}")
    def _fake_argmax(  # noqa: ANN001
        out_ids,
        hidden,
        qweight,
        group_scale,
        part_val,
        part_idx,
    ):
        return None

    _fake_registered = True


@register_patch("bcx_conv_gate", default=True)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- keep import order uniform with the other patches
    import vtl._C  # noqa: F401  -- dlopen registers the CUDA kernel + the vllm_cuda op schemas

    missing = [
        op
        for op in (_OP, _SUPPORTED_OP, _ARGMAX_OP)
        if not hasattr(torch.ops.vllm_cuda, op)
    ]
    if missing:
        log.warning("vtl: bcx_conv_gate op(s) %s not registered after importing vtl._C; disabled",
                    ", ".join(missing))
        return

    _register_fake()
    log.info("vtl: bcx_conv_gate op registered (fused short-conv decode; wired in short_conv)")


def _self_check() -> None:
    """No vLLM/torch: assert the wiring constants short_conv keys off."""
    assert _OP == "bcx_conv_gate_quant", _OP
    assert _SUPPORTED_OP == "bcx_conv_gate_supported", _SUPPORTED_OP
    assert _ARGMAX_OP == "draft_argmax", _ARGMAX_OP

    from vtl.registry import PATCH_REGISTRY  # local import so the file loads without torch

    # Registered ON by default; VTL_ENABLE_BCX_CONV_GATE=0 is the revert. The compose files still
    # set it explicitly, so the default only governs an unset environment.
    # (`make check` runs each patch file standalone, so only this module's entry is present.)
    names = {p.name: p.default for p in PATCH_REGISTRY}
    assert names.get("bcx_conv_gate") is True, names

    print("bcx_conv_gate self-check ok")


if __name__ == "__main__":
    _self_check()
