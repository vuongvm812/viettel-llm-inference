"""Register ``vllm_cuda::conv_align_fused`` -- the fused mamba-align pre-copy op.

The CUDA kernel lives in ``vtl/csrc/conv_align_fused.cu``; the CALL SITE is in the fork
(``vtl/vllm_patches/v0.25.0/mamba_align_precopy.patch`` on
``MambaHybridModelState.preprocess_state``), not here. This module exists only to make the
op reachable: importing ``vtl._C`` dlopens the extension whose static initializers register
the schema, and a fake/meta rule keeps the op traceable if it is ever reached under
``torch.compile``.

Why a fork patch and not a monkeypatch: ``preprocess_state`` runs two Triton kernels inline
inside one method, with no seam between them. Wrapping the method from a plugin would mean
re-implementing the fallback body in Python anyway, and the fallback has to stay
byte-identical to stock -- so it stays in stock's own file, with the fused path as an early
return in front of it.

The op replaces BOTH stock launches: ``preprocess_mamba_align_fused_kernel`` (state advance
+ the two scratch arrays) and ``ctx.run_fused_precopy``. Sizing honesty: <=0.05 ms/step, far
under the A/B noise floor -- acceptance is parity (``bench/test_precopy_conv_align.py``), not
a TPOT measurement.

Gates: ``VTL_ENABLE_CONV_ALIGN_FUSED=0`` skips the registration entirely (the fork's
``hasattr`` check then keeps the stock pair); ``VTL_V2_FUSED_CONV_ALIGN=0`` is the in-fork
kill-switch that leaves the op registered but unused. Either one is a complete revert.
"""

from __future__ import annotations

import logging

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl.conv_align")

_OP = "conv_align_fused"

_fake_registered = False


def _register_fake() -> None:
    """Meta rule: the op mutates ``state_idx``/``num_accepted`` in place and returns ().

    Idempotent -- registering a fake twice raises.
    """
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake(f"vllm_cuda::{_OP}")
    def _fake(state_idx, num_accepted, idx_mapping, num_computed_tokens, query_start_loc,
              block_table_ptrs, block_table_stride_req, state_base_addrs,
              state_block_strides, state_elem_sizes, state_inner_sizes, state_conv_widths,
              state_group_indices, num_reqs, mamba_block_size):  # noqa: ANN001
        return None

    _fake_registered = True


@register_patch("conv_align_fused", default=True)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- keep import order uniform
    import vtl._C  # noqa: F401  -- dlopen registers the CUDA kernel + the op schema

    if not hasattr(torch.ops.vllm_cuda, _OP):
        log.warning("vtl: conv_align_fused op not registered after importing vtl._C; disabled")
        return

    _register_fake()
    log.info("vtl: conv_align_fused op registered (call site is in mamba_hybrid.preprocess_state)")


def _self_check() -> None:
    """No torch: assert the wiring constants the fork patch keys off cannot drift."""
    assert _OP == "conv_align_fused", _OP

    from vtl.registry import PATCH_REGISTRY

    assert {p.name: p.default for p in PATCH_REGISTRY}.get("conv_align_fused") is True

    # The fork patch calls exactly these two ops; a rename here without a patch
    # regeneration would silently keep the stock two-launch path forever.
    import pathlib

    patch = (
        pathlib.Path(__file__).resolve().parents[1]
        / "vllm_patches" / "v0.25.0" / "mamba_align_precopy.patch"
    )
    text = patch.read_text(encoding="utf-8")
    for op in ("conv_align_fused", "conv_align_supported"):
        assert f"torch.ops.vllm_cuda.{op}" in text, f"fork patch does not call {op}"
    assert "VTL_V2_FUSED_CONV_ALIGN" in text, "fork patch lost its kill-switch"

    # ...and the kernel must actually define both, with the argument count the patch passes.
    src = (pathlib.Path(__file__).resolve().parents[1] / "csrc" / "conv_align_fused.cu")
    body = src.read_text(encoding="utf-8")
    assert "void conv_align_fused(" in body and "bool conv_align_supported(" in body

    print("conv_align_fused self-check ok")


if __name__ == "__main__":
    _self_check()
