"""Register the fused conv-gate op ``vllm_cuda::mul_dynamic_per_token_quant``.

Fuses the LFM2 short-conv decode epilogue ``y = C * Bx`` (bf16 elementwise) with the fp8 quant
that ``out_proj`` would otherwise do on ``y`` -- reads C and Bx once, writes fp8 + per-token scale,
so the bf16 ``y`` intermediate never lands in HBM and one launch per conv layer is saved. The CUDA
kernel lives in ``vtl/csrc/mul_quant.cu``.

Unlike silu_mul_quant this is NOT wired by an FX fusion pattern: the whole ShortConv forward runs
inside the opaque ``torch.ops.vllm.short_conv`` custom op, so the gate multiply is invisible to the
graph fusion pass. Instead the patched ``ShortConv.forward_cuda`` calls this op DIRECTLY on the
pure-decode path and feeds ``out_proj`` the pre-quantized result via ``cutlass_scaled_mm`` (see the
``_vtl_mul_quant_*`` helpers in ``short_conv.py``). That call site is gated the same way and
fails closed to the stock ``y = C*Bx`` + normal ``out_proj`` path.

Second-order vs [[shortconv_quant]] (which is the actual TPOT lever): this only removes a bf16
[T, conv_dim] round-trip + one launch per conv layer. **Default OFF** (``VTL_ENABLE_MUL_QUANT``);
enable only after an on-box A/B shows decode TPOT improves (torch.compile/inductor may already fuse
the multiply) and the eval confirms no accuracy regression. Requires the conv projections to be fp8
(shortconv_quant on) -- otherwise ``out_proj`` is bf16 and there is nothing to feed pre-quantized.
"""

from __future__ import annotations

import logging

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl")

_OP = "mul_dynamic_per_token_quant"

_fake_registered = False


def _register_fake() -> None:
    """FakeTensor/meta rule so torch.compile can trace an inserted node. The op mutates
    ``result``/``scale`` in place and returns (); the fake only has to run without touching data.
    Idempotent -- registering a fake twice raises."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake(f"vllm_cuda::{_OP}")
    def _fake(result, scale, a, b, scale_ub=None):  # noqa: ANN001
        return None

    _fake_registered = True


@register_patch("mul_quant", default=False)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- keep import order uniform with the other patches
    import vtl._C  # noqa: F401  -- dlopen registers the CUDA kernel + the vllm_cuda op schema

    if not hasattr(torch.ops.vllm_cuda, _OP):
        log.warning("vtl: mul_quant op not registered after importing vtl._C; disabled")
        return

    _register_fake()
    log.info("vtl: mul_quant op registered (conv-gate fused mul+fp8-quant; wired in short_conv)")


def _self_check() -> None:
    """No vLLM/torch: just assert the module-level wiring constants are what short_conv expects."""
    assert _OP == "mul_dynamic_per_token_quant", _OP
    # The env gate name short_conv keys off must match this patch's registered name.
    from vtl.registry import PATCH_REGISTRY  # local import so the file loads without torch

    # apply() is registered under "mul_quant" with default False (opt-in).
    names = {p.name: p.default for p in PATCH_REGISTRY}
    assert names.get("mul_quant") is False, names
    print("mul_quant self-check ok")


if __name__ == "__main__":
    _self_check()
