"""Qwen3.5 MTP-head fused-projection GEMM (cuBLASDx) for speculative decoding.

The MTP head (`Qwen3_5MultiTokenPredictor`, vLLM v0.25.0) runs every decode step when
`--speculative-config '{"method":"qwen3_next_mtp",...}'` is set. Its forward is
`pre_fc_norm_embedding(embeds) + pre_fc_norm_hidden(hidden) -> cat -> fc([M,4096]->[M,2048])`.
At `num_speculative_tokens=2`, M is tens of rows -- a small-M GEMM where cuBLASDx's *device*
GEMM avoids cuBLAS launch overhead. See vtl/csrc/mtp_fc_gemm.cu.

STAGE A (here): route only the `fc` GEMM through `vllm_cuda::mtp_fc_gemm`. The two RMSNorms
and the cat stay in torch. Instance-scoped: we wrap the specific `mtp.fc` module's `forward`,
not ColumnParallelLinear's class (which every linear shares). Stage B (folding the norms+cat
into the kernel prologue) is future work; VTL_MTP_FC_FUSION is the lever.

Gated by VTL_ENABLE_MTP_KERNELS (default OFF -- arm only after `make test-kernel` parity on the
box). The op only exists if vtl._C was built with cuBLASDx headers (nvidia-mathdx); otherwise
this patch logs and leaves stock vLLM in place. The fc weight is fp8 under --quantization=vtl_fp8,
so the dtype guard below falls back to stock unless fc runs unquantized (e.g. fc in VTL_FP8_IGNORE
or a bf16 checkpoint); an fp8 GEMM path is the follow-up.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _fc_fusion_enabled() -> bool:
    """VTL_MTP_FC_FUSION=1 wires the fc GEMM (Stage A). =0 leaves stock -- an A/B off lever."""
    return os.environ.get("VTL_MTP_FC_FUSION", "1").strip().lower() in _TRUTHY


_fake_registered = False


def _register_fake() -> None:
    """FakeTensor rule so the op traces under the MTP head's @support_torch_compile. Idempotent."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake("vllm_cuda::mtp_fc_gemm")
    def _fake(result, input, weight):  # noqa: ANN001  -- result mutated in place, returns None
        return None

    _fake_registered = True


def _wrap_fc(fc) -> None:  # noqa: ANN001
    """Replace this fc instance's forward with the cuBLASDx GEMM, guarded. Defensive: any
    unquantized/shape/dtype mismatch (incl. fp8 weights, tp>1, bias) falls back to the original."""
    import torch

    if fc is None or already_patched(fc, "forward"):
        return

    orig_forward = fc.forward

    def forward(x):  # noqa: ANN001  -- instance attr, so called as fc.forward(x), no self
        try:
            w = getattr(fc, "weight", None)
            ok = (
                getattr(fc, "bias", None) is None
                and getattr(fc, "tp_size", 1) == 1  # no all-gather across shards
                and w is not None
                and x.is_cuda
                and x.dim() == 2
                and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and w.dtype == x.dtype  # fp8-quantized weight -> skip (dtype won't match)
                and w.dim() == 2
                and x.shape[1] == w.shape[1]  # K
                and x.shape[1] % 8 == 0
            )
            if ok:
                x = x.contiguous()
                out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
                torch.ops.vllm_cuda.mtp_fc_gemm(out, x, w.contiguous())
                return out
        except Exception:
            pass
        return orig_forward(x)

    fc.forward = mark_patched(forward, orig_forward)


def _install_fc_gemm() -> bool:
    """Hook Qwen3_5MultiTokenPredictor.__init__ so each MTP head wraps its own fc after build.
    Instance-scoped -- the class-level ColumnParallelLinear is untouched."""
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor  # caller guards

    if already_patched(Qwen3_5MultiTokenPredictor, "__init__"):
        return True

    orig_init = Qwen3_5MultiTokenPredictor.__init__

    def __init__(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        orig_init(self, *args, **kwargs)
        try:
            _wrap_fc(getattr(self, "fc", None))
        except Exception:  # never break model construction
            log.warning("vtl: MTP fc wrap failed; stock fc kept", exc_info=True)

    Qwen3_5MultiTokenPredictor.__init__ = mark_patched(__init__, orig_init)
    return True


@register_patch("mtp_kernels", default=False)
def apply() -> None:
    import torch

    import vtl._C  # noqa: F401  -- registers vllm_cuda::mtp_fc_gemm (only if built with cuBLASDx)

    if not hasattr(torch.ops.vllm_cuda, "mtp_fc_gemm"):
        log.info("vtl: mtp_fc_gemm op not built (no cuBLASDx headers); skipping MTP kernels")
        return

    _register_fake()

    if not _fc_fusion_enabled():
        log.info("vtl: MTP fc GEMM available but VTL_MTP_FC_FUSION=0; not wiring (A/B off lever)")
        return

    try:
        if _install_fc_gemm():
            log.info("vtl: MTP fc cuBLASDx GEMM installed (VERIFY parity on H200)")
    except Exception as exc:
        log.warning("vtl: MTP fc GEMM install failed (%s); stock fc kept", exc)


def _self_check() -> None:
    """Runs anywhere (no torch/vLLM): registry wiring + flag parsing only."""
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "mtp_kernels")
    assert patch.default is False, "MTP kernels are opt-in until parity is verified on the box"
    assert is_enabled(patch) is False

    os.environ["VTL_ENABLE_MTP_KERNELS"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_MTP_KERNELS")

    assert _fc_fusion_enabled() is True
    os.environ["VTL_MTP_FC_FUSION"] = "0"
    assert _fc_fusion_enabled() is False
    os.environ.pop("VTL_MTP_FC_FUSION")

    print("mtp_kernels self-check ok")


if __name__ == "__main__":
    _self_check()
