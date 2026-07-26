"""Swap vLLM's standalone dynamic per-token FP8 quant CUDA kernel for ours.

`_C::dynamic_per_token_scaled_fp8_quant` is the per-token fp8 quant vLLM injects at every
fp8-linear input that is NOT preceded by a fused RMSNorm -- on the served qwen3_5 model that
is the `o_proj` input (after the full-attention `attn_output * sigmoid(gate)` multiply) and,
when the Phase-2 SiLU-mul fusion is off, the `down_proj` input. It also runs at load time on
the per-output-channel weight quant (vtl_fp8).

Importing `vtl._C` re-registers the CUDA kernel behind that op (see
vtl/csrc/torch_bindings.cpp). The op, its schema, its meta kernel and its FX node are vLLM's,
untouched -- only the kernel behind the CUDA dispatch key changes.

`vllm._C_stable_libtorch` must be imported BEFORE `vtl._C` (it defines the `_C` schema;
registering a kernel for an undefined op is silently deferred and the later definition wins).
The schema is byte-identical from vLLM v0.22.1 through v0.26.0 (re-verified against tag
v0.26.0 in csrc/libtorch_stable/torch_bindings.cpp on 2026-07-25),
so this override binds unchanged.

Set `VTL_ENABLE_DYNAMIC_PER_TOKEN_QUANT=0` to keep the stock kernel (the .so is then still
imported by a sibling patch, but this patch's log line / verification is skipped).
"""

from __future__ import annotations

import logging

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl")


OP = "_C::dynamic_per_token_scaled_fp8_quant"


def _cuda_kernel_is_ours() -> bool | None:
    """True/False from torch's dispatch table; None if it cannot be read."""
    import torch

    try:
        dump = torch._C._dispatch_dump(OP)
    except Exception:  # private API, absent or renamed
        return None
    for line in dump.splitlines():
        if line.strip().startswith("CUDA:"):
            return "vtl/csrc" in line
    return False


@register_patch("dynamic_per_token_quant", default=True)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schema, must be first

    if not hasattr(torch.ops._C, OP.split("::")[1]):
        log.warning("vtl: %s absent, leaving the stock per-token fp8 quant kernel", OP)
        return

    import vtl._C  # noqa: F401  -- dlopen side effect: TORCH_LIBRARY_IMPL(_C, CUDA)

    if _cuda_kernel_is_ours() is False:
        log.warning("vtl: %s CUDA kernel is NOT ours; stock kernel still installed", OP)
        return

    log.info("vtl: dynamic per-token fp8-quant CUDA kernel installed (o_proj/down_proj)")


def _self_check() -> None:
    """Runs anywhere: no GPU, no vLLM, no compiled extension."""
    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "dynamic_per_token_quant")
    assert patch.default is True
    assert is_enabled(patch) is True

    os.environ["VTL_ENABLE_DYNAMIC_PER_TOKEN_QUANT"] = "0"
    assert is_enabled(patch) is False
    os.environ.pop("VTL_ENABLE_DYNAMIC_PER_TOKEN_QUANT")

    try:
        apply()
    except Exception:
        pass

    print("dynamic_per_token_quant self-check ok")


if __name__ == "__main__":
    _self_check()
