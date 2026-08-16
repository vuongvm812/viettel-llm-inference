"""First production NVRTC consumer: shape-specialized group-128 fp8 block-quant kernels.

Overrides the CUDA kernels behind TWO stock ops, op-identity-preserving (HANDOFF §4.3):

* ``_C::rms_norm_per_block_quant(Tensor! result, Tensor input, Tensor weight,
  Tensor! scale, float epsilon, Tensor? scale_ub, Tensor!? residual, int group_size,
  bool is_scale_transposed) -> ()`` -- what the RMSNormQuantFusionPass emits for every
  fused-add-rmsnorm -> group-quant pair on a block-fp8 checkpoint
  (``FUSED_OPS[kFp8Dynamic128Sym]``, vllm/compilation/passes/fusion/rms_quant_fusion.py).
* ``_C::per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! output_s,
  int group_size, float eps, float fp8_min, float fp8_max, bool scale_ue8m0,
  bool dummy_is_scale_transposed, bool dummy_is_tma_aligned) -> ()`` -- the unfused
  activation quant behind ``per_token_group_quant_fp8`` (fp8_utils.py), reached both
  eagerly and through the ``triton_per_token_group_quant_fp8`` custom op.

Only the kernel behind the CUDA dispatch key changes (``torch.library.Library("_C",
"IMPL")``); schema, meta kernels, FUSED_OPS, FixFunctionalizationPass and the
torch.compile cache key are untouched.

MECHANISM. Registering a Python impl evicts the stock C++ kernel irrecoverably, so the
override is only installed once the NVRTC kernels have actually compiled for the loaded
model. ``apply()`` therefore just wraps ``BaseModelLoader.load_model`` (the same seam
``quant_w4a8``/``l2_persist`` use -- the registry's ``patch=`` names keep the stack
sound). After the model loads, the wrapper reads the REAL geometry from the config
(``model_config`` hidden size, ``quant_config.weight_block_size`` -> group), calls
``vtl.nvrtc.compile_kernel`` for the three specializations (rms with/without residual +
the plain group quant), and installs the dispatch override only if all three built.
``compile_kernel`` returning None -- NVRTC off, no cuda-python, bad source, no device --
leaves the stock ops completely untouched. Compilation happens at model load, off the
serving path, and the cubins disk-cache under ``$VTL_NVRTC_CACHE``.

If an installed wrapper ever sees a call outside the compiled envelope (different hidden
size, non-bf16 input, ue8m0 scales, int8 out...) it falls back to an eager torch
replication of the stock formula and warns once. That path is unreachable for the model
the kernels were compiled from; it exists so a surprise degrades to slow-and-right.

Gate: ``VTL_ENABLE_NVRTC_BLOCK_QUANT=1`` (default OFF until A/B'd) **and** ``VTL_NVRTC=1``
(the layer-wide switch). Numerics are the stock ops' bit for bit -- including the
double bf16 rounding of ``bf16(x*rms) * bf16(w)`` and the TRUE division by the group
scale -- see vtl/kernels/rms_norm_block_quant.cu and bench/test_nvrtc_block_quant.py.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.nvrtc_block_quant")

KERNEL = "rms_norm_block_quant"          # vtl/kernels/rms_norm_block_quant.cu
RMS_OP = "rms_norm_per_block_quant"      # torch.ops._C.<name>
GQ_OP = "per_token_group_fp8_quant"

VEC = 8                # bf16 elems per 16-byte vector, mirrors the kernel
LANES_PER_GROUP = 16   # GROUP / VEC
GQ_THREADS = 256       # group-quant entry block width (16 groups per block)
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)

# Populated by _arm(); read by the wrappers. `lib` must stay referenced or torch drops
# the registration.
_state: dict = {
    "installed": False,
    "armed": False,
    "hidden": 0,
    "group": 0,
    "threads": 0,
    "launchers": None,
    "lib": None,
}
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _geometry_ok(hidden: int, group: int) -> bool:
    """Can the shipped kernel be specialized to this (hidden, group)?

    Mirrors the kernel's static_asserts: GROUP=128 (16 lanes x vec8), one vec8 per
    thread so HIDDEN/8 must be whole warps and fit a block.
    """
    if group != 128:
        return False
    if hidden <= 0 or hidden % group != 0 or hidden % VEC != 0:
        return False
    threads = hidden // VEC
    return threads % 32 == 0 and threads <= 1024


def _threads_for(hidden: int) -> int:
    return hidden // VEC


def _defines(hidden: int, group: int, has_residual: int) -> dict:
    return {
        "HIDDEN": hidden,
        "GROUP": group,
        "THREADS": _threads_for(hidden),
        "HAS_RESIDUAL": has_residual,
        "GQ_THREADS": GQ_THREADS,
    }


def _build_launchers(hidden: int, group: int):
    """Compile all three specializations, or None (=> stock ops stay untouched)."""
    from vtl import nvrtc

    rms0 = nvrtc.compile_kernel(KERNEL, _defines(hidden, group, 0), entry=KERNEL)
    if rms0 is None:
        return None
    rms1 = nvrtc.compile_kernel(KERNEL, _defines(hidden, group, 1), entry=KERNEL)
    if rms1 is None:
        return None
    # Same cubin as rms0 (same defines), different entry point.
    gq = nvrtc.compile_kernel(KERNEL, _defines(hidden, group, 0), entry="group_quant_fp8")
    if gq is None:
        return None
    return {"rms0": rms0, "rms1": rms1, "gq": gq}


# --------------------------------------------------------------------------------------
# eager fallbacks: exact stock formulas in torch, for calls outside the compiled envelope
# --------------------------------------------------------------------------------------

def _eager_rms(result, input, weight, scale, epsilon, scale_ub, residual,
               group_size, is_scale_transposed) -> None:
    import torch

    if result.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"vtl nvrtc_block_quant: unsupported out dtype {result.dtype} "
            f"(stock kernel was replaced; only fp8e4m3 is implemented)"
        )
    hidden = input.shape[-1]
    x = input.reshape(-1, hidden).float()
    if residual is not None:
        x = x + residual.reshape(-1, hidden).float()
        residual.reshape(-1, hidden).copy_(x.to(residual.dtype))
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
    # The stock double rounding: bf16(x*rms) then a bf16 homogeneous multiply by w.
    y = ((x * rms).to(input.dtype) * weight).to(input.dtype).float()
    tokens, groups = y.shape[0], hidden // group_size
    g = y.view(tokens, groups, group_size)
    amax = g.abs().amax(-1)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub.float())
    s = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    q = torch.clamp(g / s.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(result.dtype)
    result.view(tokens, groups * group_size).copy_(q.view(tokens, -1))
    if scale.dim() == 2 and scale.shape[0] >= tokens and scale.shape[1] == groups:
        # Indexed writes respect the (possibly transposed) strides.
        scale[:tokens, :groups].copy_(s)
    else:
        scale.reshape(-1)[: tokens * groups] = s.reshape(-1)


def _eager_gq(input, output_q, output_s, group_size, eps, fp8_min, fp8_max,
              scale_ue8m0) -> None:
    import torch

    if output_q.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"vtl nvrtc_block_quant: unsupported out dtype {output_q.dtype} "
            f"(stock kernel was replaced; only fp8e4m3 is implemented)"
        )
    x = input.reshape(-1, group_size).float()
    y_s = x.abs().amax(-1).clamp_min(eps) / fp8_max   # absmax seeded with eps, stock-style
    if scale_ue8m0:
        y_s = torch.exp2(torch.ceil(torch.log2(y_s.abs().clamp_min(1e-10))))
    q = (x / y_s.unsqueeze(-1)).clamp(fp8_min, fp8_max).to(output_q.dtype)
    output_q.reshape(-1, group_size).copy_(q)
    num_groups = x.shape[0]
    gcols = output_s.shape[1] if output_s.dim() == 2 else 0
    if gcols and num_groups % gcols == 0:
        output_s[: num_groups // gcols, :gcols].copy_(y_s.view(-1, gcols))
    else:
        output_s.reshape(-1)[:num_groups] = y_s


# --------------------------------------------------------------------------------------
# dispatch-key wrappers
# --------------------------------------------------------------------------------------

def _rms_impl(result, input, weight, scale, epsilon, scale_ub, residual,
              group_size, is_scale_transposed) -> None:
    import torch
    from vtl import nvrtc

    launchers = _state["launchers"]
    hidden = _state["hidden"]
    ok = (
        launchers is not None
        and input.dtype == torch.bfloat16
        and result.dtype == torch.float8_e4m3fn
        and input.shape[-1] == hidden
        and int(group_size) == _state["group"]
        and input.stride(-1) == 1
        and scale.dtype == torch.float32
        and (residual is None or residual.dtype == torch.bfloat16)
    )
    in_stride = 0
    if ok:
        in_stride = input.reshape(-1, hidden).stride(0)
        ok = in_stride % VEC == 0   # 16-byte alignment of every token row
    if not ok:
        _warn_once("rms_eager", "vtl: %s call outside compiled envelope; eager fallback",
                   RMS_OP)
        _eager_rms(result, input, weight, scale, epsilon, scale_ub, residual,
                   group_size, is_scale_transposed)
        return

    num_tokens = input.numel() // hidden
    outer = scale.stride(1) if scale.dim() >= 2 else 1
    launcher = launchers["rms1" if residual is not None else "rms0"]
    launcher(
        grid=(num_tokens, 1, 1),
        block=(_state["threads"], 1, 1),
        args=nvrtc.pack_args(
            result.data_ptr(),
            scale.data_ptr(),
            input.data_ptr(),
            weight.data_ptr(),
            scale_ub.data_ptr() if scale_ub is not None else 0,
            residual.data_ptr() if residual is not None else 0,
            ("f", float(epsilon)),
            ("l", in_stride),
            ("i", 1 if is_scale_transposed else 0),
            ("l", max(int(outer), 1)),
        ),
    )


def _gq_impl(input, output_q, output_s, group_size, eps, fp8_min, fp8_max,
             scale_ue8m0, dummy_is_scale_transposed, dummy_is_tma_aligned) -> None:
    import torch
    from vtl import nvrtc

    launchers = _state["launchers"]
    ok = (
        launchers is not None
        and not scale_ue8m0
        and input.dtype == torch.bfloat16
        and output_q.dtype == torch.float8_e4m3fn
        and int(group_size) == _state["group"]
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.dtype == torch.float32
        and output_s.dim() == 2
        and input.numel() % int(group_size) == 0
    )
    if not ok:
        _warn_once("gq_eager", "vtl: %s call outside compiled envelope; eager fallback",
                   GQ_OP)
        _eager_gq(input, output_q, output_s, group_size, eps, fp8_min, fp8_max,
                  scale_ue8m0)
        return

    num_groups = input.numel() // int(group_size)
    groups_per_block = GQ_THREADS // LANES_PER_GROUP
    # Stock detects the layout from the strides; so do we.
    col_major = output_s.stride(0) < output_s.stride(1)
    launchers["gq"](
        grid=((num_groups + groups_per_block - 1) // groups_per_block, 1, 1),
        block=(GQ_THREADS, 1, 1),
        args=nvrtc.pack_args(
            output_q.data_ptr(),
            output_s.data_ptr(),
            input.data_ptr(),
            ("f", float(eps)),
            ("f", float(fp8_min)),
            ("f", float(fp8_max)),
            ("l", num_groups),
            ("i", 1 if col_major else 0),
            ("i", int(output_s.size(1))),
            ("i", int(output_s.stride(1))),
        ),
    )


def _install_overrides() -> bool:
    """Swap the CUDA kernels behind both ops. Op identity (schema/meta/graph) unchanged."""
    import torch

    # Defines the _C schemas; must exist before we override, or torch would defer the
    # registration and vLLM's later import would silently win the key back.
    import vllm._C_stable_libtorch  # noqa: F401

    for op in (RMS_OP, GQ_OP):
        if not hasattr(torch.ops._C, op):
            log.warning("vtl: %s absent; leaving stock block-quant kernels", op)
            return False
    lib = torch.library.Library("_C", "IMPL")  # noqa: TOR901 -- deliberate override
    lib.impl(RMS_OP, _rms_impl, "CUDA")
    lib.impl(GQ_OP, _gq_impl, "CUDA")
    _state["lib"] = lib   # keep-alive: dropping it would deregister the kernels
    return True


def _arm(vllm_config, model_config) -> None:
    """Runs once after the model loads: read geometry, compile, then (only then) install."""
    if _state["armed"]:
        return
    _state["armed"] = True

    try:
        hidden = int(model_config.get_hidden_size())
    except Exception:
        hidden = int(getattr(getattr(model_config, "hf_text_config", None),
                             "hidden_size", 0) or 0)
    quant_config = getattr(vllm_config, "quant_config", None)
    wbs = getattr(quant_config, "weight_block_size", None)
    if not wbs or len(wbs) < 2:
        log.info("vtl: nvrtc_block_quant: no weight_block_size in quant config "
                 "(not a block-fp8 model); stock ops stand")
        return
    group = int(wbs[1])   # activation groups run along K, i.e. weight_block_size[1]

    if not _geometry_ok(hidden, group):
        log.info("vtl: nvrtc_block_quant: geometry hidden=%d group=%d outside the "
                 "kernel's envelope; stock ops stand", hidden, group)
        return

    launchers = _build_launchers(hidden, group)
    if launchers is None:
        log.info("vtl: nvrtc_block_quant: NVRTC compile unavailable/failed; "
                 "stock ops stand")
        return

    _state.update(hidden=hidden, group=group, threads=_threads_for(hidden),
                  launchers=launchers)
    if not _install_overrides():
        _state["launchers"] = None
        return
    _state["installed"] = True
    # The one-line tier log `make verify`-style checks can grep for.
    log.info(
        "vtl: nvrtc block-quant tier active: %s + %s -> NVRTC "
        "(HIDDEN=%d GROUP=%d THREADS=%d GQ_THREADS=%d)",
        RMS_OP, GQ_OP, hidden, group, _threads_for(hidden), GQ_THREADS,
    )


@register_patch("nvrtc_block_quant", default=False)
def apply() -> None:
    from vtl import nvrtc

    if not nvrtc.enabled():
        log.info("vtl: nvrtc_block_quant selected but VTL_NVRTC is off; no-op")
        return

    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch="nvrtc_block_quant"):
        return
    orig = BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, *args, **kwargs):
        model = orig(self, vllm_config, model_config, *args, **kwargs)
        try:
            _arm(vllm_config, model_config)
        except Exception:
            # Never let an optimization kill the load; stock kernels remain installed
            # because _install_overrides only runs after a successful compile.
            log.exception("vtl: nvrtc_block_quant arming failed; stock ops stand")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, orig, patch="nvrtc_block_quant")
    log.info("vtl: nvrtc_block_quant armed (compiles + installs at model load)")


def _self_check() -> None:
    """Runs anywhere: no GPU, no torch, no vLLM."""
    import os

    from vtl import nvrtc
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "nvrtc_block_quant")
    assert patch.default is False, "unproven optimization: default OFF"
    assert is_enabled(patch) is False
    os.environ["VTL_ENABLE_NVRTC_BLOCK_QUANT"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_NVRTC_BLOCK_QUANT")

    # Geometry envelope: the model this ships for (3072/128) must pass; the ways a
    # foreign model could break the kernel must not.
    assert _geometry_ok(3072, 128) is True
    assert _threads_for(3072) == 384
    assert _geometry_ok(4096, 128) is True
    assert _geometry_ok(3072, 64) is False, "GROUP=128 only in this port"
    assert _geometry_ok(3000, 128) is False, "hidden must divide by group"
    assert _geometry_ok(16384, 128) is False, "HIDDEN/8 must fit one block"
    assert _geometry_ok(0, 128) is False

    # Each specialization must be a distinct cubin identity, and the residual flag or a
    # different geometry must never collide in the disk cache.
    src = nvrtc.load_source(KERNEL)
    assert src, "vtl/kernels/rms_norm_block_quant.cu missing from the package"
    assert 'extern "C"' in src and "group_quant_fp8" in src
    keys = {
        nvrtc.cache_key(src, _defines(3072, 128, 0), "90a", "12.8"),
        nvrtc.cache_key(src, _defines(3072, 128, 1), "90a", "12.8"),
        nvrtc.cache_key(src, _defines(4096, 128, 0), "90a", "12.8"),
        nvrtc.cache_key(src, _defines(3072, 128, 0), "90", "12.8"),
    }
    assert len(keys) == 4, "cache keys must be distinct across define sets and arches"
    # ...and stable across dict-order permutations.
    d = _defines(3072, 128, 0)
    perm = dict(reversed(list(d.items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(src, d, "90a", "12.8")

    # While NVRTC is disabled, compiling is a no-op and apply() must not install anything.
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(KERNEL, _defines(3072, 128, 0)) is None
    apply()   # vLLM may be absent; with VTL_NVRTC off this must return cleanly
    assert _state["installed"] is False

    # With NVRTC on but vLLM absent, apply() may raise (registry isolates it) but must
    # not corrupt module state.
    os.environ["VTL_NVRTC"] = "1"
    try:
        apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_NVRTC", None)
    assert _state["installed"] is False

    print("nvrtc_block_quant self-check ok")


if __name__ == "__main__":
    _self_check()
