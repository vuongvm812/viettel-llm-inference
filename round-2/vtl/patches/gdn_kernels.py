"""GDN (linear-attention) gated-RMSNorm custom CUDA kernels for the qwen3_5 hot path
(36 of 46 layers on Qwen3.5-122B-A10B).

The GDN output norm is ``RMSNormGated`` (per value head, head_v_dim=128), applied to the
gated-delta core output before ``out_proj``: norm_before_gate=True, group_size=None,
activation in {silu, sigmoid} => ``out = rms_norm(x)*w * act(z)``. This checkpoint is FP8
block-quant, so ``out_proj`` consumes its input quantized dynamic per-(token, 128-group)
(``kFp8Dynamic128Sym`` / ``_C::per_token_group_fp8_quant``) -- NOT per-token. The stock
CUDA path runs FLA's Triton norm and then the separate group-quant kernel, so the bf16
norm output makes a full HBM round-trip on every GDN layer, every token. Because
head_v_dim == quant group == 128, one quant group is exactly one head, and the fused
norm+quant is a single warp-local kernel.

Mechanism (same shape as round-1.1, which shipped the FUSION path with an eager fallback):

  * FUSION (default; ``VTL_GDN_RMSNORM_FUSION=1``): register a replacement pattern that
    rewrites ``RMSNormGated(x,z,w) -> reshape(-1, H*D) -> per_token_group_fp8_quant`` into
    one ``vllm_cuda::gdn_gated_rmsnorm_group_quant`` node. Structure mirrors vLLM's OWN
    ROCm ``AiterRMSNormGatedFp8GroupQuantPattern`` (rocm_aiter_fusion.py) -- stock
    ``MatcherRMSNormGated`` + ``MatcherQuantFP8(kFp8Dynamic128Sym)``, traced with
    view->reshape + consecutive-reshape folding -- but registered into the CUDA
    ``RMSNormQuantFusionPass`` (which qwen_gdn_linear_attn.py's own comment names as the
    intended fusion seam) and calling our op. Needs ``fuse_norm_quant: true`` in the serve
    --compilation-config (already set in docker-compose.yaml). VERIFY on the box that it
    fired (``Replaced N patterns``, N>0); a non-match is a silent no-op, never corruption.

  * EAGER (``VTL_GDN_RMSNORM_FUSION=0``, or fusion registration failed): monkeypatch
    ``RMSNormGated.forward`` to the standalone ``vllm_cuda::gated_rmsnorm`` kernel (norm
    only; quant stays stock). A/B lever -- NOT installed alongside the fusion, because a
    traced monkeypatched forward would replace the node the pattern needs.

The fused op resolves its kernel through a three-tier ladder ONCE, at the first call,
from the live tensors of the loaded model (this is where the NVRTC defines come from):

    NVRTC (vtl/kernels/gated_rmsnorm_group_quant.cu, shape-specialized; VTL_NVRTC=1)
      -> AOT (vtl._C :: gated_rmsnorm_per_group_quant)
        -> stock composition in torch (never raises, only slower)

Gated by ``VTL_ENABLE_GDN_KERNELS`` (default ON since 2026-08-17 -- the ladder above is
what makes that safe; set it to 0 to A/B against stock).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.gdn_kernels")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The Python-registered fused op the fusion pattern inserts. Its CUDA impl is the tier
# ladder below; keeping ONE op identity regardless of which tier wins keeps the
# torch.compile cache key stable across NVRTC availability.
OP = "vllm_cuda::gdn_gated_rmsnorm_group_quant"
# The AOT kernel behind tier 2 (registered by importing vtl._C).
AOT_OP = "gated_rmsnorm_per_group_quant"

GROUP = 128  # fp8 activation quant group of this checkpoint (kFp8Dynamic128Sym)
NVRTC_THREADS = 256


def _fusion_enabled() -> bool:
    return os.environ.get("VTL_GDN_RMSNORM_FUSION", "1").strip().lower() in _TRUTHY


def _gate_is_silu(activation) -> bool:  # noqa: ANN001
    """swish == silu; only "sigmoid" turns it off. Matches RMSNormGated.forward_static."""
    return str(activation) != "sigmoid"


# ---------------------------------------------------------------------------------------
# Pure shape helpers -- exercised by `--self-check` (no torch needed).
# ---------------------------------------------------------------------------------------

def _nvrtc_defines(dv: int, hv: int, is_silu: bool, w_fp32: bool) -> dict[str, int]:
    """The -D set for vtl/kernels/gated_rmsnorm_group_quant.cu. Values from the live model
    tensors at first call; every entry changes the cubin cache key."""
    return {
        "DV": int(dv),
        "HV": int(hv),
        "GROUP": GROUP,
        "THREADS": NVRTC_THREADS,
        "IS_SILU": 1 if is_silu else 0,
        "W_FP32": 1 if w_fp32 else 0,
    }


def _nvrtc_grid(num_rows: int, dv: int, threads: int = NVRTC_THREADS) -> tuple[int, int, int]:
    """Sub-group-per-row launch: LANES = DV/8 lanes per row, THREADS/LANES rows per block."""
    lanes = dv // 8
    rows_per_block = threads // lanes
    return ((num_rows + rows_per_block - 1) // rows_per_block, 1, 1)


def _nvrtc_supported(dv: int, hv: int, threads: int = NVRTC_THREADS) -> bool:
    """Mirror of the kernel's static_asserts, so an unsupported shape is refused HERE
    (degrade to AOT) instead of failing the compile at model load."""
    if dv <= 0 or hv <= 0 or dv % 8 != 0 or dv != GROUP:
        return False
    lanes = dv // 8
    if lanes < 2 or lanes > 32 or (lanes & (lanes - 1)) != 0:
        return False
    return threads % lanes == 0


def _scale_shape(num_tokens: int, hidden: int, group: int = GROUP) -> tuple[int, int]:
    """Row-major scale shape per_token_group_fp8_quant emits: [tokens, hidden/group]."""
    assert hidden % group == 0, (hidden, group)
    return (num_tokens, hidden // group)


# ---------------------------------------------------------------------------------------
# The fused op + its tier ladder.
# ---------------------------------------------------------------------------------------

_lib = None            # keep the torch.library.Library alive (GC would deregister the op)
_fake_registered = False
_tiers: dict = {}      # (dv, hv, is_silu, w_fp32, dtype) -> ("nvrtc", launch) | (tier, None)


def _resolve_tier(input, weight, num_heads, gate_is_silu):  # noqa: ANN001
    """First-call resolution: NVRTC -> AOT -> stock. Cached per specialization; the chosen
    tier is logged once. Never raises."""
    import torch

    key = (
        int(input.shape[-1]),
        int(num_heads),
        bool(gate_is_silu),
        weight.dtype == torch.float32,
        str(input.dtype),
    )
    hit = _tiers.get(key)
    if hit is not None:
        return hit

    dv, hv, is_silu, w_fp32, _ = key
    tier = ("stock", None)
    try:
        if hasattr(torch.ops.vllm_cuda, AOT_OP):
            tier = ("aot", None)
    except Exception:
        pass
    try:
        from vtl import nvrtc

        if input.dtype == torch.bfloat16 and _nvrtc_supported(dv, hv):
            launch = nvrtc.compile_kernel(
                "gated_rmsnorm_group_quant", _nvrtc_defines(dv, hv, is_silu, w_fp32)
            )
            if launch is not None:
                tier = ("nvrtc", launch)
    except Exception as exc:
        log.info("vtl: gdn group-quant NVRTC tier unavailable (%r)", exc)

    _tiers[key] = tier
    log.info(
        "vtl: gdn_gated_rmsnorm_group_quant -> %s tier (DV=%d HV=%d silu=%s w_fp32=%s)",
        tier[0], dv, hv, is_silu, w_fp32,
    )
    return tier


def _stock_group_quant(result, scale, input, gate, weight, epsilon, gate_is_silu):  # noqa: ANN001
    """Tier 3: the exact stock composition in torch (RMSNormGated.forward_static numerics
    + per_token_group quant numerics, double rounding included). Slow, correct, never
    raises past the caller's guard."""
    import torch

    xf = input.float()
    zf = gate.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    y = (xf * torch.rsqrt(var + epsilon)) * weight.float()
    act = zf * torch.sigmoid(zf) if gate_is_silu else torch.sigmoid(zf)
    yq = (y * act).to(input.dtype).float()  # narrow once at the op boundary, then re-widen
    grouped = yq.reshape(scale.shape[0], scale.shape[1], GROUP)
    amax = grouped.abs().amax(-1).clamp_(min=1e-10)
    s = amax / 448.0
    q = (grouped / s.unsqueeze(-1)).clamp_(-448.0, 448.0)
    result.copy_(q.reshape(result.shape).to(result.dtype))
    scale.copy_(s)


def _decode_step_epilogue(result, scale, input, weight, num_heads) -> bool:  # noqa: ANN001
    """Tier 0: the GDN decode step already produced this exact result, fused into its own
    launch. Returns True if it satisfied the call.

    TWO-SIDED HANDSHAKE with ``vtl/patches/gdn_decode_step.py`` (see its module docstring):
    reaching this function AT ALL is the proof that patch needs that this op -- rather than
    stock RMSNormGated + a separate quant -- is what consumes the core output for the layer
    owning ``weight``, so ``note_epilogue_consumer`` is called unconditionally and BEFORE
    the claim lookup. Only then may the decode step stop writing bf16 ``core_attn_out``.

    ``take_epilogue`` matches on ``input.data_ptr()``, which is the core-output buffer
    itself (``_output_projection`` reaches us through a reshape VIEW of it), and pops the
    claim so it can never be applied twice. A miss is the ordinary case on prefill, spec
    and any batch the decode step did not claim, and costs one dict lookup.

    The residual ``copy_`` is the staged fp8 landing in the buffer inductor allocated inside
    the graph -- that buffer does not exist yet when the opaque core op runs, so it cannot
    be the kernel's destination. It replaces the full norm+quant kernel, not an existing
    copy: what is gone is the [T, H*D] bf16 write and its matching read.
    """
    try:
        from vtl.patches import gdn_decode_step as decode_step
    except Exception:
        return False
    try:
        decode_step.note_epilogue_consumer(weight.data_ptr())
        staged = decode_step.take_epilogue(input, num_heads)
        if staged is None:
            return False
        fp8, scales = staged
        if tuple(fp8.shape) != tuple(result.shape) or tuple(scales.shape) != tuple(scale.shape):
            return False
        result.copy_(fp8)
        scale.copy_(scales)
        return True
    except Exception as exc:
        log.warning("vtl: gdn decode-step epilogue handoff failed (%r); running the full "
                    "norm+quant for this call", exc)
        return False


def _group_quant_impl(result, scale, input, gate, weight, epsilon, num_heads, gate_is_silu):  # noqa: ANN001
    """CUDA impl of vllm_cuda::gdn_gated_rmsnorm_group_quant -- the tier ladder."""
    import torch

    if _decode_step_epilogue(result, scale, input, weight, num_heads):
        return

    tier, launch = _resolve_tier(input, weight, num_heads, gate_is_silu)

    if tier == "nvrtc":
        try:
            from vtl import nvrtc

            num_rows = input.shape[0]
            launch(
                grid=_nvrtc_grid(num_rows, input.shape[-1]),
                block=(NVRTC_THREADS, 1, 1),
                args=nvrtc.pack_args(
                    result.data_ptr(), scale.data_ptr(), input.data_ptr(),
                    gate.data_ptr(), weight.data_ptr(),
                    ("l", scale.stride(0)), ("l", scale.stride(1)),
                    ("l", num_rows), ("f", float(epsilon)),
                ),
            )
            return
        except Exception as exc:  # demote permanently; a launch that raised once will again
            log.warning("vtl: gdn group-quant NVRTC launch failed (%r); demoting to AOT", exc)
            for k, v in list(_tiers.items()):
                if v[0] == "nvrtc":
                    _tiers[k] = ("aot", None) if hasattr(torch.ops.vllm_cuda, AOT_OP) else (
                        "stock", None)
            tier = _tiers.get(
                (int(input.shape[-1]), int(num_heads), bool(gate_is_silu),
                 weight.dtype == torch.float32, str(input.dtype)), ("stock", None))[0]

    if tier == "aot":
        try:
            getattr(torch.ops.vllm_cuda, AOT_OP)(
                result, scale, input, gate, weight, float(epsilon), int(num_heads), GROUP,
                bool(gate_is_silu),
            )
            return
        except Exception as exc:
            log.warning("vtl: gdn group-quant AOT kernel failed (%r); stock composition", exc)

    _stock_group_quant(result, scale, input, gate, weight, float(epsilon), bool(gate_is_silu))


def _register_op() -> bool:
    """Define the Python-side fused op (schema + CUDA impl + fake). Idempotent."""
    global _lib, _fake_registered
    import torch

    ns, name = OP.split("::")
    if not hasattr(getattr(torch.ops, ns, None) or object(), name):
        if _lib is None:
            _lib = torch.library.Library(ns, "FRAGMENT")  # noqa: TOR901 -- plugin seam
        _lib.define(
            f"{name}(Tensor! result, Tensor! scale, Tensor input, Tensor gate, "
            "Tensor weight, float epsilon, int num_heads, bool gate_is_silu) -> ()"
        )
        _lib.impl(name, _group_quant_impl, "CUDA")

    if not _fake_registered:
        @torch.library.register_fake(OP)
        def _fake(result, scale, input, gate, weight, epsilon, num_heads, gate_is_silu):  # noqa: ANN001
            return None

        # The AOT ops also need fakes so a traced call (tests, eager compile) runs.
        for aot in (AOT_OP, "gated_rmsnorm", "gated_rmsnorm_dynamic_per_token_quant"):
            try:
                if hasattr(torch.ops.vllm_cuda, aot):
                    torch.library.register_fake(f"vllm_cuda::{aot}")(
                        lambda *a, **k: None
                    )
            except Exception:
                pass  # already registered (re-apply) -- fine
        _fake_registered = True
    return True


# ---------------------------------------------------------------------------------------
# Fusion pattern (mirrors AiterRMSNormGatedFp8GroupQuantPattern, retargeted to CUDA).
# ---------------------------------------------------------------------------------------

def _discover_gdn_shapes(config):  # noqa: ANN001
    """(num_heads, head_dim, eps, activation) per GDN layer, the same way vLLM's own ROCm
    pass discovers them. Empty on any drift."""
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    except Exception:
        return []
    out = []
    try:
        layers = get_layers_from_vllm_config(config, GatedDeltaNetAttention)
    except Exception:
        return []
    for layer in layers.values():
        try:
            tp = getattr(layer, "tp_size", 1) or 1
            h = (getattr(layer, "num_v_heads", None) or getattr(layer, "num_heads")) // tp
            d = getattr(layer, "head_v_dim", None) or getattr(layer, "head_dim")
            norm = getattr(layer, "norm", None)
            eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
            act = str(getattr(norm, "activation", "silu"))
            out.append((int(h), int(d), eps, act))
        except Exception:
            continue
    return list(dict.fromkeys(out))  # all 36 layers share one shape; dedup


def _make_pattern_class():
    """Built lazily: the imports only exist inside a vLLM process."""
    import torch
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
    from torch._inductor import pattern_matcher as pm
    from vllm.compilation.passes.fusion.matcher_utils import (
        MatcherQuantFP8,
        MatcherRMSNormGated,
    )
    from vllm.compilation.passes.vllm_inductor_pass import (
        _fx_view_to_reshape,
        fold_consecutive_reshapes,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym

    fused_op = getattr(torch.ops.vllm_cuda, OP.split("::")[1]).default

    class _GdnGatedNormGroupQuantPattern:
        """RMSNormGated(x,z,w) -> reshape(-1, H*D) -> per_token_group_fp8_quant  ==>
        one vllm_cuda::gdn_gated_rmsnorm_group_quant node. Structure lifted from vLLM's
        AiterRMSNormGatedFp8GroupQuantPattern; the reshape chain in the real graph
        (norm -> reshape(z_shape) -> flatten(-2)) is folded by the same
        fold_consecutive_reshapes the ROCm pass uses."""

        def __init__(self, epsilon: float, num_heads: int, head_dim: int,
                     col_major_scales: bool) -> None:
            self.epsilon = epsilon
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.col_major_scales = col_major_scales
            self.norm_matcher = MatcherRMSNormGated(epsilon)
            self.quant_matcher = MatcherQuantFP8(
                kFp8Dynamic128Sym, has_col_major_scales=col_major_scales
            )

        def register(self, pm_pass) -> None:  # noqa: ANN001
            H, D, eps = self.num_heads, self.head_dim, self.epsilon
            hidden = H * D
            norm_matcher, quant_matcher = self.norm_matcher, self.quant_matcher
            col_major = self.col_major_scales

            def pattern(x, z, weight):  # noqa: ANN001
                normed = norm_matcher(x, z, weight)
                merged = normed.reshape(-1, hidden)
                quant_out, scales_out = quant_matcher(merged)
                return quant_out, scales_out

            def replacement(x, z, weight):  # noqa: ANN001
                n_tokens = x.shape[0] // H
                result = torch.empty(
                    (n_tokens, hidden), device=x.device, dtype=torch.float8_e4m3fn
                )
                if col_major:
                    scale = torch.empty(
                        (hidden // GROUP, n_tokens), device=x.device, dtype=torch.float32
                    ).permute(-1, -2)
                else:
                    scale = torch.empty(
                        (n_tokens, hidden // GROUP), device=x.device, dtype=torch.float32
                    )
                at = auto_functionalized(
                    fused_op,
                    result=result,
                    scale=scale,
                    input=x,
                    gate=z,
                    weight=weight,
                    epsilon=eps,
                    num_heads=H,
                    gate_is_silu=True,  # patterns are only registered for silu/swish layers
                )
                return at[1], at[2]  # result, scale (mutated args, schema order)

            def trace_fn(*args, **kwargs):  # noqa: ANN001
                gm = pm.fwd_only(*args, **kwargs)
                _fx_view_to_reshape(gm)
                fold_consecutive_reshapes(gm)
                return gm

            n_tokens = 2
            x = norm_matcher.empty(n_tokens * H, D)
            z = norm_matcher.empty(n_tokens * H, D)
            w = norm_matcher.empty(D)
            pm.register_replacement(pattern, replacement, [x, z, w], trace_fn, pm_pass)

    return _GdnGatedNormGroupQuantPattern


def _register_fusion_pattern() -> bool:
    """Hook RMSNormQuantFusionPass (the CUDA fuse_norm_quant pass, already enabled by the
    serve --compilation-config) to also register our gated-norm+group-quant patterns, one
    per discovered (H, D, eps) with D == 128 and a silu/swish gate. Best-effort: any
    import or discovery failure returns False and the caller falls back to eager."""
    from vllm.compilation.passes.fusion import rms_quant_fusion as rqf

    try:
        from vllm.compilation.passes.inductor_pass import enable_fake_mode
    except Exception:  # moved? degrade to identity -- registration may still work
        def enable_fake_mode(fn):  # noqa: ANN001
            return fn

    Pass = rqf.RMSNormQuantFusionPass
    if getattr(Pass, "__vtl_gdn_rmsnorm_patched__", False):
        return True

    orig_init = Pass.__init__

    @enable_fake_mode
    def _register_gdn_patterns(self, config) -> None:  # noqa: ANN001
        PatternCls = _make_pattern_class()
        shapes = _discover_gdn_shapes(config)
        registered = []
        for H, D, eps, act in shapes:
            if D != GROUP or (H * D) % GROUP != 0:
                log.warning("vtl: gdn gated-norm fusion skips H=%d D=%d (group=%d)", H, D, GROUP)
                continue
            if not _gate_is_silu(act):
                # MatcherRMSNormGated traces the default (silu) activation only; a sigmoid
                # gate could never match, so registering would be dead weight.
                log.warning("vtl: gdn gated-norm fusion skips sigmoid-gated layer set "
                            "(H=%d D=%d) -- use the eager path for A/B", H, D)
                continue
            # Both scale layouts the fp8 linear path can request; the kernel writes either
            # (it reads the scale strides), and a graph contains only one of them.
            for col_major in (False, True):
                PatternCls(eps, H, D, col_major).register(self.patterns)
            registered.append((H, D, eps, act))
        if registered:
            log.info("vtl: GDN gated-norm+group-quant fusion patterns registered for %s",
                     registered)

    def __init__(self, config):  # noqa: ANN001
        orig_init(self, config)
        try:
            _register_gdn_patterns(self, config)
        except Exception as exc:  # API drift -> stay unfused
            log.warning("vtl: GDN gated-norm fusion register failed (%s); unfused", exc)

    Pass.__init__ = __init__
    Pass.__vtl_gdn_rmsnorm_patched__ = True
    return True


# ---------------------------------------------------------------------------------------
# Eager fallback (round-1.1's monkeypatch, kept for A/B and for sigmoid-gated configs).
# ---------------------------------------------------------------------------------------

def _install_gated_rmsnorm_eager() -> bool:
    """Monkeypatch RMSNormGated.forward to the standalone kernel for the exact case it
    supports (norm_before_gate=True, group_size=None). Defensive: any unknown
    signature/shape -> original."""
    import torch

    from vllm.model_executor.layers.layernorm import RMSNormGated  # guarded by caller

    if already_patched(RMSNormGated, "forward"):
        return True

    orig_forward = RMSNormGated.forward

    def forward(self, x, gate=None):  # noqa: ANN001
        try:
            weight = self.weight
            eps = getattr(self, "eps", getattr(self, "variance_epsilon", 1e-6))
            group = getattr(self, "group_size", None)
            norm_before_gate = getattr(self, "norm_before_gate", None)
            is_silu = _gate_is_silu(getattr(self, "activation", "silu"))
            ok = (
                gate is not None
                and norm_before_gate is True  # our kernel = norm-then-gate; None/False -> bail
                and (group is None or group == x.shape[-1])  # no sub-grouping
                and x.is_cuda
                and x.is_contiguous()
                and gate.is_contiguous()
                and gate.shape == x.shape
                and weight.dtype in (torch.float32, x.dtype)
                and weight.numel() == x.shape[-1]
            )
            if not ok:
                return orig_forward(self, x, gate)
            out = torch.empty_like(x)
            torch.ops.vllm_cuda.gated_rmsnorm(out, x, gate, weight, float(eps), is_silu)
            return out
        except Exception:
            return orig_forward(self, x, gate)

    RMSNormGated.forward = mark_patched(forward, orig_forward)
    return True


@register_patch("gdn_kernels", default=True)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schemas we reference

    try:
        import vtl._C  # noqa: F401  -- registers vllm_cuda::gated_rmsnorm(+ quant variants)
    except Exception as exc:
        log.warning("vtl: vtl._C not importable (%s); GDN kernels rely on NVRTC/stock tiers", exc)

    have_aot = hasattr(torch.ops.vllm_cuda, AOT_OP)
    nvrtc_armed = os.environ.get("VTL_NVRTC", "").strip().lower() in _TRUTHY
    if not have_aot and not nvrtc_armed:
        log.warning("vtl: no gdn AOT op and NVRTC off; skipping GDN kernels")
        return

    _register_op()

    installed_fusion = False
    if _fusion_enabled():
        try:
            installed_fusion = _register_fusion_pattern()
            if installed_fusion:
                log.info("vtl: GDN gated-RMSNorm+group-quant FUSION installed "
                         "(VERIFY 'Replaced N patterns' N>0 on the H200)")
        except Exception as exc:
            log.warning("vtl: GDN gated-norm fusion install failed (%s); trying eager", exc)

    if not installed_fusion and have_aot:
        try:
            if _install_gated_rmsnorm_eager():
                log.info("vtl: GDN gated-RMSNorm eager kernel installed (VERIFY parity on H200)")
        except Exception as exc:
            log.warning("vtl: GDN gated-RMSNorm install failed (%s); stock kernels kept", exc)


def _self_check() -> None:
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "gdn_kernels")
    # Enabled by default per project decision (2026-08-17) -- the shipped compose pins
    # VTL_ENABLE_GDN_KERNELS=1, and the registry default now says the same thing rather
    # than relying on the compose line to carry it. The safety story is not the gate: the
    # fusion pass is a no-op on a graph it does not match, and the eager arm degrades
    # per-op to stock RMSNormGated, so an unfitting model loses speed, not correctness.
    assert patch.default is True, "shipped ENABLED; the per-op fallback is the safety net"
    assert is_enabled(patch) is True

    os.environ["VTL_ENABLE_GDN_KERNELS"] = "0"
    assert is_enabled(patch) is False, "the gate must still be able to turn it off"
    os.environ["VTL_ENABLE_GDN_KERNELS"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GDN_KERNELS")

    assert _fusion_enabled() is True
    os.environ["VTL_GDN_RMSNORM_FUSION"] = "0"
    assert _fusion_enabled() is False
    os.environ.pop("VTL_GDN_RMSNORM_FUSION")

    assert _gate_is_silu("silu") is True
    assert _gate_is_silu("swish") is True
    assert _gate_is_silu("sigmoid") is False

    # -- pure shape math, the part `make check` can prove without a GPU --
    # The model geometry: 64 heads x 128, group 128 -> one group per head.
    d = _nvrtc_defines(128, 64, True, False)
    assert d == {"DV": 128, "HV": 64, "GROUP": 128, "THREADS": 256, "IS_SILU": 1, "W_FP32": 0}
    assert _nvrtc_defines(128, 64, False, True)["IS_SILU"] == 0
    assert _nvrtc_defines(128, 64, False, True)["W_FP32"] == 1
    assert _nvrtc_supported(128, 64) is True
    assert _nvrtc_supported(100, 64) is False, "DV must be the quant group"
    assert _nvrtc_supported(0, 64) is False
    assert _nvrtc_supported(128, 0) is False
    # 16 lanes/row, 256 threads -> 16 rows per block.
    assert _nvrtc_grid(1, 128) == (1, 1, 1)
    assert _nvrtc_grid(16, 128) == (1, 1, 1)
    assert _nvrtc_grid(17, 128) == (2, 1, 1)
    assert _nvrtc_grid(64 * 7, 128) == (28, 1, 1)
    assert _scale_shape(7, 64 * 128) == (7, 64), "one scale per (token, head)"

    # Without vLLM/torch, apply() must raise rather than corrupt state -- apply_all()
    # isolates it and the server falls back to the stock kernels.
    os.environ["VTL_ENABLE_GDN_KERNELS"] = "1"
    try:
        apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_ENABLE_GDN_KERNELS", None)

    print("gdn_kernels self-check ok")


if __name__ == "__main__":
    _self_check()
