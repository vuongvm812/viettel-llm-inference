"""One fused NVRTC launch for the GDN decode step, replacing a Triton pair.

WHAT IT REPLACES. On a pure non-spec decode batch, ``QwenGatedDeltaNetAttention.
_forward_core_decode_non_spec`` (vllm/model_executor/layers/mamba/gdn/
qwen_gdn_linear_attn.py) runs exactly two kernels per linear-attention layer:

    causal_conv1d_update(mixed_qkv, conv_state, ...)          # Triton, mamba/ops
    fused_recurrent_gated_delta_rule_packed_decode(...)       # Triton, fla/ops

36 GDN layers x 2 launches = 72 launches per token, and the pair round-trips the whole
[T, 12288] conv output through HBM in between (kernel 1 writes it back over ``mixed_qkv``,
kernel 2 reads it). ``vtl/kernels/gdn_decode_step.cu`` is the two of them fused into one
launch that keeps that intermediate in shared memory: half the launches, and the 12288-wide
write + read per token per layer is gone. The fp32 SSM state (4 MB/seq/layer) is read and
written exactly once either way -- that traffic is the floor, and it is what makes this
HBM-bound rather than launch-bound at the concurrency the trace shows.

WHAT IT DOES NOT TOUCH. Prefill, spec/MTP decode, and mixed batches all take the stock
path, untouched, because the fast path is keyed on the metadata the stock method was
handed -- ``spec_sequence_masks is None and num_prefills == 0 and num_decodes > 0``, which
is precisely the condition ``_forward_core`` already used to route here. Padded rows of a
full-cudagraph decode batch carry ``NULL_BLOCK_ID`` (0) and the kernel early-outs on them
exactly as both Triton kernels do, so the fast path stays engaged for the graph-replay
shapes rather than falling off at the one place it matters.

MECHANISM (HANDOFF §4.3 shape, same seam as nvrtc_block_quant). ``apply()`` installs a
wrapper on the METHOD -- vLLM's op, graph and metadata are untouched; nothing about the
compiled FX graph changes, because this method is already inside the opaque
``qwen_gdn_attention_core`` custom op. The wrapper is inert until ``_arm()`` succeeds, and
``_arm()`` runs from a ``BaseModelLoader.load_model`` wrapper: geometry comes from the
layers that actually got built (``head_k_dim``/``head_v_dim``/``num_*_heads``/tp/
``conv_kernel_size``), the kernel is compiled for it, and only a successful compile arms
the fast path. That keeps NVRTC off both the serving path and cudagraph capture. It also
pre-materializes the fp32 ``A_log``/``dt_bias`` views at load, so the first decode -- which
may BE a graph capture -- allocates nothing.

``compile_kernel`` returning None (NVRTC off, no cuda-python, bad source, no device) leaves
the stock Triton pair in place and the wrapper is a straight passthrough. Anything the fast
path does not recognise at call time -- a geometry that is not the compiled one, a dtype or
stride outside the envelope, a non-silu activation -- also falls through to stock, once per
distinct reason and with a warning, so a surprise costs a log line rather than an answer.

Gate: ``VTL_ENABLE_GDN_DECODE_STEP=1`` (default OFF until A/B'd) **and** ``VTL_NVRTC=1``
(the layer-wide switch). Numerics are the Triton pair's op for op -- see the header of
vtl/kernels/gdn_decode_step.cu for the two tolerated fp32 divergences (reduction order,
transcendental lowering) and bench/test_gdn_decode_step.py for the bound on them.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.gdn_decode_step")

NAME = "gdn_decode_step"
KERNEL = "gdn_decode_step"          # vtl/kernels/gdn_decode_step.cu, entry of the same name

LAYER_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
LAYER_CLASS = "QwenGatedDeltaNetAttention"
METHOD = "_forward_core_decode_non_spec"

CONV_WIDTH = 4       # the kernel's tap sequence is written for width 4
HEAD_K_DIM = 128     # phase 3 is 32 lanes x float4; see the .cu's static_assert
NULL_BLOCK_ID = 0    # vllm/v1/attention/backends/utils.py

# Filled by _arm(); read by the wrapper. `geom` is (dk, dv, hk, hv, conv).
_state: dict = {
    "armed": False,
    "installed": False,
    "geom": None,
    "launcher": None,
}
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args, **kwargs) -> None:
    """One line per distinct reason. A per-token warning would be its own outage."""
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args, **kwargs)


def _disarm() -> None:
    _state["launcher"] = None
    _state["installed"] = False


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def _geometry_ok(dk: int, dv: int, hk: int, hv: int, conv: int) -> bool:
    """Can the shipped kernel be specialized to this layer geometry?

    Mirrors the .cu's static_asserts one for one: CONV=4 taps, THREADS==DK==DV (one thread
    per q/k channel and per v channel), DK==128 because the phase-3 dot is a 32-lane x
    float4 ladder, whole warps inside one block, and HV a multiple of HK with the sibling
    count (HV/HK, computed by the first threads) fitting the block.
    """
    if conv != CONV_WIDTH:
        return False
    if dk != HEAD_K_DIM or dv != dk:
        return False
    if dk % 32 != 0 or dk > 1024:
        return False
    if hk <= 0 or hv <= 0 or hv % hk != 0:
        return False
    return (hv // hk) <= dk


def _defines(geom: tuple[int, int, int, int, int]) -> dict:
    dk, dv, hk, hv, conv = geom
    return {"DK": dk, "DV": dv, "HK": hk, "HV": hv, "CONV": conv, "THREADS": dk}


def _conv_dim(geom: tuple[int, int, int, int, int]) -> int:
    """[q | k | v] packed width == the depthwise conv's channel count."""
    dk, dv, hk, hv, _ = geom
    return 2 * hk * dk + hv * dv


def _layer_geometry(layer):  # noqa: ANN001
    """(dk, dv, hk, hv, conv) for one built layer, per-rank (i.e. already TP-sharded)."""
    tp = int(getattr(layer, "tp_size", 1) or 1)
    return (
        int(layer.head_k_dim),
        int(layer.head_v_dim),
        int(layer.num_k_heads) // tp,
        int(layer.num_v_heads) // tp,
        int(layer.conv_kernel_size),
    )


def _decode_batch_ok(md) -> bool:  # noqa: ANN001
    """Is this metadata a PURE non-spec decode batch?

    Deliberately the same predicate ``_forward_core`` uses to route into the method we
    wrap, re-checked here so the fast path cannot outlive a future re-route. Kept free of
    torch so the off-box self-check can exercise it against fakes.

    ``num_actual_tokens > num_decodes`` is NOT a rejection: a full-cudagraph decode batch
    is token-padded and the builder fills the padded state indices with NULL_BLOCK_ID,
    which both Triton kernels and ours skip. Rejecting it would turn the fast path off for
    exactly the shape it exists for.
    """
    if getattr(md, "spec_sequence_masks", None) is not None:
        return False
    if int(getattr(md, "num_prefills", 1) or 0) != 0:
        return False
    if int(getattr(md, "num_spec_decodes", 0) or 0) != 0:
        return False
    num_decodes = int(getattr(md, "num_decodes", 0) or 0)
    if num_decodes <= 0:
        return False
    tokens = int(getattr(md, "num_actual_tokens", 0) or 0)
    return tokens > 0 and tokens >= num_decodes


# --------------------------------------------------------------------------------------
# launch planning: pure validation, no mutation
# --------------------------------------------------------------------------------------

def _f32_1d(layer, name: str):  # noqa: ANN001
    """A contiguous fp32 view of a [HV] gating parameter, cached on the layer.

    The Triton kernel loads A_log/dt_bias and immediately does ``.to(tl.float32)``, so a
    bf16 checkpoint value and its fp32 upcast are the same number; the kernel here takes
    ``const float*``. Materialized once at arm time -- never inside a decode step, where an
    allocation would be illegal under cudagraph capture.
    """
    import torch

    cached = getattr(layer, f"_vtl_gdn_{name}_f32", None)
    if cached is not None:
        return cached
    src = getattr(layer, name)
    if src.dtype == torch.float32 and src.is_contiguous():
        # .detach(), NOT src: A_log/dt_bias are nn.Parameters, and nn.Module.__setattr__
        # would REGISTER one stored under a new name -- the parameter would then show up
        # twice in named_parameters()/state_dict(). A plain tensor lands in __dict__.
        out = src.detach()
    else:
        out = src.detach().float().contiguous()
    setattr(layer, f"_vtl_gdn_{name}_f32", out)
    return out


def _plan(layer, mixed_qkv, b, a, core_attn_out, attn_metadata):  # noqa: ANN001
    """Validate everything, then return ``(grid, block, args)``. None => use stock.

    NOTHING here mutates any tensor. That is what makes the wrapper's fallback safe: if
    this returns None, or if the single launch that follows fails to submit, no state has
    moved and the stock pair can run from scratch.
    """
    import torch

    from vtl import nvrtc

    geom = _state["geom"]
    if geom is None or _state["launcher"] is None:
        return None
    if not _decode_batch_ok(attn_metadata):
        return None
    if _layer_geometry(layer) != geom:
        _warn_once("geom", "vtl: gdn_decode_step: layer geometry %s != compiled %s; stock",
                   _layer_geometry(layer), geom)
        return None

    dk, dv, hk, hv, conv = geom
    conv_dim = _conv_dim(geom)
    tokens = int(attn_metadata.num_actual_tokens)

    # -- the conv is silu-only in the kernel; stock takes activation in {silu, swish} --
    activation = getattr(layer, "activation", None)
    if activation is True:
        activation = "silu"
    if activation not in ("silu", "swish"):
        _warn_once("act", "vtl: gdn_decode_step: activation %r is not silu; stock", activation)
        return None

    # -- kv cache: exactly the two views the stock method builds --
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    except Exception:
        return None
    kv_cache = layer.kv_cache
    conv_state = kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
    ssm_state = kv_cache[1]

    idx = getattr(attn_metadata, "non_spec_state_indices_tensor", None)
    if idx is None:
        return None

    weight = layer.conv1d.weight
    if weight.dim() != 3 or weight.size(1) != 1:
        return None
    conv_w = weight.view(weight.size(0), weight.size(2))
    conv_bias = layer.conv1d.bias

    bf16 = torch.bfloat16
    ok = (
        mixed_qkv.dim() == 2
        and mixed_qkv.size(0) >= tokens
        and mixed_qkv.size(1) == conv_dim
        and mixed_qkv.dtype is bf16
        and mixed_qkv.stride(1) == 1
        # a dtype mismatch would make stock's `x.to(conv_state.dtype)` COPY, so the conv
        # output would not land back in mixed_qkv at all -- a different program.
        and conv_state.dtype is bf16
        and conv_state.dim() == 3
        and conv_state.size(1) == conv_dim
        and conv_state.size(2) >= conv - 1
        and conv_w.shape == (conv_dim, conv)
        and conv_w.dtype is bf16
        and (conv_bias is None or (conv_bias.dtype is bf16
                                   and conv_bias.numel() == conv_dim
                                   and conv_bias.stride(0) == 1))
        and a.dim() == 2 and a.size(0) >= tokens and a.size(1) == hv
        and b.dim() == 2 and b.size(0) >= tokens and b.size(1) == hv
        and a.dtype is bf16 and b.dtype is bf16
        and a.stride(1) == 1 and b.stride(1) == 1
        # ssm_state [lines, HV, DV, DK]: stock indexes i_hv*V*K + v*K + k, i.e. it assumes
        # the inner three dims are contiguous. Same assumption, checked rather than hoped.
        and ssm_state.dim() == 4
        and ssm_state.dtype is torch.float32
        and ssm_state.shape[-3:] == (hv, dv, dk)
        and ssm_state.stride(3) == 1
        and ssm_state.stride(2) == dk
        and ssm_state.stride(1) == dv * dk
        and core_attn_out.dim() == 3
        and core_attn_out.size(0) >= tokens
        and core_attn_out.shape[1:] == (hv, dv)
        and core_attn_out.dtype is bf16
        and core_attn_out[:tokens].is_contiguous()
        and idx.dim() == 1
        and idx.dtype is torch.int32
        and idx.numel() >= tokens
    )
    if not ok:
        _warn_once("envelope",
                   "vtl: gdn_decode_step: call outside the compiled envelope; stock pair "
                   "(mixed_qkv=%s/%s ssm=%s/%s out=%s/%s)",
                   tuple(mixed_qkv.shape), mixed_qkv.dtype,
                   tuple(ssm_state.shape), ssm_state.dtype,
                   tuple(core_attn_out.shape), core_attn_out.dtype)
        return None

    a_log = _f32_1d(layer, "A_log")
    dt_bias = _f32_1d(layer, "dt_bias")
    if a_log.numel() != hv or dt_bias.numel() != hv:
        return None

    args = nvrtc.pack_args(
        mixed_qkv.data_ptr(),
        conv_state.data_ptr(),
        conv_w.data_ptr(),
        conv_bias.data_ptr() if conv_bias is not None else 0,
        a.data_ptr(),
        b.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        ssm_state.data_ptr(),
        core_attn_out.data_ptr(),
        idx.data_ptr(),
        ("f", float(dk) ** -0.5),          # scale, as the stock call site computes it
        ("l", mixed_qkv.stride(0)),
        ("l", a.stride(0)),
        ("l", b.stride(0)),
        ("l", conv_state.stride(0)),
        ("l", conv_state.stride(1)),
        ("l", conv_state.stride(2)),
        ("l", conv_w.stride(0)),
        ("l", conv_w.stride(1)),
        ("l", ssm_state.stride(0)),
        # NOT 1 in the eager path: the builder hands out block_table_tensor[:, 0].
        ("l", idx.stride(0)),
    )
    # One block per KEY head, not per value head: the q/k conv channels and their rotating
    # conv state are shared by the HV/HK sibling value heads, and sibling blocks have no
    # way to order their state rotation against each other. See the .cu header.
    return (hk, tokens, 1), (dk, 1, 1), args


# --------------------------------------------------------------------------------------
# arming + install
# --------------------------------------------------------------------------------------

def _gdn_layers(vllm_config):  # noqa: ANN001
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

        return list(get_layers_from_vllm_config(vllm_config, GatedDeltaNetAttention).values())
    except Exception:
        return []


def _arm(vllm_config) -> None:
    """Once, after the model loads: read geometry off the built layers, compile, arm."""
    if _state["armed"]:
        return
    _state["armed"] = True

    from vtl import nvrtc

    layers = _gdn_layers(vllm_config)
    if not layers:
        log.info("vtl: gdn_decode_step: no GatedDeltaNet layers; stock pair stands")
        return

    geoms = set()
    for layer in layers:
        try:
            geoms.add(_layer_geometry(layer))
        except Exception:
            log.info("vtl: gdn_decode_step: layer geometry unreadable; stock pair stands")
            return
    if len(geoms) != 1:
        log.info("vtl: gdn_decode_step: %d distinct GDN geometries %s; stock pair stands",
                 len(geoms), sorted(geoms))
        return
    geom = geoms.pop()
    if not _geometry_ok(*geom):
        log.info("vtl: gdn_decode_step: geometry (dk,dv,hk,hv,conv)=%s outside the "
                 "kernel's envelope; stock pair stands", geom)
        return

    launcher = nvrtc.compile_kernel(KERNEL, _defines(geom), entry=KERNEL)
    if launcher is None:
        log.info("vtl: gdn_decode_step: NVRTC compile unavailable/failed; stock pair stands")
        return

    # Materialize the fp32 gating views now: the first decode may be a cudagraph capture,
    # where an allocation (or an h2d cast) would be illegal.
    for layer in layers:
        try:
            _f32_1d(layer, "A_log")
            _f32_1d(layer, "dt_bias")
        except Exception:
            log.info("vtl: gdn_decode_step: A_log/dt_bias not materializable; stock stands")
            return

    _state.update(geom=geom, launcher=launcher, installed=True)
    # The one-line tier log a `make verify`-style grep can key on.
    log.info(
        "vtl: gdn decode-step tier active: causal_conv1d_update + "
        "fused_recurrent_gated_delta_rule_packed_decode -> 1 NVRTC launch "
        "(DK=%d DV=%d HK=%d HV=%d CONV=%d, %d layers)",
        *geom, len(layers),
    )


@register_patch(NAME, default=False)
def apply() -> None:
    import importlib

    from vtl import nvrtc

    if not nvrtc.enabled():
        log.info("vtl: gdn_decode_step selected but VTL_NVRTC is off; no-op")
        return

    mod = importlib.import_module(LAYER_MODULE)
    cls = getattr(mod, LAYER_CLASS)
    if already_patched(cls, METHOD, patch=NAME):
        return
    orig = getattr(cls, METHOD)

    # Keyword names must match stock's: _forward_core calls this with kwargs only.
    def _forward_core_decode_non_spec(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
        if _state["launcher"] is not None:
            plan = None
            try:
                plan = _plan(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
            except Exception:
                _warn_once("plan", "vtl: gdn_decode_step: planning raised; disarming",
                           exc_info=True)
                _disarm()
            if plan is not None:
                grid, block, args = plan
                try:
                    _state["launcher"](grid=grid, block=block, args=args)
                    return None
                except Exception:
                    # cuLaunchKernel only raises on a SUBMISSION error (bad config, dead
                    # context), i.e. before any thread ran, so no state has moved and the
                    # stock pair below is still a correct from-scratch run. An error the
                    # kernel itself raises is asynchronous and surfaces elsewhere.
                    _warn_once("launch", "vtl: gdn_decode_step: launch failed; disarming "
                                         "and falling back to the stock Triton pair",
                               exc_info=True)
                    _disarm()

        return orig(
            self,
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            attn_metadata=attn_metadata,
        )

    setattr(cls, METHOD, mark_patched(_forward_core_decode_non_spec, orig, patch=NAME))

    # Compile at model load, not at first decode: NVRTC must not run on the serving path
    # and must not run inside a cudagraph capture.
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch=NAME):
        return
    orig_load = BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, *args, **kwargs):
        model = orig_load(self, vllm_config, model_config, *args, **kwargs)
        try:
            _arm(vllm_config)
        except Exception:
            log.exception("vtl: gdn_decode_step arming failed; stock pair stands")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, orig_load, patch=NAME)
    log.info("vtl: gdn_decode_step armed (compiles + engages at model load)")


def _self_check() -> None:
    """Runs anywhere: no GPU, no torch, no vLLM."""
    import os

    from vtl import nvrtc
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == NAME)
    assert patch.default is False, "unproven optimization: default OFF"
    assert is_enabled(patch) is False
    os.environ["VTL_ENABLE_GDN_DECODE_STEP"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GDN_DECODE_STEP")

    # -- geometry envelope: Qwen3.5-122B-A10B must pass, the ways a foreign model would
    #    break the kernel must not --
    qwen = (128, 128, 16, 64, 4)
    assert _geometry_ok(*qwen) is True
    assert _conv_dim(qwen) == 12288, _conv_dim(qwen)
    assert _defines(qwen) == {"DK": 128, "DV": 128, "HK": 16, "HV": 64,
                              "CONV": 4, "THREADS": 128}
    assert _geometry_ok(128, 128, 16, 64, 3) is False, "the tap sequence is width 4 only"
    assert _geometry_ok(256, 256, 16, 64, 4) is False, "phase 3 is 32 lanes x float4"
    assert _geometry_ok(128, 64, 16, 64, 4) is False, "THREADS is both DK and DV"
    assert _geometry_ok(128, 128, 16, 60, 4) is False, "HV must be a multiple of HK"
    assert _geometry_ok(128, 128, 0, 64, 4) is False
    # A key head with more sibling value heads than threads has nowhere to compute gating.
    assert _geometry_ok(128, 128, 1, 256, 4) is False

    # -- engage condition. Only a pure non-spec decode batch, and token padding is NOT a
    #    rejection (it is exactly the cudagraph-replay shape this exists for). --
    class MD:
        def __init__(self, **kw):
            self.spec_sequence_masks = None
            self.num_prefills = 0
            self.num_decodes = 4
            self.num_spec_decodes = 0
            self.num_actual_tokens = 4
            self.__dict__.update(kw)

    assert _decode_batch_ok(MD()) is True
    assert _decode_batch_ok(MD(num_actual_tokens=32)) is True, "cudagraph token padding"
    assert _decode_batch_ok(MD(spec_sequence_masks=object())) is False, "spec/MTP batch"
    assert _decode_batch_ok(MD(num_prefills=1)) is False, "mixed prefill batch"
    assert _decode_batch_ok(MD(num_spec_decodes=2)) is False
    assert _decode_batch_ok(MD(num_decodes=0)) is False
    assert _decode_batch_ok(MD(num_actual_tokens=0)) is False
    assert _decode_batch_ok(MD(num_actual_tokens=2)) is False, "fewer tokens than decodes"
    assert _decode_batch_ok(object()) is False, "an unknown metadata shape must not engage"

    # -- the kernel must ship, and its entry name must match compile_kernel(name=...) --
    src = nvrtc.load_source(KERNEL)
    assert src, "vtl/kernels/gdn_decode_step.cu missing from the package"
    assert f'extern "C" __global__ void __launch_bounds__(THREADS) {KERNEL}(' in src
    for macro in ("DK", "DV", "HK", "HV", "CONV", "THREADS"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    # The stride argument the eager path needs (block_table_tensor[:, 0] is not stride-1).
    assert "indices_stride" in src

    # -- every define set is its own cubin, and dict order is not part of the identity --
    keys = {
        nvrtc.cache_key(src, _defines(qwen), "90a", "12.8"),
        nvrtc.cache_key(src, _defines((128, 128, 8, 32, 4)), "90a", "12.8"),
        nvrtc.cache_key(src, _defines((128, 128, 16, 64, 4)), "90", "12.8"),
    }
    assert len(keys) == 3, "cache keys must be distinct across define sets and arches"
    d = _defines(qwen)
    perm = dict(reversed(list(d.items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(src, d, "90a", "12.8")

    # -- while NVRTC is disabled, compiling is a no-op and apply() installs nothing --
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(KERNEL, _defines(qwen)) is None
    apply()   # vLLM may be absent; with VTL_NVRTC off this must return cleanly
    assert _state["installed"] is False and _state["launcher"] is None

    # -- with NVRTC on but vLLM absent, apply() may raise (registry isolates it) but must
    #    not leave a half-armed module --
    os.environ["VTL_NVRTC"] = "1"
    try:
        apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_NVRTC", None)
    assert _state["installed"] is False and _state["launcher"] is None

    print("gdn_decode_step self-check ok")


if __name__ == "__main__":
    _self_check()
