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

Gate: ``VTL_ENABLE_GDN_DECODE_STEP`` (default ON since 2026-08-17) **and** ``VTL_NVRTC=1``
(the layer-wide switch). Numerics are the Triton pair's op for op -- see the header of
vtl/kernels/gdn_decode_step.cu for the two tolerated fp32 divergences (reduction order,
transcendental lowering) and bench/test_gdn_decode_step.py for the bound on them.

=========================================================================================
THE FUSED EPILOGUE (gated RMSNorm + group-128 fp8 quant, folded into the same launch)
=========================================================================================

NO NEW GATE. This rides ``VTL_ENABLE_GDN_DECODE_STEP``: arming compiles BOTH kernel
variants (``-DFUSED_EPILOGUE=1`` preferred, ``=0`` as the fallback), and the fused one is
launched only for layers where every epilogue input resolved AND the boot parity gate
passed. Anything else runs the unfused fast path, and anything outside that runs stock.

WHAT THE SEAM CAN AND CANNOT SEE (v0.25.0, qwen_gdn_linear_attn.py). The method we wrap is
handed ``(mixed_qkv, b, a, core_attn_out, attn_metadata)`` -- no gate, no norm weight
(:1644-1652). The norm+quant stage lives in a DIFFERENT method, ``_output_projection``
(:851-869), which runs AFTER the opaque ``qwen_gdn_attention_core`` custom op returns and
is therefore inside the torch.compile FX graph -- it is the very region
``vtl/patches/gdn_kernels.py`` registers its inductor fusion pattern against. Widening the
wrap to cover it would move this patch out of the opaque-op seam and into traced code,
where a runtime branch on ``attn_metadata`` is not even legal. So the wrap does NOT move.
Three things make the fusion possible anyway:

  * ``z`` IS reachable. ``forward_cuda`` builds it as ``mixed_qkv, z = mixed_qkvz.split(
    [qkv_size, z_size], dim=-1)`` (:933-935) -- both are VIEWS of one buffer, so the gate is
    the columns immediately after ``mixed_qkv``'s in the same rows. ``_z_ptr`` recovers it
    from ``mixed_qkv``'s own stride/offset/storage extent and refuses if the provenance is
    not that split (the Qwen3-Next interleaved branch builds ``mixed_qkv`` with
    ``torch.cat``, whose row stride is the conv width -- it fails the check and never
    claims).
  * the norm weight, eps and activation live on the LAYER (``layer.norm``, :536-543).
  * the destination is OURS: per-layer persistent fp8 + fp32-scale staging buffers, sized at
    arm time. No allocation, no sync and no stream juggling on the launch path, so the
    single ``cuLaunchKernel`` stays capturable and bakes into nstep's burst graphs exactly
    as the unfused one does.

NOT APPLIED TWICE, AND NEVER APPLIED ZERO TIMES. The fused kernel stops writing bf16
``core_attn_out`` -- dropping that [T, HV*DV] write and the epilogue's matching read is the
point -- so it may only do so when the single downstream reader is known to take the fp8
instead. That is an explicit two-sided handshake with ``gdn_kernels``:

  * ``gdn_kernels``' fused op calls ``note_epilogue_consumer(norm_weight.data_ptr())`` on
    every invocation. Reaching that line PROVES, for that specific layer, that the fused
    norm+quant op is what consumes the core output (it fires during the profile run, before
    any decode). If the inductor pattern never matched, or the stock RMSNormGated path is
    what runs, the set stays empty and we never claim.
  * a claiming launch then publishes ``(core_attn_out.data_ptr()) -> (fp8, scales)``, and
    the same op consumes it by pointer identity instead of re-running the norm+quant.

Reaching that op is necessary but not sufficient -- a copy inserted between the core op and
the norm (functionalization, a non-view reshape, a future refactor) would break the pointer
identity, and the failure mode of guessing wrong is SILENT: zeros in, zeros out. So the
per-layer decision is a three-state latch that makes the system prove the handoff to itself:

    None -> "probe"   the fused kernel runs with ``out`` NON-NULL: it writes fp8 AND bf16,
                      a strict superset of the unfused build. Correct whichever path the
                      downstream actually takes, and the claim's fate is observable.
         -> "fused"   after PROBE_CONSUMES claims were actually consumed: ``out = nullptr``,
                      and the write is finally gone.
         -> "plain"   a published claim was still outstanding when the next one arrived
                      (nobody is consuming), or an input stopped resolving. Permanent.

"plain" is permanent and "fused" is only ever reached forward, because a cudagraph bakes
whichever pair of decisions was live at capture: a later probe->fused flip is harmless (the
graph that baked the full norm+quant also baked the bf16 write that feeds it), but a
plain->fused flip would leave a replay reading a ``core_attn_out`` nobody writes any more.

RESIDUAL COST, STATED HONESTLY. The consumer still does a ``copy_`` of the staged fp8 into
the buffer inductor allocated inside the graph, because that buffer does not exist yet when
the opaque op runs. Per token per layer the epilogue traffic goes from
16 KB (bf16 write) + 16 KB (x read) + 16 KB (z read) + 8 KB (fp8 write)
to 16 KB (z read) + 8 KB (fp8 write) + 8 KB + 8 KB (the copy) -- the bf16 round-trip is
gone, the norm+quant arithmetic runs once instead of once-per-stage, and what is left is a
straight fp8 copy. Removing that last copy means replacing out_proj's quant seam so the GEMM
reads our buffer directly; that is a separate change and is deliberately not made here.
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

# ---- fused epilogue ----------------------------------------------------------------
GROUP = 128          # kFp8Dynamic128Sym, this checkpoint's activation quant group
EVEC = 8             # bf16 per 16-byte z load in the epilogue; mirrors the .cu
FP8_MAX = 448.0
GROUP_QUANT_EPS = 1e-10
# Claims that must be observably CONSUMED before the fused path stops writing bf16
# core_attn_out. Two, not one: one consume could be a coincidence of a recycled pointer;
# two come from two independent steps. Each probe step is already fully correct, so the only
# cost of a higher number is keeping the write a little longer.
PROBE_CONSUMES = 2
# Tokens the per-layer staging buffers serve. A wider batch simply does not claim the fused
# path (it takes the unfused one), so this is a memory cap, not a correctness bound:
# rows * HV*DV bytes of fp8 + rows * HV * 4 B of scales, per layer. 256 rows at the
# production geometry is ~2.1 MB/layer, ~76 MB across 36 layers.
MAX_STAGING_TOKENS = 256

# Filled by _arm(); read by the wrapper. `geom` is (dk, dv, hk, hv, conv).
_state: dict = {
    "armed": False,
    "installed": False,
    "geom": None,
    "launcher": None,          # the unfused (-DFUSED_EPILOGUE=0) launcher
    "launcher_fused": None,    # the fused (-DFUSED_EPILOGUE=1) launcher, or None
    "fused_defines": None,
    "staging_rows": 0,
}
_warned: set[str] = set()

# ---- the two-sided handshake with vtl/patches/gdn_kernels.py -------------------------
# Norm-weight data_ptrs whose norm+quant stage is served by gdn_kernels' fused op. A ptr is
# in here IFF that op actually ran for that layer, which is the only available proof that
# the fp8 the fused kernel writes is what gets consumed. Read by _plan, written by
# note_epilogue_consumer below.
EPILOGUE_CONSUMERS: set[int] = set()
# core_attn_out.data_ptr() -> (rows, hidden, fp8, scales) published by a claiming launch.
# Bounded: the buffers are recycled, so the same handful of pointers repeat. Under piecewise
# cudagraph REPLAY the consumer's Python never runs (its copy_ is baked into the graph), so
# entries are overwritten rather than popped -- that is normal, not a leak.
_claims: dict = {}
_MAX_CLAIMS = 64


def _warn_once(key: str, msg: str, *args, **kwargs) -> None:
    """One line per distinct reason. A per-token warning would be its own outage."""
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args, **kwargs)


def _disarm() -> None:
    _state["launcher"] = None
    _state["launcher_fused"] = None
    _state["installed"] = False
    _claims.clear()


# --------------------------------------------------------------------------------------
# the handshake gdn_kernels uses. Both functions are the PUBLIC surface of this module and
# must stay import-safe: gdn_kernels calls them from inside a CUDA op impl.
# --------------------------------------------------------------------------------------

def note_epilogue_consumer(weight_ptr: int) -> None:
    """gdn_kernels' fused norm+quant op ran for the layer owning this norm weight.

    That is the whole proof the fused decode step needs: the op is on the graph for this
    layer, so a claim it publishes will be seen. Called on EVERY invocation (it is a set
    add on an int -- cheaper than deciding whether to call it) and never raises.
    """
    try:
        EPILOGUE_CONSUMERS.add(int(weight_ptr))
    except Exception:  # pragma: no cover -- a fake weight in a test
        pass


def take_epilogue(input, num_heads: int):  # noqa: ANN001
    """Staged ``(fp8, scales)`` for this exact core output, or None. POPS the claim.

    Matched on ``input.data_ptr()``: ``_output_projection`` reaches the op through
    ``core_attn_out.reshape(-1, head_v_dim)``, which is a VIEW of the very buffer the fused
    launch was planned against, so the pointer is an exact identity -- not a heuristic. The
    row count is re-derived from the op's own view and cross-checked, so a shape the claim
    was not made for falls through to the full norm+quant instead of silently reusing it.

    A POINTER match pops the claim even when that cross-check then rejects it: a claim its
    own consumer cannot use is stale, and leaving it in the book would offer it to a LATER
    call on the same recycled buffer.
    """
    if not _claims:
        return None
    try:
        entry = _claims.pop(int(input.data_ptr()), None)
        if entry is None:
            return None
        rows, hidden, fp8, scales, epi = entry
        heads = int(num_heads)
        if heads <= 0 or int(input.shape[0]) != rows * heads:
            return None
        if int(input.shape[-1]) * heads != hidden:
            return None
        if epi is not None:
            # The handoff landed. This is the ONLY evidence that lets the layer stop
            # writing bf16 core_attn_out -- see the three-state latch in the docstring.
            epi["consumed"] = epi.get("consumed", 0) + 1
            epi["pending"] = False
        return fp8, scales
    except Exception:  # pragma: no cover
        return None


def _publish_claim(ptr: int, rows: int, hidden: int, fp8, scales, epi=None) -> None:  # noqa: ANN001
    if len(_claims) >= _MAX_CLAIMS:
        # Only reachable if the consumer's Python stopped running (piecewise replay) AND the
        # output buffers stopped repeating. Drop the book rather than grow it; the worst
        # case is one step of full norm+quant.
        _claims.clear()
    _claims[int(ptr)] = (int(rows), int(hidden), fp8, scales, epi)


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
    """The unfused build: core only, bf16 ``out``, no epilogue symbols compiled at all."""
    dk, dv, hk, hv, conv = geom
    return {"DK": dk, "DV": dv, "HK": hk, "HV": hv, "CONV": conv, "THREADS": dk,
            "FUSED_EPILOGUE": 0}


def _fused_defines(geom: tuple[int, int, int, int, int], is_silu: bool,
                   w_fp32: bool) -> dict:
    """The fused build. A DIFFERENT define set, hence a different cubin and cache key --
    the variant is chosen by which one was compiled, never by a branch inside the kernel."""
    d = _defines(geom)
    d.update({"FUSED_EPILOGUE": 1, "GROUP": GROUP,
              "IS_SILU": 1 if is_silu else 0, "W_FP32": 1 if w_fp32 else 0})
    return d


def _fused_geometry_ok(geom: tuple[int, int, int, int, int]) -> bool:
    """Mirrors the epilogue's own static_asserts on top of ``_geometry_ok``.

    One quant group per head (GROUP == DV), a power-of-two 16-byte lane count inside one
    warp, and GPB sub-groups of ELANES lanes fitting the block -- at GPB=4/ELANES=16 that is
    64 of 128 threads; a key head with 8+ siblings would overflow it and must not compile.
    """
    if not _geometry_ok(*geom):
        return False
    dk, dv, hk, hv, _ = geom
    if dv != GROUP or dv % EVEC != 0:
        return False
    lanes = dv // EVEC
    if lanes < 2 or lanes > 32 or (lanes & (lanes - 1)) != 0:
        return False
    return (hv // hk) * lanes <= dk    # THREADS == dk


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


def _norm_params(layer):  # noqa: ANN001
    """``(weight, eps, is_silu)`` of the layer's RMSNormGated, or None if it is not the
    shape the epilogue reproduces.

    ``norm_before_gate`` MUST be True and ``group_size`` must not sub-divide the row: the
    epilogue computes ``rms_norm(x)*w * act(z)`` over the whole head, which is what
    qwen_gdn_linear_attn.py:536-543 builds, and any other RMSNormGated configuration is a
    different function. ``"sigmoid"`` is the only activation that is not silu/swish --
    ``gdn_kernels._gate_is_silu`` says the same thing, and both feed the same -DIS_SILU.
    """
    norm = getattr(layer, "norm", None)
    weight = getattr(norm, "weight", None)
    if norm is None or weight is None:
        return None
    if getattr(norm, "norm_before_gate", None) is not True:
        return None
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        return None
    group = getattr(norm, "group_size", None)
    if group is not None and int(group) != int(weight.numel()):
        return None
    return weight, float(eps), str(getattr(norm, "activation", "silu")) != "sigmoid"


def _z_ptr(mixed_qkv, conv_dim: int, hidden: int, tokens: int):  # noqa: ANN001
    """``(ptr, row_stride_in_elements)`` of the gate ``z``, or None.

    ``forward_cuda`` splits one projection buffer into ``mixed_qkv`` (conv_dim columns) and
    ``z`` (hidden columns) -- both views of the same rows. So z is at
    ``mixed_qkv.data_ptr() + conv_dim*itemsize`` with mixed_qkv's own row stride, PROVIDED
    that stride is exactly ``conv_dim + hidden``: that equality is what identifies the
    buffer as the split rather than a coincidence. Qwen3-Next's interleaved branch, which
    builds ``mixed_qkv`` with ``torch.cat``, has row stride ``conv_dim`` and is refused
    here -- there is no z behind it to read.

    Also bounds-checked against the real storage extent, because reading z off the end of
    the projection buffer is exactly the kind of mistake a stride assumption makes silently.
    """
    itemsize = mixed_qkv.element_size()
    s0 = int(mixed_qkv.stride(0))
    if s0 != conv_dim + hidden:
        return None
    need = int(mixed_qkv.storage_offset()) + (tokens - 1) * s0 + conv_dim + hidden
    try:
        have = mixed_qkv.untyped_storage().nbytes() // itemsize
    except Exception:
        return None
    if have < need:
        return None
    ptr = mixed_qkv.data_ptr() + conv_dim * itemsize
    # A misaligned 16-byte vector load faults; it does not run slowly.
    if ptr % 16 or (s0 * itemsize) % 16:
        return None
    return ptr, s0


def _epilogue_for(layer):  # noqa: ANN001
    """Per-layer fused-epilogue record built at arm time, or None if the layer has none."""
    return getattr(layer, "_vtl_gdn_epilogue", None)


def _plan(layer, mixed_qkv, b, a, core_attn_out, attn_metadata):  # noqa: ANN001
    """Validate everything, then return ``(grid, block, args, launcher, claim)``.

    None => use stock. ``claim`` is None for the unfused launcher and
    ``(out_ptr, rows, hidden, fp8, scales)`` for the fused one, to be published by the
    caller only after the launch actually submitted.

    NOTHING here mutates any tensor, and nothing here allocates. That is what makes the
    wrapper's fallback safe (if this returns None, or the single launch that follows fails
    to submit, no state has moved and the stock pair can run from scratch) and what keeps
    the launch path cudagraph-capturable.
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

    fused = _fused_plan(layer, mixed_qkv, core_attn_out, geom, conv_dim, tokens)

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
        # NULL only once the fused handoff has been observed working: until then the fused
        # build writes bf16 too, so it is a strict superset of the unfused one.
        0 if (fused is not None and not fused[2]) else core_attn_out.data_ptr(),
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
        *(fused[0] if fused is not None else ()),
    )
    launcher = _state["launcher"] if fused is None else _state["launcher_fused"]
    claim = fused[1] if fused is not None else None
    # One block per KEY head, not per value head: the q/k conv channels and their rotating
    # conv state are shared by the HV/HK sibling value heads, and sibling blocks have no
    # way to order their state rotation against each other. See the .cu header.
    return (hk, tokens, 1), (dk, 1, 1), args, launcher, claim


def _fused_plan(layer, mixed_qkv, core_attn_out, geom, conv_dim, tokens):  # noqa: ANN001
    """``(extra_args, claim, write_bf16)`` if this call may take the FUSED launcher, else None.

    ``write_bf16`` is the probe state: True means ``out`` is still passed non-null, so the
    launch is a strict superset of the unfused one and correct no matter which way the
    downstream goes. It only turns False once PROBE_CONSUMES claims were actually consumed.

    Every other condition here is a reason the fused launcher may not run at all. The
    per-layer state machine only ever moves forward (see the module docstring); this still
    re-validates the per-call envelope on every step, and an eager step that falls outside
    it takes the unfused launcher, which writes ``core_attn_out`` normally and lets the
    consumer's own Python run the full norm+quant.
    """
    epi = _epilogue_for(layer)
    if epi is None or _state["launcher_fused"] is None:
        return None
    mode = epi.get("mode")
    if mode == "plain":
        return None
    if mode == "probe" and epi.get("pending"):
        # The claim we published last time is still sitting in the book: whatever consumes
        # this layer's core output is NOT gdn_kernels' op after all (a copy between the two,
        # a pattern that stopped matching, a replayed graph). Every probe step so far wrote
        # bf16 as well, so nothing is wrong -- but the write can never be dropped now.
        epi["mode"] = "plain"
        epi["pending"] = False
        _warn_once("fused:unconsumed",
                   "vtl: gdn_decode_step: a fused-epilogue claim went unconsumed; the "
                   "norm+quant stage is not reading our fp8, so the decode step stays "
                   "unfused for the rest of the run")
        return None

    dk, dv, hk, hv, _ = geom
    hidden = hv * dv
    z = None
    reason = None
    if epi["weight_ptr"] not in EPILOGUE_CONSUMERS:
        # gdn_kernels' fused op has never run for this layer, so nothing is known to read
        # the fp8. Not an error -- the fusion pattern may simply not have matched.
        reason = "no-consumer"
    elif tokens > epi["rows"]:
        reason = "staging-rows"
    elif int(core_attn_out.size(0)) != tokens:
        # The consumer norms EVERY row of core_attn_out, including any past
        # num_actual_tokens; the kernel only visits [0, tokens). Rather than reason about
        # the tail, refuse the mismatch.
        reason = "out-rows"
    if reason is None:
        z = _z_ptr(mixed_qkv, conv_dim, hidden, tokens)
        if z is None:
            reason = "gate-unresolvable"
    if reason is not None:
        if mode is None:
            # Latched on the FIRST call, and only towards "plain": the consumer registers
            # during the profile run, well before any decode, so a miss here means the
            # fusion genuinely is not there rather than that we asked too early.
            epi["mode"] = "plain"
            _warn_once("fused:" + reason,
                       "vtl: gdn_decode_step: fused epilogue not claimed (%s); the decode "
                       "step stays unfused and the norm+quant stage runs as today", reason)
        return None

    z_ptr, z_stride = z
    # None -> probe (bf16 still written); probe -> fused once the handoff has been observed
    # PROBE_CONSUMES times. Never backwards, and never straight to "fused".
    if mode is None:
        mode = "probe"
        epi["consumed"] = 0
    if mode == "probe" and int(epi.get("consumed", 0)) >= PROBE_CONSUMES:
        mode = "fused"
        log.info("vtl: gdn_decode_step: fused epilogue handoff confirmed (%d consumed); "
                 "dropping the bf16 core_attn_out write", epi["consumed"])
    epi["mode"] = mode
    write_bf16 = mode == "probe"
    epi["pending"] = write_bf16
    fp8, scales = epi["fp8"], epi["scales"]
    # Row views memoized per token count: slicing allocates no device memory, but building
    # two TensorImpls per layer per step is pure overhead on a path that repeats a handful
    # of widths forever.
    views = epi["views"].get(tokens)
    if views is None:
        views = epi["views"][tokens] = (fp8[:tokens], scales[:tokens])
    extra = (
        z_ptr,
        epi["w_ptr"],
        fp8.data_ptr(),
        scales.data_ptr(),
        ("f", epi["eps"]),
        ("l", z_stride),
        ("l", hidden),               # fp8 row stride, elements
        ("l", scales.stride(0)),     # ss_tok
        ("l", scales.stride(1)),     # ss_grp
    )
    claim = (core_attn_out.data_ptr(), tokens, hidden, views[0], views[1], epi)
    return extra, claim, write_bf16


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


def _staging_rows(vllm_config) -> int:  # noqa: ANN001
    """How many decode tokens the per-layer fp8/scale staging buffers must serve.

    The widest captured decode graph, falling back to max_num_seqs. Capped at
    MAX_STAGING_TOKENS: a batch beyond the cap simply does not claim the fused path.
    """
    sizes = []
    try:
        cc = getattr(vllm_config, "compilation_config", None)
        sizes += [int(s) for s in (getattr(cc, "cudagraph_capture_sizes", None) or [])]
    except Exception:
        pass
    try:
        sc = getattr(vllm_config, "scheduler_config", None)
        sizes.append(int(getattr(sc, "max_num_seqs", 0) or 0))
    except Exception:
        pass
    rows = max([s for s in sizes if s > 0], default=0)
    return min(rows, MAX_STAGING_TOKENS)


def _reference_epilogue(x, z, weight, eps, is_silu):  # noqa: ANN001
    """RMSNormGated.forward_static -> reshape -> per_token_group_fp8_quant, in torch.

    The oracle side of the boot parity gate, and the SAME composition
    ``gdn_kernels._stock_group_quant`` uses -- including the double rounding at the op
    boundary, the 1e-10 amax floor (NOT a min-scale clamp) and the /448 scale.
    ``x`` is [T, heads, D] bf16, ``z`` the same; returns ([T, heads*D] fp8, [T, heads] f32).
    """
    import torch

    xf = x.float()
    zf = z.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    y = (xf * torch.rsqrt(var + eps)) * weight.float()
    act = zf * torch.sigmoid(zf) if is_silu else torch.sigmoid(zf)
    yq = (y * act).to(x.dtype).float()               # narrow once, then re-widen
    amax = yq.abs().amax(-1).clamp_(min=GROUP_QUANT_EPS)
    scale = amax / FP8_MAX
    q = (yq / scale.unsqueeze(-1)).clamp_(-FP8_MAX, FP8_MAX)
    return q.reshape(x.shape[0], -1).to(torch.float8_e4m3fn), scale


def _gate_inputs(geom, tokens, eps, w_dtype, device):  # noqa: ANN001
    """Adversarial-ish fixture for the boot parity gate, at the REAL compiled geometry.

    ``mixed_qkvz`` is built as ONE buffer and split, so the gate exercises the exact z
    recovery ``_z_ptr`` does in production rather than a contiguous stand-in.
    """
    import torch

    dk, dv, hk, hv, conv = geom
    conv_dim = _conv_dim(geom)
    hidden = hv * dv
    g = torch.Generator(device=device).manual_seed(0x6D4E)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32)
                * scale).to(dtype)

    lines = tokens + 1
    qkvz = rnd(tokens, conv_dim + hidden)
    mixed_qkv, z = qkvz.split([conv_dim, hidden], dim=-1)
    # Rows the epilogue has to get exactly right rather than approximately: an all-zero
    # core row (the 1e-10 floor), a saturating row, and a saturated gate.
    z[0].zero_()
    z[1].fill_(20.0)
    idx = torch.arange(1, tokens + 1, dtype=torch.int32, device=device)
    idx[0] = NULL_BLOCK_ID       # a padded cudagraph row: zero core output, still quantized
    return dict(
        qkvz=qkvz, mixed_qkv=mixed_qkv, z=z, conv_dim=conv_dim, hidden=hidden,
        conv_state=rnd(lines, conv_dim, conv - 1),
        ssm_state=torch.randn(lines, hv, dv, dk, generator=g, device=device,
                              dtype=torch.float32) * 0.1,
        a=rnd(tokens, hv, scale=0.5), b=rnd(tokens, hv),
        A_log=torch.log(torch.rand(hv, generator=g, device=device) * 0.9 + 0.1),
        dt_bias=torch.randn(hv, generator=g, device=device, dtype=torch.float32) * 0.1,
        conv_w=rnd(conv_dim, conv), idx=idx,
        norm_w=rnd(dv, dtype=w_dtype), eps=eps,
    )


def _gate_common_args(geom):  # noqa: ANN001
    """The argument prefix both variants take, built from a ``_gate_inputs`` fixture.

    Deliberately hand-packed rather than routed through ``_plan``: the gate has to be able
    to launch the fused build BEFORE any layer is allowed to claim it, and ``_plan`` will
    not hand out the fused launcher until that has happened.
    """
    from vtl import nvrtc

    dk = geom[0]

    def build(fx, conv_state, ssm_state, out_ptr, extra=()):
        return nvrtc.pack_args(
            fx["mixed_qkv"].data_ptr(), conv_state.data_ptr(), fx["conv_w"].data_ptr(), 0,
            fx["a"].data_ptr(), fx["b"].data_ptr(),
            fx["A_log"].data_ptr(), fx["dt_bias"].data_ptr(),
            ssm_state.data_ptr(), out_ptr, fx["idx"].data_ptr(),
            ("f", float(dk) ** -0.5),
            ("l", fx["mixed_qkv"].stride(0)), ("l", fx["a"].stride(0)),
            ("l", fx["b"].stride(0)),
            ("l", conv_state.stride(0)), ("l", conv_state.stride(1)),
            ("l", conv_state.stride(2)),
            ("l", fx["conv_w"].stride(0)), ("l", fx["conv_w"].stride(1)),
            ("l", ssm_state.stride(0)), ("l", fx["idx"].stride(0)), *extra,
        )

    return build


def _boot_parity_ok(geom, eps, is_silu, w_dtype) -> bool:  # noqa: ANN001
    """Does the FUSED build reproduce the unfused build chained with the reference epilogue?

    Runs on the real device at the real compiled geometry, BEFORE any layer is allowed to
    claim the fused path. Three assertions, in increasing strength:

      1. the core is untouched -- conv_state and ssm_state must come out BIT-IDENTICAL to
         the unfused build's. There is no arithmetic difference between the two source
         paths, so anything here is a compile-time surprise, not rounding;
      2. the epilogue is EXACT against the torch reference chained onto the unfused build's
         bf16 output. Codes bit-equal, scales bit-equal -- this is the parity discipline the
         gdn_kernels oracles describe, so anything looser would stop describing it;
      3. if the stock Triton pair imports, the whole chain (Triton pair -> reference
         epilogue) is compared too, at the fp8 tolerance the standalone kernel's own tests
         use -- the core's two documented fp32 divergences from Triton live here, and they
         must not move more than a fraction of a percent of the codes.

    One failure and the fused launcher is dropped: the unfused fast path stands, which is
    exactly today's behaviour.
    """
    import torch

    t = 3
    dk, dv, hk, hv, conv = geom
    hidden = hv * dv
    device = torch.cuda.current_device()
    fx = _gate_inputs(geom, t, eps, w_dtype, f"cuda:{device}")
    build = _gate_common_args(geom)

    # -- 1/2: unfused build + reference epilogue, vs the fused build --
    cs_a, ss_a = fx["conv_state"].clone(), fx["ssm_state"].clone()
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=f"cuda:{device}")
    _state["launcher"](grid=(hk, t, 1), block=(dk, 1, 1),
                       args=build(fx, cs_a, ss_a, out.data_ptr()))

    cs_b, ss_b = fx["conv_state"].clone(), fx["ssm_state"].clone()
    q = torch.zeros(t, hidden, dtype=torch.float8_e4m3fn, device=f"cuda:{device}")
    s = torch.zeros(t, hv, dtype=torch.float32, device=f"cuda:{device}")
    # `out` NON-NULL here on purpose: the gate exercises PROBE mode, which is what the very
    # first serving steps run, and it lets the bf16 half be checked against the unfused
    # build's in the same shot. The nullable half is one predicated store away and is
    # covered by bench/test_gdn_decode_step.py on a GPU.
    probe = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=f"cuda:{device}")
    _state["launcher_fused"](
        grid=(hk, t, 1), block=(dk, 1, 1),
        args=build(fx, cs_b, ss_b, probe.data_ptr(), extra=(
            fx["z"].data_ptr(), fx["norm_w"].data_ptr(), q.data_ptr(), s.data_ptr(),
            ("f", float(eps)), ("l", fx["mixed_qkv"].stride(0)), ("l", hidden),
            ("l", s.stride(0)), ("l", s.stride(1)))),
    )
    torch.cuda.synchronize()

    if not (torch.equal(cs_a.view(torch.uint8), cs_b.view(torch.uint8))
            and torch.equal(ss_a, ss_b)
            and torch.equal(out.view(torch.uint8), probe.view(torch.uint8))):
        log.error("vtl: gdn_decode_step FUSED BOOT PARITY FAILED: the fused build moved the "
                  "conv/ssm state or the probe-mode bf16 output relative to the unfused "
                  "one; unfused path stands")
        return False

    q_ref, s_ref = _reference_epilogue(out, fx["z"].reshape(t, hv, dv), fx["norm_w"],
                                       float(eps), is_silu)
    if not torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)):
        bad = int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
        log.error("vtl: gdn_decode_step FUSED BOOT PARITY FAILED: %d/%d fp8 codes differ "
                  "from RMSNormGated+group-quant on the unfused output; unfused path stands",
                  bad, q.numel())
        return False
    if not torch.equal(s, s_ref):
        log.error("vtl: gdn_decode_step FUSED BOOT PARITY FAILED: group scales differ "
                  "(max |d| = %r); unfused path stands", float((s - s_ref).abs().max()))
        return False

    # -- 3: the chained oracle against the kernels we are actually replacing --
    chained = _chained_triton_reference(geom, fx, t, float(eps), is_silu)
    if chained is not None:
        q_t, s_t = chained
        # NOT bit-exact, and asking for it would be wrong: the core's two documented fp32
        # divergences from Triton move `o` by ~1 ulp, which can flip a code sitting on an
        # e4m3 rounding boundary. One e4m3 mantissa step bounds the flip; the 1% fraction
        # is what separates that noise from a systematic error. Same pair of bounds
        # bench/test_gdn_gated_rmsnorm.py uses, for the same reason.
        frac = float((q.view(torch.uint8) != q_t.view(torch.uint8)).float().mean())
        close = torch.allclose(q.float(), q_t.float(), rtol=0.13, atol=2.0 ** -9)
        if frac > 1e-2 or not close:
            log.error("vtl: gdn_decode_step FUSED BOOT PARITY FAILED against the stock "
                      "Triton pair chained with the reference epilogue (%.2f%% of codes "
                      "differ, within-one-step=%s); unfused path stands",
                      100.0 * frac, close)
            return False
        if not torch.allclose(s, s_t, rtol=1e-3, atol=0.0):
            log.error("vtl: gdn_decode_step FUSED BOOT PARITY FAILED: group scales drift "
                      "from the chained stock oracle by %r; unfused path stands",
                      float((s - s_t).abs().max()))
            return False
    return True


def _chained_triton_reference(geom, fx, t, eps, is_silu):  # noqa: ANN001
    """``(fp8, scales)`` from stock Triton pair -> reference epilogue, or None if the pair
    is unimportable (an old/patched FLA build must not fail the arm, only weaken the gate)."""
    try:
        import torch

        from vllm.model_executor.layers.fla.ops import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    except Exception as exc:
        log.info("vtl: gdn_decode_step: stock Triton pair unimportable (%r); the fused boot "
                 "gate runs without its chained arm", exc)
        return None
    try:
        dk, dv, hk, hv, conv = geom
        mixed = fx["mixed_qkv"].contiguous()   # kernel 1 writes its output back in place
        conv_state = fx["conv_state"].clone()
        ssm_state = fx["ssm_state"].clone()
        conv_out = causal_conv1d_update(
            mixed, conv_state, fx["conv_w"], None, "silu",
            conv_state_indices=fx["idx"], validate_data=False,
        )
        out = torch.zeros(t, 1, hv, dv, dtype=torch.bfloat16, device=mixed.device)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=conv_out, a=fx["a"], b=fx["b"], A_log=fx["A_log"],
            dt_bias=fx["dt_bias"], scale=float(dk) ** -0.5, initial_state=ssm_state,
            out=out, ssm_state_indices=fx["idx"], use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        return _reference_epilogue(out.squeeze(1), fx["z"].reshape(t, hv, dv),
                                   fx["norm_w"], eps, is_silu)
    except Exception as exc:
        log.info("vtl: gdn_decode_step: chained Triton oracle raised (%r); the fused boot "
                 "gate runs without it", exc)
        return None


def _arm_epilogue(layers, geom, rows) -> int:  # noqa: ANN001
    """Compile the fused variant and attach a staging record to every eligible layer.

    Returns the number of layers armed. Zero means the unfused fast path stands, which is
    exactly today's behaviour -- no output changes, only the second launch chain survives.
    """
    import torch

    from vtl import nvrtc

    if rows <= 0 or not _fused_geometry_ok(geom):
        log.info("vtl: gdn_decode_step: fused epilogue not applicable (staging rows=%d, "
                 "geometry %s); unfused fast path stands", rows, geom)
        return 0

    dk, dv, hk, hv, conv = geom
    hidden = hv * dv
    specs = {}
    for layer in layers:
        params = _norm_params(layer)
        if params is None:
            log.info("vtl: gdn_decode_step: a layer's RMSNormGated is not the shape the "
                     "epilogue reproduces; unfused fast path stands")
            return 0
        weight, eps, is_silu = params
        if weight.numel() != dv or not weight.is_contiguous():
            return 0
        if getattr(layer, "gqa_interleaved_layout", False):
            # z is not a split view of mixed_qkv on that layout; there is nothing to read.
            log.info("vtl: gdn_decode_step: interleaved GQA layout has no recoverable gate "
                     "at this seam; unfused fast path stands")
            return 0
        specs[layer] = (weight, eps, is_silu, weight.dtype == torch.float32)

    # ONE cubin for the whole model: all 36 GDN layers share eps/activation/weight dtype.
    variants = {(round(e, 12), s, w) for _, e, s, w in specs.values()}
    if len(variants) != 1:
        log.info("vtl: gdn_decode_step: %d distinct norm specializations %s; unfused fast "
                 "path stands", len(variants), sorted(variants))
        return 0
    eps, is_silu, w_fp32 = variants.pop()
    defines = _fused_defines(geom, is_silu, w_fp32)
    launcher = nvrtc.compile_kernel(KERNEL, defines, entry=KERNEL)
    if launcher is None:
        log.info("vtl: gdn_decode_step: fused-epilogue compile unavailable/failed; unfused "
                 "fast path stands")
        return 0
    _state.update(launcher_fused=launcher, fused_defines=defines)

    w_dtype = torch.float32 if w_fp32 else torch.bfloat16
    try:
        ok = _boot_parity_ok(geom, eps, is_silu, w_dtype)
    except Exception:
        log.exception("vtl: gdn_decode_step: fused boot parity gate raised; unfused fast "
                      "path stands")
        ok = False
    if not ok:
        _state.update(launcher_fused=None, fused_defines=None)
        return 0

    armed = 0
    for layer, (weight, l_eps, l_silu, _) in specs.items():
        # Persistent, allocated ONCE: the decode path (which may BE a cudagraph capture)
        # must not allocate, and the addresses must be stable across replays.
        layer._vtl_gdn_epilogue = {
            "mode": None,                       # None -> probe -> fused | plain
            "consumed": 0,                      # claims gdn_kernels' op actually took
            "pending": False,                   # a published claim not yet taken
            "weight_ptr": weight.data_ptr(),
            "w_ptr": weight.data_ptr(),
            "eps": float(l_eps),
            "is_silu": bool(l_silu),
            "rows": rows,
            "hidden": hidden,
            "fp8": torch.zeros(rows, hidden, dtype=torch.float8_e4m3fn,
                               device=weight.device),
            "scales": torch.zeros(rows, hv, dtype=torch.float32, device=weight.device),
            "views": {},
        }
        armed += 1
    log.info(
        "vtl: gdn decode-step FUSED EPILOGUE armed: RMSNormGated + per-(token,%d)-group fp8 "
        "quant folded into the decode launch for %d layers (rows=%d, silu=%s, w_fp32=%s, "
        "staging %.1f MB total)",
        GROUP, armed, rows, is_silu, w_fp32,
        armed * rows * (hidden + hv * 4) / 1e6,
    )
    return armed


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

    rows = _staging_rows(vllm_config)
    _state.update(geom=geom, launcher=launcher, installed=True, staging_rows=rows)
    # The one-line tier log a `make verify`-style grep can key on.
    log.info(
        "vtl: gdn decode-step tier active: causal_conv1d_update + "
        "fused_recurrent_gated_delta_rule_packed_decode -> 1 NVRTC launch "
        "(DK=%d DV=%d HK=%d HV=%d CONV=%d, %d layers)",
        *geom, len(layers),
    )

    # The fused epilogue rides the same gate: it either arms here or it does not exist, and
    # a failure at any step leaves exactly the tier that was just logged.
    try:
        _arm_epilogue(layers, geom, rows)
    except Exception:
        log.exception("vtl: gdn_decode_step: fused epilogue arming failed; unfused fast "
                      "path stands")
        _state.update(launcher_fused=None, fused_defines=None)


@register_patch(NAME, default=True)
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
                grid, block, args, launcher, claim = plan
                try:
                    launcher(grid=grid, block=block, args=args)
                    if claim is not None:
                        # AFTER the launch submitted, never before: an unconsumed claim
                        # would make the norm+quant stage read staging the kernel never
                        # filled. Publishing here means the claim exists iff the fp8 does.
                        _publish_claim(*claim)
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
    # Enabled by default per project decision (2026-08-17), matching the shipped compose.
    # The safety story is the ladder, not the gate: NVRTC compile -> stock conv1d + delta
    # rule, the geometry envelope below, the `_decode_batch_ok` engage predicate (prefill,
    # spec-decode and mixed batches never take this path), and the launch-failure latch
    # that stands the kernel down for the rest of the run after one bad launch.
    assert patch.default is True, "shipped ENABLED; NVRTC->stock + the engage predicate guard it"
    assert is_enabled(patch) is True
    os.environ["VTL_ENABLE_GDN_DECODE_STEP"] = "0"
    assert is_enabled(patch) is False, "the gate must still be able to turn it off"
    os.environ["VTL_ENABLE_GDN_DECODE_STEP"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GDN_DECODE_STEP")

    # -- geometry envelope: Qwen3.5-122B-A10B must pass, the ways a foreign model would
    #    break the kernel must not --
    qwen = (128, 128, 16, 64, 4)
    assert _geometry_ok(*qwen) is True
    assert _conv_dim(qwen) == 12288, _conv_dim(qwen)
    assert _defines(qwen) == {"DK": 128, "DV": 128, "HK": 16, "HV": 64,
                              "CONV": 4, "THREADS": 128, "FUSED_EPILOGUE": 0}
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

    # -- the fused epilogue's own define set and envelope --
    assert _fused_defines(qwen, True, False) == {
        "DK": 128, "DV": 128, "HK": 16, "HV": 64, "CONV": 4, "THREADS": 128,
        "FUSED_EPILOGUE": 1, "GROUP": 128, "IS_SILU": 1, "W_FP32": 0}
    assert _fused_defines(qwen, False, True)["IS_SILU"] == 0
    assert _fused_defines(qwen, False, True)["W_FP32"] == 1
    assert _fused_geometry_ok(qwen) is True
    assert _fused_geometry_ok((128, 128, 2, 8, 4)) is True, "GPB=4 -> 64 of 128 threads"
    # GPB*ELANES must fit the block: 8 siblings x 16 lanes = 128 fits, 16 do not.
    assert _fused_geometry_ok((128, 128, 8, 64, 4)) is True
    assert _fused_geometry_ok((128, 128, 4, 64, 4)) is False, "GPB=16 -> 256 > THREADS"
    assert _fused_geometry_ok((128, 128, 16, 60, 4)) is False
    # DV must BE the quant group -- the whole fusion is 8192/128 == 64 == HV.
    assert GROUP == 128 and _fused_geometry_ok((128, 64, 16, 64, 4)) is False

    # -- the kernel must ship, and its entry name must match compile_kernel(name=...) --
    src = nvrtc.load_source(KERNEL)
    assert src, "vtl/kernels/gdn_decode_step.cu missing from the package"
    assert f'extern "C" __global__ void __launch_bounds__(THREADS) {KERNEL}(' in src
    for macro in ("DK", "DV", "HK", "HV", "CONV", "THREADS", "FUSED_EPILOGUE"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    # The stride argument the eager path needs (block_table_tensor[:, 0] is not stride-1).
    assert "indices_stride" in src
    # The epilogue must be compiled OUT of the unfused build, not branched around at
    # runtime, and must carry the same constants as its standalone twin.
    assert "#if FUSED_EPILOGUE" in src and "gdn_norm_quant_epilogue" in src
    assert "kGroupQuantEps = 1e-10f" in src and "kFp8Max = 448.0f" in src
    assert "__nv_cvt_float2_to_fp8x2" in src

    # -- every define set is its own cubin, and dict order is not part of the identity.
    #    The fused/unfused pair MUST NOT collide: they are different programs. --
    keys = {
        nvrtc.cache_key(src, _defines(qwen), "90a", "12.8"),
        nvrtc.cache_key(src, _defines((128, 128, 8, 32, 4)), "90a", "12.8"),
        nvrtc.cache_key(src, _defines((128, 128, 16, 64, 4)), "90", "12.8"),
        nvrtc.cache_key(src, _fused_defines(qwen, True, False), "90a", "12.8"),
        nvrtc.cache_key(src, _fused_defines(qwen, False, False), "90a", "12.8"),
        nvrtc.cache_key(src, _fused_defines(qwen, True, True), "90a", "12.8"),
    }
    assert len(keys) == 6, "cache keys must be distinct across define sets and arches"
    d = _defines(qwen)
    perm = dict(reversed(list(d.items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(src, d, "90a", "12.8")

    # -- the claim book: pointer identity, popped once, and shape-checked --
    class _T:
        def __init__(self, ptr, shape):
            self._p, self.shape = ptr, shape

        def data_ptr(self):
            return self._p

    EPILOGUE_CONSUMERS.clear()
    _claims.clear()
    assert take_epilogue(_T(0x1000, (256, 128)), 64) is None, "no claim -> no handoff"
    _publish_claim(0x1000, 4, 8192, "FP8", "SCALES")
    assert take_epilogue(_T(0x2000, (256, 128)), 64) is None, "another buffer must not match"
    assert _claims, "a pointer MISS must leave the claim in the book"
    # ...but a pointer HIT consumes it, even when the cross-check refuses the shape: a claim
    # its consumer cannot use is stale and must not be offered to the next call.
    assert take_epilogue(_T(0x1000, (128, 128)), 64) is None, "row count must match"
    assert not _claims, "a pointer hit consumes the claim either way"
    for shape, why in (((256, 64), "hidden must match"), ((129, 128), "rows must match")):
        _publish_claim(0x1000, 4, 8192, "FP8", "SCALES")
        assert take_epilogue(_T(0x1000, shape), 64) is None, why
    _publish_claim(0x1000, 4, 8192, "FP8", "SCALES")
    assert take_epilogue(_T(0x1000, (256, 128)), 64) == ("FP8", "SCALES")
    assert take_epilogue(_T(0x1000, (256, 128)), 64) is None, "a claim is consumed ONCE"
    _publish_claim(0x1000, 4, 8192, "FP8", "SCALES")
    assert take_epilogue(_T(0x1000, (256, 128)), 0) is None, "a nonsense head count declines"
    _claims.clear()

    # -- a consumed claim is the ONLY thing that credits the probe counter, and only a
    #    successful take does it --
    epi = {"consumed": 0, "pending": True}
    _publish_claim(0x1000, 4, 8192, "FP8", "SCALES", epi)
    assert take_epilogue(_T(0x1000, (128, 128)), 64) is None
    assert epi == {"consumed": 0, "pending": True}, "a refused take must credit nothing"
    _publish_claim(0x1000, 4, 8192, "FP8", "SCALES", epi)
    assert take_epilogue(_T(0x1000, (256, 128)), 64) == ("FP8", "SCALES")
    assert epi == {"consumed": 1, "pending": False}
    _claims.clear()

    # -- the three-state latch. Driven through _fused_plan with fakes so the transitions
    #    are pinned without a GPU: probe first (bf16 STILL written), fused only after
    #    PROBE_CONSUMES observed handoffs, and plain the moment one goes unconsumed. --
    class _FakeQkv:
        """A stand-in for the split view: stride(0) == conv_dim + hidden is what identifies
        the projection buffer, and the storage must actually reach that far."""

        def __init__(self, ptr, stride0, tokens, nbytes):
            self._ptr, self._s0, self._n, self._bytes = ptr, stride0, tokens, nbytes

        def element_size(self):
            return 2

        def stride(self, d):
            return self._s0 if d == 0 else 1

        def storage_offset(self):
            return 0

        def data_ptr(self):
            return self._ptr

        def untyped_storage(self):
            return type("S", (), {"nbytes": lambda _s: self._bytes})()

    class _FakeOut:
        def __init__(self, rows, ptr=0x9000):
            self._rows, self._ptr = rows, ptr

        def size(self, d):
            return self._rows

        def data_ptr(self):
            return self._ptr

    class _FakeStage:
        def __init__(self, ptr):
            self._ptr = ptr

        def data_ptr(self):
            return self._ptr

        def stride(self, d):
            return (64, 1)[d]

        def __getitem__(self, k):
            return self

    conv_dim, hidden, tok = 12288, 8192, 4
    qkv = _FakeQkv(0x2000, conv_dim + hidden, tok, (conv_dim + hidden) * tok * 2)

    WEIGHT_PTR = 0xC0FFEE

    def _fresh_layer(rows=32):
        lay = type("L", (), {})()
        lay._vtl_gdn_epilogue = {
            "mode": None, "consumed": 0, "pending": False,
            "weight_ptr": WEIGHT_PTR, "w_ptr": WEIGHT_PTR, "eps": 1e-6, "is_silu": True,
            "rows": rows, "hidden": hidden, "fp8": _FakeStage(0x5000),
            "scales": _FakeStage(0x6000), "views": {},
        }
        return lay

    _state["launcher_fused"] = object()
    EPILOGUE_CONSUMERS.clear()
    EPILOGUE_CONSUMERS.add(WEIGHT_PTR)
    # Every decline below logs its reason once, by design. That is the point of the code and
    # noise in `make check`, so mute the module logger for the duration rather than weaken
    # the warnings.
    logging.disable(logging.WARNING)
    try:
        lay = _fresh_layer()
        e = lay._vtl_gdn_epilogue
        for step in range(PROBE_CONSUMES):
            got = _fused_plan(lay, qkv, _FakeOut(tok), qwen, conv_dim, tok)
            assert got is not None and got[2] is True, "probe mode must keep the bf16 write"
            assert e["mode"] == "probe" and e["pending"] is True
            _publish_claim(*got[1])
            assert take_epilogue(_T(0x9000, (tok * 64, 128)), 64) is not None
        got = _fused_plan(lay, qkv, _FakeOut(tok), qwen, conv_dim, tok)
        assert got[2] is False, "after PROBE_CONSUMES handoffs the bf16 write is dropped"
        assert e["mode"] == "fused"

        # ...and the other branch: a claim nobody consumes latches the layer to plain.
        lay = _fresh_layer()
        e = lay._vtl_gdn_epilogue
        assert _fused_plan(lay, qkv, _FakeOut(tok), qwen, conv_dim, tok)[2] is True
        _claims.clear()          # the consumer never ran
        assert _fused_plan(lay, qkv, _FakeOut(tok), qwen, conv_dim, tok) is None
        assert e["mode"] == "plain"
        assert _fused_plan(lay, qkv, _FakeOut(tok), qwen, conv_dim, tok) is None, "permanent"

        # ...and every claim condition latches plain on the FIRST call, never to fused.
        # (name, layer, mixed_qkv, out_rows, consumer_registered)
        flat = _FakeQkv(0x2000, conv_dim, tok, conv_dim * tok * 2)
        for name, lay, qk, out_rows, consumer in (
            ("no-consumer", _fresh_layer(), qkv, tok, False),
            ("staging-rows", _fresh_layer(rows=1), qkv, tok, True),
            ("out-rows", _fresh_layer(), qkv, tok + 1, True),
            # a contiguous mixed_qkv is NOT the split view: there is no gate behind it, and
            # reading one anyway would be reading whatever follows the projection buffer.
            ("gate-unresolvable", _fresh_layer(), flat, tok, True),
        ):
            EPILOGUE_CONSUMERS.clear()
            if consumer:
                EPILOGUE_CONSUMERS.add(WEIGHT_PTR)
            assert _fused_plan(lay, qk, _FakeOut(out_rows), qwen, conv_dim, tok) is None, name
            assert lay._vtl_gdn_epilogue["mode"] == "plain", name

        # -- _z_ptr's own refusals, directly --
        assert _z_ptr(qkv, conv_dim, hidden, tok) == (0x2000 + conv_dim * 2, conv_dim + hidden)
        assert _z_ptr(flat, conv_dim, hidden, tok) is None, "row stride identifies the split"
        short = _FakeQkv(0x2000, conv_dim + hidden, tok, (conv_dim + hidden) * tok * 2 - 16)
        assert _z_ptr(short, conv_dim, hidden, tok) is None, "z must be inside the storage"
        odd = _FakeQkv(0x2004, conv_dim + hidden, tok, (conv_dim + hidden) * tok * 2)
        assert _z_ptr(odd, conv_dim, hidden, tok) is None, "a 16B vector load must be aligned"
    finally:
        logging.disable(logging.NOTSET)
        _state["launcher_fused"] = None
        EPILOGUE_CONSUMERS.clear()
        _claims.clear()
        _warned.clear()   # the one-shot warnings must still be available to production
    for i in range(_MAX_CLAIMS * 2):
        _publish_claim(0x3000 + i, 1, 8192, None, None)
    assert len(_claims) <= _MAX_CLAIMS, "the claim book must stay bounded"
    _claims.clear()

    # -- the consumer set: only gdn_kernels' op may open the fused path --
    assert 0xDEAD not in EPILOGUE_CONSUMERS
    note_epilogue_consumer(0xDEAD)
    assert 0xDEAD in EPILOGUE_CONSUMERS
    note_epilogue_consumer(object())   # must never raise from inside an op impl
    EPILOGUE_CONSUMERS.clear()

    # -- the norm reader: only the RMSNormGated configuration the epilogue reproduces --
    class _W:
        def __init__(self, n=128):
            self._n = n

        def numel(self):
            return self._n

    class _Norm:
        def __init__(self, **kw):
            self.weight = _W()
            self.eps = 1e-6
            self.norm_before_gate = True
            self.group_size = None
            self.activation = "silu"
            self.__dict__.update(kw)

    class _L:
        def __init__(self, norm):
            self.norm = norm

    assert _norm_params(_L(_Norm()))[1:] == (1e-6, True)
    assert _norm_params(_L(_Norm(activation="sigmoid")))[2] is False
    assert _norm_params(_L(_Norm(norm_before_gate=False))) is None, "gate-then-norm differs"
    assert _norm_params(_L(_Norm(norm_before_gate=None))) is None
    assert _norm_params(_L(_Norm(group_size=32))) is None, "sub-grouped norm is not this one"
    assert _norm_params(_L(_Norm(group_size=128)))[1] == 1e-6, "group == row is no grouping"
    assert _norm_params(_L(None)) is None
    n = _Norm()
    del n.eps
    n.variance_epsilon = 1e-5
    assert _norm_params(_L(n))[1] == 1e-5, "the older attribute name must still be read"

    # -- staging sizing: the widest captured graph, capped, and 0 means "do not arm" --
    class _CC:
        cudagraph_capture_sizes = [1, 2, 4, 8, 32]

    class _SC:
        max_num_seqs = 5

    class _Cfg:
        compilation_config = _CC()
        scheduler_config = _SC()

    assert _staging_rows(_Cfg()) == 32
    cfg = _Cfg()
    cfg.compilation_config = None
    assert _staging_rows(cfg) == 5, "no captured sizes -> max_num_seqs"
    cfg2 = _Cfg()
    cfg2.compilation_config = type("X", (), {"cudagraph_capture_sizes": [4096]})()
    assert _staging_rows(cfg2) == MAX_STAGING_TOKENS, "capped, not unbounded"
    assert _staging_rows(object()) == 0

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
