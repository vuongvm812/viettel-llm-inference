"""``moe_decode_gemv`` -- a grouped-GEMV MoE decode path for the fp8-block checkpoint.

WHY. Qwen3.5-122B-A10B is ~90% routed-expert weight: 48 layers x 256 experts x (w13 +
w2), and every decoded token reads top-8 of them. At the trace's decode concurrency (peak
~6 sequences, MTP adds a few more) that is ~3.6 GB of expert weight per token and it is the
entire TPOT floor. The stock fp8 path builds grouped-GEMM machinery for that -- align/sort
tokens by expert, materialize a permuted activation buffer, run a tile-based grouped GEMM
whose M dimension is 1..16. The tiles are almost entirely padding, and the whole job is
really "stream 8 expert slices from HBM once and do one FMA per loaded value".

So: a GEMV. One block per (token, expert-slot, output-group), weights streamed with 16-byte
read-only loads, fp32 accumulation, no permutation buffer, no atomics. The kernel is
``vtl/kernels/moe_decode_gemv.cu``; this module is the seam, the weight preparation and the
launch config.

THE SEAM is ``Fp8MoEMethod.apply`` --
``vllm/model_executor/layers/quantization/fp8.py:833`` in v0.25.0. It is the narrowest
stable point in the stack:

  * Everything routing-shaped has already happened. ``RoutedExperts.forward_modular``
    (fused_moe/routed_experts.py:1175) calls it with ``topk_weights``/``topk_ids`` already
    selected by the stock router, so the gate, the top-k, the score correction bias and the
    routed-scaling factor are all untouched stock code. We never see router logits.
  * The expert weights are plain attributes of ``layer`` (``w13_weight``, ``w2_weight`` and
    their ``*_weight_scale_inv``), so there is nothing to reach into.
  * Below it is ``self.moe_kernel.apply`` -> a modular-kernel object whose internals
    (prepare/finalize, expert-map, DP/EP all-to-all) are exactly the things that change
    between vLLM releases. Above it is the model file.

Declining is free: we return ``original(...)`` and the stock kernel runs, byte for byte.

WHEN IT ENGAGES. All of these, or stock:
  * ``M <= VTL_MOE_GEMV_MAX_M`` (default 16). Above it the GEMV loses to the grouped GEMM,
    which is the whole reason the threshold exists.
  * geometry armed at load: fp8 ``w13 [E, 2N, K]`` / ``w2 [E, K, N]`` with the checkpoint's
    ``[128,128]`` block scales, K and N multiples of 128.
  * ``shared_experts is None`` -- the shared expert is stock, and in the stock path it is
    applied INSIDE the modular kernel's finalize (modular_kernel.py:1346) as a side effect,
    not through our return value. Rather than reproduce that ordering we decline the call.
  * ``expert_map is None`` (no EP: topk_ids must be global expert ids),
    ``apply_router_weight_on_input`` False, ``activation == "silu"``, x bf16 and 2-D.
CUDA-graph capture needs no special case: the NVRTC compile happens at LOAD (``_arm_layer``),
so by the time anything is captured there is nothing left to JIT. Moving the compile to
first-use would break that, and the note in ``moe_apply`` says so.

TWO WEIGHT ARMS, ``VTL_MOE_GEMV_WEIGHTS``:

  ``fp8``  (default) The kernel reads the checkpoint's own fp8 expert weights and
           ``[128,128]`` block scales. ZERO extra memory, and stock stays available for
           every M above the threshold. This is the arm to ship first: it wins by deleting
           the grouped-GEMM overhead at tiny M, not by moving fewer bytes.
  ``int4`` Requantize each expert row to RTN int4 group-128 at load (see PACKING below) and
           stream 4 bits per weight instead of 8 -- the actual bandwidth halving, and the
           only way the 122 GB of weights fits with room to spare. It costs a second copy of
           the expert weights unless ``VTL_MOE_GEMV_FREE_FP8=1`` releases the fp8 originals,
           and releasing them takes stock fused_moe away: big-M calls then run the GEMV in
           token chunks, which is correct and SLOW. On a trace that is 11:1
           prefill:decode that is a bad trade, so FREE_FP8 defaults to 0 and the int4 arm is
           an experiment you run on a box with headroom, gated by ``bench/eval_quality.py``.

PACKING (int4 arm). Defined once, here, and consumed by ``moe_decode_gemv.cu``'s unpack and
by ``bench/test_moe_decode.py``. For a weight ``[E, R, L]`` whose LAST dim is the reduction
dim:

    scale[e, r, g] = max(|max(w_g)| / 7, |min(w_g)| / 8)      g = L/128 groups along L
    q    [e, r, l] = clamp(round(w / scale), -8, 7)           signed int4, no zero point
    packed[e, r, j] = sum_i (q[e, r, 8j+i] & 0xF) << (4*i)    int32, j = l/8, i = l%8

The nibble convention is [[quant_w4a8]]'s ``pack_int4_rows`` -- imported, not restated, so
the two int4 paths in this tree cannot drift. The scale FORMULA is vLLM's own
``quantize_weights(..., scalar_types.int4, group_size=128, zero_points=False)``, restated in
torch only because that helper wants a 2-D K-major matrix and these are 3-D expert stacks.
What we do NOT share with quant_w4a8 is the ENCODING: it feeds
``cutlass_encode_and_reorder_int4b``, which permutes the packed words into CUTLASS's
mixed-input tile order. A GEMV wants plain rows, so this path stops at the packed int32.

ACCURACY. int4 is a requant of already-fp8 weights, but the two group geometries nest: an
int4 group is 128 consecutive elements of one row, and the checkpoint's ``[128,128]`` block
scale is CONSTANT across it. So the requant is exactly RTN-int4 of the fp8 values times that
block scale -- no interaction between the two quantizations, and the error versus the
original bf16 is (int4 RTN error) + (fp8 error), not a product. Same argument as
[[w4a8_from_fp8]]; ``bench/test_w4a8_from_fp8.py --self-check`` is where it is checked.

KNOBS.
    VTL_ENABLE_MOE_DECODE_GEMV=1  registry gate (default off)
    VTL_NVRTC=1                   REQUIRED -- the kernel is NVRTC-compiled; without this
                                  compile_kernel returns None and the patch is a no-op
    VTL_MOE_GEMV_WEIGHTS=fp8|int4 which weight arm to prepare at load (default fp8)
    VTL_MOE_GEMV_MAX_M=16         engage only at or below this token count
    VTL_MOE_GEMV_THREADS=256      block width (power of two, >= 128)
    VTL_MOE_GEMV_CHUNK=64         token chunk, i.e. the staging buffer cap
    VTL_MOE_GEMV_FREE_FP8=0       int4 arm: release the fp8 originals after packing
    VTL_MOE_GEMV_PACK_CHUNK=4     experts per packing step at load (peak-memory knob)

DEGRADE, NEVER DIE. Every failure path -- no cuda-python, NVRTC off, a kernel that will not
compile, an unexpected weight layout, a model with no MoE at all -- ends in "the stock
method runs". Nothing here raises into vLLM.
"""

from __future__ import annotations

import logging
import os

from vtl import nvrtc
from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.moe_decode_gemv")

WEIGHTS_ENV = "VTL_MOE_GEMV_WEIGHTS"
MAX_M_ENV = "VTL_MOE_GEMV_MAX_M"
THREADS_ENV = "VTL_MOE_GEMV_THREADS"
CHUNK_ENV = "VTL_MOE_GEMV_CHUNK"
FREE_FP8_ENV = "VTL_MOE_GEMV_FREE_FP8"
PACK_CHUNK_ENV = "VTL_MOE_GEMV_PACK_CHUNK"

KERNEL = "moe_decode_gemv"
#: Entry points in the .cu, in launch order. moe_int4_to_fp8 is deliberately NOT here -- see
#: the RESERVED note in the kernel header.
ENTRIES = ("moe_x_quant", "moe_gemv_up", "moe_gemv_down", "moe_finalize")

# The kernel's one group size, along the reduction dim, for BOTH the weight groups and the
# activation groups. Not a tunable: the [128,128] block-scale indexing in the .cu assumes it,
# and so does the checkpoint.
GROUP = 128
PACK_FACTOR = 8              # int4 values per int32 word
INT4_MAX, INT4_MIN = 7, -8   # signed int4, no zero point -- matches vLLM's scalar_types.int4
BLOCK = (128, 128)           # the only weight_block_size this patch understands

DEFAULT_MAX_M = 16
DEFAULT_THREADS = 256
DEFAULT_CHUNK = 64
DEFAULT_PACK_CHUNK = 4
VALID_WEIGHTS = ("fp8", "int4")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Counters, not one-shot latches -- same reasoning as quant_w4a8's: "we believed the kernel
# was live and it never fired once" is this patch's silent failure mode, and only totals can
# tell one declined layer from all 48.
_armed_layers = 0
_declined_layers = 0
_engaged_calls = 0
_declined_calls = 0
_first_decline = ""
_summary_logged = False


# ---------------------------------------------------------------------------------------
# Pure helpers. No torch, no vLLM: _self_check() exercises all of this on a bare host.
# ---------------------------------------------------------------------------------------


def _env_int(name: str, default: int, minimum: int, raw: str | None = None) -> int:
    """An env int that never raises and never returns something the kernel cannot launch."""
    if raw is None:
        raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        log.warning("vtl: %s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < minimum:
        log.warning("vtl: %s=%d is below the floor %d; using %d", name, value, minimum, minimum)
        return minimum
    return value


def parse_weights(raw: str | None) -> str:
    """``VTL_MOE_GEMV_WEIGHTS`` -> ``"fp8"`` or ``"int4"``. Anything else falls back to fp8.

    fp8 is the default because it is the arm with NO memory cost and NO loss of the stock
    big-M path; int4 has to be asked for.
    """
    if raw is None or not raw.strip():
        return "fp8"
    value = raw.strip().lower()
    if value not in VALID_WEIGHTS:
        log.warning(
            "vtl: %s=%r is not one of %s; using fp8", WEIGHTS_ENV, raw, list(VALID_WEIGHTS)
        )
        return "fp8"
    return value


def parse_threads(raw: str | None) -> int:
    """Block width. Must be a power of two in [GROUP, 1024] -- the kernel static_asserts it,
    and an NVRTC static_assert failure is a silent fall back to stock, not a crash. Rounding
    an invalid value here means the operator's typo costs nothing."""
    value = _env_int(THREADS_ENV, DEFAULT_THREADS, GROUP, raw)
    value = min(value, 1024)
    if value & (value - 1):  # round DOWN to a power of two, never below GROUP
        power = 1
        while power * 2 <= value:
            power *= 2
        log.warning("vtl: %s=%d is not a power of two; using %d", THREADS_ENV, value, power)
        value = max(power, GROUP)
    return value


def weights_mode() -> str:
    return parse_weights(os.environ.get(WEIGHTS_ENV))


def max_m() -> int:
    return _env_int(MAX_M_ENV, DEFAULT_MAX_M, 0)


def chunk_tokens() -> int:
    return _env_int(CHUNK_ENV, DEFAULT_CHUNK, 1)


def pack_chunk() -> int:
    return _env_int(PACK_CHUNK_ENV, DEFAULT_PACK_CHUNK, 1)


def free_fp8() -> bool:
    return os.environ.get(FREE_FP8_ENV, "").strip().lower() in _TRUTHY


def kernel_defines(k: int, n: int, topk: int, weights: str, threads: int) -> dict[str, int]:
    """The -D set for one specialization. This IS the cache identity of the cubin
    (nvrtc.cache_key hashes the rendered defines), so every value that changes the emitted
    code has to be in here -- WEIGHT_FP8 above all, because the int4 and fp8 arms differ only
    inside ``#if`` and would otherwise collide on one cubin."""
    return {
        "K": int(k),
        "N": int(n),
        "TOPK": int(topk),
        "GROUP": GROUP,
        "THREADS": int(threads),
        "WEIGHT_FP8": 1 if weights == "fp8" else 0,
    }


def validate_geometry(w13_shape, w2_shape, w13_scale_shape, w2_scale_shape):
    """Shapes -> ``(num_experts, k, n)``, or a string saying why this layer is not ours.

    Pure shape math so every rejection is reachable from the self-check. The vLLM layout
    being matched (fp8.py:576-640):
        w13_weight        [E, 2N, K]        w2_weight        [E, K, N]
        w13_weight_scale_inv [E, 2N/128, K/128]
        w2_weight_scale_inv  [E, K/128, N/128]
    A backend that shuffles the weights at load (DeepGEMM, Marlin, FlashInfer) produces
    something that does not match, which is exactly the intent: we only claim the plain
    Hopper/Triton layout.
    """
    if len(w13_shape) != 3 or len(w2_shape) != 3:
        return f"expert weights are {len(w13_shape)}-D/{len(w2_shape)}-D, expected 3-D"
    e, two_n, k = (int(v) for v in w13_shape)
    e2, k2, n = (int(v) for v in w2_shape)
    if e2 != e:
        return f"expert counts disagree: w13 has {e}, w2 has {e2}"
    if k2 != k:
        return f"hidden size disagrees: w13 reduces over {k}, w2 outputs {k2}"
    if two_n != 2 * n:
        return f"w13 has {two_n} rows, expected 2*{n} for a stacked [gate; up]"
    if k % GROUP or n % GROUP:
        return f"K={k}, N={n}: both must be multiples of {GROUP}"
    want13 = (e, two_n // GROUP, k // GROUP)
    want2 = (e, k // GROUP, n // GROUP)
    if tuple(int(v) for v in w13_scale_shape) != want13:
        return f"w13 block scales are {tuple(w13_scale_shape)}, expected {want13}"
    if tuple(int(v) for v in w2_scale_shape) != want2:
        return f"w2 block scales are {tuple(w2_scale_shape)}, expected {want2}"
    return (e, k, n)


def activation_name(activation) -> str:
    """``layer.activation`` -> a lowercase string, whatever vLLM is calling it this release.

    In v0.25.0 it is a ``MoEActivation`` Enum whose ``.value`` is ``"silu"``; before the enum
    landed it was that bare string. ``MoEActivation`` does NOT subclass ``str``, so the
    obvious ``layer.activation != "silu"`` is true for BOTH, which would decline every layer
    and look exactly like "the kernel just never fires".

    The kernel implements ``silu(w13[:N] . x) * (w13[N:] . x)`` -- silu of the FIRST half,
    matching ``torch.ops._C.silu_and_mul`` (activation.py:144), which is what stock calls for
    MoEActivation.SILU on a packed [gate; up] w13. Any other name is a different function.
    """
    return str(getattr(activation, "value", activation)).strip().lower()


def token_chunks(m: int, chunk: int) -> list[tuple[int, int]]:
    """Split M tokens into ``(start, count)`` slices no wider than ``chunk``.

    The staging tensor is ``[chunk*TOPK, K]`` fp32, so this is the only thing bounding its
    size. At the decode sizes this patch engages for it is one chunk; it matters only for
    the FREE_FP8 arm, where a whole prefill has to come through the same kernel.
    """
    if m <= 0:
        return []
    chunk = max(1, int(chunk))
    return [(s, min(chunk, m - s)) for s in range(0, m, chunk)]


def slot_row(token: int, slot: int, topk: int) -> int:
    """Row of the ``[M*TOPK, ...]`` intermediate/staging tensors for one (token, slot).

    The kernel derives the same index from ``blockIdx.x``: ``ts = token*TOPK + slot``,
    ``tok = ts / TOPK``. Stating it here (and asserting the round trip in the self-check) is
    what keeps a future launch-config edit from silently transposing the two.
    """
    return token * topk + slot


def staging_elems(m: int, topk: int, k: int) -> int:
    """fp32 elements of the private per-slot accumulator. Private = atomic-free = the same
    answer every run, which is what makes a numeric regression distinguishable from a flake."""
    return m * topk * k


def launch_grids(m: int, topk: int, k: int, n: int) -> dict[str, tuple[int, int, int]]:
    """Grid for each entry, at one token chunk of ``m``. Blocks are ``(THREADS,1,1)``.

    Every entry is one block per (unit of work, output group of GROUP) so that the group a
    block requants is exactly the group it produced -- that is why the absmax is a single
    block reduction and not a second pass over HBM.
    """
    return {
        "moe_x_quant": (m * (k // GROUP), 1, 1),
        "moe_gemv_up": (m * topk, n // GROUP, 1),
        "moe_gemv_down": (m * topk, k // GROUP, 1),
        "moe_finalize": (m, 1, 1),
    }


def _note_decline(reason: str) -> None:
    global _declined_layers, _first_decline
    _declined_layers += 1
    if not _first_decline:
        _first_decline = reason
        log.warning("vtl: moe_decode_gemv declined a layer (%s); it stays on stock fp8", reason)


# ---------------------------------------------------------------------------------------
# Weight preparation (torch). Imported lazily; nothing here runs on the serving path.
# ---------------------------------------------------------------------------------------


def quantize_int4_group(w):
    """fp32 ``[..., L]`` -> ``(q int8 in [-8,7] [..., L], scale fp32 [..., L/128])``.

    The scale is vLLM's own asymmetric-range formula for a SIGNED int4 with no zero point
    (``quant_utils.quantize_weights``): ``max(|max|/7, |min|/8)``, not ``absmax/7``. Getting
    this wrong is a ~7% systematic scale error on every negative-dominated group, which is
    small enough to look like "int4 is just lossy" instead of like a bug.

    One deliberate divergence: an all-zero group gets a floored scale instead of 0. vLLM
    divides by it and produces NaN; an expert row of zeros is not hypothetical in a
    256-expert model, and a NaN weight poisons every token routed to that expert.
    """
    import torch

    lead = tuple(w.shape[:-1])
    length = w.shape[-1]
    g = w.reshape(*lead, length // GROUP, GROUP)
    scale = torch.maximum(
        g.amax(-1).abs() / float(INT4_MAX), g.amin(-1).abs() / float(-INT4_MIN)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.clamp(torch.round(g / scale.unsqueeze(-1)), INT4_MIN, INT4_MAX)
    return q.reshape(*lead, length).to(torch.int8), scale.to(torch.float32)


def pack_int4_last_dim(q):
    """int4-valued ``[..., L]`` -> int32 ``[..., L/8]``; nibble i of word j is element 8j+i.

    [[quant_w4a8]]'s ``pack_int4_rows`` packs along dim 0 of a 2-D ``[K, N]``; the transpose
    sandwich turns that into "pack along the last dim" without a second implementation of the
    nibble order. IMPORTED on purpose: the dense CUTLASS path and this GEMV path must agree
    about what a packed word means, and the only way to guarantee that is to have one packer.
    """
    from vtl.patches import quant_w4a8

    lead = tuple(q.shape[:-1])
    length = q.shape[-1]
    flat = q.reshape(-1, length).t().contiguous()          # [L, R]
    packed = quant_w4a8.pack_int4_rows(flat, PACK_FACTOR)  # [L/8, R]
    return packed.t().contiguous().reshape(*lead, length // PACK_FACTOR)


def dequant_block_fp8(w_fp8, block_scale):
    """fp8 ``[E, R, L]`` x fp32 ``[E, R/128, L/128]`` -> fp32 ``[E, R, L]``.

    The checkpoint's block dequant, expressed for the 3-D expert stack. Both dims are exact
    multiples of 128 here (validate_geometry guarantees it), so this is a plain
    repeat_interleave with no ragged edge -- unlike the dense [[w4a8_from_fp8]] case.
    """
    import torch

    s = block_scale.to(torch.float32)
    s = s.repeat_interleave(GROUP, dim=1).repeat_interleave(GROUP, dim=2)
    return w_fp8.to(torch.float32) * s


def requantize_experts(w_fp8, block_scale, experts_per_step: int):
    """fp8 block-quant expert stack ``[E, R, L]`` -> ``(int32 [E, R, L/8], fp32 [E, R, L/128])``.

    Chunked over experts, and the output is preallocated rather than ``cat``-ed, because the
    peak here lands at the worst possible moment -- the tail of model load, on a card already
    ~90% full of weights. Two terms dominate per chunk: the fp32 dequant (chunk*R*L*4 B) and
    the int64 intermediate inside ``pack_int4_rows`` (chunk*R*L*8 B, and it builds a shifted
    copy before summing). At the default 4 experts of this model's w13 that is ~100 MB +
    ~200 MB transient; at 64 it would be ~5 GB. ``VTL_MOE_GEMV_PACK_CHUNK`` trades that peak
    against load time -- it is ~6k small ops for a 48-layer model, tens of seconds, once.
    """
    import torch

    e, rows, length = (int(v) for v in w_fp8.shape)
    step = max(1, int(experts_per_step))
    packed = torch.empty((e, rows, length // PACK_FACTOR), dtype=torch.int32,
                         device=w_fp8.device)
    scales = torch.empty((e, rows, length // GROUP), dtype=torch.float32,
                         device=w_fp8.device)
    for start in range(0, e, step):
        stop = min(start + step, e)
        deq = dequant_block_fp8(w_fp8[start:stop], block_scale[start:stop])
        q, s = quantize_int4_group(deq)
        del deq
        packed[start:stop] = pack_int4_last_dim(q)
        scales[start:stop] = s
        del q, s
    return packed, scales


# ---------------------------------------------------------------------------------------
# Compilation. One cubin per (source, defines, arch, toolkit); the four entries are four
# module loads of that ONE cubin, because nvrtc.cache_key does not include the entry name.
# ---------------------------------------------------------------------------------------

_kernel_cache: dict[tuple, dict | None] = {}


def compile_entries(defines: dict[str, int]):
    """``{entry: launchable}`` for one specialization, or None if anything at all failed."""
    key = tuple(sorted(defines.items()))
    if key in _kernel_cache:
        return _kernel_cache[key]
    built: dict = {}
    for entry in ENTRIES:
        k = nvrtc.compile_kernel(KERNEL, defines, entry=entry)
        if k is None:
            log.info(
                "vtl: moe_decode_gemv could not build %s for %s; staying on stock fused_moe",
                entry, nvrtc.render_defines(defines),
            )
            _kernel_cache[key] = None
            return None
        built[entry] = k
    _kernel_cache[key] = built
    return built


class _Armed:
    """Everything the apply path needs, resolved once at load so the hot path branches on
    plain attributes instead of reading os.environ or re-deriving shapes."""

    __slots__ = (
        "num_experts", "k", "n", "topk", "weights", "threads", "max_m", "chunk",
        "kernels", "w13", "w13_scale", "w2", "w2_scale",
    )

    def __init__(self, **kw) -> None:
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


# ---------------------------------------------------------------------------------------
# Arming, at the end of the stock process_weights_after_loading.
# ---------------------------------------------------------------------------------------


def _arm_layer(layer) -> None:
    """Validate + prepare one RoutedExperts. Sets ``layer._vtl_moe_gemv`` or leaves it None.

    Runs AFTER the stock method, on purpose: the stock ``_setup_kernel`` may hand the layer's
    weights to a backend-specific shuffler (DeepGEMM, Marlin, FlashInfer), and reading them
    BEFORE that would arm on tensors the layer is about to stop using -- pinning a second
    copy of 116 GB of experts alive. Reading them after means a shuffled layout simply fails
    validate_geometry and we decline, which is the correct answer for those backends anyway.
    """
    import torch

    global _armed_layers

    w13 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    w13_s = getattr(layer, "w13_weight_scale_inv", None)
    w2_s = getattr(layer, "w2_weight_scale_inv", None)
    if any(t is None for t in (w13, w2, w13_s, w2_s)):
        _note_decline("layer has no w13/w2 weight_scale_inv pair (not an fp8-block MoE)")
        return
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        _note_decline(f"expert weights are {w13.dtype}/{w2.dtype}, not float8_e4m3fn")
        return
    if w13_s.dtype != torch.float32 or w2_s.dtype != torch.float32:
        # An e8m0-scale checkpoint's bytes are not fp32 and must not be read as if they were.
        _note_decline(f"block scales are {w13_s.dtype}/{w2_s.dtype}, not float32")
        return
    if not (w13.is_contiguous() and w2.is_contiguous()):
        _note_decline("expert weights are not contiguous")
        return

    geometry = validate_geometry(w13.shape, w2.shape, w13_s.shape, w2_s.shape)
    if isinstance(geometry, str):
        _note_decline(geometry)
        return
    num_experts, k, n = geometry

    topk = int(getattr(layer, "top_k", 0) or 0)
    if topk <= 0:
        _note_decline("layer.top_k is unset")
        return
    if int(getattr(layer, "global_num_experts", num_experts)) != num_experts:
        _note_decline("global_num_experts != the local expert count (expert parallelism)")
        return

    weights = weights_mode()
    threads = parse_threads(os.environ.get(THREADS_ENV))
    kernels = compile_entries(kernel_defines(k, n, topk, weights, threads))
    if kernels is None:
        _note_decline("NVRTC could not build the kernel (is VTL_NVRTC=1?)")
        return

    armed = _Armed(
        num_experts=num_experts, k=k, n=n, topk=topk, weights=weights, threads=threads,
        max_m=max_m(), chunk=chunk_tokens(), kernels=kernels,
        w13=w13.data, w13_scale=w13_s.data, w2=w2.data, w2_scale=w2_s.data,
    )

    if weights == "int4":
        step = pack_chunk()
        w13_q, w13_gs = requantize_experts(w13.data, w13_s.data, step)
        w2_q, w2_gs = requantize_experts(w2.data, w2_s.data, step)
        # register_parameter, not bare attributes: named_parameters() walkers (memory
        # accounting, the load summary, weight-reload checks) skip plain tensors. Same
        # reasoning as quant_w4a8's weight_packed. requires_grad=False is what lets an int32
        # tensor be a Parameter at all.
        for name, tensor in (("vtl_w13_int4", w13_q), ("vtl_w13_int4_scale", w13_gs),
                             ("vtl_w2_int4", w2_q), ("vtl_w2_int4_scale", w2_gs)):
            layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
        armed.w13, armed.w13_scale = w13_q, w13_gs
        armed.w2, armed.w2_scale = w2_q, w2_gs
        if free_fp8():
            # Rebind to empty rather than del: plenty of vLLM code does a bare
            # `layer.w13_weight` lookup, and an AttributeError at that point is a dead
            # engine. The stock big-M path is gone from here on -- see the module docstring.
            for name in ("w13_weight", "w2_weight"):
                old = getattr(layer, name)
                setattr(layer, name, torch.nn.Parameter(
                    torch.empty(0, dtype=old.dtype, device=old.device), requires_grad=False
                ))
            layer._vtl_moe_gemv_owns_all_m = True

    layer._vtl_moe_gemv = armed
    # Counted once per LAYER, not once per call: process_weights_after_loading runs again on
    # an RL weight reload, and a summary that says "armed 96 of 48 layers" is worse than no
    # summary at all.
    if not getattr(layer, "_vtl_moe_gemv_counted", False):
        layer._vtl_moe_gemv_counted = True
        _armed_layers += 1


# ---------------------------------------------------------------------------------------
# The serving path.
# ---------------------------------------------------------------------------------------


def _can_engage(armed, layer, x, topk_weights, topk_ids, shared_experts) -> str | None:
    """None if this call is ours; otherwise the (logged-once) reason it is not."""
    import torch

    if armed is None:
        return "layer not armed"
    if shared_experts is not None:
        return "shared_experts is fused into the stock finalize"
    if getattr(layer, "expert_map", None) is not None:
        return "expert_map is set (expert parallelism)"
    if getattr(layer, "apply_router_weight_on_input", False):
        return "apply_router_weight_on_input"
    if activation_name(getattr(layer, "activation", "silu")) != "silu":
        return f"activation is {getattr(layer, 'activation', None)!r}, not silu"
    # A clamped/affine SwiGLU (gpt-oss, MiniMax) routes SILU through
    # silu_and_mul_with_clamp instead -- different math, same enum value.
    if any(getattr(layer, name, None) is not None
           for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")):
        return "the layer carries swiglu clamp/affine parameters"
    if getattr(layer, "w13_bias", None) is not None or getattr(layer, "w2_bias", None) is not None:
        return "biased MoE (the GEMV has no bias term)"
    if x.dim() != 2 or x.shape[1] != armed.k:
        return f"x is {tuple(x.shape)}, expected [M, {armed.k}]"
    if x.dtype != torch.bfloat16:
        return f"x is {x.dtype}, the kernel reads bf16"
    if topk_ids.dim() != 2 or topk_ids.shape[1] != armed.topk:
        return f"topk_ids is {tuple(topk_ids.shape)}, expected [M, {armed.topk}]"
    if topk_ids.shape[0] != x.shape[0] or topk_weights.shape[0] != x.shape[0]:
        return "topk_ids/topk_weights disagree with x about M"
    if x.shape[0] > armed.max_m and not getattr(layer, "_vtl_moe_gemv_owns_all_m", False):
        return f"M={x.shape[0]} > {MAX_M_ENV}={armed.max_m}"
    return None


def _run(armed, x, topk_weights, topk_ids):
    """Launch the four entries over ``x``, one token chunk at a time. Returns ``[M, K]`` bf16."""
    import torch

    from vtl.nvrtc import pack_args

    m, k, n, topk = x.shape[0], armed.k, armed.n, armed.topk
    dev = x.device
    fp8 = torch.float8_e4m3fn
    block = (armed.threads, 1, 1)
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    wts = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(torch.float32)
    ids = ids.contiguous()
    wts = wts.contiguous()
    x = x.contiguous()
    out = torch.empty((m, k), dtype=torch.bfloat16, device=dev)

    kx = armed.kernels["moe_x_quant"]
    kup = armed.kernels["moe_gemv_up"]
    kdown = armed.kernels["moe_gemv_down"]
    kfin = armed.kernels["moe_finalize"]

    for start, count in token_chunks(m, armed.chunk):
        xc = x[start:start + count]
        idc = ids[start:start + count]
        wtc = wts[start:start + count]
        outc = out[start:start + count]
        xq = torch.empty((count, k), dtype=fp8, device=dev)
        x_scale = torch.empty((count, k // GROUP), dtype=torch.float32, device=dev)
        h = torch.empty((count * topk, n), dtype=fp8, device=dev)
        h_scale = torch.empty((count * topk, n // GROUP), dtype=torch.float32, device=dev)
        staging = torch.empty((count * topk, k), dtype=torch.float32, device=dev)
        grids = launch_grids(count, topk, k, n)

        kx(grid=grids["moe_x_quant"], block=block,
           args=pack_args(xq.data_ptr(), x_scale.data_ptr(), xc.data_ptr()))
        kup(grid=grids["moe_gemv_up"], block=block,
            args=pack_args(h.data_ptr(), h_scale.data_ptr(), xq.data_ptr(),
                           x_scale.data_ptr(), armed.w13.data_ptr(),
                           armed.w13_scale.data_ptr(), idc.data_ptr(),
                           ("i", armed.num_experts)))
        kdown(grid=grids["moe_gemv_down"], block=block,
              args=pack_args(staging.data_ptr(), h.data_ptr(), h_scale.data_ptr(),
                             armed.w2.data_ptr(), armed.w2_scale.data_ptr(),
                             idc.data_ptr(), ("i", armed.num_experts)))
        kfin(grid=grids["moe_finalize"], block=block,
             args=pack_args(outc.data_ptr(), staging.data_ptr(), wtc.data_ptr()))
    return out


def log_summary() -> None:
    """One line saying whether this ever fired. Idempotent; grepped by ``make verify``."""
    global _summary_logged
    if _summary_logged:
        return
    _summary_logged = True
    if _armed_layers == 0:
        log.warning(
            "vtl: moe_decode_gemv armed 0 MoE layers (declined %d; first: %s) -- decode is "
            "running the stock fused_moe path",
            _declined_layers, _first_decline or "n/a",
        )
    else:
        log.info(
            "vtl: moe_decode_gemv armed %d MoE layers (weights=%s, max_m=%d, declined=%d)",
            _armed_layers, weights_mode(), max_m(), _declined_layers,
        )


def _install_load_summary() -> None:
    """``BaseModelLoader.load_model``, the same seam quant_w4a8 and l2_persist use. Named-patch
    marked because several patches legitimately wrap this one attribute."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch="moe_decode_gemv"):
        return
    original = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        try:
            log_summary()
        except Exception:
            log.exception("vtl: moe_decode_gemv summary failed")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, original, patch="moe_decode_gemv")


@register_patch("moe_decode_gemv", default=False)
def apply() -> None:
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    if not already_patched(Fp8MoEMethod, "process_weights_after_loading", patch="moe_decode_gemv"):
        original_pwal = Fp8MoEMethod.process_weights_after_loading

        def process_weights_after_loading(self, layer):
            if getattr(layer, "_vtl_moe_gemv_owns_all_m", False):
                # FREE_FP8 released the checkpoint tensors; the stock method would now
                # process empty weights and re-arming has nothing to read. A weight reload
                # is simply not supported in that mode, and pretending otherwise would
                # silently serve the OLD int4 weights against NEW routing.
                log.warning("vtl: moe_decode_gemv cannot re-arm a layer whose fp8 originals "
                            "were freed (VTL_MOE_GEMV_FREE_FP8=1); skipping the reload")
                return
            original_pwal(self, layer)
            layer._vtl_moe_gemv = None
            if not getattr(self, "block_quant", False):
                _note_decline("not a block-quant fp8 MoE method")
                return
            if list(getattr(self, "weight_block_size", None) or ()) != list(BLOCK):
                _note_decline(f"weight_block_size {getattr(self, 'weight_block_size', None)}")
                return
            try:
                _arm_layer(layer)
            except Exception as exc:
                # A failed arm must cost the optimization, never the model load.
                layer._vtl_moe_gemv = None
                _note_decline(f"arming raised: {exc!r}")

        Fp8MoEMethod.process_weights_after_loading = mark_patched(
            process_weights_after_loading, original_pwal, patch="moe_decode_gemv"
        )

    if not already_patched(Fp8MoEMethod, "apply", patch="moe_decode_gemv"):
        original_apply = Fp8MoEMethod.apply

        def moe_apply(self, layer, x, topk_weights, topk_ids, shared_experts=None,
                      shared_experts_input=None, **kwargs):
            # **kwargs is a VERSION GUARD, not convenience: RoutedExperts.forward_modular
            # calls this all-keyword, so a vLLM release that adds one more argument would
            # otherwise TypeError inside the engine. An argument we do not understand means
            # we do not understand the call, so we hand it straight back to stock.
            global _engaged_calls, _declined_calls
            armed = getattr(layer, "_vtl_moe_gemv", None)
            if kwargs:
                return original_apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                                      shared_experts_input, **kwargs)
            try:
                # NOTE ON CUDA-GRAPH CAPTURE: every NVRTC compile happened at load
                # (_arm_layer), so nothing here can stall a capture. The per-chunk tensors
                # come from torch's caching allocator, which is capture-aware, and their
                # shapes are a function of the captured M -- so a replay reuses the same
                # addresses. Do NOT move the compile to first-use without adding an
                # is_current_stream_capturing() bail here.
                reason = _can_engage(armed, layer, x, topk_weights, topk_ids, shared_experts)
                if reason is not None:
                    _declined_calls += 1
                    if _declined_calls == 1:
                        log.info("vtl: moe_decode_gemv declined a call (%s); stock runs", reason)
                    return original_apply(self, layer, x, topk_weights, topk_ids,
                                          shared_experts, shared_experts_input)
                out = _run(armed, x, topk_weights, topk_ids)
                _engaged_calls += 1
                if _engaged_calls == 1:
                    # The load summary runs BEFORE any forward, so "armed" is not proof the
                    # kernel ever ran. This line is. One INFO, once per process, off the
                    # steady-state path -- `make verify` greps for it.
                    log.info(
                        "vtl: moe_decode_gemv ENGAGED (weights=%s, M=%d, K=%d, N=%d, topk=%d)",
                        armed.weights, x.shape[0], armed.k, armed.n, armed.topk,
                    )
                return out
            except Exception as exc:
                _declined_calls += 1
                if _declined_calls == 1:
                    log.warning("vtl: moe_decode_gemv raised (%r); falling back to stock", exc)
                if getattr(layer, "_vtl_moe_gemv_owns_all_m", False):
                    # The fp8 originals are gone; there is no stock path to fall back TO, so
                    # re-raising is more honest than returning a silently wrong tensor.
                    raise
                return original_apply(self, layer, x, topk_weights, topk_ids,
                                      shared_experts, shared_experts_input)

        Fp8MoEMethod.apply = mark_patched(moe_apply, original_apply, patch="moe_decode_gemv")

    try:
        _install_load_summary()
    except Exception:
        log.exception("vtl: could not install the moe_decode_gemv load summary hook")

    log.info(
        "vtl: moe_decode_gemv installed (weights=%s, max_m=%d, threads=%d, chunk=%d, "
        "free_fp8=%s, nvrtc=%s)",
        weights_mode(), max_m(), parse_threads(os.environ.get(THREADS_ENV)),
        chunk_tokens(), free_fp8(), nvrtc.enabled(),
    )


def _self_check() -> None:
    # -- weight arm parsing: fp8 unless int4 is asked for, by name --
    assert parse_weights(None) == "fp8"
    assert parse_weights("") == "fp8"
    assert parse_weights(" INT4 ") == "int4"
    assert parse_weights("fp8") == "fp8"
    assert parse_weights("w4a8") == "fp8", "an unknown arm must not silently become int4"

    # -- ints: bad input costs nothing, floors hold --
    assert _env_int("X", 16, 0, None) == 16
    assert _env_int("X", 16, 0, "") == 16
    assert _env_int("X", 16, 0, "32") == 32
    assert _env_int("X", 16, 0, "nope") == 16
    assert _env_int("X", 16, 4, "1") == 4
    assert _env_int("X", 16, 0, "0") == 0, "max_m=0 must be expressible (disarm without unloading)"

    # -- threads: powers of two only, never below GROUP, never above 1024 --
    assert parse_threads(None) == DEFAULT_THREADS
    assert parse_threads("512") == 512
    assert parse_threads("2048") == 1024
    assert parse_threads("300") == 256, "must round DOWN to a power of two"
    assert parse_threads("64") == GROUP, "a block narrower than GROUP cannot requant a group"
    assert parse_threads("garbage") == DEFAULT_THREADS

    # -- defines: the model's real geometry, and every arm distinguishable --
    d = kernel_defines(3072, 1024, 8, "int4", 256)
    assert d == {"K": 3072, "N": 1024, "TOPK": 8, "GROUP": 128, "THREADS": 256,
                 "WEIGHT_FP8": 0}, d
    assert kernel_defines(3072, 1024, 8, "fp8", 256)["WEIGHT_FP8"] == 1

    # -- cache-key distinctness. The int4 and fp8 arms differ ONLY inside #if, so if
    #    WEIGHT_FP8 ever fell out of the define set they would share a cubin and one of the
    #    two would silently run the other's code. Every axis gets its own assertion. --
    base = kernel_defines(3072, 1024, 8, "int4", 256)
    src, arch, tk = "// source", "90a", "12.4"
    key = nvrtc.cache_key(src, base, arch, tk)
    assert nvrtc.cache_key(src, dict(base), arch, tk) == key, "must be stable"
    variants = {
        "weights": kernel_defines(3072, 1024, 8, "fp8", 256),
        "k": kernel_defines(2048, 1024, 8, "int4", 256),
        "n": kernel_defines(3072, 512, 8, "int4", 256),
        "topk": kernel_defines(3072, 1024, 4, "int4", 256),
        "threads": kernel_defines(3072, 1024, 8, "int4", 512),
    }
    keys = {name: nvrtc.cache_key(src, dv, arch, tk) for name, dv in variants.items()}
    for name, other in keys.items():
        assert other != key, f"{name} must change the cubin identity"
    assert len(set(keys.values())) == len(keys), keys
    # ...and the arch/toolkit axes nvrtc owns still bite through our defines.
    assert nvrtc.cache_key(src, base, "80", tk) != key
    assert nvrtc.cache_key(src, base, arch, "12.5") != key

    # -- geometry validation: the real shapes pass, every rejection names itself --
    E, K, N = 256, 3072, 1024
    ok = validate_geometry((E, 2 * N, K), (E, K, N), (E, 2 * N // 128, K // 128),
                           (E, K // 128, N // 128))
    assert ok == (E, K, N), ok
    assert "3-D" in validate_geometry((E, 2 * N), (E, K, N), (E, 16, 24), (E, 24, 8))
    assert "expert counts" in validate_geometry((E, 2 * N, K), (E - 1, K, N),
                                                (E, 16, 24), (E, 24, 8))
    assert "hidden size" in validate_geometry((E, 2 * N, K), (E, K + 128, N),
                                              (E, 16, 24), (E, 24, 8))
    assert "stacked" in validate_geometry((E, N, K), (E, K, N), (E, 8, 24), (E, 24, 8))
    assert "multiples" in validate_geometry((E, 2 * 100, K), (E, K, 100), (E, 1, 24), (E, 24, 1))
    # The DeepGEMM/Marlin case: weights are fine, the scale grid is not what we expect.
    assert "block scales" in validate_geometry((E, 2 * N, K), (E, K, N), (E, 24, 16), (E, 24, 8))
    assert "block scales" in validate_geometry((E, 2 * N, K), (E, K, N), (E, 16, 24), (E, 8, 24))

    # -- activation naming: the enum, the bare string, and anything else --
    class _FakeEnum:
        value = "silu"

    assert activation_name("silu") == "silu"
    assert activation_name(" SiLU ") == "silu"
    assert activation_name(_FakeEnum()) == "silu", "MoEActivation is an Enum, not a str"
    assert activation_name("silu_no_mul") != "silu", "the non-gated variant is different math"
    assert activation_name("gelu") == "gelu"

    # -- slot bookkeeping. The kernel recovers (token, slot) from one blockIdx, so the map
    #    has to be a bijection onto [0, M*TOPK) with the token in the HIGH digit. --
    topk = 8
    seen = set()
    for token in range(5):
        for slot in range(topk):
            row = slot_row(token, slot, topk)
            assert row not in seen, (token, slot, row)
            seen.add(row)
            assert row // topk == token and row % topk == slot, (token, slot, row)
    assert seen == set(range(5 * topk))
    assert slot_row(0, 0, topk) == 0 and slot_row(1, 0, topk) == topk
    assert staging_elems(16, 8, 3072) == 16 * 8 * 3072
    assert staging_elems(16, 8, 3072) * 4 == 1572864, "M=16 staging is 1.5 MB, i.e. L2-resident"

    # -- chunking covers every token exactly once, in order, with no chunk over the cap --
    for m, chunk in ((0, 4), (1, 4), (4, 4), (5, 4), (16, 64), (300, 64), (7, 1)):
        parts = token_chunks(m, chunk)
        assert sum(c for _, c in parts) == m, (m, chunk, parts)
        assert all(0 < c <= chunk for _, c in parts), (m, chunk, parts)
        covered = [i for s, c in parts for i in range(s, s + c)]
        assert covered == list(range(m)), (m, chunk, parts)
    assert token_chunks(10, 0) == token_chunks(10, 1), "a zero chunk must not divide by zero"

    # -- launch grids: one block per output group, and the .cu's blockIdx arithmetic --
    g = launch_grids(16, 8, 3072, 1024)
    assert g["moe_x_quant"] == (16 * 24, 1, 1)      # one block per (token, K group)
    assert g["moe_gemv_up"] == (128, 8, 1)          # (token*slot, N group)
    assert g["moe_gemv_down"] == (128, 24, 1)       # (token*slot, K group)
    assert g["moe_finalize"] == (16, 1, 1)          # one block per token
    assert launch_grids(1, 8, 3072, 1024)["moe_gemv_up"] == (8, 8, 1)
    # gridDim.y counts requant groups, so it must be exactly the scale-tensor width.
    assert g["moe_gemv_up"][1] == 1024 // GROUP and g["moe_gemv_down"][1] == 3072 // GROUP

    # -- the kernel source must ship with the package, and declare what we pass it --
    src_text = nvrtc.load_source(KERNEL)
    assert src_text, "vtl/kernels/moe_decode_gemv.cu missing from the package"
    for entry in ENTRIES + ("moe_int4_to_fp8",):
        assert f'extern "C" __global__ void __launch_bounds__(THREADS) {entry}(' in src_text, entry
    for name in ("K", "N", "TOPK"):
        assert f'#error "NVRTC: -D{name}=' in src_text, f"{name} must be #error-guarded"
    for name in ("GROUP", "THREADS", "WEIGHT_FP8"):
        assert f"#define {name} " in src_text, f"{name} must have a default"
    assert set(kernel_defines(1, 1, 1, "int4", 128)) == {
        "K", "N", "TOPK", "GROUP", "THREADS", "WEIGHT_FP8"
    }, "the define set and the .cu's macro list must not drift"

    # -- the packing contract, in pure python, against the .cu's unpack expression --
    #    word j nibble i is element 8j+i, two's-complement signed. This is the one thing
    #    that has to agree across three files (this module, the .cu, the test).
    def pack_reference(values: list[int]) -> list[int]:
        words = []
        for j in range(len(values) // PACK_FACTOR):
            word = 0
            for i in range(PACK_FACTOR):
                word |= (values[PACK_FACTOR * j + i] & 0xF) << (4 * i)
            words.append(word)
        return words

    def unpack_reference(word: int) -> list[int]:
        # exactly `((int)(w << (28 - 4*i))) >> 28` in the kernel
        out = []
        for i in range(PACK_FACTOR):
            nib = (word >> (4 * i)) & 0xF
            out.append(nib - 16 if nib >= 8 else nib)
        return out

    vals = [-8, -1, 0, 1, 7, -5, 3, -8]
    assert unpack_reference(pack_reference(vals)[0]) == vals
    assert pack_reference([0] * 8) == [0]
    assert pack_reference([-1] * 8)[0] == 0xFFFFFFFF
    assert unpack_reference(0xFFFFFFFF) == [-1] * 8
    assert pack_reference([7] + [0] * 7)[0] == 7
    assert pack_reference([0] * 7 + [7])[0] == 7 << 28

    print("moe_decode_gemv self-check ok")


if __name__ == "__main__":
    _self_check()
