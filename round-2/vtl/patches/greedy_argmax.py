"""Fused greedy argmax: three NVRTC launches replace ``torch.argmax`` on the sampling path.

WHAT IT REPLACES. On a provably-greedy step the whole sampler collapses to one reduction
over ``[num_reqs, 248320]`` bf16 logits. ``torch.argmax`` already does that in a single
fused MULTI-CTA pass at the memory roofline, so the win here is NOT the reduction -- the
kernel's split-row grid exists to MATCH torch's occupancy, not to beat it. What is left is
everything around it:

  * VOCAB is a compile-time ``-D`` constant, so the row loop's trip count and every offset
    fold to immediates and the loads are 16-byte vectors (vtl/kernels/greedy_argmax.cu);
  * the launches are three ``cuLaunchKernel``s off pre-built ctypes argument arrays, with no
    TensorIterator build, no dtype dispatch and no reduction-config heuristic;
  * the OUT variant writes straight into the caller's persistent int64 buffer -- no
    allocation and no ``out=`` copy, which is what lets the N-step burst capture it.

That is a few microseconds of HOST dispatch per step in eager mode and approximately
nothing inside a replayed graph, where the point is instead that the token pick is
capturable at all. bench/test_greedy_argmax.py's ``VTL_BENCH=1`` arm measures both.

TWO CONSUMERS, BOTH PRE-EXISTING SEAMS -- this patch only fills them:

  * the forked V2 sampler (``vtl/vllm_patches/v0.25.0/v2_greedy_sampler.patch``) resolves
    ``torch.ops.vllm_cuda.greedy_argmax_i64`` ONCE, on the first greedy step, behind
    ``VTL_V2_GREEDY_ARGMAX_KERNEL``; if the op is not registered it keeps
    ``logits.argmax(dim=-1)`` forever. So "did not compile" costs nothing and needs no
    coordination -- we simply never register the op.
  * ``vtl/patches/nstep_decode.py`` calls its module-level ``_ARGMAX`` indirection at the
    three places it picks a token (burst body, prologue, eager token-1). We rebind that
    global at model load, i.e. BEFORE ``_capture_burst_graphs`` runs, so the fused launches
    are what get captured into the burst graphs -- and therefore what the Rust runner
    replays too, since it replays the same execs.

WHY IT MUST BE BIT-EXACT, and WHAT PROVES IT. The kernel reproduces ``torch.argmax``'s
contract exactly: NaN is the maximum, and every tie -- equal maxima, an all-``-inf`` masked
row, several NaN, a ``-0.0``/``+0.0`` pair -- resolves to the LOWEST index. Two things hold
it to that:

  * the BOOT PARITY GATE in ``_arm`` below: adversarial rows at the REAL compiled vocab on
    the REAL device, bit-compared against ``torch.argmax`` BEFORE either op is registered.
    One failure and nothing is installed, so both consumers keep torch.argmax. ~1 ms, once.
  * bench/test_greedy_argmax.py on a GPU in CI.

What does NOT prove it is nstep's ``_graph_matches_eager``. Once ``_ARGMAX`` is rebound,
both of its arms call this kernel, so it compares fused against fused: it is a
pointer-staleness smoke test for graph capture and nothing more.

ARMING. Same ``BaseModelLoader.load_model`` seam as ``nvrtc_block_quant`` /
``quant_w4a8`` / ``l2_persist`` (the registry's ``patch=`` names keep the stack sound).
After the model loads we read the REAL vocab off the config, compile, run the parity gate,
and only then register the ops and rebind nstep. ``compile_kernel`` returning None -- NVRTC
off, no cuda-python, bad source, no device -- leaves both consumers on ``torch.argmax``,
which is the behaviour they already have today.

Gate: ``VTL_ENABLE_GREEDY_ARGMAX`` (default ON since 2026-08-17) **and** ``VTL_NVRTC=1`` (the layer-wide
switch). Block width: ``VTL_GREEDY_ARGMAX_THREADS`` (default 256, a whole number of warps up
to 1024). Row slices: ``VTL_GREEDY_ARGMAX_SPLIT`` (default 16, 1..1024).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.greedy_argmax")

KERNEL = "greedy_argmax"                  # vtl/kernels/greedy_argmax.cu
ENTRY_RESET = "greedy_argmax_reset"       # extern "C" symbols inside it, in launch order
ENTRY_PARTIAL = "greedy_argmax_partial"
ENTRY_FINALIZE = "greedy_argmax_finalize"
ENTRIES = (ENTRY_RESET, ENTRY_PARTIAL, ENTRY_FINALIZE)
NS = "vllm_cuda"                  # our own op namespace (vtl/csrc/torch_bindings.cpp)
OP = "greedy_argmax_i64"          # torch.ops.vllm_cuda.<OP>(Tensor) -> Tensor
OP_OUT = "greedy_argmax_i64_out"  # torch.ops.vllm_cuda.<OP_OUT>(Tensor, Tensor!) -> ()

DEFAULT_THREADS = 256
DEFAULT_SPLIT = 16
# Block width for the two tiny kernels. They stride on blockDim, so this is a launch
# constant rather than a -D: one block covers any `rows` at any width.
EDGE_THREADS = 256
# Rows the persistent u64 workspace serves. Allocated ONCE at arm time -- a per-call
# allocation would be both a sync risk and uncapturable. The served config is
# --max-num-seqs 5 and the widest captured decode graph is 32, so 256 is ~50x headroom for
# 2 KiB; a batch beyond it leaves the envelope and takes torch.argmax.
#
# ONE workspace, so two OVERLAPPING calls would race on it. That is not a scenario here --
# every caller (the V2 sampler, nstep's burst, nstep's eager token-1) runs on the engine
# thread and the three launches are ordered by the stream they share -- but it is the reason
# this is a module-level buffer rather than a free-for-all, and the reason a second sampling
# thread would need its own.
MAX_WS_ROWS = 256
VEC = 8          # bf16 elems per 16-byte load, mirrors the kernel
ALIGN = 16       # bytes the vectorized row path needs

# Populated by _arm(); read by the op impls. `lib` must stay referenced or torch drops the
# registration along with it.
#
# THE ENVELOPE IS `vocab`. It is published LAST, after the parity gate passed and both ops
# registered, and zeroed on every failure path -- so a half-armed module reports vocab 0,
# `_logits_reason` answers "vocab" for every real tensor, and both impls degrade to
# torch.argmax without ever reaching a launcher.
_state: dict = {
    "installed": False,
    "armed": False,
    "disabled": False,   # latched by a driver error in _launch; see _argmax_impl
    "vocab": 0,
    "threads": 0,
    "split": 0,
    "vec": False,        # is the 16-byte vector path the one that got compiled?
    "reset": None,       # the three launchers, in launch order
    "partial": None,
    "finalize": None,
    "ws": None,          # persistent [MAX_WS_ROWS] int64 workspace tensor (keep-alive)
    "ws_ptr": 0,
    "ws_dev": -1,
    "ws_rows": 0,        # rows the workspace serves; a wider batch leaves the envelope
    "block": None,       # the partial kernel's block tuple
    "edge_block": None,  # the reset/finalize block tuple
    "raw_stream": None,  # torch._C._cuda_getCurrentRawStream, bound once
    "args": None,        # {"reset"/"partial"/"finalize": (ctypes array, holders)}
    "grids": None,       # rows -> the partial kernel's grid tuple
    "lib": None,
    "op_out": None,
    "nstep": False,
}
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _int_env(name: str, default: int, lo: int, hi: int, step: int = 1) -> int:
    """A bounded int knob. A bad value falls back rather than failing the compile."""
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    if value < lo or value > hi or value % step != 0:
        log.info("vtl: greedy_argmax: %s=%s is out of range; using %d", name, value, default)
        return default
    return value


def _threads_from_env() -> int:
    """Block width of the partial kernel. Must be a whole number of warps."""
    return _int_env("VTL_GREEDY_ARGMAX_THREADS", DEFAULT_THREADS, 32, 1024, 32)


def _split_from_env() -> int:
    """Row slices per row, i.e. the partial kernel's gridDim.x.

    This is the knob that turned one CTA per row (<= 8 of 132 SMs at the served batch) into
    a grid that actually fills the device. Sweepable because the right value depends on
    VOCAB and on how many rows are in flight, not on anything we can know here.
    """
    return _int_env("VTL_GREEDY_ARGMAX_SPLIT", DEFAULT_SPLIT, 1, 1024)


def _defines(vocab: int, threads: int, split: int = DEFAULT_SPLIT) -> dict:
    return {"VOCAB": vocab, "THREADS": threads, "SPLIT": split}


def _vocab_ok(vocab: int) -> bool:
    """Can the kernel serve this vocab at all?

    Every positive vocab is servable -- the kernel has a scalar path for the non-multiple
    of 8 case -- so this only rejects nonsense the config reader may hand us. The upper
    bound mirrors the source's ``VOCAB < 0x7fffffff`` static_assert (the kNoIndex sentinel).
    """
    return isinstance(vocab, int) and 0 < vocab < 0x7fffffff


# --------------------------------------------------------------------------------------
# the envelope: what the compiled kernel is allowed to see, re-checked on every call
# --------------------------------------------------------------------------------------

def _logits_reason(logits) -> str | None:
    """None if the compiled kernel may run on these logits, else why not (one word)."""
    import torch

    if logits.device.type != "cuda":
        return "device"
    if logits.dtype is not torch.bfloat16:
        return "dtype"
    if logits.dim() != 2:
        return "rank"
    if not logits.is_contiguous():
        return "stride"
    if int(logits.shape[1]) != _state["vocab"]:
        return "vocab"
    if int(logits.shape[0]) > _state["ws_rows"]:
        return "rows"
    if logits.device.index != _state["ws_dev"]:
        # The workspace is one device's memory; another device's stream cannot atomicMax
        # into it. Single-GPU is the served config, so this is a guard, not a feature.
        return "ws-device"
    # A misaligned 16-byte load faults; it does not run slowly. Torch's caching allocator
    # hands out 512-byte-aligned blocks, so this only ever fires on a narrowed view. Gated
    # on the VECTOR path actually being what compiled: a scalar specialization
    # (vocab % 8 != 0) reads one bf16 at a time and needs only the 2-byte alignment every
    # torch tensor has by construction.
    if _state["vec"] and logits.data_ptr() % ALIGN:
        return "align"
    return None


def _out_reason(out, rows: int) -> str | None:
    import torch

    if out.device.type != "cuda":
        return "out-device"
    if out.dtype is not torch.int64:
        return "out-dtype"
    if out.dim() != 1 or int(out.shape[0]) != rows:
        return "out-shape"
    if not out.is_contiguous():
        return "out-stride"
    return None


# --------------------------------------------------------------------------------------
# the launch
# --------------------------------------------------------------------------------------

_ONE_BLOCK = (1, 1, 1)


def _launch(logits, out, rows: int) -> None:
    """reset -> partial -> finalize, back to back on whatever stream the caller is on.

    NOTHING IS BUILT HERE. The three ctypes argument arrays were constructed at arm time and
    only their holders' ``.value`` fields move; the grid tuples are memoized per row count;
    the stream accessor is a bound function, not an attribute walk through
    ``torch.cuda.current_stream()``. Measured on this host: three fresh ``pack_args`` calls
    at the pre-fix cost were ~17 us of Python per token pick, a fresh pack after nvrtc's
    ``addressof`` fix ~4.5 us, and the cached-and-mutated arrays below ~0.3 us. At ~1 ms of
    GPU per decode step that difference is the entire reason this patch is worth having.

    The stream is passed EXPLICITLY rather than left to the launcher's default so the intent
    survives a future launcher change: under ``torch.cuda.graph`` capture this is the capture
    stream, which is the only way the launches land in the graph. All three are ordinary
    kernel launches on one stream, so all three are capture-legal and the ordering between
    them is the stream's.
    """
    st = _state
    stream = st["raw_stream"](logits.device.index)
    ws_ptr = st["ws_ptr"]
    args = st["args"]

    arr, hold = args["reset"]
    hold[0].value = ws_ptr
    hold[1].value = rows
    st["reset"](grid=_ONE_BLOCK, block=st["edge_block"], args=arr, stream=stream)

    arr, hold = args["partial"]
    hold[0].value = logits.data_ptr()
    hold[1].value = ws_ptr
    hold[2].value = rows
    grid = st["grids"].get(rows)
    if grid is None:
        grid = st["grids"][rows] = (st["split"], rows, 1)
    st["partial"](grid=grid, block=st["block"], args=arr, stream=stream)

    arr, hold = args["finalize"]
    hold[0].value = ws_ptr
    hold[1].value = out.data_ptr()
    hold[2].value = rows
    st["finalize"](grid=_ONE_BLOCK, block=st["edge_block"], args=arr, stream=stream)


# --------------------------------------------------------------------------------------
# op impls. Both degrade to torch.argmax INLINE -- a surprise must be slow, never wrong.
# --------------------------------------------------------------------------------------

def _torch_out(logits, out) -> None:
    """``out := torch.argmax(logits, -1)`` for ANY ``out`` the caller handed us.

    ``torch.argmax(..., out=out)`` RAISES for precisely the shapes that route here: an int32
    out (torch refuses the dtype outright) and a length mismatch (the implied resize lands on
    a view and torch refuses that too). Raising is the one thing a fallback must not do, so
    the degrade is compute-then-copy: ``copy_`` casts, the slice trims, and both write
    THROUGH to the caller's storage -- which is why the contiguous case goes through
    ``reshape(-1)`` (a view) and the strided case through ``copy_`` on ``out`` itself
    (``reshape`` on a non-contiguous tensor may return a COPY, and writing into that would
    silently leave the caller's buffer stale).
    """
    import torch

    # `.reshape(-1)` so a 0-d result (which is what a rank-refused input produces) still
    # slices; this path exists precisely for inputs the kernel would not touch.
    res = torch.argmax(logits, dim=-1).reshape(-1)
    try:
        if out.is_contiguous():
            dst = out.reshape(-1)
            n = min(int(dst.numel()), int(res.numel()))
            dst[:n].copy_(res[:n])
        else:
            out.copy_(res.reshape(out.shape))
    except Exception:
        log.exception("vtl: greedy_argmax fallback could not write out (%s %s <- %s %s)",
                      tuple(out.shape), out.dtype, tuple(res.shape), res.dtype)


def _fail() -> None:
    """Latch the whole kernel off for the run, from inside an ``except`` block.

    A driver error is not a one-off: whatever made ``cuLaunchKernel`` fail (a corrupted
    context, an unloaded module, an illegal address left behind by another kernel) will make
    the NEXT token pick fail identically, so an unlatched impl turns one bad launch into an
    engine that dies on every step forever. This is the ONLY message that says "for the rest
    of the run"; an envelope miss is per-call and deliberately does not latch.
    """
    _state["disabled"] = True
    log.exception("vtl: greedy_argmax launch failed; torch.argmax for the rest of the run")


def _argmax_impl(logits):
    """``greedy_argmax_i64(Tensor logits) -> Tensor``: allocates [rows] int64.

    The eager sampler's shape. It allocates, so it is NOT the variant nstep captures.
    """
    import torch

    if _state["disabled"]:
        return torch.argmax(logits, dim=-1)
    reason = _logits_reason(logits)
    if reason is not None:
        _warn_once("impl:" + reason,
                   "vtl: greedy_argmax called outside its envelope (%s); "
                   "torch.argmax for this call", reason)
        return torch.argmax(logits, dim=-1)
    rows = int(logits.shape[0])
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows:
        try:
            _launch(logits, out, rows)
        except Exception:
            # A driver error here is not a one-off: the next token pick would raise the
            # same way and kill the engine step again, forever. Latch, then still return a
            # value so THIS step completes.
            _fail()
            return torch.argmax(logits, dim=-1)
    return out


def _argmax_out_impl(logits, out) -> None:
    """``greedy_argmax_i64_out(Tensor logits, Tensor! out) -> ()``: no alloc, no sync.

    The capturable variant, and what nstep's ``_ARGMAX`` is rebound to DIRECTLY (see
    ``_rebind_nstep``). Everything it touches is either the caller's persistent buffer, our
    persistent workspace or a compile-time constant, so a replay reruns exactly the launches
    that were captured.
    """
    if _state["disabled"]:
        _torch_out(logits, out)
        return
    # BEFORE the shape read, deliberately: `logits.shape[0]` is an IndexError on a 0-d
    # tensor, and "hand us something absurd" must degrade, not raise.
    reason = _logits_reason(logits)
    if reason is None:
        reason = _out_reason(out, int(logits.shape[0]))
    if reason is not None:
        _warn_once("out:" + reason,
                   "vtl: greedy_argmax_out called outside its envelope (%s); "
                   "torch.argmax for this call", reason)
        _torch_out(logits, out)
        return
    rows = int(logits.shape[0])
    if rows:
        try:
            _launch(logits, out, rows)
        except Exception:
            _fail()
            _torch_out(logits, out)


def _register_ops() -> bool:
    """Define both ops (schema + CUDA impl + fake). Idempotent across re-apply.

    The Library is created ONLY when something actually needs defining, and ``_state["lib"]``
    is written once and never replaced: a second Library object assigned over the first would
    drop the first, and its destructor DEREGISTERS every op it defined -- i.e. a re-apply
    would silently delete the live ops. Same shape as gdn_kernels._register_op.
    """
    import torch

    ns = getattr(torch.ops, NS, None)
    missing = [name for name in (OP, OP_OUT) if not hasattr(ns or object(), name)]
    if missing:
        lib = _state["lib"]
        if lib is None:
            lib = torch.library.Library(NS, "FRAGMENT")  # noqa: TOR901 -- plugin seam
            _state["lib"] = lib   # keep-alive, set ONCE; see the docstring
        if OP in missing:
            lib.define(f"{OP}(Tensor logits) -> Tensor")
            lib.impl(OP, _argmax_impl, "CUDA")
        if OP_OUT in missing:
            lib.define(f"{OP_OUT}(Tensor logits, Tensor! out) -> ()")
            lib.impl(OP_OUT, _argmax_out_impl, "CUDA")

    # Fakes so a traced/compiled call (tests, torch.compile) can run without a device. ONE
    # try/except EACH: sharing one would let a failure on the first silently skip the second.
    def _fake(logits):  # noqa: ANN001
        return logits.new_empty((logits.shape[0],), dtype=torch.int64)

    def _fake_out(logits, out):  # noqa: ANN001
        return None

    for name, fn in ((OP, _fake), (OP_OUT, _fake_out)):
        try:
            torch.library.register_fake(f"{NS}::{name}")(fn)
        except Exception as exc:
            # Already registered on a re-apply -- fine, but say which one and why.
            log.debug("vtl: greedy_argmax: fake for %s::%s not registered (%r)", NS, name, exc)

    live = getattr(torch.ops, NS, None)
    if not (hasattr(live or object(), OP) and hasattr(live or object(), OP_OUT)):
        return False
    # Resolved ONCE, down to the OVERLOAD (the house idiom -- silu_mul_quant.py:88): calling
    # `.default` skips the packet's per-call overload resolution.
    _state["op_out"] = getattr(live, OP_OUT).default
    return True


# --------------------------------------------------------------------------------------
# nstep indirection
# --------------------------------------------------------------------------------------

def _rebind_nstep() -> bool:
    """Point nstep's ``_ARGMAX`` at ``_argmax_out_impl``. False if there is nothing to point.

    STRAIGHT AT THE PYTHON FUNCTION, not at ``torch.ops.vllm_cuda.greedy_argmax_i64_out``.
    The op stays registered -- it is the documented seam the forked sampler and any external
    caller resolve -- but nstep's eager token-1 path calls this once per token on the
    critical path, and the dispatcher hop costs 5-10 us there for nothing: the impl this
    would dispatch INTO is exactly the function bound here, there is no second kernel behind
    the schema, and nstep is not a functionalized/compiled caller.

    IMPORT-SAFE ON PURPOSE: we look nstep up in ``sys.modules`` instead of importing it.
    If that module failed to import (a vLLM version bump moved a symbol) the registry has
    already logged and skipped it, and force-importing it here would resurrect the failure
    inside OUR arming path. No nstep, no rebind, no harm -- the burst is not running either.
    """
    import sys

    mod = sys.modules.get("vtl.patches.nstep_decode")
    if mod is None or not hasattr(mod, "_ARGMAX"):
        return False
    mod._ARGMAX = _argmax_out_impl
    return True


# --------------------------------------------------------------------------------------
# arming
# --------------------------------------------------------------------------------------

def _vocab_from(model_config) -> int:
    try:
        return int(model_config.get_vocab_size())
    except Exception:
        return int(getattr(getattr(model_config, "hf_text_config", None),
                           "vocab_size", 0) or 0)


def _reset_state() -> None:
    """Back to the hard-fallback envelope: vocab 0, no launchers, no workspace.

    ``lib`` and ``op_out`` are deliberately NOT cleared -- once torch owns a Library the
    registration is live for the process and dropping our reference would deregister it.
    With vocab 0 the registered impls answer "vocab" for every tensor and call torch.argmax,
    which is exactly the behaviour a failed arm should leave behind.
    """
    _state.update(
        installed=False, vocab=0, threads=0, split=0, vec=False,
        reset=None, partial=None, finalize=None,
        ws=None, ws_ptr=0, ws_dev=-1, ws_rows=0,
        raw_stream=None, args=None, grids=None, block=None, edge_block=None,
        nstep=False,
    )


def _build_launch_state(vocab: int, threads: int, split: int, launchers) -> None:
    """Everything ``_launch`` reads, built ONCE. Leaves ``vocab`` unpublished on purpose."""
    import torch

    from vtl import nvrtc

    dev = torch.cuda.current_device()
    # int64 storage, read as unsigned long long by the kernel: torch's uint64 support is
    # version-dependent and the bit pattern is what matters, not the signedness torch prints.
    ws = torch.zeros(MAX_WS_ROWS, dtype=torch.int64, device=f"cuda:{dev}")
    _state.update(
        threads=threads, split=split, vec=(vocab % VEC == 0),
        reset=launchers[0], partial=launchers[1], finalize=launchers[2],
        ws=ws, ws_ptr=ws.data_ptr(), ws_dev=dev, ws_rows=MAX_WS_ROWS,
        block=(threads, 1, 1), edge_block=(EDGE_THREADS, 1, 1),
        grids={},
        # Bound once. `torch.cuda.current_stream().cuda_stream` builds a Stream object and
        # walks three attributes per call; this is the same number the driver wants, in one
        # C call taking the device ordinal. Private, hence the guarded fallback: a torch
        # that renamed it should cost the microseconds, not the kernel.
        raw_stream=_raw_stream_fn(torch),
        args={
            # Built with real packing so the slot-holds-addressof invariant is nvrtc's, not
            # a second copy of it here; only `.value` moves afterwards, and the holders do
            # not, so the void** array stays valid forever.
            "reset": _holder_pair(nvrtc.pack_args(0, ("i", 0))),
            "partial": _holder_pair(nvrtc.pack_args(0, 0, ("i", 0))),
            "finalize": _holder_pair(nvrtc.pack_args(0, 0, ("i", 0))),
        },
    )


def _holder_pair(arr):
    return (arr, arr._vtl_holders)


def _raw_stream_fn(torch):
    """``device_index -> cudaStream_t`` in one C call, or the portable long way round."""
    fn = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if fn is not None:
        return fn
    log.info("vtl: greedy_argmax: torch._C._cuda_getCurrentRawStream is gone; "
             "falling back to torch.cuda.current_stream() per launch")
    return lambda index: torch.cuda.current_stream(index).cuda_stream


# The boot parity gate's row book. Each entry is (name, builder) and builders get
# (torch, vocab, device) -- every one of them is a shape where a hand-written reduction and
# torch can legally disagree, and every one of them has cost somebody a debugging day.
def _gate_rows(torch, vocab: int, device):
    import math

    lo, mid, hi = 0, vocab // 2, vocab - 1

    def full(fill):
        return torch.full((1, vocab), fill, dtype=torch.bfloat16, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(0x6772)
    rand = torch.randn(4, vocab, generator=gen, dtype=torch.float32,
                       device=device).to(torch.bfloat16)

    rows = [("random", rand)]
    for fill in (0.0, 1.0, -1.0):
        rows.append((f"all-equal({fill})", full(fill)))
    rows.append(("all--inf", full(float("-inf"))))

    dup = full(-3.0)
    for i in {lo, mid, hi}:
        dup[0, i] = 2.5
    rows.append(("duplicate-maxima", dup))

    nan_first = rand[:1].clone()
    nan_first[0, lo] = float("nan")
    rows.append(("nan-first", nan_first))
    # NOT at index 0: a reduction that treats NaN as "skip" gets index 0 right by accident.
    nan_late = rand[:1].clone()
    nan_late[0, hi] = float("nan")
    rows.append(("nan-non-first", nan_late))

    nan_inf = full(float("-inf"))
    nan_inf[0, mid] = float("nan")
    nan_inf[0, hi] = float("inf")
    rows.append(("nan-vs-+inf", nan_inf))

    many = rand[:1].clone()
    for i in {min(31, hi), mid, hi}:
        many[0, i] = float("nan")
    rows.append(("several-nan", many))

    zeros = full(-1.0)
    zeros[0, min(3, hi)] = -0.0
    zeros[0, mid] = 0.0
    zeros[0, hi] = -0.0
    rows.append(("+-0.0-mix", zeros))
    alt = torch.where(
        torch.arange(vocab, device=device) % 2 == 0,
        torch.zeros(vocab, device=device),
        torch.full((vocab,), -0.0, device=device),
    ).to(torch.bfloat16).view(1, vocab)
    rows.append(("+-0.0-alternating", alt))

    edge = full(float("-inf"))
    edge[0, hi] = -math.inf if vocab == 1 else -1e30
    rows.append(("masked-except-last", edge))
    return rows


def _boot_parity_ok(vocab: int) -> bool:
    """One shot, ~1 ms: does the compiled kernel agree with torch.argmax bit for bit?

    Runs on the ACTUAL device at the ACTUAL compiled vocab, through ``_launch`` directly --
    NOT through the ops, which are not registered yet and whose envelope check would read
    the still-unpublished ``vocab`` and refuse. A single mismatch means we register nothing:
    the fork resolver keeps ``logits.argmax``, nstep keeps its own ``torch.argmax``, and the
    only cost of the whole patch is this millisecond.

    This is the parity oracle. nstep's ``_graph_matches_eager`` cannot be one -- after the
    rebind both of its arms run this kernel.
    """
    import torch

    for name, logits in _gate_rows(torch, vocab, f"cuda:{_state['ws_dev']}"):
        rows = int(logits.shape[0])
        got = torch.empty(rows, dtype=torch.int64, device=logits.device)
        _launch(logits, got, rows)
        want = torch.argmax(logits, dim=-1)
        torch.cuda.synchronize()
        if not torch.equal(got, want):
            bad = (got != want).nonzero().flatten().tolist()
            log.error(
                "vtl: greedy_argmax BOOT PARITY FAILED on row set %r (VOCAB=%d THREADS=%d "
                "SPLIT=%d): rows %s differ, kernel=%s torch=%s; registering nothing, "
                "torch.argmax stands",
                name, vocab, _state["threads"], _state["split"], bad[:8],
                got.tolist()[:8], want.tolist()[:8],
            )
            return False
    return True


def _arm(model_config) -> None:
    """Runs once after the model loads: read the vocab, compile, PROVE parity, then install."""
    if _state["armed"]:
        return
    _state["armed"] = True

    from vtl import nvrtc

    vocab = _vocab_from(model_config)
    if not _vocab_ok(vocab):
        log.info("vtl: greedy_argmax: no usable vocab_size on the loaded config "
                 "(got %r); torch.argmax stands", vocab)
        return
    threads = _threads_from_env()
    split = _split_from_env()

    defines = _defines(vocab, threads, split)
    # Three entries out of ONE cubin: build_cubin's disk cache makes calls 2 and 3 a read.
    launchers = [nvrtc.compile_kernel(KERNEL, defines, entry=e) for e in ENTRIES]
    if any(k is None for k in launchers):
        # Not an error: the sampler fork's resolver and nstep's _ARGMAX both already mean
        # torch.argmax, and neither is touched until an op exists.
        log.info("vtl: greedy_argmax: NVRTC compile unavailable/failed "
                 "(VOCAB=%d THREADS=%d SPLIT=%d); torch.argmax stands",
                 vocab, threads, split)
        return

    try:
        _build_launch_state(vocab, threads, split, launchers)
        if not _boot_parity_ok(vocab):
            _reset_state()
            return
        if not _register_ops():
            raise RuntimeError(f"torch.ops.{NS}.{OP} did not appear after registration")
    except Exception:
        log.exception("vtl: greedy_argmax: arming failed after compile; torch.argmax stands")
        _reset_state()
        return

    # Published LAST: `vocab` is what opens the envelope, so nothing before this line can be
    # served by the kernel even if an op were somehow already registered.
    _state.update(vocab=vocab, installed=True)
    _state["nstep"] = _rebind_nstep()
    # The one-line tier log `make verify`-style checks grep for. Rebind and nstep's own
    # enabled state are SEPARATE facts: a rebind onto a module whose burst never arms is
    # still a correct rebind, and reporting them as one hides which of the two is missing.
    try:
        import sys

        nstep_mod = sys.modules.get("vtl.patches.nstep_decode")
        nstep_on = bool(getattr(getattr(nstep_mod, "BURST", None), "armed", False))
    except Exception:
        nstep_on = False
    log.info(
        "vtl: greedy_argmax tier active: %s::%s + %s -> NVRTC "
        "(VOCAB=%d THREADS=%d SPLIT=%d vec=%s nstep=%s)",
        NS, OP, OP_OUT, vocab, threads, split, vocab % VEC == 0,
        "absent" if not _state["nstep"] else ("rebound" if nstep_on
                                              else "rebound-but-nstep-disabled"),
    )


@register_patch("greedy_argmax", default=True)
def apply() -> None:
    from vtl import nvrtc

    if not nvrtc.enabled():
        log.info("vtl: greedy_argmax selected but VTL_NVRTC is off; no-op")
        return

    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch="greedy_argmax"):
        return
    orig = BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, *args, **kwargs):
        model = orig(self, vllm_config, model_config, *args, **kwargs)
        try:
            _arm(model_config)
        except Exception:
            # Never let an optimization kill the load. Nothing is registered until the
            # compile AND the parity gate succeeded, so both consumers are still on
            # torch.argmax here.
            log.exception("vtl: greedy_argmax arming failed; torch.argmax stands")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, orig, patch="greedy_argmax")
    log.info("vtl: greedy_argmax armed (compiles + parity-gates + registers at model load)")


def _self_check() -> None:
    """Runs anywhere: no GPU, no torch, no vLLM."""
    import sys
    import types
    from pathlib import Path

    from vtl import nvrtc
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "greedy_argmax")
    # Enabled by default per project decision (2026-08-17), matching the shipped compose.
    # The safety story is the ladder, not the gate: NVRTC compile -> nothing registered, and
    # even a successful compile registers the op ONLY after the boot parity gate matches
    # torch.argmax bit for bit on adversarial rows. Every consumer (the V2 fast path, nstep's
    # three call sites) resolves through _ARGMAX and falls back to torch.argmax when the op
    # is absent, so default-on can cost a compile, never a wrong token.
    assert patch.default is True, "shipped ENABLED; the boot parity gate is the safety net"
    assert is_enabled(patch) is True
    os.environ["VTL_ENABLE_GREEDY_ARGMAX"] = "0"
    assert is_enabled(patch) is False, "the gate must still be able to turn it off"
    os.environ["VTL_ENABLE_GREEDY_ARGMAX"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GREEDY_ARGMAX")

    # -- block width / split: the defaults ship, nonsense does not reach the compiler --
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS", None)
    assert _threads_from_env() == DEFAULT_THREADS
    for good in ("256", "512", "1024"):
        os.environ["VTL_GREEDY_ARGMAX_THREADS"] = good
        assert _threads_from_env() == int(good)
    for bad in ("0", "17", "2048", "-32", "banana", ""):
        os.environ["VTL_GREEDY_ARGMAX_THREADS"] = bad
        assert _threads_from_env() == DEFAULT_THREADS, bad
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS")

    os.environ.pop("VTL_GREEDY_ARGMAX_SPLIT", None)
    assert _split_from_env() == DEFAULT_SPLIT
    for good in ("1", "8", "32", "1024"):
        os.environ["VTL_GREEDY_ARGMAX_SPLIT"] = good
        assert _split_from_env() == int(good)
    for bad in ("0", "-4", "2048", "banana", ""):
        os.environ["VTL_GREEDY_ARGMAX_SPLIT"] = bad
        assert _split_from_env() == DEFAULT_SPLIT, bad
    os.environ.pop("VTL_GREEDY_ARGMAX_SPLIT")

    # -- vocab envelope --
    assert _vocab_ok(248320) is True
    assert _vocab_ok(1) is True      # the scalar path serves any positive vocab
    assert _vocab_ok(0) is False
    assert _vocab_ok(-8) is False
    assert _vocab_ok(0x7fffffff) is False, "kNoIndex must stay outside the index space"

    # -- the source ships, and its three entry symbols are the ones we ask the loader for --
    src = nvrtc.load_source(KERNEL)
    assert src, "vtl/kernels/greedy_argmax.cu missing from the package"
    for entry in ENTRIES:
        assert f'extern "C" __global__' in src and f" {entry}(" in src, entry
    assert '#error "NVRTC: -DVOCAB=<vocab_size> is required"' in src
    assert f"#define THREADS {DEFAULT_THREADS}" in src, "kernel/patch THREADS default drift"
    assert f"#define SPLIT {DEFAULT_SPLIT}" in src, "kernel/patch SPLIT default drift"

    # -- one cubin identity per specialization: a collision would serve a cubin compiled
    #    for ANOTHER vocab, which reads off the end of the row --
    sets = [_defines(248320, 512, 16), _defines(248320, 256, 16),
            _defines(151936, 512, 16), _defines(248320, 512, 1)]
    keys = {nvrtc.cache_key(src, s, "90a", "12.8") for s in sets}
    assert len(keys) == len(sets), "define sets must not collide in the cubin cache"
    assert nvrtc.cache_key(src, sets[0], "90", "12.8") not in keys, "arch is identity too"

    # -- the config reader: three shapes it must survive --
    class _Works:
        def get_vocab_size(self):
            return 248320

    class _RaisesWithFallback:
        hf_text_config = types.SimpleNamespace(vocab_size=151936)

        def get_vocab_size(self):
            raise AttributeError("moved in this vLLM version")

    class _Neither:
        def get_vocab_size(self):
            raise RuntimeError("nope")

    assert _vocab_from(_Works()) == 248320
    assert _vocab_from(_RaisesWithFallback()) == 151936
    assert _vocab_from(_Neither()) == 0 and _vocab_ok(_vocab_from(_Neither())) is False

    # -- while NVRTC is off, compiling is a no-op and apply() installs nothing --
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(KERNEL, sets[0], entry=ENTRY_PARTIAL) is None
    apply()   # vLLM may be absent; with VTL_NVRTC off this must return cleanly
    assert _state["installed"] is False

    # -- with NVRTC on but vLLM absent, apply() may raise (registry isolates it) but must
    #    not leave the module half-installed --
    os.environ["VTL_NVRTC"] = "1"
    try:
        apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_NVRTC", None)
    assert _state["installed"] is False
    assert _state["partial"] is None and _state["vocab"] == 0
    assert _state["disabled"] is False, "nothing ran, so nothing may be latched off"

    # -- the nstep rebind is import-safe: nstep_decode is not imported here, so this must
    #    report "nothing to rebind" rather than dragging vLLM in --
    if "vtl.patches.nstep_decode" not in sys.modules:
        assert _rebind_nstep() is False

    # ...and the POSITIVE half, with a stand-in module: the rebind must land on the module
    # global by name and must be the impl FUNCTION, not the dispatcher (see _rebind_nstep).
    stub = types.ModuleType("vtl.patches.nstep_decode")
    stub._ARGMAX = object()
    sys.modules["vtl.patches.nstep_decode"] = stub
    try:
        assert _rebind_nstep() is True
        assert stub._ARGMAX is _argmax_out_impl
    finally:
        del sys.modules["vtl.patches.nstep_decode"]

    # -- the module is actually in the patch package's apply list. POSITION IS NOT LOAD-
    #    BEARING (see the comment there): the rebind goes through sys.modules at model load,
    #    so it does not matter whether nstep_decode was imported before or after us. --
    from vtl import patches as _patches

    assert "greedy_argmax" in _patches._MODULES

    print("greedy_argmax self-check ok")


if __name__ == "__main__":
    _self_check()
