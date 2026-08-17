"""Fused greedy argmax: one NVRTC launch replaces ``torch.argmax`` on the sampling path.

WHAT IT REPLACES. On a provably-greedy step the whole sampler collapses to one reduction
over ``[num_reqs, 248320]`` bf16 logits. ``torch.argmax`` already does that in a single
fused pass, so the win here is NOT the reduction -- it is everything around it:

  * VOCAB is a compile-time ``-D`` constant, so the row loop's trip count and every offset
    fold to immediates and the loads are 16-byte vectors (vtl/kernels/greedy_argmax.cu);
  * the launch is one ``cuLaunchKernel``, with no TensorIterator build, no dtype dispatch
    and no reduction-config heuristic between the caller and the SM;
  * the OUT variant writes straight into the caller's persistent int64 buffer -- no
    allocation and no ``out=`` copy, which is what lets the N-step burst capture it.

TWO CONSUMERS, BOTH PRE-EXISTING SEAMS -- this patch only fills them:

  * the forked V2 sampler (``vtl/vllm_patches/v0.25.0/v2_greedy_sampler.patch``) resolves
    ``torch.ops.vllm_cuda.greedy_argmax_i64`` ONCE, on the first greedy step, behind
    ``VTL_V2_GREEDY_ARGMAX_KERNEL``; if the op is not registered it keeps
    ``logits.argmax(dim=-1)`` forever. So "did not compile" costs nothing and needs no
    coordination -- we simply never register the op.
  * ``vtl/patches/nstep_decode.py`` calls its module-level ``_ARGMAX`` indirection at the
    three places it picks a token (burst body, prologue, eager token-1). We rebind that
    global at model load, i.e. BEFORE ``_capture_burst_graphs`` runs, so the fused launch
    is what gets captured into the burst graphs -- and therefore what the Rust runner
    replays too, since it replays the same execs.

WHY IT MUST BE BIT-EXACT, not merely close. nstep's boot-time ``_graph_matches_eager``
bit-compares the graph's tokens against an eager step and demotes the whole burst on a
single differing token. So this kernel reproduces ``torch.argmax``'s contract exactly:
NaN is the maximum, and every tie -- equal maxima, an all-``-inf`` masked row, several
NaN -- resolves to the LOWEST index. bench/test_greedy_argmax.py holds that to
bit-equality against torch on a GPU.

ARMING. Same ``BaseModelLoader.load_model`` seam as ``nvrtc_block_quant`` /
``quant_w4a8`` / ``l2_persist`` (the registry's ``patch=`` names keep the stack sound).
After the model loads we read the REAL vocab off the config, compile, and only then
register the ops and rebind nstep. ``compile_kernel`` returning None -- NVRTC off, no
cuda-python, bad source, no device -- leaves both consumers on ``torch.argmax``, which is
the behaviour they already have today.

Gate: ``VTL_ENABLE_GREEDY_ARGMAX=1`` (default OFF) **and** ``VTL_NVRTC=1`` (the layer-wide
switch). Block width: ``VTL_GREEDY_ARGMAX_THREADS`` (default 512, must be a whole number
of warps up to 1024).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.greedy_argmax")

KERNEL = "greedy_argmax"          # vtl/kernels/greedy_argmax.cu
ENTRY = "greedy_argmax_i64"       # extern "C" symbol inside it
NS = "vllm_cuda"                  # our own op namespace (vtl/csrc/torch_bindings.cpp)
OP = "greedy_argmax_i64"          # torch.ops.vllm_cuda.<OP>(Tensor) -> Tensor
OP_OUT = "greedy_argmax_i64_out"  # torch.ops.vllm_cuda.<OP_OUT>(Tensor, Tensor!) -> ()

DEFAULT_THREADS = 512
VEC = 8          # bf16 elems per 16-byte load, mirrors the kernel
ALIGN = 16       # bytes the vectorized row path needs

# Populated by _arm(); read by the op impls. `lib` must stay referenced or torch drops the
# registration along with it.
_state: dict = {
    "installed": False,
    "armed": False,
    "vocab": 0,
    "threads": 0,
    "launch": None,
    "lib": None,
    "op_out": None,
    "nstep": False,
}
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _threads_from_env() -> int:
    """Block width, validated. A bad value falls back rather than failing the compile."""
    try:
        threads = int(os.environ.get("VTL_GREEDY_ARGMAX_THREADS", "").strip()
                      or DEFAULT_THREADS)
    except ValueError:
        return DEFAULT_THREADS
    if threads < 32 or threads > 1024 or threads % 32 != 0:
        log.info("vtl: greedy_argmax: VTL_GREEDY_ARGMAX_THREADS=%s is not a legal block "
                 "width; using %d", threads, DEFAULT_THREADS)
        return DEFAULT_THREADS
    return threads


def _defines(vocab: int, threads: int) -> dict:
    return {"VOCAB": vocab, "THREADS": threads}


def _vocab_ok(vocab: int) -> bool:
    """Can the kernel serve this vocab at all?

    Every positive vocab is servable -- the kernel has a scalar path for the non-multiple
    of 8 case -- so this only rejects nonsense the config reader may hand us.
    """
    return isinstance(vocab, int) and 0 < vocab < (1 << 31)


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
    # A misaligned 16-byte load faults; it does not run slowly. Torch's caching allocator
    # hands out 512-byte-aligned blocks, so this only ever fires on a narrowed view.
    if logits.data_ptr() % ALIGN:
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


def _launch(logits, out, rows: int) -> None:
    """One kernel, one row per block, on whatever stream the caller is on.

    The stream is passed EXPLICITLY rather than left to the launcher's default so the
    intent survives a future launcher change: under ``torch.cuda.graph`` capture this is
    the capture stream, which is the only way the launch lands in the graph.
    """
    import torch

    from vtl import nvrtc

    _state["launch"](
        grid=(rows, 1, 1),
        block=(_state["threads"], 1, 1),
        args=nvrtc.pack_args(logits.data_ptr(), out.data_ptr(), ("i", rows)),
        stream=torch.cuda.current_stream().cuda_stream,
    )


# --------------------------------------------------------------------------------------
# op impls. Both degrade to torch.argmax INLINE -- a surprise must be slow, never wrong.
# --------------------------------------------------------------------------------------

def _argmax_impl(logits):
    """``greedy_argmax_i64(Tensor logits) -> Tensor``: allocates [rows] int64.

    The eager sampler's shape. It allocates, so it is NOT the variant nstep captures.
    """
    import torch

    reason = _logits_reason(logits)
    if reason is not None:
        _warn_once("impl:" + reason,
                   "vtl: greedy_argmax called outside its envelope (%s); "
                   "falling back to torch.argmax for the rest of the run", reason)
        return torch.argmax(logits, dim=-1)
    rows = int(logits.shape[0])
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows:
        _launch(logits, out, rows)
    return out


def _argmax_out_impl(logits, out) -> None:
    """``greedy_argmax_i64_out(Tensor logits, Tensor! out) -> ()``: no alloc, no sync.

    The capturable variant. Everything it touches is either the caller's persistent buffer
    or a compile-time constant, so a replay reruns exactly the launch that was captured.
    """
    import torch

    rows = int(logits.shape[0])
    reason = _logits_reason(logits) or _out_reason(out, rows)
    if reason is not None:
        _warn_once("out:" + reason,
                   "vtl: greedy_argmax_out called outside its envelope (%s); "
                   "falling back to torch.argmax for the rest of the run", reason)
        torch.argmax(logits, dim=-1, out=out)
        return
    if rows:
        _launch(logits, out, rows)


def _register_ops() -> bool:
    """Define both ops (schema + CUDA impl + fake). Idempotent across re-apply."""
    import torch

    ns = getattr(torch.ops, NS, None)
    lib = torch.library.Library(NS, "FRAGMENT")  # noqa: TOR901 -- plugin seam
    if not hasattr(ns or object(), OP):
        lib.define(f"{OP}(Tensor logits) -> Tensor")
        lib.impl(OP, _argmax_impl, "CUDA")
    if not hasattr(ns or object(), OP_OUT):
        lib.define(f"{OP_OUT}(Tensor logits, Tensor! out) -> ()")
        lib.impl(OP_OUT, _argmax_out_impl, "CUDA")

    # Fakes so a traced/compiled call (tests, torch.compile) can run without a device.
    try:
        @torch.library.register_fake(f"{NS}::{OP}")
        def _fake(logits):  # noqa: ANN001
            return logits.new_empty((logits.shape[0],), dtype=torch.int64)

        @torch.library.register_fake(f"{NS}::{OP_OUT}")
        def _fake_out(logits, out):  # noqa: ANN001
            return None
    except Exception:
        pass  # already registered on a re-apply -- fine

    _state["lib"] = lib   # keep-alive: dropping it would deregister both ops
    live = getattr(torch.ops, NS, None)
    if not (hasattr(live or object(), OP) and hasattr(live or object(), OP_OUT)):
        return False
    # Resolved ONCE, the same way the forked sampler resolves its own backend: nstep's
    # rebind then costs one dict lookup per token instead of an attribute walk.
    _state["op_out"] = getattr(live, OP_OUT)
    return True


# --------------------------------------------------------------------------------------
# nstep indirection
# --------------------------------------------------------------------------------------

def _nstep_argmax(logits, out) -> None:
    """What ``nstep_decode._ARGMAX`` becomes once the kernel is live.

    Through the dispatcher rather than straight at ``_argmax_out_impl``: the op is the
    documented seam, it keeps the two consumers on ONE code path, and a functionalized or
    compiled caller sees a registered op instead of an opaque Python function.
    """
    _state["op_out"](logits, out)


def _rebind_nstep() -> bool:
    """Point nstep's ``_ARGMAX`` at the fused op. False if there is nothing to point.

    IMPORT-SAFE ON PURPOSE: we look nstep up in ``sys.modules`` instead of importing it.
    If that module failed to import (a vLLM version bump moved a symbol) the registry has
    already logged and skipped it, and force-importing it here would resurrect the failure
    inside OUR arming path. No nstep, no rebind, no harm -- the burst is not running either.
    """
    import sys

    mod = sys.modules.get("vtl.patches.nstep_decode")
    if mod is None or not hasattr(mod, "_ARGMAX"):
        return False
    mod._ARGMAX = _nstep_argmax
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


def _arm(model_config) -> None:
    """Runs once after the model loads: read the vocab, compile, then (only then) install."""
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

    launch = nvrtc.compile_kernel(KERNEL, _defines(vocab, threads), entry=ENTRY)
    if launch is None:
        # Not an error: the sampler fork's resolver and nstep's _ARGMAX both already mean
        # torch.argmax, and neither is touched until an op exists.
        log.info("vtl: greedy_argmax: NVRTC compile unavailable/failed "
                 "(VOCAB=%d THREADS=%d); torch.argmax stands", vocab, threads)
        return

    _state.update(vocab=vocab, threads=threads, launch=launch)
    try:
        if not _register_ops():
            raise RuntimeError(f"torch.ops.{NS}.{OP} did not appear after registration")
    except Exception:
        log.exception("vtl: greedy_argmax: op registration failed; torch.argmax stands")
        _state.update(launch=None, lib=None, op_out=None)
        return

    _state["installed"] = True
    _state["nstep"] = _rebind_nstep()
    # The one-line tier log `make verify`-style checks grep for.
    log.info(
        "vtl: greedy_argmax tier active: %s::%s + %s -> NVRTC "
        "(VOCAB=%d THREADS=%d vec=%s nstep=%s)",
        NS, OP, OP_OUT, vocab, threads, vocab % VEC == 0,
        "rebound" if _state["nstep"] else "absent",
    )


@register_patch("greedy_argmax", default=False)
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
            # compile succeeded, so both consumers are still on torch.argmax here.
            log.exception("vtl: greedy_argmax arming failed; torch.argmax stands")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, orig, patch="greedy_argmax")
    log.info("vtl: greedy_argmax armed (compiles + registers at model load)")


def _self_check() -> None:
    """Runs anywhere: no GPU, no torch, no vLLM."""
    from vtl import nvrtc
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "greedy_argmax")
    assert patch.default is False, "unproven optimization: default OFF"
    assert is_enabled(patch) is False
    os.environ["VTL_ENABLE_GREEDY_ARGMAX"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GREEDY_ARGMAX")

    # -- block width: the default ships, nonsense does not reach the compiler --
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS", None)
    assert _threads_from_env() == DEFAULT_THREADS
    for good in ("256", "512", "1024"):
        os.environ["VTL_GREEDY_ARGMAX_THREADS"] = good
        assert _threads_from_env() == int(good)
    for bad in ("0", "17", "2048", "-32", "banana", ""):
        os.environ["VTL_GREEDY_ARGMAX_THREADS"] = bad
        assert _threads_from_env() == DEFAULT_THREADS, bad
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS")

    # -- vocab envelope --
    assert _vocab_ok(248320) is True
    assert _vocab_ok(1) is True      # the scalar path serves any positive vocab
    assert _vocab_ok(0) is False
    assert _vocab_ok(-8) is False

    # -- the source ships, and its entry symbol is the one we ask the loader for --
    src = nvrtc.load_source(KERNEL)
    assert src, "vtl/kernels/greedy_argmax.cu missing from the package"
    assert f'extern "C" __global__ void __launch_bounds__(THREADS) {ENTRY}(' in src
    assert '#error "NVRTC: -DVOCAB=<vocab_size> is required"' in src

    # -- one cubin identity per specialization: a collision would serve a cubin compiled
    #    for ANOTHER vocab, which reads off the end of the row --
    sets = [_defines(248320, 512), _defines(248320, 256), _defines(151936, 512)]
    keys = {nvrtc.cache_key(src, s, "90a", "12.8") for s in sets}
    assert len(keys) == len(sets), "define sets must not collide in the cubin cache"
    assert nvrtc.cache_key(src, sets[0], "90", "12.8") not in keys, "arch is identity too"

    # -- while NVRTC is off, compiling is a no-op and apply() installs nothing --
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(KERNEL, sets[0], entry=ENTRY) is None
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
    assert _state["launch"] is None

    # -- the nstep rebind is import-safe: nstep_decode is not imported here, so this must
    #    report "nothing to rebind" rather than dragging vLLM in --
    import sys

    if "vtl.patches.nstep_decode" not in sys.modules:
        assert _rebind_nstep() is False

    print("greedy_argmax self-check ok")


if __name__ == "__main__":
    _self_check()
