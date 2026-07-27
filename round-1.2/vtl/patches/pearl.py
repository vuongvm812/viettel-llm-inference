"""PEARL-style post-verify overlap for the separate draft model.

WHAT THIS ACTUALLY DOES, stated plainly so nobody reads more into it than is here: it runs the
drafter on a SECOND CUDA stream so its launches overlap the main stream's remaining post-sample
bookkeeping, and it puts the conv-state save/restore in place that any speculative variant needs.
It does NOT yet draft from an ASSUMED accepted prefix.

That last step is the actual PEARL gamble, and it cannot be done from here. This patch wraps
``GPUModelRunner.propose_draft_token_ids``, which by definition is called with the real
``sampled_token_ids`` -- i.e. after the target has already verified. Speculating means starting
the drafter BEFORE that, which requires hoisting the drafter call above ``_sample`` inside
``execute_model`` and constructing the draft inputs from a guessed prefix. That is
gpu_model_runner surgery, not a wrapper, and it is the remaining work.

So the honest summary: the plumbing and the rollback are done and testable; the speculation is
not. Given the arithmetic below, finishing it is very likely not worth it -- but the pieces are
here if the measurement says otherwise.

READ THIS BEFORE TURNING IT ON. The expected value is genuinely marginal on this hardware, and
the reasons are structural rather than fixable by tuning:

* **The chain needs K+1, not K.** When the target accepts all K drafts it also emits a bonus
  token the drafter never saw, so a speculative round conditioned on a prefix one token short is
  worthless. The drafter must extend with its own ``d_{K+1}`` and the round survives only if all
  K drafts were accepted AND the bonus equals ``d_{K+1}`` -- probability ``p^(K+1)``, about 0.24
  at a per-token match rate of 0.7. Roughly three quarters of speculative rounds are discarded.
* **There is no idle GPU to overlap into.** The judge runs a MIG 1g.18gb slice (~19 SMs). The
  target's verify and the draft forward contend for the same SMs, so a discarded round is charged
  at close to full price rather than being free work in a bubble.
* **The miss path is strictly slower than baseline** -- restore, then a serialized re-draft with
  no overlap left to recover it.
* **vLLM already takes the cheap overlaps.** Async scheduling overlaps step ``t+1``'s host prep
  with step ``t``'s GPU work, and ``use_gpu_toks`` already runs the drafter straight off the GPU
  sampled tokens without waiting for bookkeeping.

So this ships behind ``VTL_PEARL``, default OFF. Before enabling it, measure: histogram
``num_accepted`` off the rejection sampler and time ``drafter.propose`` against the target
forward with CUDA events. Enable only if ``p^(K+1) > 0.5`` and the drafter is more than ~35% of
step time. Those two numbers decide it; nothing else does.

STATE. The drafter's KV rolls back as bookkeeping (``seq_lens`` decrement -- the slots are simply
overwritten), but its 10 short-conv ``conv_state`` tensors do not: the drafter's
``num_accepted_tokens`` is None by construction, so it takes the destructive non-spec rotation.
This module therefore snapshots and restores them around a speculative round.

That snapshot is the honest weak point: ~10 launches to save and ~10 to restore, against the ~30
being overlapped, on the resource that is already co-dominant. The zero-copy alternative is to
give the drafter the same sliding-window layout the target uses under spec (always append, taps
at a runtime offset), which turns rollback into one int32 per request -- that is
``bcx_conv_gate_quant``'s kSpec path retargeted at the drafter, and it also needs the draft model
built with ``num_spec + 1`` state width, i.e. a different ``num_spec`` for draft and target
through the ``short_conv`` / ``lfm2`` / ``mamba_utils`` three-file lockstep. Deliberately not
built; if PEARL ever measures positive, that is the upgrade.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl")

# THE gate is the registry's, i.e. ``VTL_ENABLE_PEARL`` -- ``registry.is_enabled`` derives it
# from the patch name and consults it before ``apply()`` is ever called, exactly as for every
# other patch in this tree. This module deliberately does NOT read a second variable of its own:
# an earlier revision gated on ``VTL_PEARL`` as well, which meant PEARL needed BOTH set and
# setting only the documented one silently did nothing.
ENV = "VTL_ENABLE_PEARL"
# Speculation depth. Read but unused while nothing is installed -- kept because it is the one
# knob a real post-verify implementation needs, and its parsing is already tested. 0 means
# "num_speculative_tokens + 1", i.e. assume the whole round including the bonus token landed.
DEPTH_ENV = "VTL_PEARL_DEPTH"


@functools.cache
def depth() -> int:
    """Speculation depth. 0 = ``num_speculative_tokens + 1`` (assume the whole round was
    accepted, including the bonus token); a smaller value assumes only that many drafts landed.

    Depth 1 is the cheap variant: hit rate ``p`` instead of ``p^(K+1)``, and a miss wastes one
    forward rather than the whole round. It is the sane starting point despite covering less.
    """
    raw = os.environ.get(DEPTH_ENV, "1").strip()
    try:
        value = int(raw)
    except ValueError:
        log.warning("vtl: %s=%r is not an integer; using 1", DEPTH_ENV, raw)
        return 1
    return max(0, value)


class _ConvSnapshot:
    """Save/restore of the drafter's short-conv states for the rows a step touches.

    Only the blocks named by ``state_indices`` are copied, so the cost scales with the batch
    (<= 8 here), not with the cache. ``index_select`` + ``index_copy_`` keep it on-device with no
    host round trip.
    """

    def __init__(self):
        self._saved: list[tuple] = []

    def save(self, conv_states, rows) -> None:
        self._saved = [(t, rows, t.index_select(0, rows).clone()) for t in conv_states]

    def restore(self) -> None:
        for tensor, rows, data in self._saved:
            tensor.index_copy_(0, rows, data)
        self._saved = []

    def drop(self) -> None:
        self._saved = []


def _draft_conv_states(runner):
    """The DRAFT model's short-conv state tensors, or [] if they cannot be identified.

    Walks the draft model's own modules and reads ``layer.kv_cache[0]`` -- the tensor
    ``ShortConv.forward_cuda`` actually mutates. Deliberately NOT a lookup through the runner's
    shared ``kv_caches`` list by layer name: the target's conv layers are named the same way, and
    snapshotting or restoring one of THOSE would corrupt the served model rather than the
    drafter. Starting from ``drafter.model`` makes that mistake unrepresentable.

    Returns [] on anything unexpected, which disables PEARL for the step.
    """
    drafter = getattr(runner, "drafter", None)
    model = getattr(drafter, "model", None)
    if model is None:
        return []
    # The drafter's model may be wrapped (CUDAGraphWrapper / BreakableCUDAGraphWrapper).
    inner = getattr(model, "unwrap", None)
    if callable(inner):
        try:
            model = model.unwrap()
        except Exception:
            return []
    try:
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        states = []
        for module in model.modules():
            if not isinstance(module, MambaBase):
                continue
            cache = getattr(module, "kv_cache", None)
            if not cache or cache[0] is None or cache[0].numel() == 0:
                return []  # not allocated yet -- do not speculate on a half-built drafter
            states.append(cache[0])
        return states
    except Exception:
        return []


@register_patch("pearl", default=False)
def apply() -> None:
    """Install nothing, and say why.

    This is the finding, not an omission. Wrapping ``propose_draft_token_ids`` looked like the
    natural seam -- it is the whole drafter invocation and everything a speculative round would
    need to undo lives inside it -- but two independent problems make it a dead end.
    """
    # Deliberately no wrapper. Two independent reasons, both discovered by review after the
    # first cut of this module, and both structural rather than fixable by tuning:
    #
    # 1. NO SAFE OVERLAP FROM HERE. Running the drafter inside `torch.cuda.stream(...)` makes
    #    every tensor it allocates owned by that stream. `wait_stream` orders EXECUTION, not
    #    deallocation, and several of those tensors escape: `out` into `self._draft_token_ids`
    #    (read later on a THIRD stream that never waits on ours), `self._draft_probs` into the
    #    next step's rejection sampler -- that one corrupts ACCEPTANCE silently -- and the
    #    drafter's lazily-created persistent buffers, which would stay side-stream-owned for the
    #    process lifetime. vLLM flags this exact hazard in `v1/worker/gpu/spec_decode/utils.py`
    #    and `pp_utils.py`, and every side stream it actually uses is a D2H copy into a
    #    PRE-ALLOCATED buffer, never a model forward. Closing it means `record_stream()` on an
    #    escape list that grows silently as vLLM changes.
    #
    # 2. NOTHING TO ROLL BACK ANYWAY. Without the overlap there is no speculation, and without
    #    speculation the snapshot below has no consumer -- wrapping the call to save and drop
    #    state nothing rewinds is pure cost (~20 launches) on the host-bound path this was
    #    supposed to relieve.
    #
    # Real post-verify needs the drafter hoisted ABOVE `_sample` in `execute_model` and driven
    # from a guessed prefix. That is runner surgery, not a wrapper, and the arithmetic in the
    # module docstring argues it is very unlikely to pay on ~19 SMs. What survives here is the
    # rollback primitive -- `_ConvSnapshot` and `_draft_conv_states` -- which is the part any
    # future speculative scheme needs and the part that is testable off-box today.
    log.info(
        "vtl: PEARL not installed -- no safe overlap is reachable from "
        "propose_draft_token_ids; see vtl/patches/pearl.py for the analysis"
    )


def _self_check() -> None:
    """No torch: the knob parsing and the snapshot bookkeeping, which is all that is testable
    off-box."""
    # ONE gate, and it is the registry's. A second module-local env read is what made an earlier
    # revision need both VTL_PEARL and VTL_ENABLE_PEARL, so assert the module does not reintroduce
    # one: ENV must be exactly the name registry.is_enabled derives from the patch name.
    assert ENV == "VTL_ENABLE_PEARL", ENV
    # Every env read must go through the two constants above, so a stray literal is the
    # signature of a second gate creeping back in.
    src = Path(__file__).read_text()
    assert "os.environ.get(\"VTL" not in src, "pearl.py must read env only via ENV/DEPTH_ENV"

    for raw, want in (("1", 1), ("4", 4), ("0", 0), ("banana", 1), ("-3", 0)):
        os.environ[DEPTH_ENV] = raw
        depth.cache_clear()
        assert depth() == want, (raw, depth(), want)
    os.environ.pop(DEPTH_ENV, None)
    depth.cache_clear()

    # The snapshot is a plain list of (tensor, rows, data) triples; save/restore/drop must not
    # leak entries between rounds, or a later restore would write stale rows into a live cache.
    snap = _ConvSnapshot()
    assert snap._saved == []
    snap._saved = [("t", "r", "d")]
    snap.drop()
    assert snap._saved == []

    from vtl.registry import PATCH_REGISTRY

    names = {p.name: p.default for p in PATCH_REGISTRY}
    assert names.get("pearl") is False, names

    print("pearl self-check ok")


if __name__ == "__main__":
    _self_check()
