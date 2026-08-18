"""Stall watchdog: when requests are RUNNING but no counter moves, dump why.

WHY THIS EXISTS. The N-step burst hang (2026-08-18, on-box): generation wedges mid-decode --
requests pinned RUNNING, 0 tok/s, no exception anywhere -- reproduced with
``VTL_NSTEP=1`` under BOTH ``VTL_NSTEP_MODE=graph`` and ``eager``, and gone with
``VTL_NSTEP=0``. A silent liveness bug in an async two-steps-in-flight pipeline cannot be
root-caused from throughput logs; it needs the thread stacks and the request counters AT the
moment of the wedge, and ``py-spy`` is not always attachable inside the served container.
So the engine carries its own flight recorder.

WHAT IT DOES. ``apply()`` wraps ``update_from_output`` on the scheduler classes for exactly
one purpose: capturing the live scheduler instance (weakly) and starting one daemon thread.
The thread polls every ``_POLL`` seconds and computes a progress signature over the running
requests -- ``(req_id, num_computed_tokens, num_output_tokens, num_output_placeholders,
status)``. If there ARE running requests and the signature has not changed for
``VTL_STALL_DUMP_SECS`` (default 20), it emits, once per ``_RELATCH`` window:

  * one WARNING line per running request with the burst-relevant counters
    (``num_computed_tokens``, ``num_output_placeholders``, ``num_output_tokens``,
    ``is_prefill_chunk``, status) plus the waiting-queue length;
  * the ``nstep_decode.BURST`` handshake state (armed/ready/key/n/mode and the per-step
    commitments pending/pending1/pending_seq/rust_steps) and the ``rust_runner.STATE``
    handshake (live/refused/why/inflight/pending/stash seq), whichever of the two modules
    is loaded -- between them they say whether a step was committed and then never sampled;
  * ``faulthandler.dump_traceback(all_threads=True)`` to stderr -- the Python stack of every
    thread in the EngineCore process, which is the piece that names the wedged line.

DIAGNOSTICS ONLY. Nothing on the hot path but one weakref store per engine step; the poll
thread only READS attributes (racily, guarded) and only ever logs. It cannot un-wedge
anything and does not try: recovering by force (uncommitting placeholders under async
scheduling) can mis-position tokens, which is worse than the hang it would hide.

    VTL_ENABLE_STALL_DUMP=0     remove the wrapper and the thread entirely
    VTL_STALL_DUMP_SECS=<s>     stall threshold before the first dump (default 20)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.stall_dump")

_POLL = 2.0       # seconds between signature polls
_RELATCH = 120.0  # seconds between dumps while a stall persists

_sched_ref: weakref.ref | None = None
_thread_started = False
_lock = threading.Lock()


def _threshold() -> float:
    raw = os.environ.get("VTL_STALL_DUMP_SECS", "").strip()
    try:
        return max(float(raw), _POLL) if raw else 20.0
    except ValueError:
        return 20.0


def signature(running) -> tuple:
    """Progress fingerprint of the running set. Pure -- self-checked.

    Any sampled token, computed-token advance, placeholder reconcile, admission or
    finish changes it; a wedge does not.
    """
    sig = []
    for r in running:
        sig.append((
            str(getattr(r, "request_id", id(r))),
            int(getattr(r, "num_computed_tokens", -1)),
            int(getattr(r, "num_output_tokens", -1)),
            int(getattr(r, "num_output_placeholders", -1)),
            str(getattr(r, "status", "?")),
        ))
    return tuple(sig)


def stalled(now: float, last_change: float, threshold: float, n_running: int) -> bool:
    """Whether the current quiet spell counts as a stall. Pure -- self-checked."""
    return n_running > 0 and (now - last_change) >= threshold


def _burst_state() -> str:
    """The two handshakes a wedge lives in, one line each. Never raises.

    ``sys.modules.get``, never an import: this runs on the watchdog thread against a
    process that is already sick, and importing a patch module from here would either be a
    no-op or a brand-new import with side effects at the worst possible moment. A module
    that was never loaded is itself a finding ("absent").

    THE FIELDS ARE THE WEDGE, not a dump of the objects. ``_Burst`` is a ``__slots__``
    class, so a name that does not exist reads back as the ``'?'`` default rather than
    raising -- which is how the previous list came to print ``disabled``/``reason``, two
    fields that have never existed on it. Every name below is asserted against the real
    ``__slots__`` in ``_self_check``:

      * ``armed``/``ready``/``key``/``n``/``mode`` -- is the burst installed at all, and
        which batch did the worker last publish as burstable;
      * ``pending``/``pending1``/``pending_seq`` -- the per-step commitments. Non-zero here
        while nothing moves means a ``sample_tokens`` that never ran (or never returned)
        after an ``execute_model`` that did;
      * ``rust_steps`` -- how many launches a burst step is allowed to chain.

    The runner line answers the other half: whether the Rust runner is live, whether it
    stood down and why, how many scheduled steps it thinks are unapplied (``inflight``),
    how many launches are waiting for their commit (``pending``), and which scheduled step
    owns the stash it is holding (``step.seq``) -- a stash that never moves is a launch
    site that never came back for it.
    """
    lines = []
    try:
        import sys

        nd = sys.modules.get("vtl.patches.nstep_decode")
        b = getattr(nd, "BURST", None)
        if b is None:
            lines.append("BURST: absent")
        else:
            fields = ("armed", "ready", "key", "n", "mode", "pending", "pending1",
                      "pending_seq", "rust_steps")
            lines.append("BURST: " + " ".join(f"{f}={getattr(b, f, '?')!r}" for f in fields))

        rr = sys.modules.get("vtl.patches.rust_runner")
        st = getattr(rr, "STATE", None)
        if st is None:
            lines.append("RUNNER: absent")
        else:
            step = getattr(st, "step", None)
            lines.append(
                "RUNNER: live=%r refused=%r why=%r inflight=%r pending=%r stash_seq=%r"
                % (
                    getattr(st, "live", "?"),
                    getattr(st, "refused", "?"),
                    getattr(st, "why", "?"),
                    getattr(st, "inflight", "?"),
                    len(getattr(st, "pending", ()) or ()),
                    getattr(step, "seq", None) if step is not None else None,
                )
            )
    except Exception as exc:
        lines.append(f"BURST/RUNNER: unreadable ({exc!r})")
    return "\n".join(lines)


def _dump(sched) -> None:
    import faulthandler
    import sys

    try:
        running = list(getattr(sched, "running", ()) or ())
        waiting = getattr(sched, "waiting", None)
        n_waiting = len(waiting) if waiting is not None else -1
        log.warning(
            "vtl: STALL -- %d running request(s) made no progress for %.0fs (waiting=%d). "
            "Counters below; all-thread stacks follow on stderr.",
            len(running), _threshold(), n_waiting,
        )
        for r in running:
            log.warning(
                "vtl: STALL req=%s status=%s num_tokens=%s computed=%s placeholders=%s "
                "output=%s prefill_chunk=%s",
                getattr(r, "request_id", "?"), getattr(r, "status", "?"),
                getattr(r, "num_tokens", "?"), getattr(r, "num_computed_tokens", "?"),
                getattr(r, "num_output_placeholders", "?"),
                getattr(r, "num_output_tokens", "?"),
                getattr(r, "is_prefill_chunk", "?"),
            )
        for line in _burst_state().splitlines():
            log.warning("vtl: STALL %s", line)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        log.exception("vtl: stall dump itself failed")


def _monitor() -> None:
    last_sig: tuple = ()
    last_change = time.monotonic()
    next_dump_ok = 0.0
    while True:
        time.sleep(_POLL)
        try:
            sched = _sched_ref() if _sched_ref is not None else None
            if sched is None:
                continue
            try:
                sig = signature(list(getattr(sched, "running", ()) or ()))
            except Exception:
                continue  # racing the engine thread mid-mutation; next poll wins
            now = time.monotonic()
            if sig != last_sig:
                last_sig = sig
                last_change = now
                continue
            if stalled(now, last_change, _threshold(), len(sig)) and now >= next_dump_ok:
                _dump(sched)
                next_dump_ok = now + _RELATCH
        except Exception:
            log.exception("vtl: stall monitor iteration failed")


def _note(sched) -> None:
    """Per-step hook body: remember the scheduler, start the thread once."""
    global _sched_ref, _thread_started
    if _sched_ref is None or _sched_ref() is not sched:
        _sched_ref = weakref.ref(sched)
    if not _thread_started:
        with _lock:
            if not _thread_started:
                threading.Thread(target=_monitor, name="vtl-stall-dump", daemon=True).start()
                _thread_started = True
                log.info("vtl: stall watchdog armed (threshold %.0fs)", _threshold())


@register_patch("stall_dump", default=True)
def apply() -> None:
    # Wrap AFTER rust_sched (this module sits below it in _MODULES) so whatever
    # update_from_output is finally bound -- stock, async, or the Rust-scheduler one --
    # is the one that stamps the watchdog.
    import vllm.v1.core.sched.scheduler as sched_mod

    classes = [sched_mod.Scheduler]
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        classes.append(AsyncScheduler)
    except Exception:
        pass

    for cls in classes:
        if "update_from_output" not in vars(cls):
            continue  # inherits it; the base-class wrapper covers it
        if already_patched(cls, "update_from_output", patch="stall_dump"):
            continue
        inner = cls.update_from_output

        def wrapped(self, *a, _inner=inner, **kw):
            _note(self)
            return _inner(self, *a, **kw)

        mark_patched(wrapped, inner, patch="stall_dump")
        cls.update_from_output = wrapped
    log.info("vtl: stall_dump installed on update_from_output")


def _self_check() -> None:
    """Runs anywhere: no GPU, no vLLM."""
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "stall_dump")
    assert patch.default is True
    assert is_enabled(patch) is True
    os.environ["VTL_ENABLE_STALL_DUMP"] = "0"
    assert is_enabled(patch) is False
    os.environ.pop("VTL_ENABLE_STALL_DUMP")

    # -- signature: any counter movement, admission or finish changes it --
    class R:
        def __init__(self, rid, computed, out, ph):
            self.request_id = rid
            self.num_computed_tokens = computed
            self.num_output_tokens = out
            self.num_output_placeholders = ph
            self.status = "RUNNING"

    a = R("a", 10, 2, 0)
    s1 = signature([a])
    assert signature([a]) == s1, "no movement -> same signature"
    a.num_output_tokens += 1
    assert signature([a]) != s1, "a sampled token must change the signature"
    a.num_output_tokens -= 1
    a.num_output_placeholders += 3
    assert signature([a]) != s1, "a burst commit must change the signature"
    a.num_output_placeholders -= 3
    assert signature([a, R("b", 0, 0, 0)]) != s1, "admission must change the signature"
    assert signature([]) == (), "empty running set"

    # -- stall predicate: needs running requests AND a full quiet threshold --
    assert stalled(now=30.0, last_change=5.0, threshold=20.0, n_running=1)
    assert not stalled(now=30.0, last_change=15.0, threshold=20.0, n_running=1)
    assert not stalled(now=30.0, last_change=5.0, threshold=20.0, n_running=0), \
        "an idle engine is not a stall"

    # -- threshold parsing never raises and never undercuts the poll interval --
    os.environ["VTL_STALL_DUMP_SECS"] = "nonsense"
    assert _threshold() == 20.0
    os.environ["VTL_STALL_DUMP_SECS"] = "0.1"
    assert _threshold() == _POLL, "threshold below the poll interval is clamped up"
    os.environ.pop("VTL_STALL_DUMP_SECS")
    assert _threshold() == 20.0

    # -- the two handshake summaries. Both halves report, loaded or not --
    state = _burst_state()
    assert "BURST" in state and "RUNNER" in state, state
    assert len(state.splitlines()) == 2, state

    # ...and every field named is a field that EXISTS. This is the whole reason the check
    # is here: `_Burst` is a `__slots__` class and `getattr(..., '?')` swallows a typo
    # silently, which is how the previous list came to print two fields (`disabled`,
    # `reason`) that the class has never had. Importing is safe here (no vLLM, no torch at
    # module scope) and is exactly what the watchdog itself refuses to do.
    from vtl.patches import nstep_decode, rust_runner

    for f in ("armed", "ready", "key", "n", "mode", "pending", "pending1", "pending_seq",
              "rust_steps"):
        assert f in nstep_decode._Burst.__slots__, f
    for f in ("live", "refused", "why", "inflight", "pending", "step"):
        assert f in rust_runner._State.__slots__, f
    assert "seq" in rust_runner._Stash._fields
    live = _burst_state()
    assert "BURST: absent" not in live and "RUNNER: absent" not in live, live
    assert "unreadable" not in live, live
    assert "pending_seq=" in live and "stash_seq=" in live, live

    print("stall_dump self-check ok")


if __name__ == "__main__":
    _self_check()
