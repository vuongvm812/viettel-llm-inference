"""vLLM ``vllm.general_plugins`` entry point.

vLLM calls ``register()`` once per process (API server, engine core, every worker)
before it does any real work. Three hard constraints follow:

* No top-level ``import vllm`` -- this module must import in a vLLM-less env.
* Idempotent -- vLLM may call us more than once.
* Never raises. A broken patch degrades us to stock vLLM; it never takes the
  server down.
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("vtl")

_REGISTERED = False


def _tune_gil_switch_interval() -> None:
    """Shorten CPython's GIL handoff so the engine's output thread is not stalled 5ms.

    The engine-core main thread runs a tight pure-Python loop that never blocks while it has
    work (``vllm/v1/engine/core.py`` ``run_busy_loop``), while the output IO thread sits in
    ``queue.Queue.get()`` and needs the GIL to forward each step's tokens. CPython only forces
    a handoff after ``sys.getswitchinterval()`` -- 5ms by default, i.e. potentially longer
    than a whole decode step. Per-token latency is the scored metric, so we trade a little
    switching overhead for a much tighter handoff.

    ``VTL_GIL_SWITCH_INTERVAL`` is in SECONDS; 0 (or a bad value) leaves the interpreter
    default alone, which is the A/B control arm.
    """
    raw = os.environ.get("VTL_GIL_SWITCH_INTERVAL", "0.0002")
    try:
        interval = float(raw)
    except ValueError:
        log.warning("vtl: VTL_GIL_SWITCH_INTERVAL=%r is not a number; leaving default", raw)
        return
    if interval <= 0:
        return
    sys.setswitchinterval(interval)
    log.info("vtl: GIL switch interval set to %gs (was 0.005 default)", interval)


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    try:
        if os.environ.get("VTL_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
            log.info("vtl: disabled via VTL_DISABLE, running stock vLLM")
            return

        # Before the patches: this is process-wide interpreter state, not a patch, and it
        # matters in every process vLLM calls us from (engine core most of all).
        _tune_gil_switch_interval()

        from vtl import patches  # noqa: F401  -- import side effect fills the registry
        from vtl.registry import PATCH_REGISTRY, apply_all

        applied, selected = apply_all()
        # bench/ and the smoke test grep for this line. Its absence means the
        # plugin never ran and any benchmark below it measured stock vLLM.
        log.info(
            "vtl: applied %d/%d patches (%d registered)",
            applied,
            selected,
            len(PATCH_REGISTRY),
        )
    except Exception:
        log.exception("vtl: register() failed, continuing with stock vLLM")
