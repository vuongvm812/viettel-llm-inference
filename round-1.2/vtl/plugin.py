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

log = logging.getLogger("vllm.vtl.plugin")

_REGISTERED = False
_LOGGING_READY = False


def _install_log_handler() -> None:
    """Give ``vllm.vtl`` its own INFO handler, independent of ``VLLM_LOGGING_LEVEL``.

    Two separate gates used to eat every vtl INFO record, and between them ``make verify``
    could not pass on a healthy server:

    * Name. vLLM attaches its handler to the ``vllm`` logger with ``propagate: False``
      (``vllm/logger.py`` DEFAULT_LOGGING_CONFIG) and configures nothing on root, so records
      on a bare ``vtl`` logger fell through to ``logging.lastResort``, which is WARNING-only.
      Fixed by naming every vtl logger ``vllm.vtl.<module>``.
    * Level. Being a child of ``vllm`` then inherits its level, and the submission compose pins
      ``VLLM_LOGGING_LEVEL: WARNING`` -- so INFO is still dropped, now one gate later.

    Lowering vLLM's own handler would un-gate all of vLLM's INFO output, which is not what we
    want in a scored run. A dedicated handler on the ``vllm.vtl`` subtree is the narrow fix: it
    makes exactly our lines visible, at their real severity, and leaves vLLM's verbosity alone.
    ``propagate = False`` so records stop here rather than being emitted a second time by the
    ``vllm`` handler whenever VLLM_LOGGING_LEVEL happens to be INFO or below.

    ``VTL_LOG_LEVEL`` overrides; it is read once, and a bad value leaves INFO.
    """
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    _LOGGING_READY = True

    vtl_log = logging.getLogger("vllm.vtl")
    level = getattr(logging, os.environ.get("VTL_LOG_LEVEL", "INFO").strip().upper(), logging.INFO)
    vtl_log.setLevel(level)
    if not vtl_log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        # Same shape as vLLM's own formatter so the two interleave readably in `docker logs`.
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s", "%m-%d %H:%M:%S")
        )
        vtl_log.addHandler(handler)
    vtl_log.propagate = False


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
        # First, before anything can log: every line below this is otherwise invisible.
        _install_log_handler()

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


def _self_check() -> None:
    """The thing `make verify` needs: a vtl INFO record survives a WARNING-level `vllm` parent."""
    import io

    global _LOGGING_READY
    parent = logging.getLogger("vllm")
    parent.setLevel(logging.WARNING)  # what the submission compose pins

    _LOGGING_READY = False
    logging.getLogger("vllm.vtl").handlers.clear()
    _install_log_handler()

    sink = io.StringIO()
    handler = logging.getLogger("vllm.vtl").handlers[0]
    handler.setStream(sink)
    logging.getLogger("vllm.vtl.w4a8").info("vtl: w4a8 device = FakeGPU, 16 SMs")
    assert "16 SMs" in sink.getvalue(), "vtl INFO dropped under VLLM_LOGGING_LEVEL=WARNING"

    # Idempotent: a second register() must not double-emit (vLLM may call us more than once).
    _install_log_handler()
    assert len(logging.getLogger("vllm.vtl").handlers) == 1
    assert logging.getLogger("vllm.vtl").propagate is False

    print("vtl/plugin.py: ok")


if __name__ == "__main__":
    _self_check()
