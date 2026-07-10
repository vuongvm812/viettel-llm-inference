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

log = logging.getLogger("vtl")

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    try:
        if os.environ.get("VTL_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
            log.info("vtl: disabled via VTL_DISABLE, running stock vLLM")
            return

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
