"""Patch registry.

Every patch is a zero-arg callable registered under a name and gated by an env var
``VTL_ENABLE_<NAME>``. A patch that raises is logged and skipped -- it must never
prevent vLLM from serving.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, NamedTuple

# MUST stay under the "vllm." prefix. vLLM's dictConfig attaches a handler to the "vllm"
# logger only (propagate=False); the root logger gets none. A bare getLogger("vtl") therefore
# falls through to logging.lastResort, which drops everything below WARNING -- so every
# log.info in this tree (patch applied, N layers quantized, lm_head outcome, fusion installed)
# vanished and `make verify` could never confirm the overlay ran at all. As a child of "vllm"
# we inherit its handler and its VLLM_LOGGING_LEVEL.
log = logging.getLogger("vllm.vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class Patch(NamedTuple):
    name: str
    apply: Callable[[], None]
    default: bool


PATCH_REGISTRY: list[Patch] = []


def register_patch(name: str, *, default: bool = False):
    """Decorator: add ``fn`` to PATCH_REGISTRY under ``name``."""

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        PATCH_REGISTRY.append(Patch(name, fn, default))
        return fn

    return decorator


def is_enabled(patch: Patch) -> bool:
    raw = os.environ.get(f"VTL_ENABLE_{patch.name.upper()}")
    if raw is None:
        return patch.default
    return raw.strip().lower() in _TRUTHY


def already_patched(obj: object, attr: str) -> bool:
    """True if ``obj.attr`` is one of our wrappers, so re-applying is a no-op."""
    return hasattr(getattr(obj, attr, None), "__vtl_wrapped__")


def mark_patched(wrapper: Callable, original: Callable) -> Callable:
    wrapper.__vtl_wrapped__ = original
    return wrapper


def apply_all() -> tuple[int, int]:
    """Apply every enabled patch. Returns ``(applied, selected)``."""
    selected = [p for p in PATCH_REGISTRY if is_enabled(p)]
    applied = 0
    for patch in selected:
        try:
            patch.apply()
        except Exception:
            log.exception("vtl: patch %s FAILED, skipping", patch.name)
            continue
        applied += 1
        log.info("vtl: patch %s applied", patch.name)
    return applied, len(selected)


def _self_check() -> None:
    saved_registry = PATCH_REGISTRY[:]
    PATCH_REGISTRY.clear()
    try:
        calls: list[str] = []

        @register_patch("good", default=True)
        def _good() -> None:
            calls.append("good")

        @register_patch("bad", default=True)
        def _bad() -> None:
            calls.append("bad")
            raise RuntimeError("boom")

        @register_patch("off_by_default")
        def _off() -> None:
            calls.append("off")

        os.environ.pop("VTL_ENABLE_GOOD", None)
        os.environ.pop("VTL_ENABLE_BAD", None)
        os.environ.pop("VTL_ENABLE_OFF_BY_DEFAULT", None)

        # A raising patch is isolated: it counts as selected, not applied.
        applied, selected = apply_all()
        assert calls == ["good", "bad"], calls
        assert (applied, selected) == (1, 2), (applied, selected)

        # Env var overrides the in-code default, both directions.
        os.environ["VTL_ENABLE_OFF_BY_DEFAULT"] = "on"
        os.environ["VTL_ENABLE_GOOD"] = "0"
        assert is_enabled(PATCH_REGISTRY[2]) is True
        assert is_enabled(PATCH_REGISTRY[0]) is False

        class Target:
            def method(self) -> None: ...

        assert not already_patched(Target, "method")
        Target.method = mark_patched(lambda self: None, Target.method)
        assert already_patched(Target, "method")

        # Every vtl logger must be a "vllm." child or its INFO records are dropped on the
        # floor (see the comment on `log` above). Guards the whole tree, not just this file.
        import pathlib

        for py in pathlib.Path(__file__).parent.rglob("*.py"):
            for lineno, line in enumerate(py.read_text().splitlines(), 1):
                if "= logging.getLogger(" in line and 'getLogger("vllm.vtl")' not in line:
                    raise AssertionError(f"{py}:{lineno}: logger must be 'vllm.vtl': {line}")

        print("registry self-check ok")
    finally:
        PATCH_REGISTRY[:] = saved_registry
        for key in ("VTL_ENABLE_GOOD", "VTL_ENABLE_OFF_BY_DEFAULT"):
            os.environ.pop(key, None)


if __name__ == "__main__":
    _self_check()
