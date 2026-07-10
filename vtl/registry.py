"""Patch registry.

Every patch is a zero-arg callable registered under a name and gated by an env var
``VTL_ENABLE_<NAME>``. A patch that raises is logged and skipped -- it must never
prevent vLLM from serving.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, NamedTuple

log = logging.getLogger("vtl")

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

        print("registry self-check ok")
    finally:
        PATCH_REGISTRY[:] = saved_registry
        for key in ("VTL_ENABLE_GOOD", "VTL_ENABLE_OFF_BY_DEFAULT"):
            os.environ.pop(key, None)


if __name__ == "__main__":
    _self_check()
