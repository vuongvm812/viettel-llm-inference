#!/usr/bin/env python3
"""compose <-> registry parity for the ``VTL_ENABLE_*`` patch gates.

docker-compose.yaml pins EVERY patch gate, and its header block claims that every pinned
value equals the patch's ``@register_patch(default=)`` in code -- with exactly two named
exceptions (``shm_ipc`` and ``rust_sched``, which are ``default=False`` and forced on because
the Rust data/control plane is the shipped configuration and its own master switches are the
real gate). That claim is the whole reason the pins are trustworthy documentation of what
ships, and nothing enforced it: a `default=` flipped in a refactor, or a gate added to the
registry and never pinned, drifts silently and the comment keeps asserting otherwise.

This check enforces it, three ways:

  1. every registered patch has a pin in compose  (a new patch cannot ship un-pinned),
  2. every pin in compose names a registered patch (a stale pin cannot linger after a
     patch is renamed or deleted -- it would read as "shipped ON" and do nothing),
  3. every pinned value equals the code default, except for the two-name exception list,
     for which the OPPOSITE is asserted (default False, pinned on) so that the exception
     list itself cannot rot: flip one of those to `default=True` and this fails, telling
     you to delete the exception rather than leaving a lie in the compose header.

Deliberately dependency-free: a regex over the YAML text (so it does not need PyYAML) plus
``import vtl.patches`` (which needs neither torch nor vLLM -- every patch module defers those
imports into ``apply()``). Runs under ``make check`` on a bare host.

The regex accepts both quoted and unquoted ints because compose uses both (`"1"` in the
registry-gates block, bare `1` further down); comment lines are skipped so the prose in that
header -- which names gates in passing -- cannot be mistaken for a pin.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.yaml"

# Pinned ON in compose while `default=False` in code, ON PURPOSE. Names are the compose
# suffix (i.e. the patch name upper-cased). Keep this list SHORT and justified:
#   shm_ipc, rust_sched -- the Rust data/control plane. Registered opt-in so a bare
#   `import vtl.patches` on a box without the Rust wheel does nothing, and forced on here
#   because the shipped configuration is the Rust stack; VTL_RUST_SCHED_* / VTL_SHM_IPC_*
#   below them are the real per-feature gates.
FORCED_ON = frozenset({"SHM_IPC", "RUST_SCHED"})

_PIN = re.compile(r'^\s*VTL_ENABLE_([A-Z0-9_]+):\s*"?([01])"?\s*(?:#.*)?$')


def parse_compose_pins(text: str) -> dict[str, bool]:
    """``{GATE_SUFFIX: bool}`` for every ``VTL_ENABLE_*`` pin in the compose text."""
    pins: dict[str, bool] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _PIN.match(line)
        if m is None:
            continue
        name, value = m.group(1), m.group(2)
        assert name not in pins, f"VTL_ENABLE_{name} is pinned twice in docker-compose.yaml"
        pins[name] = value == "1"
    return pins


def registry_defaults() -> dict[str, bool]:
    """``{GATE_SUFFIX: default}`` for every patch in PATCH_REGISTRY."""
    import vtl.patches  # noqa: F401  -- importing the package is what registers them
    from vtl.registry import PATCH_REGISTRY

    return {p.name.upper(): p.default for p in PATCH_REGISTRY}


def compare(pins: dict[str, bool], defaults: dict[str, bool]) -> list[str]:
    """Every way the two can disagree, as a list of human-readable problems."""
    problems: list[str] = []

    for name in sorted(defaults.keys() - pins.keys()):
        problems.append(
            f"patch {name.lower()!r} is registered but NOT pinned in docker-compose.yaml -- "
            f"add `VTL_ENABLE_{name}: \"{int(defaults[name])}\"` to the registry-gates block"
        )
    for name in sorted(pins.keys() - defaults.keys()):
        problems.append(
            f"docker-compose.yaml pins VTL_ENABLE_{name} but no patch registers that name -- "
            "a renamed or deleted patch left a pin that reads as shipped-ON and does nothing"
        )

    for name in sorted(pins.keys() & defaults.keys()):
        pinned, default = pins[name], defaults[name]
        if name in FORCED_ON:
            if default is not False:
                problems.append(
                    f"{name} is on the FORCED_ON exception list but is now `default=True` -- "
                    "the exception (and the compose header's claim about it) is stale; "
                    "delete it from FORCED_ON and from the compose comment"
                )
            if pinned is not True:
                problems.append(
                    f"{name} is on the FORCED_ON exception list but compose pins it to 0 -- "
                    "either ship it on or stop calling it a forced-on exception"
                )
            continue
        if pinned != default:
            problems.append(
                f"VTL_ENABLE_{name} is pinned {int(pinned)} in docker-compose.yaml but the "
                f"patch registers default={default}. compose's registry-gates header claims "
                "these agree for every gate outside FORCED_ON -- fix one or the other"
            )
    return problems


def _self_check() -> None:
    text = COMPOSE.read_text(encoding="utf-8")

    # -- the parser itself, on synthetic text: both spellings, comments inert --
    sample = "\n".join(
        (
            '      VTL_ENABLE_ALPHA: "1"',
            "      VTL_ENABLE_BETA: 0",
            '      VTL_ENABLE_GAMMA: "0" # trailing comment',
            "      # VTL_ENABLE_DELTA: 1   <- commented out, not a pin",
            "      #   prose about VTL_ENABLE_EPSILON: 1",
            "      VTL_GREEDY_ARGMAX_SPLIT: 26",
        )
    )
    got = parse_compose_pins(sample)
    assert got == {"ALPHA": True, "BETA": False, "GAMMA": False}, got

    # -- the comparison, on synthetic inputs: each failure mode names itself --
    assert compare({"A": True}, {"A": True}) == []
    assert "NOT pinned" in compare({}, {"A": True})[0]
    assert "no patch registers" in compare({"A": True}, {})[0]
    assert "registers default=False" in compare({"A": True}, {"A": False})[0]
    # The exception list: default False + pinned on is the ONLY shape it accepts.
    ex = next(iter(FORCED_ON))
    assert compare({ex: True}, {ex: False}) == []
    assert "stale" in compare({ex: True}, {ex: True})[0]
    assert "forced-on exception" in compare({ex: False}, {ex: False})[0]

    # -- and now the real thing --
    pins = parse_compose_pins(text)
    assert pins, "no VTL_ENABLE_* pins found in docker-compose.yaml -- did the file move?"
    defaults = registry_defaults()
    assert defaults, "PATCH_REGISTRY is empty -- vtl.patches imported nothing"
    problems = compare(pins, defaults)
    if problems:
        print("compose <-> registry gate parity FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"compose gates self-check ok ({len(pins)} pins == {len(defaults)} registry defaults; "
        f"forced-on exceptions: {', '.join(sorted(FORCED_ON))})"
    )


if __name__ == "__main__":
    if "--self-check" in sys.argv or len(sys.argv) == 1:
        _self_check()
    else:
        raise SystemExit(f"usage: {sys.argv[0]} --self-check")
