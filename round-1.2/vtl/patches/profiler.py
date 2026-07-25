"""In-process torch.profiler, driven from the worker -- because this build stripped it.

The served ``awesome-badger`` vLLM image has NO ``VLLM_TORCH_PROFILER_DIR`` env and NO
``/start_profile`` route (both greps came back empty on the box), so the documented
profiler path (https://docs.vllm.ai/en/stable/contributing/profiling/) can't register.
This patch reinstates it without the endpoint or an image rebuild: ``VLLM_PLUGINS=vtl``
already loads us in the worker process, so we wrap ``GPUModelRunner.execute_model`` and
run ``torch.profiler`` around a fixed window of engine steps, then dump a chrome trace to
``VTL_PROFILE_DIR``. ``bench/profile_trace.py`` buckets that trace unchanged.

Arming is a trigger FILE, not a step counter from process start: startup warmup (the
healthcheck replay) burns an unknown number of execute_model calls, and we want the
REPLAY captured, not warmup. So `make profile` boots + waits for healthy, then
``touch $VTL_PROFILE_DIR/.arm`` right before the load replay; the worker starts profiling
on the next step, captures ``VTL_PROFILE_STEPS`` steps, writes the trace, and disarms.

A chrome trace only names ATen ops, and on this workload those account for 2.3 of the
6.5 ms host time per decode step -- the other ~4 ms is interpreter frames the C++ profiler
cannot see (with_stack only decorates ops that exist). So the same window optionally drives
``cProfile`` on the engine-core thread and dumps a tottime-ranked ``vtl-pyprof-<pid>.txt``
next to the trace. Both profilers inflate absolute time; the RANKING is what's usable.

Env (all read directly; the patch is inert unless the dir is set):
  VTL_ENABLE_PROFILER   registry gate (overlay sets =1; default OFF)
  VTL_PROFILE_DIR       output dir for the chrome trace (also the .arm trigger dir)
  VTL_PROFILE_STEPS     execute_model calls to capture after arming (default 20)
  VTL_PROFILE_PY        also run cProfile over the window (default 1; "0" = trace only)

Fail-closed: any profiler error disables further profiling and NEVER breaks execute_model.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# One-shot state machine for the single worker process.
_state = {"prof": None, "pyprof": None, "started": False, "since": 0, "done": False}


def _profile_dir() -> str | None:
    d = os.environ.get("VTL_PROFILE_DIR", "").strip()
    return d or None


def _steps() -> int:
    try:
        return max(1, int(os.environ.get("VTL_PROFILE_STEPS", "20")))
    except ValueError:
        return 20


def _py_enabled() -> bool:
    return os.environ.get("VTL_PROFILE_PY", "1").strip().lower() in _TRUTHY


def _new_pyprofiler():  # split out so the self-check can drive it with a fake
    import cProfile

    return cProfile.Profile()


def _dump_pyprof(pyprof, path: str) -> None:
    """tottime-ranked text dump. tottime (self, excluding callees) is the one that
    localises interpreter cost; cumtime just re-reports the call stack."""
    import pstats

    with open(path, "w") as f:
        st = pstats.Stats(pyprof, stream=f)
        st.sort_stats("tottime").print_stats(60)
        st.sort_stats("cumtime").print_stats(40)


def _arm_path(d: str) -> str:
    return os.path.join(d, ".arm")


def _armed(d: str) -> bool:
    try:
        return os.path.exists(_arm_path(d))
    except Exception:
        return False


def _disarm(d: str) -> None:
    try:
        os.remove(_arm_path(d))
    except Exception:
        pass


def _new_profiler():  # pragma: no cover -- needs torch; self-check injects a fake
    import torch

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    )


def _make_wrapper(original):
    """Wrap execute_model: arm on the trigger file, capture _steps() calls, dump, disarm.
    Module-level state + pure-ish control flow so the self-check can drive it with fakes."""

    def execute_model(self, *args, **kwargs):
        d = _profile_dir()
        # Start: armed, not yet running, not already done.
        if d and not _state["done"] and not _state["started"] and _armed(d):
            try:
                _state["prof"] = _new_profiler()
                _state["prof"].start()
                if _py_enabled():
                    # After the torch profiler so its own startup is not attributed to
                    # the model. enable() is per-thread and this IS the engine-core
                    # busy-loop thread (scheduler + runner), which is what we want.
                    _state["pyprof"] = _new_pyprofiler()
                    _state["pyprof"].enable()
                _state["started"] = True
                _state["since"] = 0
                log.info("vtl: profiler ARMED -> capturing %d execute_model steps", _steps())
            except Exception:
                log.exception("vtl: profiler start failed; disabling")
                _state["done"] = True
                _state["prof"] = None

        out = original(self, *args, **kwargs)

        # Stop after the window; export once, then disarm and never profile again.
        if _state["started"] and not _state["done"]:
            _state["since"] += 1
            if _state["since"] >= _steps():
                try:
                    if _state["pyprof"] is not None:
                        _state["pyprof"].disable()
                    _state["prof"].stop()
                    path = os.path.join(d, f"vtl-trace-{os.getpid()}.json")
                    _state["prof"].export_chrome_trace(path)
                    log.info("vtl: profiler wrote %s (%d steps)", path, _state["since"])
                    if _state["pyprof"] is not None:
                        py_path = os.path.join(d, f"vtl-pyprof-{os.getpid()}.txt")
                        _dump_pyprof(_state["pyprof"], py_path)
                        log.info("vtl: python profiler wrote %s", py_path)
                except Exception:
                    log.exception("vtl: profiler export failed")
                finally:
                    _state["done"] = True
                    _state["prof"] = None
                    _state["pyprof"] = None
                    if d:
                        _disarm(d)
        return out

    return execute_model


# BOTH runner classes, because gpu_worker.py picks one at construction time from
# `vllm_config.use_v2_model_runner` and we ship VLLM_USE_V2_MODEL_RUNNER=1. Patching only the
# V1 module (what this file did until 2026-07-25) meant the wrapper was installed on a class
# the worker never instantiates: `make profile` armed, captured nothing, and died on "no
# vtl-trace-*.json" -- i.e. the repo's only profiling tool was dead in the shipped config.
# Both are patched rather than the config being read here: the module-level _state machine is
# one-shot per process and only one of these classes is ever constructed, so double-patching
# costs nothing and cannot double-capture.
_RUNNER_MODULES = (
    "vllm.v1.worker.gpu.model_runner",  # V2 -- what we serve
    "vllm.v1.worker.gpu_model_runner",  # V1 -- VLLM_USE_V2_MODEL_RUNNER=0
)


@register_patch("profiler", default=False)
def apply() -> None:
    # Inert unless a profile dir is configured (the overlay sets it). Keeps this a no-op
    # in every normal serve even if VTL_ENABLE_PROFILER leaks on.
    if _profile_dir() is None:
        log.info("vtl: profiler enabled but VTL_PROFILE_DIR unset; nothing to do")
        return
    import importlib

    installed = []
    for mod_name in _RUNNER_MODULES:
        try:
            runner = importlib.import_module(mod_name).GPUModelRunner
        except Exception as exc:  # not the worker process, or class moved -> skip cleanly
            log.warning("vtl: %s.GPUModelRunner not importable (%s)", mod_name, exc)
            continue
        if already_patched(runner, "execute_model"):
            continue
        original = runner.execute_model
        runner.execute_model = mark_patched(_make_wrapper(original), original)
        installed.append(mod_name)
    if not installed:
        log.warning("vtl: no GPUModelRunner patched; profiler not installed")
        return
    log.info(
        "vtl: profiler installed on %s -> touch %s to capture %d steps",
        ", ".join(installed),
        _arm_path(_profile_dir()),
        _steps(),
    )


def _self_check() -> None:
    """No torch, no vLLM: fakes exercise the arm -> capture N -> dump -> disarm machine."""
    import tempfile

    d = tempfile.mkdtemp()
    os.environ["VTL_PROFILE_DIR"] = d
    os.environ["VTL_PROFILE_STEPS"] = "3"
    _fresh = {"prof": None, "pyprof": None, "started": False, "since": 0, "done": False}
    for k in _state:
        _state[k] = _fresh[k]

    events: list[str] = []

    class FakeProf:
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

        def export_chrome_trace(self, path):
            events.append("export")
            with open(path, "w") as f:
                f.write("{}")

    class FakePyProf:
        def enable(self):
            events.append("py-enable")

        def disable(self):
            events.append("py-disable")

    # inject the fakes + a no-op original
    global _new_profiler, _new_pyprofiler, _dump_pyprof
    saved, saved_py, saved_dump = _new_profiler, _new_pyprofiler, _dump_pyprof
    _new_profiler = lambda: FakeProf()  # noqa: E731
    _new_pyprofiler = lambda: FakePyProf()  # noqa: E731

    def _fake_dump(p, path):
        events.append("py-dump")
        with open(path, "w") as f:
            f.write("")

    _dump_pyprof = _fake_dump
    try:
        wrapped = _make_wrapper(lambda self, *a, **k: "out")
        obj = object()

        # Not armed: no profiling, calls pass through.
        assert wrapped(obj) == "out"
        assert events == [], events
        assert not _state["started"]

        # Arm, then drive: start on the next call, capture exactly STEPS calls, export once.
        open(_arm_path(d), "w").close()
        expected = ["start", "py-enable", "py-disable", "stop", "export", "py-dump"]
        for _ in range(5):
            assert wrapped(obj) == "out"
        assert events == expected, events
        assert _state["done"] is True
        assert not os.path.exists(_arm_path(d)), "should disarm after export"
        assert os.path.exists(os.path.join(d, f"vtl-trace-{os.getpid()}.json"))
        assert os.path.exists(os.path.join(d, f"vtl-pyprof-{os.getpid()}.txt"))

        # Re-arming after done is a no-op (one-shot per process).
        open(_arm_path(d), "w").close()
        wrapped(obj)
        assert events == expected, events

        # VTL_PROFILE_PY=0 keeps the chrome trace but skips cProfile entirely.
        for k in _state:
            _state[k] = _fresh[k]
        events.clear()
        os.environ["VTL_PROFILE_PY"] = "0"
        wrapped = _make_wrapper(lambda self, *a, **k: "out")
        open(_arm_path(d), "w").close()
        for _ in range(4):
            wrapped(obj)
        assert events == ["start", "stop", "export"], events
    finally:
        _new_profiler, _new_pyprofiler, _dump_pyprof = saved, saved_py, saved_dump
        os.environ.pop("VTL_PROFILE_PY", None)
        os.environ.pop("VTL_PROFILE_DIR", None)
        os.environ.pop("VTL_PROFILE_STEPS", None)

    # The runner the compose actually serves (VLLM_USE_V2_MODEL_RUNNER=1) must be covered,
    # or the capture silently produces nothing. Regression guard for the V1-only bug.
    assert "vllm.v1.worker.gpu.model_runner" in _RUNNER_MODULES, _RUNNER_MODULES
    assert "vllm.v1.worker.gpu_model_runner" in _RUNNER_MODULES, _RUNNER_MODULES

    print("profiler self-check ok")


if __name__ == "__main__":
    _self_check()
