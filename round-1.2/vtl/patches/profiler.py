"""In-process torch.profiler, driven from the worker -- because this build stripped it.

The served ``awesome-badger`` vLLM image has NO ``VLLM_TORCH_PROFILER_DIR`` env and NO
``/start_profile`` route (both greps came back empty on the box), so the documented
profiler path (https://docs.vllm.ai/en/stable/contributing/profiling/) can't register.
This patch reinstates it without the endpoint or an image rebuild: ``VLLM_PLUGINS=vtl``
already loads us in the worker process, so we wrap ``GPUModelRunner.execute_model`` and
run ``torch.profiler`` around a fixed window of engine steps, then dump a chrome trace to
``VTL_PROFILE_DIR``. ``bench/profile_trace.py`` buckets that trace unchanged.

There are TWO ``GPUModelRunner`` classes (V1 and V2) and the worker picks one at construction
time, so ``apply()`` wraps both -- see the comment there. SCOPE: this wraps the WORKER's
``execute_model``, so the trace covers model execution only. The scheduler runs in the
engine-core process and does NOT appear here, which is why scheduler-side changes cannot be
sized with ``make profile``.

Arming is a trigger FILE, not a step counter from process start: startup warmup (the
healthcheck replay) burns an unknown number of execute_model calls, and we want the
REPLAY captured, not warmup. So `make profile` boots + waits for healthy, then
``touch $VTL_PROFILE_DIR/.arm`` right before the load replay; the worker starts profiling
on the next step, captures ``VTL_PROFILE_STEPS`` steps, writes the trace, and disarms.

Env (all read directly; the patch is inert unless the dir is set):
  VTL_ENABLE_PROFILER   registry gate (overlay sets =1; default OFF)
  VTL_PROFILE_DIR       output dir for the chrome trace (also the .arm trigger dir)
  VTL_PROFILE_STEPS     execute_model calls to capture after arming (default 20)

Fail-closed: any profiler error disables further profiling and NEVER breaks execute_model.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# One-shot state machine for the single worker process.
_state = {"prof": None, "started": False, "since": 0, "done": False}


def _profile_dir() -> str | None:
    d = os.environ.get("VTL_PROFILE_DIR", "").strip()
    return d or None


def _steps() -> int:
    try:
        return max(1, int(os.environ.get("VTL_PROFILE_STEPS", "20")))
    except ValueError:
        return 20


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
                    _state["prof"].stop()
                    path = os.path.join(d, f"vtl-trace-{os.getpid()}.json")
                    _state["prof"].export_chrome_trace(path)
                    log.info("vtl: profiler wrote %s (%d steps)", path, _state["since"])
                except Exception:
                    log.exception("vtl: profiler export failed")
                finally:
                    _state["done"] = True
                    _state["prof"] = None
                    if d:
                        _disarm(d)
        return out

    return execute_model


@register_patch("profiler", default=False)
def apply() -> None:
    # Inert unless a profile dir is configured (the overlay sets it). Keeps this a no-op
    # in every normal serve even if VTL_ENABLE_PROFILER leaks on.
    if _profile_dir() is None:
        log.info("vtl: profiler enabled but VTL_PROFILE_DIR unset; nothing to do")
        return
    # Patch BOTH runner classes rather than resolving which one is live. There are two
    # (vllm/v1/worker/gpu/model_runner.py for V2, vllm/v1/worker/gpu_model_runner.py for V1)
    # and gpu_worker.py:402-410 picks one at construction time -- long after this runs. Until
    # 2026-07-26 this patched only the V1 class while the server ran
    # VLLM_USE_V2_MODEL_RUNNER=1, so it logged "installed" every boot and never once fired;
    # `make profile` produced no trace at all. The worker builds exactly ONE runner, so only
    # one wrapper can ever be called and the one-shot _state below needs no change.
    installed = []
    for mod_path, label in (
        ("vllm.v1.worker.gpu.model_runner", "V2"),
        ("vllm.v1.worker.gpu_model_runner", "V1"),
    ):
        try:
            runner = __import__(mod_path, fromlist=["GPUModelRunner"]).GPUModelRunner
        except Exception as exc:  # not the worker process, or the class moved
            log.info("vtl: profiler skipping %s runner (%s)", label, exc)
            continue
        if already_patched(runner, "execute_model"):
            installed.append(label)
            continue
        original = runner.execute_model
        runner.execute_model = mark_patched(_make_wrapper(original), original)
        installed.append(label)

    if not installed:
        # Loud, not silent: a rename here is exactly how this patch died the first time.
        log.warning("vtl: no GPUModelRunner importable; profiler NOT installed")
        return
    log.info(
        "vtl: profiler installed on %s runner(s) -> touch %s to capture %d steps",
        "+".join(installed),
        _arm_path(_profile_dir()),
        _steps(),
    )


def _self_check() -> None:
    """No torch, no vLLM: fakes exercise the arm -> capture N -> dump -> disarm machine."""
    import tempfile

    d = tempfile.mkdtemp()
    os.environ["VTL_PROFILE_DIR"] = d
    os.environ["VTL_PROFILE_STEPS"] = "3"
    for k in _state:
        _state[k] = {"prof": None, "started": False, "since": 0, "done": False}[k]

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

    # inject the fake profiler + a no-op original
    global _new_profiler
    saved = _new_profiler
    _new_profiler = lambda: FakeProf()  # noqa: E731
    try:
        wrapped = _make_wrapper(lambda self, *a, **k: "out")
        obj = object()

        # Not armed: no profiling, calls pass through.
        assert wrapped(obj) == "out"
        assert events == [], events
        assert not _state["started"]

        # Arm, then drive: start on the next call, capture exactly STEPS calls, export once.
        open(_arm_path(d), "w").close()
        for _ in range(5):
            assert wrapped(obj) == "out"
        assert events == ["start", "stop", "export"], events
        assert _state["done"] is True
        assert not os.path.exists(_arm_path(d)), "should disarm after export"
        assert os.path.exists(os.path.join(d, f"vtl-trace-{os.getpid()}.json"))

        # Re-arming after done is a no-op (one-shot per process).
        open(_arm_path(d), "w").close()
        wrapped(obj)
        assert events == ["start", "stop", "export"], events

        # apply() must wrap BOTH runner classes. Patching only the V1 class while the server
        # ran VLLM_USE_V2_MODEL_RUNNER=1 is exactly how this patch sat dead and silent, so
        # stand fake modules up at both import paths and assert both get wrapped.
        import sys
        import types

        fakes = {}
        for path in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
            mod = types.ModuleType(path)

            class _Runner:
                def execute_model(self, *a, **k):
                    return "stock"

            mod.GPUModelRunner = _Runner
            fakes[path] = _Runner
            sys.modules[path] = mod
        try:
            apply()
            for path, cls in fakes.items():
                assert already_patched(cls, "execute_model"), f"{path} left unpatched"
        finally:
            for path in fakes:
                sys.modules.pop(path, None)
    finally:
        _new_profiler = saved
        os.environ.pop("VTL_PROFILE_DIR", None)
        os.environ.pop("VTL_PROFILE_STEPS", None)

    print("profiler self-check ok")


if __name__ == "__main__":
    _self_check()
