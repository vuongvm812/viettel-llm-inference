"""C1 -- the on-box probes the Rust steady-state runner depends on.

The whole design in ``round-2/RUST-RUNNER.md`` rests on one fact: torch exposes the
instantiated CUDA graph as a raw ``cudaGraphExec_t`` that Rust can hand straight to
``cuGraphLaunch``, and that handle stays valid for the process lifetime. That is checkable
off-box by reading torch's source -- ``raw_cuda_graph_exec`` exists at
``torch/cuda/graphs.py:174-179`` in the pinned torch 2.11.0, vLLM constructs its graphs with
the default ``keep_graph=False`` (``cudagraph_utils.py:345``) and never calls
``.instantiate()`` -- but "the API exists and the docstring permits it" is not the same
claim as "launching through it produces the same bytes as ``graph.replay()`` on this box".

These probes make that a test result instead of a boot-time surprise, and they stay as the
regression against a torch bump. They gate a runtime DEFAULT, never correctness:
``VTL_RUST_RUNNER`` degrades to the Python step on any refusal.

    pytest bench/test_cuda_graph_handle_probe.py -q    # on the H200, inside the image
    make test-kernel                                    # runs them with everything else

PROBE 1 -- the handle exists and is non-zero after ``capture_end``.
    FAIL => the Rust runner is impossible as designed; ``VTL_RUST_RUNNER`` must stay 0 and
    the graph-launch half of the design needs a cuda-python shim instead.

PROBE 2 -- the handle is STABLE across replays and across re-reads.
    The docstring warns the exec is destroyed by a later ``instantiate()``. vLLM never calls
    it, so the handle should be constant -- but the runner caches it at boot and would
    launch a freed handle if that were ever untrue, which is a use-after-free, not a wrong
    number. FAIL => the boot-time export must re-read every step (and the design loses).

PROBE 3 -- ``cuGraphLaunch`` on that handle == ``graph.replay()``.
    The actual equivalence the runner substitutes.

PROBE 4 -- the output half: ``cuMemcpyDtoHAsync`` into pinned memory + an event.
    Phase 2 replaces ``AsyncOutput``'s fresh-tensor D2H with a pre-planned copy into a
    Rust-owned pinned buffer, synchronised on a blocking event. This proves that sequence
    without any Rust.

PROBE 5 (live tier) -- against the real booted runner: every FULL descriptor yields a
    non-zero handle, and the stream the runner captured on is the current stream.
    Needs ``VTL_TEST_MODEL``; skipped otherwise, same convention as
    ``bench/test_nstep_parity.py``.
"""

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _driver():
    """cuda-python's driver bindings, or skip.

    ``vtl/nvrtc.py`` already drives the CUDA driver API from Python this way, and
    ``round-2/Dockerfile`` best-effort installs ``cuda-python>=12.3`` -- so this is the
    in-image precedent, not a new dependency. The Rust side dlopens libcuda instead; the
    entry points and their semantics are the same.
    """
    try:
        from cuda.bindings import driver
    except Exception:  # pragma: no cover - import shape varies across cuda-python versions
        try:
            from cuda import cuda as driver  # cuda-python < 12.3 layout
        except Exception:
            pytest.skip("cuda-python not installed in this image")
    return driver


def _ck(driver, ret):
    """Unwrap cuda-python's ``(CUresult, *values)`` tuples, asserting success."""
    err, *rest = ret if isinstance(ret, tuple) else (ret,)
    ok = getattr(driver.CUresult, "CUDA_SUCCESS", 0)
    assert err == ok, f"CUDA driver call failed: {err}"
    return rest[0] if len(rest) == 1 else tuple(rest)


def _simple_graph():
    """A graph whose only effect is ``y += 1`` on a persistent device tensor.

    Deliberately trivial: these probes ask about the LAUNCH MECHANISM, not about what the
    model's graph contains -- ``test_nstep_capture_probe.py`` already covers capturability
    of the real step body.
    """
    y = torch.zeros(8, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    # Warm up on a side stream first: the first launch allocates workspaces, which is not
    # capturable. Same discipline as `_capture_burst_graphs`.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    y.zero_()
    with torch.cuda.graph(g):
        y.add_(1)
    y.zero_()
    return g, y


# ---------------------------------------------------------------------------------------
# Probe 1 / 2 -- the handle exists, and is stable.
# ---------------------------------------------------------------------------------------


def test_probe1_raw_cuda_graph_exec_exists_and_is_nonzero():
    assert hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph_exec"), (
        f"torch {torch.__version__} does not expose raw_cuda_graph_exec; the Rust runner "
        "cannot get at the cudaGraphExec_t and VTL_RUST_RUNNER must stay 0"
    )
    g, _y = _simple_graph()
    assert g.raw_cuda_graph_exec() != 0


def test_probe2_handle_is_stable_across_replays_and_rereads():
    g, y = _simple_graph()
    first = g.raw_cuda_graph_exec()
    for _ in range(8):
        g.replay()
    torch.cuda.synchronize()
    assert g.raw_cuda_graph_exec() == first, (
        "the cudaGraphExec_t moved after replay -- the boot-time export caches it, so a "
        "moving handle is a use-after-free in the Rust runner, not a wrong number"
    )
    assert int(y[0]) == 8, "sanity: the replays should have run"


# ---------------------------------------------------------------------------------------
# Probe 3 -- cuGraphLaunch is equivalent to graph.replay().
# ---------------------------------------------------------------------------------------


def test_probe3_cugraphlaunch_matches_replay():
    driver = _driver()
    g, y = _simple_graph()
    exec_handle = g.raw_cuda_graph_exec()
    stream = torch.cuda.current_stream().cuda_stream

    g.replay()
    torch.cuda.synchronize()
    via_replay = int(y[0])

    y.zero_()
    _ck(driver, driver.cuGraphLaunch(exec_handle, stream))
    torch.cuda.synchronize()
    via_driver = int(y[0])

    assert via_replay == via_driver == 1, (
        f"cuGraphLaunch and graph.replay() disagree ({via_driver} vs {via_replay}); the "
        "Rust runner substitutes exactly this call"
    )


# ---------------------------------------------------------------------------------------
# Probe 4 -- the output half: async D2H into pinned memory, synchronised on an event.
# ---------------------------------------------------------------------------------------


def test_probe4_async_d2h_into_pinned_with_blocking_event():
    driver = _driver()
    g, y = _simple_graph()
    stream = torch.cuda.current_stream().cuda_stream

    # Stands in for the Rust-owned pinned buffer (cuMemHostAlloc on that side).
    host = torch.empty(8, dtype=torch.int64, pin_memory=True)
    host.fill_(-1)

    # CU_EVENT_BLOCKING_SYNC == 1: park in the driver rather than spin. On the 3-vCPU judge
    # box a spinning wait starves the shm ingest thread (RUST-RUNNER.md hazard 5).
    event = _ck(driver, driver.cuEventCreate(1))

    _ck(driver, driver.cuGraphLaunch(g.raw_cuda_graph_exec(), stream))
    _ck(driver, driver.cuMemcpyDtoHAsync(
        host.data_ptr(), y.data_ptr(), y.numel() * y.element_size(), stream))
    _ck(driver, driver.cuEventRecord(event, stream))
    _ck(driver, driver.cuEventSynchronize(event))

    assert host.tolist() == [1] * 8, (
        f"pre-planned D2H did not land the graph's output ({host.tolist()}); this is the "
        "sequence that replaces AsyncOutput's fresh-tensor copies"
    )
    _ck(driver, driver.cuEventDestroy(event))


# ---------------------------------------------------------------------------------------
# Probe 5 -- the live runner. Needs a model.
# ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("VTL_TEST_MODEL"), reason="set VTL_TEST_MODEL")
def test_probe5_live_runner_graphs_all_expose_handles():
    from vllm import LLM
    from vllm.config.compilation import CUDAGraphMode

    llm = LLM(
        model=os.environ["VTL_TEST_MODEL"],
        max_model_len=2048,
        gpu_memory_utilization=0.60,
        enforce_eager=False,
    )
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    mgr = getattr(runner, "cudagraph_manager", None)
    if mgr is None or not getattr(mgr, "graphs", None):
        pytest.skip("no cudagraph manager / no captured graphs in this configuration")

    full = [d for d in mgr.graphs if d.cg_mode == CUDAGraphMode.FULL]
    assert full, "no FULL descriptors captured -- the Rust runner has nothing to drive"
    for desc in full:
        handle = mgr.graphs[desc].raw_cuda_graph_exec()
        assert handle != 0, f"FULL descriptor {desc} yielded a null cudaGraphExec_t"

    # The runner exports ONE stream at boot and launches every graph on it. If the captured
    # main stream is not the current stream, that export is wrong and ordering breaks.
    assert runner.main_stream.cuda_stream == torch.cuda.current_stream().cuda_stream, (
        "runner.main_stream is not the current stream; the boot-time export would hand "
        "Rust a stream the graphs were not captured against"
    )
