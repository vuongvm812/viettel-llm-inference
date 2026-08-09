"""Phase 2 -- hand the captured CUDA graphs to the Rust steady-state runner.

Not a registered patch: this module owns no monkeypatch. It is called from
``nstep_decode``'s ``capture_model`` wrapper (inside that wrapper's try/except, so any
failure here sets ``BURST.armed = False`` and the boot keeps the stock path) and from
``rust_sched``'s step commit.

WHAT IT EXPORTS, and why each piece:

  * the GRAPH TABLE -- one row per captured descriptor, and ``mgr._candidates`` flattened.
    The candidate lists are copied rather than re-derived because ``_init_candidates``
    (cudagraph_utils.py:181) is a policy that can change under us; a second implementation
    that drifted would silently pick a DIFFERENT graph than vLLM would, which is a wrong
    answer rather than a crash.

    THE HANDLE IN EACH ROW IS NOT STOCK'S. ``mgr.graphs[desc]`` is the FORWARD-ONLY graph:
    it ends with hidden states and never writes ``BURST.accum``, which is the buffer
    ``arm()`` points the pre-planned D2H at. The graph that ends a step with tokens in
    ``accum`` is nstep's UNROLL graph (``nstep_decode._burst_all``: in-graph ``step.zero_``
    -> prologue -> N-1 bodies -> accumulator tail), so that is the handle the runner is
    given. Descriptors with no unroll graph get a row (dispatch fidelity) with a NULL
    handle, which makes the runner decline that shape instead of launching a graph that
    would leave `accum` holding the previous step's tokens.
  * the STREAM the graphs were captured on, and nothing else -- the runner uses one stream.
  * the OUTPUT BUFFER: ``BURST.accum``'s device pointer and shape, so the per-step D2H is
    pre-planned into pinned memory the crate owns, replacing ``AsyncOutput``'s freshly
    allocated CPU tensors.

Then it PROVES the table before arming: every reachable ``(num_reqs, num_tokens, uniform,
loras)`` tuple is dispatched through both vLLM's manager and the Rust table, and any
disagreement refuses the arm. That check is the whole reason it is safe to have a second
dispatcher at all.

THE PER-STEP HANDSHAKE. Module globals, same lifetime story as ``BURST``:

  * ``STATE.step`` -- the STASH. ``rust_sched.commit_burst`` writes it at schedule time
    with everything the launch needs that only the scheduler knows (slot layout, the
    committed burst width, the publish hint). ``nstep_decode``'s ``sample_tokens`` takes it,
    matches it against the batch it is actually holding, and either launches or drops it.
  * ``STATE.pending`` -- update mode: the FIFO of ``(ring slot, stash)`` for launches whose
    ``update_from_output`` has not arrived yet. At most two, see below.
  * ``STATE.done`` -- sample mode: the RESULT. Written by the launch, applied by
    ``rust_sched.update_from_output`` through the existing ``r9_apply`` residue loop.
  * ``STATE.inflight`` -- sample mode: scheduled steps not yet applied.

WHERE THE COMMIT HAPPENS, and why there are two modes (``VTL_RUST_RUNNER_COMMIT``).
With async scheduling the engine runs a batch queue of depth 2
(``vllm_config.max_concurrent_batches`` = pp_size + 1 on the V2 runner), and its loop is

    schedule(k) -> execute(k) -> sample(k) -> update(k-1)

so ``sample(k)`` runs BEFORE step k-1 has been applied.

  * ``"sample"`` (legacy) -- the crate's ``run_steps`` launches AND commits inside
    ``sample_tokens``. That would append step k's tokens ahead of step k-1's, so the launch
    is interlocked on ``STATE.inflight == 1`` (this step is the only unapplied one). Under
    the served config that is only true when the batch queue drains, so the runner arms,
    self-checks and idles. Its one advantage is the multi-step residency loop
    (``VTL_RUST_RUNNER_STEPS`` > 1), which needs the whole burst chain inside one FFI call.
  * ``"update"`` (default, shipped) -- the crate's ``launch`` and ``commit`` split the step
    in two. ``sample(k)`` only enqueues: ``cuGraphLaunch`` + the pre-planned D2H into the
    free ring slot + an event record, no wait. The commit runs in ``update_from_output``, and
    UPDATES are strictly ordered even though samples are not, so there is no ordering
    hazard and no interlock -- the depth-2 overlap is preserved. Two pinned buffers is
    exactly what the depth-2 queue needs; a third outstanding launch cannot exist, and if
    one ever did the launch side simply declines and Python runs the step.

    What is kept: ``cuGraphLaunch`` in place of torch's replay dispatch, the pre-planned
    pinned D2H, and the ``decide()`` collapse at update time (event sync + gather +
    ``step_pack_locked`` instead of the per-request Python loop and its numpy crossing).
    What is given up: the multi-step loop, and ``AsyncOutput`` stays (its D2H is now
    redundant with the ring's) -- which is also the property that makes every decline safe,
    since a dropped launch just means Python's ``decide()`` commits the same tokens.

``VTL_RUST_RUNNER``: "0" off, "1" on.
``VTL_RUST_RUNNER_COMMIT``: "update" (default) | "sample" (legacy interlocked path).
``VTL_RUST_RUNNER_STEPS``: burst launches per FFI call. SAMPLE MODE ONLY.
``VTL_RUST_RUNNER_REQUIRE``: "1" makes a refusal a BOOT FAILURE instead of a silent
degrade -- see ``_State.refuse``.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

log = logging.getLogger("vllm.vtl.runner")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Pinned ring depth, mirroring ``RING`` in python.rs. The batch queue is depth 2, so two
#: launches is the most that can be outstanding; the crate allocates exactly this many
#: buffers and events at ``arm()``.
RING = 2


class PostLaunchError(RuntimeError):
    """A failure AFTER a successful update-mode ``launch``.

    The graph has run and its commit is pending in the FIFO, so the one degrade every
    other refusal on this path uses -- "fall through and let Python run the step" -- is
    the one thing that must NOT happen: stock sampling would run the step a second time
    on top of state the graph already advanced. ``wrap_sample_tokens`` re-raises this
    instead of falling back; an honest engine error beats a silently garbled stream.
    """


def mode() -> str:
    """"off" | "on"."""
    raw = os.environ.get("VTL_RUST_RUNNER", "1").strip().lower()
    return "on" if raw in _TRUTHY else "off"


def commit_mode() -> str:
    """"update" (default) | "sample". See the module docstring.

    Anything unrecognised reads as the default rather than as an error: this knob picks
    between two working paths, and a typo must not be the thing that turns the shipped one
    off silently.
    """
    raw = os.environ.get("VTL_RUST_RUNNER_COMMIT", "update").strip().lower()
    return "sample" if raw == "sample" else "update"


def require() -> bool:
    """``VTL_RUST_RUNNER_REQUIRE``: turn a refusal into a raise. Mirrors
    ``VTL_RUST_SCHED_REQUIRE`` (rust_sched.refuse) and exists for the same reason: a runner
    that quietly never armed looks, in every latency number, exactly like a runner that ran
    and did not help, so an A/B arm that measures the stock path while looking fine is the
    failure this flag makes impossible. Default OFF -- in the submission, serving-but-slower
    beats not serving."""
    return os.environ.get("VTL_RUST_RUNNER_REQUIRE", "").strip().lower() in _TRUTHY


def max_steps() -> int:
    """Phase 3's ceiling, SAMPLE MODE ONLY. 1 = one step per FFI call.

    Update mode splits the step across two FFI calls and commits at update time, so there is
    no residency loop for this to bound -- ``nstep_decode.apply`` pins it to 1 there, which
    is also what keeps the continuation graphs (and their burst-graph memory) uncaptured.
    """
    try:
        n = int(os.environ.get("VTL_RUST_RUNNER_STEPS", "8").strip() or 8)
    except ValueError:
        return 1
    return max(1, min(n, 64))


class _Stash(NamedTuple):
    """What the scheduler committed, handed to the launch site. One per scheduled step.

    ``key`` is the request ids in ``slots`` ORDER, not in ``num_scheduled_tokens`` order --
    built that way on purpose: the launch site checks ``key == tuple(ib.req_ids)``, and
    because the key was derived from the slot list, a passing check is the proof that
    ``slots[i]`` is the slot of the batch row ``i`` the runner is about to sample. There is
    no separate ordering assert to keep in sync.
    """

    key: tuple
    slots: tuple
    reqs: tuple           # the Request objects, in the same order (residue needs them)
    n: int                # tokens per burst launch
    steps: int            # launches committed; the step's token budget is n * steps
    max_model_len: int
    fold: bool            # fold cache_blocks into the crossing (R9.live)
    publish: bool         # inline shm publish is safe this step
    seq: int = 0          # update mode: which scheduled step this is. See `_State.pending`.


class _Done(NamedTuple):
    """What the launch produced, handed back to ``update_from_output``."""

    stash: _Stash
    verdicts: list        # (num_accepted, status, stop_reason) for the LAST launch
    records: list         # R8 records that were NOT published inline, in order
    exit: str             # LoopExit: "budget" | "stopped" | "unpacked"
    ran: int              # launches that actually ran (<= stash.steps)


class _State:
    """Module-global, like ``BURST`` and ``decode_fastpath._C``: one runner per process."""

    __slots__ = ("runner", "live", "why", "step", "done", "inflight", "engaged",
                 "update", "pending")

    def __init__(self) -> None:
        self.runner = None      # the vtl_sched.Runner, once armed
        self.live = False       # may the runner be called?
        self.why = "not armed"  # the one reason, logged once
        self.step = None        # the pending _Stash, or None
        self.done = None        # sample mode: the pending _Done, or None
        self.inflight = 0       # sample mode: scheduled steps not yet applied
        self.engaged = 0        # launches taken this boot, for the once-in-a-while log
        self.update = True      # commit at UPDATE time (the default mode); set by `export`
        self.pending = []       # update mode: FIFO of (ring slot, _Stash) not yet committed

    def refuse(self, why: str) -> None:
        """Permanent stand-down for the boot.

        Raises only under ``VTL_RUST_RUNNER_REQUIRE`` (see ``require``); otherwise never
        raises -- the caller keeps the stock path. The latch is dropped and the crate-side
        runner disarmed BEFORE the raise, so even the raising arm leaves consistent state
        if something upstream swallows it.
        """
        if self.live:
            log.error("vtl: rust runner disabled for this boot -- %s", why)
        else:
            log.info("vtl: rust runner not armed -- %s", why)
        self.live = False
        self.why = why
        self.step = None
        self.done = None
        # Dropping the pending launches is SAFE, and that is a property of update mode
        # rather than a hope: a launch commits nothing, so a step whose entry is dropped is
        # committed by Python's `decide()` off the `AsyncOutput` the worker kept.
        self.pending = []
        if self.runner is not None:
            try:
                self.runner.disarm()
            except Exception:  # pragma: no cover - a disarm that throws must not propagate
                log.exception("vtl: rust runner disarm failed")
        if require():
            raise RuntimeError(
                f"VTL_RUST_RUNNER_REQUIRE=1 but the Rust runner cannot run: {why}"
            )

    def take_step(self) -> "_Stash | None":
        """Pop the stash. Always popped, launch or no launch: a stash names a SCHEDULED
        step, and the next step's batch can have the identical request-id key, so a
        leftover would match a step it was never written for."""
        stash, self.step = self.step, None
        return stash

    def free_slot(self) -> "int | None":
        """The ring index no outstanding launch holds, or ``None`` when both are taken.

        THE OWNERSHIP RULE, in one place: a slot is owned from the moment its
        ``(slot, stash)`` goes into ``pending`` until ``update_from_output`` pops it, and the
        crate only ever indexes what it is handed. ``None`` is not an error -- the launch
        site declines and Python runs the step, which is what happens whenever a third step
        would otherwise be in flight.
        """
        used = [slot for slot, _ in self.pending]
        for slot in range(RING):
            if slot not in used:
                return slot
        return None


STATE = _State()


def _descriptor_fields(desc):
    """``(num_tokens, num_reqs, uniform_token_count, num_active_loras, is_full)``.

    ``num_reqs`` / ``uniform_token_count`` stay ``None`` where the descriptor leaves them
    ``None`` -- that is what makes a PIECEWISE entry match any uniform count, and collapsing
    it to a number here would change dispatch.
    """
    from vllm.config.compilation import CUDAGraphMode

    return (
        int(desc.num_tokens),
        None if desc.num_reqs is None else int(desc.num_reqs),
        None if desc.uniform_token_count is None else int(desc.uniform_token_count),
        int(desc.num_active_loras),
        desc.cg_mode == CUDAGraphMode.FULL,
    )


def _graph_execs(graphs: dict, what: str) -> dict:
    """``{desc: exec_handle}`` for one of nstep's whole-burst graph dicts.

    ``graphs`` is ``BURST.unroll_graphs`` / ``BURST.continue_graphs``: ``rows -> (graph,
    desc, n)``. A graph whose handle cannot be read is DROPPED rather than exported as 0,
    so the descriptor falls through to the null-handle row ``_build_table`` writes for
    everything else -- same outcome, one code path.
    """
    execs = {}
    for entry in graphs.values():
        graph, desc = entry[0], entry[1]
        try:
            handle = int(graph.raw_cuda_graph_exec())
        except Exception as exc:
            log.warning("vtl: no raw_cuda_graph_exec for the %s graph at %s: %s",
                        what, desc, exc)
            continue
        if handle:
            execs[desc] = handle
    return execs


def _build_table(runner_obj, mgr, b) -> tuple[dict, dict]:
    """Register every captured graph and candidate list.

    Returns ``({desc: row_index}, {desc: unroll_exec})`` -- the second is what
    ``_verify_dispatch`` must compare against, since it is what the rows actually hold.
    """
    index = {}
    execs = _graph_execs(getattr(b, "unroll_graphs", {}), "unroll")
    conts = _graph_execs(getattr(b, "continue_graphs", {}), "continuation")

    def add(desc):
        if desc in index:
            return index[desc]
        num_tokens, num_reqs, uniform, loras, is_full = _descriptor_fields(desc)
        # The whole-burst handle or nothing: see the module docstring. A 0 handle is never
        # launchable, so the runner declines that shape instead of replaying a graph that
        # does not fill the accumulator.
        idx = runner_obj.register_graph(
            num_tokens, num_reqs, uniform, loras,
            execs.get(desc, 0), is_full, conts.get(desc, 0),
        )
        index[desc] = idx
        return idx

    # Stock's own descriptors first, so dispatch's priority order can reference them.
    for desc in mgr.graphs:
        add(desc)

    # The candidate lists, verbatim. A descriptor appearing here that stock never captured
    # (possible in principle) still gets a row, with a null handle.
    for (num_tokens, loras), descs in getattr(mgr, "_candidates", {}).items():
        idx = [add(d) for d in descs]
        runner_obj.set_candidates(int(num_tokens), int(loras), idx)

    runner_obj.set_lora_dispatch(
        {int(k): int(v) for k, v in getattr(mgr, "_lora_dispatch_map", {}).items()},
        int(getattr(mgr, "_max_lora_case", 0)),
    )
    runner_obj.set_captured(bool(getattr(mgr, "_graphs_captured", False)))
    return index, execs


def _verify_dispatch(runner_obj, mgr, max_num_reqs: int, execs: dict) -> str | None:
    """Reason the Rust table disagrees with vLLM's dispatcher, or ``None``.

    Exhaustive over what the live config can actually produce: every captured token count
    (plus a couple of misses), every request count up to ``max_num_reqs``, and the uniform
    counts a decode step can have. A few thousand tuples, once, at boot.
    """
    from vllm.config.compilation import CUDAGraphMode

    token_counts = sorted({d.num_tokens for d in mgr.graphs} | {0, 1})
    uniforms = (None, 1, 2)
    checked = 0
    for num_tokens in token_counts:
        for num_reqs in range(0, min(max_num_reqs, num_tokens) + 1):
            for uniform in uniforms:
                theirs = mgr.dispatch(num_reqs, num_tokens, uniform, 0)
                mine = runner_obj.dispatch_probe(num_reqs, num_tokens, uniform, 0)
                checked += 1
                if theirs.cg_mode == CUDAGraphMode.NONE:
                    # vLLM fell back to eager: the Rust table must miss too, or the runner
                    # would launch a graph where stock would not.
                    if mine is not None:
                        return (
                            f"rust dispatched {mine} where vLLM fell back to eager at "
                            f"(num_reqs={num_reqs}, num_tokens={num_tokens}, "
                            f"uniform={uniform})"
                        )
                    continue
                if mine is None:
                    return (
                        f"rust missed at (num_reqs={num_reqs}, num_tokens={num_tokens}, "
                        f"uniform={uniform}) where vLLM chose {theirs}"
                    )
                want_full = theirs.cg_mode == CUDAGraphMode.FULL
                if mine[1] != want_full:
                    return (
                        f"rust cg_mode disagrees at (num_reqs={num_reqs}, "
                        f"num_tokens={num_tokens}, uniform={uniform}): full={mine[1]} "
                        f"vs {theirs.cg_mode}"
                    )
                if want_full:
                    # Against the UNROLL handle, which is what the row holds -- comparing
                    # against `mgr.graphs[theirs]` would now fail on every decode size.
                    handle = execs.get(theirs, 0)
                    if handle and mine[0] != handle:
                        return (
                            f"rust picked a different graph at (num_reqs={num_reqs}, "
                            f"num_tokens={num_tokens}, uniform={uniform})"
                        )
    log.info("vtl: rust runner dispatch parity OK over %d tuples", checked)
    return None


def export(runner, b) -> None:
    """Arm the Rust runner against this boot's graphs. Never raises.

    Called from ``nstep_decode``'s ``capture_model`` wrapper AFTER the burst graphs are
    captured, so ``b.accum`` exists and ``mgr.graphs`` is complete.
    """
    want = mode()
    if want == "off":
        STATE.refuse("VTL_RUST_RUNNER=0")
        return

    try:
        import vtl_sched
    except Exception as exc:
        STATE.refuse(f"vtl_sched not importable: {exc}")
        return
    if not getattr(vtl_sched, "HAS_CUDA_RUNNER", False):
        STATE.refuse("this vtl_sched wheel was built without the `cuda` feature")
        return

    from vtl.patches.shm_ipc import rust_kv

    kv = rust_kv()
    if kv is None:
        STATE.refuse("the Rust KV manager is not the authority this boot")
        return

    if not vtl_sched.Runner.driver_available():
        STATE.refuse(f"libcuda unavailable: {vtl_sched.Runner.driver_error()}")
        return

    # Hazard 7: `run_fullgraph` calls `get_offloader().sync_prev_onload()` before every
    # replay. That is a no-op for the served config (NoopOffloader), and the runner does not
    # reproduce it -- so prove the premise rather than assume it.
    try:
        from vllm.v1.offloading.offloader import NoopOffloader, get_offloader

        if type(get_offloader()) is not NoopOffloader:
            STATE.refuse(
                f"offloader is {type(get_offloader()).__name__}, not NoopOffloader; "
                "run_fullgraph's sync_prev_onload is not a no-op here"
            )
            return
    except ImportError:
        log.info("vtl: no offloader module in this build; nothing to sync")

    mgr = getattr(runner, "cudagraph_manager", None)
    if mgr is None or not getattr(mgr, "graphs", None):
        STATE.refuse("no captured cudagraphs")
        return
    if b.accum is None or b.rows is None:
        STATE.refuse("the burst accumulator was never allocated")
        return
    if not b.unroll_graphs:
        # M1's ordering enforcement. Without the unroll rung the only handles on offer are
        # stock's forward-only graphs, which never write `accum` -- a runner armed against
        # those would launch a step and then commit whatever tokens were left over from the
        # previous one. Never arm.
        STATE.refuse("nstep captured no unroll graphs; there is nothing launchable")
        return

    try:
        rust_runner = vtl_sched.Runner(kv)
        _, execs = _build_table(rust_runner, mgr, b)
        if not execs:
            STATE.refuse("no unroll graph exposed a usable cudaGraphExec_t")
            return

        why = _verify_dispatch(rust_runner, mgr, int(runner.max_num_reqs), execs)
        if why is not None:
            STATE.refuse(f"dispatch parity failed: {why}")
            return

        import torch

        stream = int(torch.cuda.current_stream().cuda_stream)
        main = int(getattr(runner, "main_stream", torch.cuda.current_stream()).cuda_stream)
        if stream != main:
            STATE.refuse(
                "runner.main_stream is not the current stream; the export would bind a "
                "stream the graphs were not captured against"
            )
            return

        rows, cols = int(b.accum.shape[0]), int(b.accum.shape[1])
        rust_runner.arm(stream, int(b.accum.data_ptr()), rows, cols)

        # Handle stability (RUST-RUNNER.md 0): torch destroys the exec on a later
        # instantiate(), and vLLM never calls one -- so this should be constant. The runner
        # CACHES it, which makes a moving handle a use-after-free, not a wrong number. Read
        # every unroll handle a SECOND time and require the same number.
        for rows, entry in b.unroll_graphs.items():
            if execs.get(entry[1]) != int(entry[0].raw_cuda_graph_exec()):
                STATE.refuse(f"the unroll cudaGraphExec_t at num_reqs={rows} moved")
                return
    except Exception as exc:
        STATE.refuse(f"export raised: {exc}")
        return

    STATE.runner = rust_runner
    STATE.live = True
    STATE.why = ""
    # Read ONCE, here: the mode picks which of two code paths every later step takes, and a
    # per-step getenv that could flip mid-boot would mean a launch in one mode and a commit
    # in the other.
    STATE.update = commit_mode() == "update"
    STATE.pending = []
    # The armed SIZES, not just the count: 17% of the scored trace's tokens decode at a
    # batch size outside the captured set, so which sizes armed is the number an H200 boot
    # log has to be auditable for.
    log.info(
        "vtl: rust runner armed (%d rows, launchable sizes %s, continuation sizes %s, "
        "mode=%s, commit=%s, max_steps=%d)",
        rust_runner.num_graphs(), sorted(b.unroll_graphs),
        sorted(getattr(b, "continue_graphs", {})), want,
        "update" if STATE.update else "sample",
        1 if STATE.update else max_steps(),
    )


def _self_check() -> None:
    """No torch, no vLLM, no crate: the env parsing and the refusal latch."""
    saved = os.environ.get("VTL_RUST_RUNNER")
    try:
        for raw, want in (("1", "on"), ("0", "off"), ("check", "off"),
                          ("yes", "on"), ("nonsense", "off")):
            os.environ["VTL_RUST_RUNNER"] = raw
            assert mode() == want, (raw, mode(), want)
        os.environ.pop("VTL_RUST_RUNNER", None)
        assert mode() == "on", "compose ships it enabled; the default must match"
    finally:
        if saved is None:
            os.environ.pop("VTL_RUST_RUNNER", None)
        else:
            os.environ["VTL_RUST_RUNNER"] = saved

    saved_r = os.environ.get("VTL_RUST_RUNNER_REQUIRE")
    try:
        os.environ.pop("VTL_RUST_RUNNER_REQUIRE", None)
        assert require() is False, "the submission must degrade, not fail"
        for raw, want in (("1", True), ("on", True), ("0", False), ("", False)):
            os.environ["VTL_RUST_RUNNER_REQUIRE"] = raw
            assert require() is want, raw
        # ...and then a refusal is a raise, not a latch. The latch still drops first, so
        # a caller that swallows the raise cannot be left with a live-but-broken runner.
        os.environ["VTL_RUST_RUNNER_REQUIRE"] = "1"
        s = _State()
        s.live = True
        logging.disable(logging.CRITICAL)
        try:
            s.refuse("required")
        except RuntimeError as exc:
            assert "VTL_RUST_RUNNER_REQUIRE" in str(exc), exc
        else:
            raise AssertionError("REQUIRE must not degrade quietly")
        finally:
            logging.disable(logging.NOTSET)
        assert not s.live and s.why == "required"
    finally:
        if saved_r is None:
            os.environ.pop("VTL_RUST_RUNNER_REQUIRE", None)
        else:
            os.environ["VTL_RUST_RUNNER_REQUIRE"] = saved_r

    saved_c = os.environ.get("VTL_RUST_RUNNER_COMMIT")
    try:
        for raw, want in (("update", "update"), ("sample", "sample"), ("SAMPLE", "sample"),
                          ("", "update"), ("nonsense", "update")):
            os.environ["VTL_RUST_RUNNER_COMMIT"] = raw
            assert commit_mode() == want, (raw, commit_mode(), want)
        os.environ.pop("VTL_RUST_RUNNER_COMMIT", None)
        assert commit_mode() == "update", "compose ships update; the default must match"
    finally:
        if saved_c is None:
            os.environ.pop("VTL_RUST_RUNNER_COMMIT", None)
        else:
            os.environ["VTL_RUST_RUNNER_COMMIT"] = saved_c

    saved_n = os.environ.get("VTL_RUST_RUNNER_STEPS")
    try:
        for raw, want in (("8", 8), ("1", 1), ("0", 1), ("-5", 1), ("999", 64), ("x", 1)):
            os.environ["VTL_RUST_RUNNER_STEPS"] = raw
            assert max_steps() == want, (raw, max_steps(), want)
        os.environ.pop("VTL_RUST_RUNNER_STEPS", None)
        assert max_steps() == 8
    finally:
        if saved_n is None:
            os.environ.pop("VTL_RUST_RUNNER_STEPS", None)
        else:
            os.environ["VTL_RUST_RUNNER_STEPS"] = saved_n

    # `_graph_execs` is the M1 override: only WHOLE-BURST graphs are launchable, and a
    # graph whose handle cannot be read drops out rather than exporting a 0.
    class Graph:
        def __init__(self, handle):
            self.handle = handle

        def raw_cuda_graph_exec(self):
            if self.handle is None:
                raise RuntimeError("no exec")
            return self.handle

    assert _graph_execs({}, "unroll") == {}
    logging.disable(logging.CRITICAL)  # the "no raw_cuda_graph_exec" warning is expected
    try:
        got = _graph_execs(
            {1: (Graph(0x11), "d1", 4), 2: (Graph(None), "d2", 4), 4: (Graph(0), "d4", 4)},
            "unroll",
        )
    finally:
        logging.disable(logging.NOTSET)
    assert got == {"d1": 0x11}, got

    # The refusal latch is permanent and must survive a runner that throws on disarm.
    class Boom:
        def disarm(self):
            raise RuntimeError("boom")

    s = _State()
    s.runner, s.live = Boom(), True
    stash = _Stash(("a",), (0,), (object(),), 4, 1, 32768, True, True, 7)
    s.step = stash
    s.done = _Done(stash, [(4, 0, -1)], [], "budget", 1)
    s.pending = [(0, stash)]
    logging.disable(logging.CRITICAL)  # the traceback below is expected; don't print it
    try:
        s.refuse("testing")
    finally:
        logging.disable(logging.NOTSET)
    assert not s.live and s.why == "testing"
    assert s.step is None and s.done is None, "a refusal must strand no handshake"
    assert s.pending == [], "a dropped launch is committed by decide(); a stranded one is not"

    # The stash is POPPED, launch or no launch: the next step's batch can have the exact
    # same request-id key, so a leftover would match a step it was never written for.
    s = _State()
    s.step = stash
    assert s.take_step() is stash
    assert s.take_step() is None and s.step is None

    # `inflight` is the ordering interlock: a launch is only legal when this step is the
    # only one not yet applied. The counter itself is scheduler-side; what has to hold
    # here is that it starts clean and clamps at zero (an unpaired update must not make
    # the interlock read as open on the step after it).
    assert _State().inflight == 0
    s.inflight = max(0, s.inflight - 1)
    assert s.inflight == 0

    # Update mode's ring accounting: the FIFO is what owns the slots, so two outstanding
    # launches must leave nothing free (a third would overwrite a buffer whose D2H has not
    # been read yet), and popping the head must hand that slot straight back.
    s = _State()
    assert s.update and s.pending == [], "update is the default mode"
    assert s.free_slot() == 0
    s.pending.append((0, stash))
    assert s.free_slot() == 1
    s.pending.append((1, stash))
    assert s.free_slot() is None, "a third launch has nowhere to land; decline it"
    assert s.pending.pop(0) == (0, stash), "updates arrive in step order, so FIFO"
    assert s.free_slot() == 0
    # ...and the slots are reused in whatever order they free, not round-robin: the head
    # popped is the only one that matters.
    s.pending = [(0, stash)]
    assert s.free_slot() == 1

    print("rust_runner self-check ok")


if __name__ == "__main__":
    _self_check()
