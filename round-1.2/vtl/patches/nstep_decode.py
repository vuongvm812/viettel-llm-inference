"""N-step decode burst: run N decode iterations per engine step, host work paid once.

WHY. At batch 1-8 on a MIG slice the decode step is ~1 ms of GPU and ~2 ms of host Python
(schedule -> execute_model -> sample_tokens -> update_from_output -> output IPC). Every one
of those 2 ms is per ENGINE STEP, not per token. Emitting N tokens per engine step divides
the host term by N; the GPU term is unchanged. At N=4 that is TPOT 3 ms -> ~1.5 ms, which
at the current operating point is worth more than every other item in this batch combined.

ARCHITECTURE (decided; the alternatives and why they lose are in
docs/round-1.2-nstep-r8-microopt-plan.md):

  * A REPLAY LOOP of one burst body, not an N-unrolled graph. N stays a plain loop bound,
    so the sweep is an env var and not a re-capture.
  * The SCHEDULER commits the burst (``rust_sched.py``). One bookkeeping authority: it
    advances ``num_computed_tokens`` / ``num_output_placeholders`` by N-1 at schedule time
    and reconciles the surplus in ``update_from_output`` if fewer than N tokens come back.
    This module never touches request state.
  * NOTHING SPEC-DECODE-SHAPED is set. ``num_scheduled_tokens`` stays 1,
    ``scheduled_spec_decode_tokens`` stays empty, ``num_speculative_tokens`` stays 0 --
    the Rust schedule loop refuses a spec config outright (rust_sched.py:229), the UFO gate
    refuses spec tokens, and a spec-decode submission was flagged as cheating. The burst
    reuses the *mechanisms* (multi-token sampler output, the ``post_update`` N-loop) with
    none of the spec configuration.

THE HANDSHAKE, and why it is this shape. The scheduler must never commit a burst the
runner will not execute: under async scheduling the next step is already scheduled off the
optimistic ``num_computed``, so a silently-short burst writes the following step's tokens at
the wrong positions. So the runner publishes, at the end of every step, "a batch with THIS
exact request-id key just executed as a clean FULL-cudagraph pure-decode step"
(``BURST.ready`` / ``BURST.key``). The scheduler commits only when the step it is building
has the same key. Same shape in, same cudagraph descriptor out -- so a committed burst is
executable by construction. Both patches live in the EngineCore process (UniProcExecutor
runs the worker in-process); under any out-of-process executor the flag simply never turns
on and no burst is ever committed. Fail closed.

ALIGN GATE (the escape hatch that makes the burst structurally safe). The scheduler only
commits when ``num_computed % block_size + N <= block_size``. That single inequality means
the whole burst lives inside ONE KV block and ONE mamba state column, so across the N
iterations:

  * no new KV block is allocated mid-burst -> the block tables the graph baked stay valid;
  * ``(seq_len - 1) // block_size`` is constant -> the mamba ``state_indices_tensor_d``
    written by step 1 stays correct and the align pre-copy (``preprocess_state``) is a
    provable no-op, so neither is re-run;
  * ``compute_slot_mappings`` only walks within the block it already has.

At block_size 16 and N=4 that covers 13/16 of decode steps.

WHAT EACH ITERATION ACTUALLY DOES (``_burst_body``): one Triton kernel to feed the sampled
token back (input_ids, positions+1, seq_lens+1, and the token into the output accumulator),
``compute_slot_mappings``, the FA3 AOT schedule (``decode_fastpath._fa_write`` -- shared, not
copied), the FULL cudagraph replay, then ``compute_logits`` + ``argmax``. Zero device syncs,
zero host reads of device memory.

MODES (``VTL_NSTEP_MODE``), a ladder that degrades on its own:
  ``graph``  -- the whole body above captured ONCE per decode size into a patch-owned CUDA
                graph and replayed. Capture happens after stock ``capture_model()``, into
                the same pool and against the attention state stock captured, with
                ``max_seqlen_k`` baked at ``max_model_len``. Any failure (OOM, an
                uncapturable host sync inside the FA3 scheduler call) logs once and drops to
                eager for the rest of the boot.
  ``eager``   -- the same body, host-launched. Ships most of the win.
  ``off``     -- nothing installed.

REVERTS: ``VTL_ENABLE_NSTEP_DECODE=0`` (runner does nothing), ``VTL_NSTEP=0`` (the scheduler
never commits, so this module never fires), ``VTL_NSTEP_MODE=eager`` (no capture),
``VTL_NSTEP_N=1`` (a burst of 1 is a normal step).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.nstep")

# Decode cudagraph sizes worth a burst graph, and therefore the largest batch a burst is
# ever committed for. Beyond 8 concurrent generations the trace does not go (the scored
# replay peaks at 8), and each extra size is another captured graph at boot.
BURST_SIZES = (1, 2, 4, 8)
MAX_BURST_REQS = BURST_SIZES[-1]


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------------------
# The scheduler-facing handshake. Module state, read by rust_sched.py in the same process.
# ---------------------------------------------------------------------------------------


class _Burst:
    """Everything the burst owns. One runner per process, so a module global is the whole
    lifetime story."""

    __slots__ = (
        "armed", "ready", "key", "n", "mode", "runner", "graphs",
        "accum", "tok", "step", "num_sampled", "num_rejected", "desc", "pending",
        "bursts", "iters", "advance", "bump", "idx_map", "max_seq_len",
    )

    def __init__(self) -> None:
        self.advance = None      # the jitted feedback kernel
        self.bump = None         # the jitted step-counter increment
        self.idx_map = None      # [max_num_reqs] i32, persistent copy of ib.idx_mapping
        self.max_seq_len = 0     # host scalar the FA3 AOT schedule needs, per iteration
        self.armed = False       # apply() ran and the patch is live
        self.ready = False       # the LAST executed step could have carried a burst
        self.key: tuple = ()     # ...and this was its request-id tuple
        self.n = 1               # VTL_NSTEP_N
        self.mode = "off"
        self.runner = None
        self.graphs: dict = {}
        self.accum = None        # [max_num_reqs, N] int64 -- the step's sampled tokens
        self.tok = None          # [max_num_reqs] int64 -- the current iteration's argmax
        self.step = None         # [1] int32 device counter (accum column)
        self.num_sampled = None  # [max_num_reqs] int32, filled with N
        self.num_rejected = None # [max_num_reqs] int32, filled with -(N-1)
        self.desc = None         # the FULL descriptor the last step replayed
        self.pending = 0         # burst factor committed for the step being executed
        self.bursts = 0
        self.iters = 0

    def disable(self, why: str) -> None:
        """Permanent, process-lifetime fallback to one token per step."""
        if self.ready or self.armed:
            log.error("vtl: nstep burst disabled for this boot -- %s", why)
        self.armed = False
        self.ready = False
        self.pending = 0


BURST = _Burst()


def commit_key(num_scheduled_tokens) -> tuple:
    """The key both sides agree on: the scheduled request ids, in dict order.

    Identical to ``decode_fastpath.decode_key``'s tuple for a 1-token-per-request step,
    which is exactly the case a burst is committed for -- so a key match also means the
    runner's decode fast path will be eligible.
    """
    return tuple(num_scheduled_tokens)


def burst_factor(num_scheduled_tokens) -> int:
    """The scheduler's question: how many tokens may this step emit? 1 = a normal step.

    Pure and side-effect free so ``rust_sched.py`` can call it once per step and so the
    self-check can exercise it without torch.
    """
    b = BURST
    if not b.armed or not b.ready or b.n < 2:
        return 1
    if b.key != commit_key(num_scheduled_tokens):
        return 1
    return b.n


# ---------------------------------------------------------------------------------------
# The Triton kernels. Modelled on speculator.py's `_update_draft_inputs_kernel`.
# ---------------------------------------------------------------------------------------


def _build_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _advance(
        tok_ptr,          # [max_num_reqs] i64  -- this iteration's sampled tokens
        accum_ptr,        # [max_num_reqs, N]  i64
        accum_stride,
        step_ptr,         # [1] i32 -- which accum column to write
        input_ids_ptr,    # [max_num_batched_tokens] i32
        positions_ptr,    # [max_num_batched_tokens] i64
        seq_lens_ptr,     # [max_num_reqs] i32
        max_model_len,
        FEED: tl.constexpr,
    ):
        req = tl.program_id(0)
        step = tl.load(step_ptr)
        tok = tl.load(tok_ptr + req)
        tl.store(accum_ptr + req * accum_stride + step, tok)
        if FEED:
            # One token per request, so batch index == token index. Clamps mirror the
            # speculator's (speculator.py:706-731): a request that walks into
            # max_model_len must not index past the block table.
            tl.store(input_ids_ptr + req, tok)
            pos = tl.load(positions_ptr + req)
            tl.store(positions_ptr + req, tl.minimum(pos + 1, max_model_len - 1))
            seq = tl.load(seq_lens_ptr + req)
            tl.store(seq_lens_ptr + req, tl.minimum(seq + 1, max_model_len))
        # PADDED ROWS ARE NOT TOUCHED: the grid is num_reqs, so rows >= num_reqs keep the
        # seq_len 0 the stock step left them at. (The speculator re-masks because its
        # kernel writes the whole padded range; ours does not.)

    @triton.jit
    def _bump(step_ptr):
        # A separate 1-program launch, not a `if req == 0` branch in `_advance`: blocks in
        # one grid have no ordering, so an in-kernel increment would race the other
        # programs' `tl.load(step_ptr)`. Two launches per iteration in eager mode, zero
        # extra cost in graph mode (both nodes are baked).
        tl.store(step_ptr, tl.load(step_ptr) + 1)

    return _advance, _bump


# ---------------------------------------------------------------------------------------
# One burst iteration.
# ---------------------------------------------------------------------------------------


def _burst_body(runner, b: _Burst, rows: int, pad: int, forward):
    """Feed back one token and produce the next. Leaves the argmax in ``b.tok[:rows]``.

    EVERY tensor it touches is PERSISTENT, which is the whole reason graph mode is possible:

      * ``input_buffers.{input_ids, positions, seq_lens, query_start_loc}`` -- persistent
        by construction (the FULL cudagraph's own inputs);
      * ``block_tables.slot_mappings`` -- ``compute_slot_mappings`` writes IN PLACE and
        returns a view (block_table.py:169-185), so no allocation;
      * the FA builder's ``scheduler_metadata`` -- likewise written in place;
      * ``b.idx_map`` / ``b.tok`` / ``b.accum`` / ``b.step`` -- ours.

    ``InputBatch.idx_mapping`` and ``InputBatch.logits_indices`` are NOT persistent
    (``async_copy_to_gpu`` and ``torch.empty`` allocate a fresh one per slow step), so this
    body must not use them: ``b.idx_map`` is a per-step copy of the former, and the latter
    is provably ``arange(num_reqs)`` on a 1-token-per-request decode step -- request i's
    only token is token i -- so a plain ``hidden[:rows]`` slice replaces the gather.

    TWO WIDTHS, deliberately. ``rows`` is the REAL request count -- the kernel grid and the
    logits width -- so the feedback never touches a padded row (the speculator has to
    re-mask precisely because its kernel does). ``pad`` is
    ``num_reqs_after_padding``, which is what the attention metadata must be sized for: the
    captured graph's FA3 kernel reads a ``scheduler_metadata`` built for the PADDED batch,
    exactly as ``decode_fastpath._fast_model_attn`` builds it. Getting these two the same
    way round is the difference between a correct schedule and a silently wrong one.

    In graph mode ``rows == pad`` by construction (a burst graph is only used when the real
    batch is exactly the captured size), so the capture and the replay see one width.
    """
    import torch

    from vtl.patches import decode_fastpath as dfp

    ibuf = runner.input_buffers
    b.advance[(rows,)](
        b.tok, b.accum, b.accum.stride(0), b.step,
        ibuf.input_ids, ibuf.positions, ibuf.seq_lens,
        runner.max_model_len,
        FEED=True,
    )
    b.bump[(1,)](b.step)

    query_start_loc = ibuf.query_start_loc[: pad + 1]
    # Same argument widths as `decode_fastpath._fast_attn`: a REAL-length idx_mapping, a
    # padded query_start_loc / positions, and the padded token count -- which is what makes
    # the kernel's last program refill the pad region with PAD_SLOT_ID.
    runner.block_tables.compute_slot_mappings(
        b.idx_map[:rows], query_start_loc, ibuf.positions[:pad],
        num_tokens_padded=pad,
    )
    fa, _mamba = dfp._C.builders
    seq_lens = ibuf.seq_lens[:pad]
    for builder, _gid in fa:
        dfp._fa_write(builder, pad, seq_lens, query_start_loc, b.max_seq_len)
    # The mamba `state_indices_tensor_d` and the align pre-copy are DELIBERATELY not
    # re-run: the align gate makes `(seq_len - 1) // block_size` constant for the whole
    # burst, so step 1's values are still exact. See the module docstring.

    hidden = forward()
    logits = runner.model.compute_logits(hidden[:rows])
    torch.argmax(logits, dim=-1, out=b.tok[:rows])


# ---------------------------------------------------------------------------------------
# Graph mode: capture the body once per decode size.
# ---------------------------------------------------------------------------------------


def _graph_matches_eager(runner, b: _Burst, graph, forward, rows: int) -> bool:
    """Boot-time shadow: one graph replay vs one eager body, from the same state.

    The graph bakes every pointer at capture. If any input the body reads turns out NOT to
    be a persistent buffer, the replay silently computes from stale memory and produces
    plausible-but-wrong tokens -- the worst failure mode in this batch, because nothing
    crashes and the output only drifts. Running both once at boot on the dummy batch and
    comparing the sampled tokens is the cheapest thing that catches it. A mismatch drops
    the whole boot to eager.

    NOT a proof of runtime correctness (the dummy batch has no padded rows and no stale
    block tables); it is a smoke test for the pointer-staleness class specifically.
    """
    import torch

    ibuf = runner.input_buffers
    snap = (
        ibuf.input_ids[:rows].clone(),
        ibuf.positions[:rows].clone(),
        ibuf.seq_lens[:rows].clone(),
        b.tok[:rows].clone(),
        b.step.clone(),
    )

    def restore():
        ibuf.input_ids[:rows].copy_(snap[0])
        ibuf.positions[:rows].copy_(snap[1])
        ibuf.seq_lens[:rows].copy_(snap[2])
        b.tok[:rows].copy_(snap[3])
        b.step.copy_(snap[4])

    restore()
    graph.replay()
    torch.cuda.synchronize()
    from_graph = b.tok[:rows].clone()

    restore()
    _burst_body(runner, b, rows, rows, forward)
    torch.cuda.synchronize()
    from_eager = b.tok[:rows].clone()

    if not torch.equal(from_graph, from_eager):
        log.error(
            "vtl: nstep burst graph at num_reqs=%d DIVERGED from the eager body "
            "(graph=%s eager=%s); eager mode for this boot",
            rows, from_graph.tolist(), from_eager.tolist(),
        )
        return False
    restore()
    return True


def _capture_burst_graphs(runner, b: _Burst) -> None:
    """One CUDA graph per FULL decode size, holding the whole ``_burst_body``.

    Runs after stock ``capture_model()``, into the SAME memory pool, replaying against the
    attention state stock captured for that descriptor -- so the forward inside our graph
    is bit-for-bit the forward inside stock's, just with the metadata kernels and the
    argmax bolted on in front and behind.

    ``max_seqlen_k`` is baked at ``max_model_len``: the FA3 AOT schedule is a function of
    it, and a graph cannot take a host scalar. That is the shipped pattern
    (mamba_hybrid.py:247-251 does the same for capture, and the speculator relies on it).

    ANY failure -- OOM, or an uncapturable host synchronisation inside
    ``get_scheduler_metadata`` (probe B in bench/test_nstep_capture_probe.py) -- drops the
    whole boot to eager mode. Never crashes.
    """
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vtl.patches import decode_fastpath as dfp

    mgr = runner.cudagraph_manager
    states = getattr(mgr, "_vtl_attn_states", None)
    if not states:
        log.warning("vtl: nstep graph mode off -- stock capture published no attention state")
        return
    if not dfp._C.builders:
        classified = dfp._classify(runner)
        if not classified:
            log.warning("vtl: nstep graph mode off -- attention builders unclassified")
            return
        dfp._C.builders = classified

    before = torch.cuda.memory_allocated()
    captured = 0
    for desc, pair in states.items():
        if desc.cg_mode != CUDAGraphMode.FULL or desc.uniform_token_count != 1:
            continue
        if desc.num_reqs not in BURST_SIZES or desc.num_active_loras:
            continue
        if desc.num_tokens != desc.num_reqs:
            # One token per request is what a decode graph is; anything else is not a
            # shape a burst can ride.
            continue
        rows = desc.num_reqs
        attn_metadata = pair.captured.attn_metadata
        slot_mappings = pair.captured.slot_mappings
        # make_dummy primes seq_lens / query_start_loc / positions in the persistent
        # buffers, which is what the body below reads. Its `idx_mapping` is arange, which
        # is also what `b.idx_map` gets at capture time -- the real values are copied in
        # per step before the replay.
        InputBatch.make_dummy(rows, rows, runner.input_buffers)
        b.idx_map[:rows].copy_(torch.arange(rows, dtype=torch.int32, device=runner.device))
        b.max_seq_len = runner.max_model_len
        model_inputs = {
            "input_ids": runner.input_buffers.input_ids[:rows],
            "positions": runner.input_buffers.positions[:rows],
            **runner.model_state.prepare_dummy_inputs(rows, rows),
        }

        def forward(_mi=model_inputs, _am=attn_metadata, _sm=slot_mappings, _nt=rows):
            with set_forward_context(
                _am,
                runner.vllm_config,
                num_tokens=_nt,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=_sm,
                is_padding=runner.input_buffers.is_padding[:_nt],
            ):
                return runner.model(**_mi)

        try:
            # Warm up outside the capture: the first call allocates workspaces and JITs
            # the Triton kernels, neither of which is capturable.
            b.step.zero_()
            _burst_body(runner, b, rows, rows, forward)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            b.step.zero_()
            with torch.cuda.graph(graph, mgr.pool):
                _burst_body(runner, b, rows, rows, forward)
            if not _graph_matches_eager(runner, b, graph, forward, rows):
                b.graphs.clear()
                b.mode = "eager"
                return
            b.graphs[rows] = (graph, desc)
            captured += 1
        except Exception:
            log.exception(
                "vtl: nstep burst capture failed at num_reqs=%d; eager mode for this boot",
                rows,
            )
            b.graphs.clear()
            b.mode = "eager"
            return
    if not captured:
        log.warning("vtl: nstep graph mode off -- no FULL decode descriptor matched %s",
                    BURST_SIZES)
        b.mode = "eager"
        return
    log.info(
        "vtl: nstep captured %d burst graph(s) for sizes %s (+%.1f MiB)",
        captured, sorted(b.graphs), (torch.cuda.memory_allocated() - before) / (1 << 20),
    )


# ---------------------------------------------------------------------------------------
# sample_tokens: the burst itself.
# ---------------------------------------------------------------------------------------


def _run_burst(runner, b: _Burst, n: int, ib, hidden):
    """Sample N tokens. Returns the ``SamplerOutput`` the stock tail expects."""
    import torch
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    num_reqs = ib.num_reqs
    # The one host scalar FA3 needs. `seq_lens_cpu_upper_bound` is this step's post-step-1
    # value, so iteration j wants base + j. Read once, incremented on the host -- no sync.
    base_max_seq_len = int(ib.seq_lens_cpu_upper_bound[:num_reqs].max())
    b.idx_map[:num_reqs].copy_(ib.idx_mapping, non_blocking=True)

    b.step.zero_()
    # Token 1 comes from the hidden states the stock forward already produced. On a
    # 1-token-per-request decode step `logits_indices` is arange(num_reqs), so the slice
    # is the gather -- and unlike `logits_indices` it is not a fresh allocation.
    torch.argmax(
        runner.model.compute_logits(hidden[:num_reqs]), dim=-1, out=b.tok[:num_reqs]
    )

    # The burst graph is used ONLY when the real batch is exactly the captured size. With
    # padded rows the graph's baked grid would advance positions/seq_lens for rows that do
    # not exist, so those steps take the eager loop -- which recomputes the grid per
    # iteration and is always correct. `==`, not `is`: BatchExecutionDescriptor is a
    # frozen dataclass, so equal descriptors need not be the same object.
    entry = b.graphs.get(num_reqs) if b.mode == "graph" else None
    if (
        entry is not None
        and entry[1] == b.desc
        and ib.num_reqs_after_padding == num_reqs
        and ib.num_tokens_after_padding == num_reqs
    ):
        graph = entry[0]
        for _ in range(n - 1):
            graph.replay()
    else:
        forward = lambda: runner.cudagraph_manager.run_fullgraph(b.desc)  # noqa: E731
        for j in range(1, n):
            b.max_seq_len = base_max_seq_len + j
            _burst_body(runner, b, num_reqs, ib.num_reqs_after_padding, forward)
    # The last token was sampled after the final feedback, so nothing wrote it into the
    # accumulator; the column index is n-1 by construction.
    b.accum[:num_reqs, n - 1].copy_(b.tok[:num_reqs])

    b.bursts += 1
    b.iters += n
    return SamplerOutput(
        # CLONED, not a view of the persistent accumulator: `AsyncOutput` copies this to
        # the host on a SECOND stream and the copy is still in flight when the next step
        # starts writing. Stock is safe because its argmax output is a fresh allocation
        # every step; this is the same thing, at 8x4x8 bytes.
        sampled_token_ids=b.accum[:num_reqs, :n].clone(),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=b.num_sampled[:num_reqs],
        # ponytail: post_update computes `computed_delta = query_len - num_rejected`
        # (input_batch.py:509) and query_len is 1 on a decode step, so a num_rejected of
        # -(N-1) is how a plain N is expressed with no kernel change. It is used for
        # nothing else on this path -- there is no speculator and no PP broadcast.
        num_rejected=b.num_rejected[:num_reqs],
    )


# ---------------------------------------------------------------------------------------
# Install.
# ---------------------------------------------------------------------------------------


def _alloc(runner, b: _Burst) -> None:
    import torch

    dev = runner.device
    m = runner.max_num_reqs
    b.accum = torch.zeros((m, b.n), dtype=torch.int64, device=dev)
    b.tok = torch.zeros(m, dtype=torch.int64, device=dev)
    b.step = torch.zeros(1, dtype=torch.int32, device=dev)
    b.idx_map = torch.zeros(m, dtype=torch.int32, device=dev)
    b.num_sampled = torch.full((m,), b.n, dtype=torch.int32, device=dev)
    b.num_rejected = torch.full((m,), -(b.n - 1), dtype=torch.int32, device=dev)


def _patch_runner(b: _Burst) -> None:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, ModelCudaGraphManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vtl.patches import decode_fastpath as dfp

    def _install(cls, name, wrapper):
        if already_patched(cls, name, patch="nstep_decode"):
            return
        original = getattr(cls, name)
        bound = wrapper(original)
        mark_patched(bound, original, patch="nstep_decode")
        setattr(cls, name, bound)

    # --- remember which FULL descriptor the step replayed ---------------------------
    def wrap_run_fullgraph(original):
        def run_fullgraph(self, desc):
            b.desc = desc
            return original(self, desc)

        return run_fullgraph

    # --- keep the attention state stock captured (graph mode needs it) --------------
    def wrap_capture(original):
        def capture(self, *args, **kwargs):
            states = original(self, *args, **kwargs)
            self._vtl_attn_states = states
            return states

        return capture

    # --- capture the burst graphs once the stock ones exist -------------------------
    def wrap_capture_model(original):
        def capture_model(self):
            size = original(self)
            b.runner = self
            try:
                _alloc(self, b)
                if b.mode == "graph":
                    _capture_burst_graphs(self, b)
                b.armed = True
                log.info("vtl: nstep burst armed (N=%d, mode=%s)", b.n, b.mode)
            except Exception:
                log.exception("vtl: nstep burst setup failed; one token per step")
                b.armed = False
            return size

        return capture_model

    # --- read the commitment off the SchedulerOutput --------------------------------
    def wrap_execute_model(original):
        def execute_model(self, scheduler_output, *args, **kwargs):
            b.pending = getattr(scheduler_output, "vtl_burst_n", 0) or 0
            return original(self, scheduler_output, *args, **kwargs)

        return execute_model

    # --- the burst, plus the readiness signal for the NEXT step ---------------------
    def wrap_sample_tokens(original):
        def sample_tokens(self, grammar_output):
            n = b.pending
            b.pending = 0
            state = self.execute_model_state
            # `grammar_output is not None` cannot happen on a committed burst (the gate
            # refuses structured output), but the burst's bare argmax would ignore the
            # bitmask if it ever did -- so refuse here too rather than sample past a
            # grammar.
            if n < 2 or state is None or grammar_output is not None:
                out = original(self, grammar_output)
                _publish_ready(self, state)
                return out
            try:
                out = _sample_burst(self, b, n, state, grammar_output, original)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                # A committed burst that produced fewer than N tokens IS reconciled by
                # the scheduler (rust_sched's `_burst_uncommit`), so this is recoverable
                # -- but it means the burst body is broken, so never try again.
                log.exception("vtl: nstep burst failed mid-step; permanently disabled")
                b.disable("burst body raised")
                self.execute_model_state = state
                out = original(self, grammar_output)
            _publish_ready(self, state)
            return out

        return sample_tokens

    def _publish_ready(runner, state) -> None:
        """Tell the scheduler whether the NEXT step with this exact shape may burst."""
        if not b.armed:
            b.ready = False
            return
        ib = None if state is None else state.input_batch
        try:
            b.ready = bool(
                ib is not None
                and b.desc is not None
                and b.desc.cg_mode == CUDAGraphMode.FULL
                and b.desc.num_reqs is not None
                and ib.num_reqs_after_padding == b.desc.num_reqs
                and ib.num_reqs <= MAX_BURST_REQS
                and ib.num_reqs == len(ib.req_ids)
                and runner.speculator is None
                and ib.num_draft_tokens == 0
                and not ib.has_structured_output_reqs
                and not ib.is_prefilling_np.any()
                and int(ib.num_scheduled_tokens.max()) == 1
                and dfp._C.builders
            )
            b.key = tuple(ib.req_ids) if b.ready else ()
        except Exception:
            log.exception("vtl: nstep readiness probe failed; no more bursts")
            b.disable("readiness probe raised")

    def _sample_burst(runner, b, n, state, grammar_output, original):
        """``sample_tokens``' last-rank tail, with the burst in place of one argmax.

        Reproduced rather than wrapped because the burst has to sit between
        ``self.sample()`` and ``AsyncOutput``. Every branch this drops is one the
        readiness probe and the scheduler's commit gate have already excluded: PP
        (``is_last_pp_rank``), pooling, the speculator, draft tokens, structured output,
        and prompt logprobs (which need a prefilling row).
        """
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.worker.gpu.async_utils import AsyncOutput

        ib = state.input_batch
        runner.execute_model_state = None
        sampler_output = _run_burst(runner, b, n, ib, state.hidden_states)

        model_runner_output = ModelRunnerOutput(
            req_ids=ib.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(ib.req_ids)},
            sampled_token_ids=None,  # type: ignore[arg-type]
            prompt_logprobs_dict={},  # type: ignore[arg-type]
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=sampler_output.num_sampled,
            main_stream=runner.main_stream,
            copy_stream=runner.output_copy_stream,
        )
        runner.postprocess_sampled(
            ib.idx_mapping,
            sampler_output.sampled_token_ids,
            sampler_output.num_sampled,
            sampler_output.num_rejected,
            ib.query_start_loc,
        )
        model_runner_output.kv_connector_output = runner.kv_connector.post_forward(
            state.finished_req_ids
        )
        return async_output

    _install(CudaGraphManager, "run_fullgraph", wrap_run_fullgraph)
    _install(ModelCudaGraphManager, "capture", wrap_capture)
    _install(GPUModelRunner, "capture_model", wrap_capture_model)
    _install(GPUModelRunner, "execute_model", wrap_execute_model)
    _install(GPUModelRunner, "sample_tokens", wrap_sample_tokens)


@register_patch("nstep_decode", default=True)
def apply() -> None:
    b = BURST
    b.mode = os.environ.get("VTL_NSTEP_MODE", "graph").strip().lower() or "graph"
    b.n = max(1, min(_int("VTL_NSTEP_N", 4), 16))
    if b.mode not in ("graph", "eager"):
        log.info("vtl: nstep decode off (VTL_NSTEP_MODE=%s)", b.mode)
        return
    if b.n < 2:
        log.info("vtl: nstep decode off (VTL_NSTEP_N=%d)", b.n)
        return
    b.advance, b.bump = _build_kernels()
    _patch_runner(b)
    log.info("vtl: nstep decode installed (N=%d, mode=%s); the scheduler arms it",
             b.n, b.mode)


# ---------------------------------------------------------------------------------------


def _self_check() -> None:
    """No torch, no vLLM: the handshake and the eligibility matrix are what can be wrong."""
    b = BURST
    saved = (b.armed, b.ready, b.key, b.n)
    try:
        b.n = 4
        b.armed = True
        b.ready = True
        b.key = ("a", "b")

        nst = {"a": 1, "b": 1}
        assert commit_key(nst) == ("a", "b")
        assert burst_factor(nst) == 4

        # Every disqualifier, one at a time.
        assert burst_factor({"b": 1, "a": 1}) == 1, "order is load-bearing (it is idx_mapping)"
        assert burst_factor({"a": 1}) == 1, "a request left the batch"
        assert burst_factor({"a": 1, "b": 1, "c": 1}) == 1, "a request joined the batch"
        b.ready = False
        assert burst_factor(nst) == 1, "the runner has not published a clean step"
        b.ready = True
        b.armed = False
        assert burst_factor(nst) == 1, "the patch is not installed"
        b.armed = True
        b.n = 1
        assert burst_factor(nst) == 1, "N=1 is a normal step"
        b.n = 4
        assert burst_factor(nst) == 4

        # disable() is permanent and closes both gates.
        log.disabled = True
        b.disable("test")
        log.disabled = False
        assert burst_factor(nst) == 1
        assert not b.armed and not b.ready
    finally:
        b.armed, b.ready, b.key, b.n = saved

    # num_rejected arithmetic: post_update's `computed_delta = query_len - num_rejected`
    # must come out as exactly N for a decode step (query_len == 1).
    for n in (2, 4, 8):
        num_rejected = -(n - 1)
        assert 1 - num_rejected == n, n
    assert 1 - 0 == 1, "a non-burst step must still advance by exactly one"

    # The accumulator column the host fills after the loop.
    for n in (2, 4, 8):
        # n-1 iterations run the body; each writes column step, starting at 0.
        assert len(range(1, n)) == n - 1
        assert (n - 1) not in range(0, n - 1), "the last column is written on the host"

    assert MAX_BURST_REQS == 8 and BURST_SIZES[0] == 1

    # `__slots__` must be able to hold everything apply() and the burst assign. Omitting a
    # name here is an AttributeError at BOOT, after the model has been loaded -- i.e. the
    # most expensive place to find out.
    for name in ("advance", "bump", "idx_map", "max_seq_len", "accum", "tok", "step",
                 "num_sampled", "num_rejected", "desc", "pending", "graphs", "runner",
                 "armed", "ready", "key", "n", "mode", "bursts", "iters"):
        setattr(BURST, name, getattr(BURST, name))

    assert _int("VTL_NOT_SET_ANYWHERE", 4) == 4
    os.environ["VTL_NOT_SET_ANYWHERE"] = "not-a-number"
    try:
        assert _int("VTL_NOT_SET_ANYWHERE", 4) == 4, "a bad literal falls back, never raises"
        os.environ["VTL_NOT_SET_ANYWHERE"] = "2"
        assert _int("VTL_NOT_SET_ANYWHERE", 4) == 2
    finally:
        del os.environ["VTL_NOT_SET_ANYWHERE"]

    from vtl.registry import PATCH_REGISTRY

    assert {p.name: p.default for p in PATCH_REGISTRY}.get("nstep_decode") is True
    print("nstep_decode self-check ok")


if __name__ == "__main__":
    _self_check()
