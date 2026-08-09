"""N-step decode burst: run N decode iterations per engine step, host work paid once.

WHY. At batch 1-8 on a MIG slice the decode step is ~1 ms of GPU and ~2 ms of host Python
(schedule -> execute_model -> sample_tokens -> update_from_output -> output IPC). Every one
of those 2 ms is per ENGINE STEP, not per token. Emitting N tokens per engine step divides
the host term by N; the GPU term is unchanged. At N=4 that is TPOT 3 ms -> ~1.5 ms, which
at the current operating point is worth more than every other item in this batch combined.

ARCHITECTURE (decided; the alternatives and why they lose are in
docs/round-2-nstep-r8-microopt-plan.md):

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

IN-GRAPH LADDER (each rung is captured independently at boot and each failure demotes to the
rung below, logging once; nothing is ever retried):

  0. ``VTL_RUST_RUNNER=1`` -- the unroll graph is launched by the Rust runner
     (``rust_runner.py``) instead of by ``replay()``, which also moves the D2H and the whole
     step commit into the crate; with ``VTL_RUST_RUNNER_STEPS>1`` it launches the
     CONTINUATION graph (rung 0b, below) for launches 2..k without returning to Python.
     Strictly a rung above the unroll graph: it launches exactly that exec, and every
     refusal falls through to rung 1.
  1. ``VTL_NSTEP_UNROLL=1``  -- ONE graph per burst size holding the prologue, all N-1
     bodies AND the final accumulator store. The whole burst is a single ``replay()``.
     With the runner on, a CONTINUATION graph is captured alongside it: N bodies and the
     accumulator store with NO prologue, so it feeds off ``b.tok`` (the previous launch's
     last token) instead of the hidden states, which is what makes back-to-back launches
     legal.
  2. ``VTL_NSTEP_FOLD_T1=1`` -- the PROLOGUE graph: ``compute_logits`` on the
     ``cudagraph_manager.hidden_states`` persistent buffer plus the ``argmax`` for token 1.
     Replayed once ahead of the N-1 body replays, so no logits/argmax is host-launched.
     This is also the graph ``VTL_SAMPLE_IN_GRAPH`` reuses.
  3. per-iteration body graphs (the shipped behaviour) -- token 1 host-launched.
  4. eager body.

``VTL_SAMPLE_IN_GRAPH=1`` is the generalisation that pays even when a burst never fires: on a
plain greedy 1-token decode step the SCHEDULER commits ``vtl_sample_in_graph`` (same handshake
as the burst, minus the align gate and the queue-empty gate -- N=1 crosses no block boundary
and delays no admission), and ``sample_tokens`` replays the prologue graph and synthesises the
``SamplerOutput`` instead of calling ``self.sample()``. That skips the V2 sampler's numpy
gathers, its ``new_ones`` alloc and its SamplerOutput build. It does NOT re-run the forward:
``execute_model`` already replayed the stock FULL graph and left the hidden states in the
persistent buffer, which the prologue graph reads -- proved per step by a pointer check
against ``execute_model_state.hidden_states``, not assumed.

REVERTS: ``VTL_ENABLE_NSTEP_DECODE=0`` (runner does nothing), ``VTL_NSTEP=0`` (the scheduler
never commits a burst), ``VTL_NSTEP_MODE=eager`` (no capture, so rungs 1-2 and
``VTL_SAMPLE_IN_GRAPH`` are all inert), ``VTL_NSTEP_N=1`` (a burst of 1 is a normal step;
in-graph N=1 sampling still installs), ``VTL_NSTEP_UNROLL=0`` / ``VTL_NSTEP_FOLD_T1=0`` /
``VTL_SAMPLE_IN_GRAPH=0`` (one rung each).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.nstep")

# Decode cudagraph sizes worth a burst graph, and therefore the largest batch a burst is
# ever committed for. Beyond 8 concurrent generations the trace does not go (the scored
# replay peaks at 8), and each extra size is another captured graph at boot. The trace sim
# says 17% of tokens decode at batch 3/5/6; adding those sizes HERE AND to compose's
# cudagraph_capture_sizes would lift graph coverage 82%->99%, but stock FULL graphs at
# never-before-captured sizes carry unvalidated-path risk and wait for a run that can
# afford to fail (see the compose comment).
BURST_SIZES = (1, 2, 4, 8)
MAX_BURST_REQS = BURST_SIZES[-1]


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------------------
# The scheduler-facing handshake. Module state, read by rust_sched.py in the same process.
# ---------------------------------------------------------------------------------------


class _Burst:
    """Everything the burst owns. One runner per process, so a module global is the whole
    lifetime story."""

    __slots__ = (
        "armed", "ready", "key", "n", "mode", "runner", "graphs",
        "accum", "tok", "step", "num_sampled", "num_rejected", "desc", "pending",
        "advance", "bump", "idx_map", "rows",
        "max_seq_len",
        # in-graph ladder (VTL_NSTEP_FOLD_T1 / VTL_NSTEP_UNROLL / VTL_SAMPLE_IN_GRAPH)
        "fold", "unroll", "sample1", "pro_graphs", "unroll_graphs",
        "ones", "zeros", "pending1",
        # the Rust runner's rung (VTL_RUST_RUNNER / _STEPS)
        "continue_graphs", "rust_steps", "nsampled_k", "nrejected_k",
    )

    def __init__(self) -> None:
        self.advance = None      # the jitted feedback kernel
        self.bump = None         # the jitted step-counter increment
        self.idx_map = None      # [max_num_reqs] i32, persistent copy of ib.idx_mapping
        self.rows = None         # [1] i32 device scalar -- the REAL request count
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
        # --- the in-graph ladder ---
        self.fold = False        # capture the token-1 prologue (VTL_NSTEP_FOLD_T1)
        self.unroll = False      # capture prologue + N-1 bodies as ONE graph
        self.sample1 = False     # in-graph greedy sampling for plain N=1 decode steps
        self.pro_graphs: dict = {}     # rows -> (graph, desc): the prologue alone
        self.unroll_graphs: dict = {}  # rows -> (graph, desc, n): the whole burst
        self.ones = None         # [max_num_reqs] int32 filled with 1  (num_sampled, N=1)
        self.zeros = None        # [max_num_reqs] int32 filled with 0  (num_rejected, N=1)
        self.pending1 = False    # in-graph N=1 sampling committed for this step
        # --- the Rust runner's rung ---
        self.continue_graphs: dict = {}  # rows -> (graph, desc, n): launches 2..k
        self.rust_steps = 1      # VTL_RUST_RUNNER_STEPS, or 1 when the runner is off
        self.nsampled_k = None   # [max_num_reqs] int32, a multi-launch step's real total
        self.nrejected_k = None  # [max_num_reqs] int32, -(total - 1)

    def disable(self, why: str) -> None:
        """Permanent, process-lifetime fallback to one token per step."""
        if self.ready or self.armed:
            log.error("vtl: nstep burst disabled for this boot -- %s", why)
        self.armed = False
        self.ready = False
        self.pending = 0
        self.pending1 = False

    def demote(self, rung: str, why: str) -> None:
        """Drop ONE rung of the in-graph ladder for the boot. Never touches the others."""
        log.warning("vtl: nstep %s disabled for this boot -- %s", rung, why)
        if rung == "unroll":
            self.unroll = False
            self.unroll_graphs = {}
            # The Rust runner launches the unroll exec, so it cannot outlive it either.
            self.continue_graphs = {}
        elif rung == "fold":
            self.fold = False
            self.sample1 = False
            self.pro_graphs = {}
            # The unroll graph CONTAINS the prologue, so it cannot outlive it.
            self.unroll = False
            self.unroll_graphs = {}
            self.continue_graphs = {}


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


def sample_in_graph_ready(num_scheduled_tokens) -> bool:
    """Same handshake as ``burst_factor``, for the N=1 in-graph sampling commit.

    Independent of ``b.n``: a burst of 1 is a normal step but its SAMPLING can still ride
    the prologue graph, which is the whole point of this rung (it pays even when the align
    gate keeps refusing bursts). ``pro_graphs`` non-empty is the boot-time proof that the
    capture succeeded -- the runner re-checks the exact size and descriptor per step.
    """
    b = BURST
    if not b.armed or not b.ready or not b.sample1 or not b.pro_graphs:
        return False
    return b.key == commit_key(num_scheduled_tokens)


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
        rows_ptr,         # [1] i32 -- the REAL request count; see below
        max_model_len,
        FEED: tl.constexpr,
    ):
        req = tl.program_id(0)
        step = tl.load(step_ptr)
        tok = tl.load(tok_ptr + req)
        tl.store(accum_ptr + req * accum_stride + step, tok)
        # PADDED ROWS MUST NOT BE FED. Eager mode could express that by sizing the grid at
        # num_reqs, but a captured graph bakes its grid -- so to let ONE graph serve a padded
        # batch (grid = num_reqs_after_padding) the real count has to arrive as a device
        # scalar, exactly like `step_ptr`. Feeding a padded row would advance the seq_len the
        # stock step left at 0, and the FA3 schedule (built for the padded width) would then
        # dispatch real work against a block-table row `gather_block_tables` zero-filled.
        #
        # The `accum` store above stays unguarded: `_burst_output` slices [:num_reqs], so a
        # padded row's column is written and never read.
        if FEED:
            if req < tl.load(rows_ptr):
                # One token per request, so batch index == token index. Clamps mirror the
                # speculator's (speculator.py:706-731): a request that walks into
                # max_model_len must not index past the block table.
                tl.store(input_ids_ptr + req, tok)
                pos = tl.load(positions_ptr + req)
                tl.store(positions_ptr + req, tl.minimum(pos + 1, max_model_len - 1))
                seq = tl.load(seq_lens_ptr + req)
                tl.store(seq_lens_ptr + req, tl.minimum(seq + 1, max_model_len))

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
    # Grid is PAD, not rows: one captured graph must serve every real count that pads to
    # this size, so the kernel masks on `b.rows` instead of on the grid extent.
    b.advance[(pad,)](
        b.tok, b.accum, b.accum.stride(0), b.step,
        ibuf.input_ids, ibuf.positions, ibuf.seq_lens, b.rows,
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


def _prologue(runner, b: _Burst, rows: int) -> None:
    """Token 1: logits + argmax off the PERSISTENT hidden-state buffer. Capturable.

    ``ModelCudaGraphManager.run_fullgraph`` returns ``self.hidden_states[:num_tokens]``
    (cudagraph_utils.py:576-577) -- a slice of a buffer allocated once at capture and
    overwritten in place by every FULL replay. So the pair below reads a fixed address and
    can live inside a graph, which is what deletes the last two host dispatches from
    ``_run_burst``'s graph arm.

    On a 1-token-per-request decode step ``logits_indices`` is ``arange(num_reqs)``, so the
    ``[:rows]`` slice IS stock's gather -- and unlike ``logits_indices`` it allocates nothing.
    """
    import torch

    hidden = runner.cudagraph_manager.hidden_states
    torch.argmax(
        runner.model.compute_logits(hidden[:rows]), dim=-1, out=b.tok[:rows]
    )


def _accum_tail(runner, b: _Burst, rows: int, pad: int | None = None) -> None:
    """Store the LAST iteration's token into the accumulator, in-graph.

    The final token is sampled after the final feedback, so no ``_advance`` wrote it. Running
    the same kernel with ``FEED=False`` writes column ``step`` (which is ``n-1`` after the
    N-1 bumps) and touches nothing else -- so the unroll graph ends the burst completely and
    ``_run_burst`` has no tail work left.
    """
    ibuf = runner.input_buffers
    # FEED=False, so this touches only `accum` -- the padded columns it writes are never
    # read. Grid still goes to `pad` in graph mode so the captured node matches `_burst_body`.
    b.advance[(pad or rows,)](
        b.tok, b.accum, b.accum.stride(0), b.step,
        ibuf.input_ids, ibuf.positions, ibuf.seq_lens, b.rows,
        runner.max_model_len,
        FEED=False,
    )


def _burst_all(runner, b: _Burst, rows: int, pad: int, forward, n: int) -> None:
    """The WHOLE burst: reset, prologue, N-1 bodies, accumulator tail. One capture unit.

    The counter reset is INSIDE so the graph is self-contained -- ``_run_burst`` then has
    literally nothing left to launch on the host but the idx_map copy.
    """
    b.step.zero_()
    _prologue(runner, b, rows)
    for _ in range(n - 1):
        _burst_body(runner, b, rows, pad, forward)
    _accum_tail(runner, b, rows, pad)


def _burst_continue(runner, b: _Burst, rows: int, pad: int, forward, n: int) -> None:
    """The burst AGAIN, continuing from the token the previous launch left in ``b.tok``.

    NO PROLOGUE, which is the entire reason this graph exists. ``_prologue`` argmaxes
    ``cudagraph_manager.hidden_states``, a buffer only the stock FULL replay ahead of the
    burst refreshes -- so relaunching the UNROLL graph back-to-back would re-read stale
    hidden states and re-emit the previous burst's first token. Here the feedback source is
    ``b.tok``, written by the previous launch's own final argmax, so ``n`` bodies produce
    ``n`` fresh tokens with no dependency on anything outside the graph.

    NO ``step.zero_()`` either -- the unroll graph does one and this must not. The column
    counter keeps running, so launch i fills accumulator columns ``[i*n, (i+1)*n)`` and the
    whole multi-launch step is ONE contiguous token block: ``postprocess_sampled`` writes
    every token into ``req_states.all_token_ids``, so a launch whose tokens were not visible
    to it would leave a hole a later recompute would read. The first body rewrites column
    ``i*n - 1`` with the value ``_accum_tail`` already stored there, which is a no-op by
    construction.
    """
    for _ in range(n):
        _burst_body(runner, b, rows, pad, forward)
    _accum_tail(runner, b, rows, pad)


def _hidden_is_persistent(runner, hidden, rows: int) -> bool:
    """Is ``hidden`` the cudagraph manager's persistent buffer, at exactly ``rows`` rows?

    The proof that replaying a prologue graph is equivalent to ``compute_logits(hidden)``:
    the graph baked the buffer's address, so it is only correct while THIS step's hidden
    states live at that address. Checked per step rather than inferred from
    ``desc.cg_mode == FULL``, because a stale ``b.desc`` (a step that never called
    ``run_fullgraph``) is exactly the failure this must not silently accept.
    """
    buf = getattr(runner.cudagraph_manager, "hidden_states", None)
    return (
        buf is not None
        and hidden is not None
        and hidden.shape[0] == rows
        and hidden.data_ptr() == buf.data_ptr()
    )


# ---------------------------------------------------------------------------------------
# Graph mode: capture the body once per decode size.
# ---------------------------------------------------------------------------------------


def _graph_matches_eager(runner, b: _Burst, graph, eager, rows: int, what: str,
                         launches: int = 1) -> bool:
    """Boot-time shadow: ``launches`` graph replays vs the same work host-launched, from
    ONE state.

    The graph bakes every pointer at capture. If any input it reads turns out NOT to be a
    persistent buffer, the replay silently computes from stale memory and produces
    plausible-but-wrong tokens -- the worst failure mode in this batch, because nothing
    crashes and the output only drifts. Running both once at boot on the dummy batch and
    comparing the sampled tokens is the cheapest thing that catches it.

    THE HIDDEN STATES ARE PART OF THE SNAPSHOT, which is what makes the token-1 fold-in
    covered: its whole premise is that ``cudagraph_manager.hidden_states`` is the address the
    graph may bake, so the two arms must provably start from the same bytes there. ``accum``
    is compared too -- the unroll graph is the only thing that fills it in-graph.

    ``launches`` > 1 shadows the graph BACK-TO-BACK. A single replay cannot catch a graph
    that is right once and stale on the next launch, which is exactly the failure mode the
    continuation rung is built to avoid (the unroll graph HAS it: its prologue reads hidden
    states nothing refreshed in between). Both arms run the same number of times from the
    same snapshot, so a second-launch divergence shows up in ``tok``/``accum``.

    NOT a proof of runtime correctness (the dummy batch has no padded rows and no stale
    block tables); it is a smoke test for the pointer-staleness class specifically.
    """
    import torch

    ibuf = runner.input_buffers
    hidden = getattr(runner.cudagraph_manager, "hidden_states", None)
    snap = (
        ibuf.input_ids[:rows].clone(),
        ibuf.positions[:rows].clone(),
        ibuf.seq_lens[:rows].clone(),
        b.tok[:rows].clone(),
        b.step.clone(),
        b.accum[:rows].clone(),
        None if hidden is None else hidden[:rows].clone(),
    )

    def restore():
        ibuf.input_ids[:rows].copy_(snap[0])
        ibuf.positions[:rows].copy_(snap[1])
        ibuf.seq_lens[:rows].copy_(snap[2])
        b.tok[:rows].copy_(snap[3])
        b.step.copy_(snap[4])
        b.accum[:rows].copy_(snap[5])
        if snap[6] is not None:
            hidden[:rows].copy_(snap[6])

    restore()
    for _ in range(launches):
        graph.replay()
    torch.cuda.synchronize()
    from_graph = (b.tok[:rows].clone(), b.accum[:rows].clone())

    restore()
    for _ in range(launches):
        eager()
    torch.cuda.synchronize()
    from_eager = (b.tok[:rows].clone(), b.accum[:rows].clone())

    restore()
    for i, name in enumerate(("tok", "accum")):
        if not torch.equal(from_graph[i], from_eager[i]):
            log.error(
                "vtl: nstep %s graph at num_reqs=%d DIVERGED from eager in %s "
                "(graph=%s eager=%s)",
                what, rows, name, from_graph[i].tolist(), from_eager[i].tolist(),
            )
            return False
    return True


def _capture_burst_graphs(runner, b: _Burst) -> None:
    """Up to THREE CUDA graphs per FULL decode size: body, prologue, whole unrolled burst.

    All three go through the same local ``capture()``: warm up (the first call allocates
    workspaces and JITs the Triton kernels, neither of which is capturable), capture, then
    boot-shadow against the identical work host-launched. A failure returns ``None`` and
    DEMOTES exactly one rung -- the body graph is the floor of graph mode, so losing it drops
    the boot to eager, while losing the prologue or the unroll only costs that rung.

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
        # Every captured FULL decode size gets a graph, not just BURST_SIZES. Compose
        # captures [1,2,4,8,16,32] while BURST_SIZES stopped at 8, so sizes 16/32 previously
        # fell back to the eager loop; with `b.rows` masking the feedback there is no longer
        # a reason to skip them. LoRA graphs stay out -- the burst has no LoRA story.
        if desc.num_active_loras:
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
        # Capture with the mask wide open (rows == pad here), so the captured graph exercises
        # the same code path a full batch takes. Per-step `b.rows.fill_()` narrows it.
        b.rows.fill_(rows)
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

        def capture(work, what, prime=None, launches=1):
            """Warm up, capture, shadow-compare. ``None`` on any failure -- never raises.

            ``prime`` puts the accumulator column counter where the graph will find it at
            runtime; the default zeroes it, which is what every rung but the continuation
            wants. ``launches`` is passed straight to the shadow (see
            ``_graph_matches_eager``).
            """
            reset = prime or b.step.zero_
            try:
                # Warm up outside the capture: the first call allocates workspaces and JITs
                # the Triton kernels, neither of which is capturable.
                reset()
                work()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                reset()
                with torch.cuda.graph(graph, mgr.pool):
                    work()
            except Exception:
                log.exception("vtl: nstep %s capture failed at num_reqs=%d", what, rows)
                return None

            # `work` IS the eager arm: the shadow's own restore() puts every buffer -- the
            # step counter included -- back to the snapshot between the two arms, so a
            # reset inside the arm would only hide a graph that depends on one.
            reset()
            if not _graph_matches_eager(runner, b, graph, work, rows, what, launches):
                return None
            return graph

        graph = capture(lambda: _burst_body(runner, b, rows, rows, forward), "burst body")
        if graph is None:
            # The body graph is the floor of graph mode for THIS size only -- skip it
            # (fold/unroll below are the same iteration, so `continue` skips those rungs
            # too) and let the other captured sizes still ride graph mode. Whole-boot
            # demotion only happens below, after the loop, if NOTHING captured.
            log.error("vtl: nstep body capture failed at num_reqs=%d; size skipped", rows)
            continue
        b.graphs[rows] = (graph, desc)
        captured += 1

        if b.fold:
            pro = capture(lambda: _prologue(runner, b, rows), "token-1 prologue")
            if pro is None:
                b.demote("fold", "the prologue did not capture")
            else:
                b.pro_graphs[rows] = (pro, desc)

        if b.unroll and b.fold and b.n > 1:
            whole = capture(
                lambda: _burst_all(runner, b, rows, rows, forward, b.n), "unrolled burst"
            )
            if whole is None:
                b.demote("unroll", "the unrolled burst did not capture")
            else:
                b.unroll_graphs[rows] = (whole, desc, b.n)

        # The Rust runner's second exec class: launches 2..k of a multi-burst step. Not a
        # ladder rung of its own -- losing it at one size only costs THAT size its
        # multi-launch step (the crate falls back to a single launch and the scheduler
        # reconciles the surplus), so it never demotes anything.
        if b.rust_steps > 1 and rows in b.unroll_graphs:
            cont = capture(
                lambda: _burst_continue(runner, b, rows, rows, forward, b.n),
                "burst continuation",
                # Where the counter stands after a first launch, so the capture writes the
                # columns it will write at runtime rather than the first burst's.
                prime=lambda: b.step.fill_(b.n - 1),
                # TWO launches: a single-replay shadow cannot see back-to-back staleness,
                # and back-to-back staleness is the only thing this graph exists to avoid.
                launches=2,
            )
            if cont is None:
                log.warning(
                    "vtl: nstep continuation capture failed at num_reqs=%d; the Rust "
                    "runner takes one burst per step at that size", rows,
                )
            else:
                b.continue_graphs[rows] = (cont, desc, b.n)

    if not captured:
        log.warning("vtl: nstep graph mode off -- no FULL 1-token-per-request descriptor")
        b.demote("fold", "no FULL decode descriptor matched")
        b.mode = "eager"
        return
    # BURST_SIZES is now an OUTCOME, not a filter: whatever FULL decode sizes the manager
    # captured are the sizes a burst can ride. Published back to the module because
    # `rust_sched.py:3088` reads `MAX_BURST_REQS` to gate its commit, and `_publish_ready`
    # reads it per step -- both would otherwise still be capped at the old hardcoded 8.
    global BURST_SIZES, MAX_BURST_REQS
    BURST_SIZES = tuple(sorted(b.graphs))
    MAX_BURST_REQS = BURST_SIZES[-1]
    log.info(
        "vtl: nstep captured %d burst body graph(s) for sizes %s "
        "(+%.1f MiB; prologue=%s unroll=%s continuation=%s sample1=%s)",
        captured, sorted(b.graphs), (torch.cuda.memory_allocated() - before) / (1 << 20),
        sorted(b.pro_graphs), sorted(b.unroll_graphs), sorted(b.continue_graphs), b.sample1,
    )


# ---------------------------------------------------------------------------------------
# sample_tokens: the burst itself.
# ---------------------------------------------------------------------------------------


def _graph_shape_ok(b: _Burst, ib, num_reqs: int, entry) -> bool:
    """Is ``entry``'s captured graph usable for THIS step's batch?

    PADDED BATCHES ARE NOW ALLOWED. The graph's baked grid runs to the padded width, but
    ``_advance`` masks the feedback on the device scalar ``b.rows``, so a padded row keeps the
    ``seq_len`` 0 the stock step left it at. What still has to hold is that the descriptor
    matches -- it already encodes BOTH widths, so an equal descriptor means the captured
    attention metadata was built for exactly this padded shape.

    ``==``, not ``is``: ``BatchExecutionDescriptor`` is a frozen dataclass, so equal
    descriptors need not be the same object.
    """
    return entry is not None and entry[1] == b.desc


def _run_burst(runner, b: _Burst, n: int, ib, hidden):
    """Sample N tokens. Returns the ``SamplerOutput`` the stock tail expects."""
    import torch
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    num_reqs = ib.num_reqs
    # The one host scalar FA3 needs. `seq_lens_cpu_upper_bound` is this step's post-step-1
    # value, so iteration j wants base + j. Read once, incremented on the host -- no sync.
    base_max_seq_len = int(ib.seq_lens_cpu_upper_bound[:num_reqs].max())
    b.idx_map[:num_reqs].copy_(ib.idx_mapping, non_blocking=True)
    # The padded-batch mask the captured `_advance` reads. Must be written EVERY step, before
    # any replay: the graph baked the grid at the padded width, so a stale value here feeds a
    # padded row (or starves a real one).
    b.rows.fill_(num_reqs)
    b.step.zero_()

    graph_mode = b.mode == "graph"
    # Graphs are keyed by the CAPTURED (padded) size, not the real request count -- a batch
    # of 5 rides the size-8 graph with `b.rows` masking rows 5..7. Before padded batches were
    # allowed these were the same number, which is why this used to read `num_reqs`.
    pad = ib.num_reqs_after_padding
    entry = b.graphs.get(pad) if graph_mode else None
    shape_ok = _graph_shape_ok(b, ib, num_reqs, entry)
    # The prologue/unroll graphs bake the address of `cudagraph_manager.hidden_states`, so
    # they are only usable when this step's hidden states actually live there. The persistent
    # buffer is sliced to the PADDED width by the stock FULL replay, so that is what the
    # prologue captured against.
    folded = shape_ok and b.fold and _hidden_is_persistent(runner, hidden, pad)

    if folded:
        whole = b.unroll_graphs.get(pad)
        if whole is not None and whole[1] == b.desc and whole[2] == n:
            # ONE replay for the whole burst: prologue, N-1 bodies, accumulator tail.
            whole[0].replay()
            return _burst_output(b, num_reqs, n, SamplerOutput)

    pro = b.pro_graphs.get(pad) if folded else None
    if pro is not None and pro[1] == b.desc:
        pro[0].replay()
    else:
        # Token 1 from the hidden states the stock forward already produced, host-launched.
        torch.argmax(
            runner.model.compute_logits(hidden[:num_reqs]), dim=-1, out=b.tok[:num_reqs]
        )

    if shape_ok:
        graph = entry[0]
        for _ in range(n - 1):
            graph.replay()
    else:
        forward = lambda: runner.cudagraph_manager.run_fullgraph(b.desc)  # noqa: E731
        for j in range(1, n):
            b.max_seq_len = base_max_seq_len + j
            _burst_body(runner, b, num_reqs, ib.num_reqs_after_padding, forward)
    # The last token was sampled after the final feedback, so nothing wrote it into the
    # accumulator; the column index is n-1 by construction. (The unroll graph did this
    # itself, in-graph, and returned above.)
    b.accum[:num_reqs, n - 1].copy_(b.tok[:num_reqs])

    return _burst_output(b, num_reqs, n, SamplerOutput)


def _run_burst_rust(runner, b: _Burst, n: int, steps: int, ib, hidden):
    """The same burst, launched by the Rust runner. ``None`` = declined.

    THE ONLY DIFFERENCE FROM ``_run_burst``'s graph arm is who calls ``cuGraphLaunch`` and
    what happens after it: the crate replays the very same unroll exec on the very same
    stream from the very same thread (``python.rs`` re-checks ``cuCtxGetCurrent`` against
    arm-time), then issues its pre-planned pinned D2H.

    WHERE THE TOKENS ARE COMMITTED depends on ``VTL_RUST_RUNNER_COMMIT`` (see rust_runner's
    module docstring):

      * update mode (default) -- ``launch`` only enqueues, into ring slot ``free_slot()``,
        and the FIFO entry it pushes is what ``rust_sched.update_from_output`` commits with
        one step's delay. That is the only ordering the depth-2 batch queue allows, and it
        needs no interlock, so the runner actually engages under the served config.
      * sample mode -- ``run_steps`` launches AND commits here through ``step_pack_locked``,
        which requires this step to be the only unapplied one (``inflight == 1``).

    EVERY refusal below falls through to ``_run_burst``, which is a complete, correct step:
    the scheduler committed ``n * steps`` tokens, the Python arm produces ``n``, and the
    shortfall is reconciled by the same ``burst_uncommit`` path a stopped burst uses. After
    a SUCCESSFUL launch there is no falling back -- the graph has fed its tokens into the
    persistent input buffers -- so nothing that can fail sits between the launch and the
    return.

    Host prep stays here rather than moving into the crate (RUST-RUNNER.md 2, "few host
    inputs"): two writes, both into buffers the graph baked.
    """
    import time

    from vtl.patches import rust_runner

    stash = rust_runner.STATE.take_step()
    state = rust_runner.STATE
    if not state.live or stash is None or state.done is not None:
        return None
    slot = None
    if state.update:
        # Both ring buffers still hold a launch nobody has read back: a third would
        # overwrite one of them before its D2H had been gathered.
        slot = state.free_slot()
        if slot is None:
            return None
    elif state.inflight != 1:
        # Sample mode: Rust commits here and Python commits at update time, so a launch
        # while an earlier step is still unapplied would put this step's tokens into the
        # store ahead of that one's.
        return None
    num_reqs = ib.num_reqs
    if stash.key != tuple(ib.req_ids) or stash.n != n or stash.steps != steps:
        return None
    pad = ib.num_reqs_after_padding
    entry = b.unroll_graphs.get(pad)
    # The same three predicates `_run_burst`'s graph arm applies before it replays: the
    # descriptor the step actually dispatched, the burst width the graph was captured for,
    # and this step's hidden states living at the address the prologue baked.
    if entry is None or entry[1] != b.desc or entry[2] != n:
        return None
    if not _hidden_is_persistent(runner, hidden, pad):
        return None
    # No check for `continue_graphs` here on purpose: the crate holds both exec classes per
    # table row and CLAMPS itself to one launch when this shape has no continuation graph,
    # which is strictly better than refusing the whole step (the surplus is reconciled).

    # Rows `num_reqs..pad` keep whatever an earlier (wider) step left in them, and the
    # captured `compute_slot_mappings` grid does read them -- but `prepare_inputs` pads
    # `query_start_loc[num_reqs + 1:]` with the real token count (input_batch.py:137), so
    # each padded row's token loop is `range(t, t)` and its stale `idx_map` value is a
    # loaded-but-dead operand; the kernel's sentinel program then PAD_SLOT_IDs the token
    # gap. Same argument the Python arm's padded replay rests on.
    b.idx_map[:num_reqs].copy_(ib.idx_mapping, non_blocking=True)
    b.rows.fill_(num_reqs)
    # No `b.step.zero_()`: the unroll graph does it in-graph, and the continuation graphs
    # deliberately do NOT (they keep filling the next `n` accumulator columns).

    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    if state.update:
        # Ownership BEFORE the launch: from this append until `update_from_output` pops
        # it, the slot is taken (`free_slot` reads occupancy off `pending`), so there is
        # no instant where a launched slot is untracked. A decline pops it right back.
        state.pending.append((slot, stash))
        try:
            launched = state.runner.launch(
                slot, num_reqs, pad, 1, 0, list(stash.slots),
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            # Nothing is committed by a launch, so unlike sample mode this is unambiguous:
            # either the graph ran (and this step's tokens are also in the AsyncOutput the
            # Python arm is about to build) or it did not. Either way the Python arm below
            # produces the step. Stand down for the boot regardless -- a driver call that
            # raised once will raise again. (`refuse` clears `pending`, ours included.)
            log.exception("vtl: rust runner launch raised; disabled for this boot")
            rust_runner.STATE.refuse("launch raised")
            return None
        if not launched:
            # Declined inside the crate: not armed, no launchable graph for this shape, or
            # a slot in the batch that cannot pack. Ours is the tail; older entries stay.
            state.pending.pop()
            return None
        # From here on the graph HAS run and its commit is pending. Falling back to stock
        # sampling would run the step a SECOND time, so a failure below is fatal for the
        # step, not a degrade -- see PostLaunchError.
        try:
            state.engaged += 1
            if state.engaged == 1 or not state.engaged & 0x1FFF:
                log.info(
                    "vtl: rust runner engaged -- %d launches (n=%d, batch=%d, slot=%d)",
                    state.engaged, n, num_reqs, slot,
                )
            return _burst_output(b, num_reqs, n, SamplerOutput)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, rust_runner.PostLaunchError)):
                raise
            log.exception("vtl: rust runner post-launch failure; the step cannot fall back")
            raise rust_runner.PostLaunchError(f"post-launch: {exc}") from exc

    try:
        out = rust_runner.STATE.runner.run_steps(
            steps, num_reqs, pad, 1, 0,
            list(range(num_reqs)), [n] * num_reqs, list(stash.slots),
            stash.max_model_len, 0, time.monotonic(), (),
            stash.fold, stash.publish,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        # Includes a crate panic (PanicException derives from BaseException). The tokens
        # were either all committed or none were -- `step_pack_locked` is one locked call
        # per launch -- but we cannot tell which from here, so stand down for the boot and
        # let this step run in Python. A double-commit would show up immediately as garbled
        # output rather than silently.
        log.exception("vtl: rust runner launch raised; disabled for this boot")
        rust_runner.STATE.refuse("run_steps raised")
        return None
    if out is None:
        # Declined inside the crate: not armed, or the shape has no launchable graph.
        return None

    verdicts, records, exit_reason, ran = out
    if not ran:
        # Nothing was launched, so nothing was committed and the Python arm can still run
        # this step whole. (Unreachable: the loop always takes its first launch.)
        return None
    state.done = rust_runner._Done(stash, verdicts, list(records), exit_reason, int(ran))
    state.engaged += 1
    if state.engaged == 1 or not state.engaged & 0x1FFF:
        log.info(
            "vtl: rust runner engaged -- %d launches (n=%d, steps=%d, ran=%d, batch=%d)",
            state.engaged, n, steps, ran, num_reqs,
        )
    total = ran * n
    if total == b.n:
        # The k=1 case, i.e. every step until VTL_RUST_RUNNER_STEPS>1 has headroom: the
        # pre-filled pair already says exactly this.
        return _burst_output(b, num_reqs, total, SamplerOutput)
    # `post_update` reads BOTH of these off the device, so a multi-launch step has to say
    # how many tokens it really produced (`ran`, not `steps` -- a stop verdict exits early).
    return _burst_output(
        b, num_reqs, total, SamplerOutput,
        b.nsampled_k.fill_(total)[:num_reqs],
        b.nrejected_k.fill_(-(total - 1))[:num_reqs],
    )


def _sample_one_in_graph(runner, b: _Burst, ib, hidden):
    """In-graph greedy sampling for a plain 1-token decode step. ``None`` = use stock.

    ``execute_model`` already replayed the stock FULL graph, so the forward is DONE and its
    hidden states are in the persistent buffer; the only thing left to put in a graph is the
    ``compute_logits`` + ``argmax`` pair, which is exactly the prologue graph the fold-in
    already captured. One replay replaces three host dispatch points and the whole V2 sampler
    (its numpy gathers, its ``new_ones`` allocation and its SamplerOutput build).

    Deliberately NOT a second capture of the forward: re-running it would double the step's
    GPU work to save host work, which is the wrong trade at 1 ms of GPU per step.
    """
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    rows = ib.num_reqs
    # Keyed by the padded width, like the burst graphs: the prologue was captured against the
    # persistent hidden-states buffer sliced to `desc.num_tokens`.
    pad = ib.num_reqs_after_padding
    entry = b.pro_graphs.get(pad)
    if not _graph_shape_ok(b, ib, rows, entry):
        return None
    if not _hidden_is_persistent(runner, hidden, pad):
        return None
    entry[0].replay()
    return SamplerOutput(
        # CLONED for the same reason the burst's accumulator is: `AsyncOutput` copies it on
        # a second stream and the copy outlives this call.
        sampled_token_ids=b.tok[:rows].view(-1, 1).clone(),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=b.ones[:rows],
        num_rejected=b.zeros[:rows],
    )


def _burst_output(b: _Burst, num_reqs: int, n: int, SamplerOutput,
                  num_sampled=None, num_rejected=None):
    """``n`` tokens per request out of the accumulator's first ``n`` columns.

    ``num_sampled`` / ``num_rejected`` default to the pre-filled ``BURST.n`` pair; the Rust
    multi-launch path passes its own, because there ``n`` is ``ran * width`` and not the
    burst width the constant pair was filled with.
    """
    return SamplerOutput(
        # CLONED, not a view of the persistent accumulator: `AsyncOutput` copies this to
        # the host on a SECOND stream and the copy is still in flight when the next step
        # starts writing. Stock is safe because its argmax output is a fresh allocation
        # every step; this is the same thing, at 8x4x8 bytes.
        sampled_token_ids=b.accum[:num_reqs, :n].clone(),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=b.num_sampled[:num_reqs] if num_sampled is None else num_sampled,
        # ponytail: post_update computes `computed_delta = query_len - num_rejected`
        # (input_batch.py:509) and query_len is 1 on a decode step, so a num_rejected of
        # -(N-1) is how a plain N is expressed with no kernel change. It is used for
        # nothing else on this path -- there is no speculator and no PP broadcast.
        num_rejected=b.num_rejected[:num_reqs] if num_rejected is None else num_rejected,
    )


# ---------------------------------------------------------------------------------------
# Install.
# ---------------------------------------------------------------------------------------


def _alloc(runner, b: _Burst) -> None:
    import torch

    dev = runner.device
    m = runner.max_num_reqs
    # WIDER THAN ONE BURST when the Rust runner may drive several launches per step: the
    # continuation graph does NOT reset the accumulator column counter, so launch i fills
    # columns [i*n, (i+1)*n) and `postprocess_sampled` gets every token the step produced
    # (it writes them into `req_states.all_token_ids`, so a missing one is a wrong resume).
    # `+ 2` is capture headroom: the boot shadow launches the continuation graph twice from
    # a step counter primed at n-1, i.e. up to column 3n-1.
    b.accum = torch.zeros((m, b.n * (b.rust_steps + 2)), dtype=torch.int64, device=dev)
    b.tok = torch.zeros(m, dtype=torch.int64, device=dev)
    b.step = torch.zeros(1, dtype=torch.int32, device=dev)
    # The real request count, as a DEVICE scalar so a captured graph (whose grid is baked at
    # the padded width) can still mask the padded rows out of the feedback. See `_advance`.
    b.rows = torch.zeros(1, dtype=torch.int32, device=dev)
    b.idx_map = torch.zeros(m, dtype=torch.int32, device=dev)
    b.num_sampled = torch.full((m,), b.n, dtype=torch.int32, device=dev)
    b.num_rejected = torch.full((m,), -(b.n - 1), dtype=torch.int32, device=dev)
    # The N=1 in-graph sampling shapes: one token, nothing rejected -- i.e. exactly what
    # `post_update`'s `computed_delta = query_len - num_rejected` reads on a normal step.
    b.ones = torch.ones(m, dtype=torch.int32, device=dev)
    b.zeros = torch.zeros(m, dtype=torch.int32, device=dev)
    # Filled per multi-launch Rust step; unused (and untouched) on every other path.
    b.nsampled_k = torch.full((m,), b.n, dtype=torch.int32, device=dev)
    b.nrejected_k = torch.full((m,), -(b.n - 1), dtype=torch.int32, device=dev)


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
                # Phase 2: hand the captured graphs to the Rust runner. Its OWN try/except,
                # nested inside this one, because the runner is strictly downstream of the
                # burst: `export` already swallows everything and refuses on failure, so the
                # only thing left that can throw is the import -- and a missing runner module
                # must not cost the burst that was just armed above.
                try:
                    from vtl.patches import rust_runner

                    rust_runner.export(self, b)
                except Exception:
                    log.exception("vtl: rust runner export failed; burst continues without it")
            except Exception:
                log.exception("vtl: nstep burst setup failed; one token per step")
                b.armed = False
            # VTL_RUST_RUNNER_REQUIRE, checked HERE: every guard above swallows on purpose
            # (a runner that cannot arm must not cost the boot), so the loud arm has to sit
            # outside all of them. An unimportable module raises on its own, which is the
            # same answer. Only PERMANENT refusals (and a burst that never armed) fail the
            # boot at this point -- on a server boot the Scheduler does not exist yet
            # (capture runs inside `_initialize_kv_caches`), so "waiting for the Rust KV
            # manager" is the expected state; `rust_sched` completes the rendezvous at
            # scheduler init and enforces REQUIRE for that half.
            if _flag("VTL_RUST_RUNNER_REQUIRE", "0"):
                from vtl.patches import rust_runner

                if rust_runner.STATE.refused or not b.armed:
                    raise RuntimeError(
                        "VTL_RUST_RUNNER_REQUIRE=1 but the Rust runner cannot arm: "
                        f"{rust_runner.STATE.why if b.armed else 'the burst never armed'}"
                    )
                if not rust_runner.STATE.live:
                    log.info(
                        "vtl: rust runner REQUIRE deferred to scheduler init -- %s",
                        rust_runner.STATE.why,
                    )
            return size

        return capture_model

    # --- read the commitment off the SchedulerOutput --------------------------------
    def wrap_execute_model(original):
        def execute_model(self, scheduler_output, *args, **kwargs):
            b.pending = getattr(scheduler_output, "vtl_burst_n", 0) or 0
            b.pending1 = bool(getattr(scheduler_output, "vtl_sample_in_graph", False))
            return original(self, scheduler_output, *args, **kwargs)

        return execute_model

    # --- the burst, plus the readiness signal for the NEXT step ---------------------
    def wrap_sample_tokens(original):
        def sample_tokens(self, grammar_output):
            n = b.pending
            one = b.pending1
            b.pending = 0
            b.pending1 = False
            state = self.execute_model_state
            # `grammar_output is not None` cannot happen on a committed burst or a committed
            # in-graph sample (the gate refuses structured output), but a bare argmax would
            # ignore the bitmask if it ever did -- so refuse here too rather than sample past
            # a grammar.
            if state is None or grammar_output is not None or (n < 2 and not one):
                out = original(self, grammar_output)
                _publish_ready(self, state)
                return out
            try:
                if n < 2:
                    out = _sample_in_graph(self, b, state, original)
                else:
                    out = _sample_burst(self, b, n, state)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                from vtl.patches import rust_runner
                if isinstance(exc, rust_runner.PostLaunchError):
                    # A Rust launch already fed this step back and its commit is pending
                    # in the FIFO -- `original()` would sample the step a second time on
                    # top of it. No degrade exists here; surface the engine error.
                    raise
                # A committed burst that produced fewer than N tokens IS reconciled by
                # the scheduler (rust_sched's `_burst_uncommit`), so this is recoverable
                # -- but it means the burst body is broken, so never try again.
                log.exception("vtl: nstep sampling failed mid-step; permanently disabled")
                b.disable("burst body raised")
                self.execute_model_state = state
                out = original(self, grammar_output)
            _publish_ready(self, state)
            return out

        return sample_tokens

    def _sample_in_graph(runner, b, state, original):
        """One replay in place of ``self.sample()``. Falls through to stock on any refusal.

        The refusals are shape checks only (captured size, matching descriptor, hidden states
        in the persistent buffer), so falling through costs nothing and changes nothing: no
        request state has been touched at that point.
        """
        sampler_output = _sample_one_in_graph(
            runner, b, state.input_batch, state.hidden_states
        )
        if sampler_output is None:
            return original(runner, None)
        runner.execute_model_state = None
        return _emit(runner, state, sampler_output)

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

    def _sample_burst(runner, b, n, state):
        """``n`` is the whole step's committed token BUDGET, which the Rust runner's
        multi-step commit makes a multiple of the burst width.

        The width is always ``b.n`` -- it is the only width any burst graph was captured
        for -- so the launch count follows from the budget and needs no second field on the
        SchedulerOutput to disagree with. The Rust runner takes all the launches in one FFI
        call; the Python arm runs ONE burst and the scheduler reconciles the rest through
        ``burst_uncommit``, the same shortfall path a stopped burst already uses.
        """
        ib = state.input_batch
        width = b.n if n >= b.n and not n % b.n else n
        steps = n // width
        out = _run_burst_rust(runner, b, width, steps, ib, state.hidden_states)
        if out is None:
            out = _run_burst(runner, b, width, ib, state.hidden_states)
        runner.execute_model_state = None
        return _emit(runner, state, out)

    def _emit(runner, state, sampler_output):
        """``sample_tokens``' last-rank tail, with ``sampler_output`` already produced.

        Reproduced rather than wrapped because the burst (and the in-graph N=1 sample) has to
        sit between ``self.sample()`` and ``AsyncOutput``. Every branch this drops is one the
        readiness probe and the scheduler's commit gate have already excluded: PP
        (``is_last_pp_rank``), pooling, the speculator, draft tokens, structured output,
        and prompt logprobs (which need a prefilling row).
        """
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.worker.gpu.async_utils import AsyncOutput

        ib = state.input_batch
        # Same idea as decode_fastpath's reused InputBatch: when the request set hasn't
        # changed, `ib.req_ids` is often the SAME list object across steps, so an identity
        # hit skips rebuilding this dict every burst emit. A miss (new object -- composition
        # changed, or fastpath not engaged) just rebuilds it once, same as before.
        if getattr(runner, "_vtl_r2i_ids", None) is ib.req_ids:
            r2i = runner._vtl_r2i
        else:
            r2i = {req_id: i for i, req_id in enumerate(ib.req_ids)}
            runner._vtl_r2i = r2i
            runner._vtl_r2i_ids = ib.req_ids
        model_runner_output = ModelRunnerOutput(
            req_ids=ib.req_ids,
            req_id_to_index=r2i,
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
    graph = b.mode == "graph"
    # Every in-graph rung needs a capture, so eager mode turns all three off rather than
    # pretending they are available and refusing per step.
    b.fold = graph and _flag("VTL_NSTEP_FOLD_T1")
    b.unroll = b.fold and _flag("VTL_NSTEP_UNROLL")
    b.sample1 = b.fold and _flag("VTL_SAMPLE_IN_GRAPH")
    # The Rust runner rides the unroll rung (it launches exactly that graph), so it can
    # never outlive it. Read here rather than at capture time so `_alloc` can size the
    # accumulator for the widest multi-launch step the runner may ever take.
    #
    # UPDATE-MODE COMMIT PINS THIS TO 1, which is also what keeps `continue_graphs`
    # uncaptured: the multi-launch residency loop only exists inside the sample-time
    # `run_steps` call, and a continuation graph per burst size doubles the burst-graph
    # memory for a rung nothing would launch.
    b.rust_steps = 1
    if b.unroll:
        try:
            from vtl.patches import rust_runner

            if rust_runner.mode() == "on" and rust_runner.commit_mode() == "sample":
                b.rust_steps = rust_runner.max_steps()
        except Exception:
            log.exception("vtl: rust runner module not importable; one launch per step")
    if b.mode not in ("graph", "eager"):
        log.info("vtl: nstep decode off (VTL_NSTEP_MODE=%s)", b.mode)
        return
    if b.n < 2 and not b.sample1:
        log.info("vtl: nstep decode off (VTL_NSTEP_N=%d, no in-graph sampling)", b.n)
        return
    b.advance, b.bump = _build_kernels()
    _patch_runner(b)
    log.info(
        "vtl: nstep decode installed (N=%d, mode=%s, fold_t1=%s, unroll=%s, sample1=%s, "
        "rust_steps=%d); the scheduler arms it",
        b.n, b.mode, b.fold, b.unroll, b.sample1, b.rust_steps,
    )


# ---------------------------------------------------------------------------------------


def _self_check() -> None:
    """No torch, no vLLM: the handshake and the eligibility matrix are what can be wrong."""
    b = BURST
    saved = (b.armed, b.ready, b.key, b.n, b.sample1, dict(b.pro_graphs))
    try:
        b.n = 4
        b.armed = True
        b.ready = True
        b.key = ("a", "b")

        nst = {"a": 1, "b": 1}
        assert commit_key(nst) == ("a", "b")
        assert burst_factor(nst) == 4

        # --- the N=1 in-graph sampling handshake ------------------------------------
        # Same key agreement as the burst, but independent of N *and* gated on a graph
        # actually having been captured -- `sample1` alone is only an intent.
        b.sample1 = True
        b.pro_graphs = {}
        assert sample_in_graph_ready(nst) is False, "no captured prologue graph"
        b.pro_graphs = {2: (object(), object())}
        assert sample_in_graph_ready(nst) is True
        assert sample_in_graph_ready({"b": 1, "a": 1}) is False, "order is load-bearing"
        b.n = 1
        assert sample_in_graph_ready(nst) is True, "N=1 sampling does not need a burst"
        assert burst_factor(nst) == 1
        b.n = 4
        b.sample1 = False
        assert sample_in_graph_ready(nst) is False
        b.sample1 = True
        b.ready = False
        assert sample_in_graph_ready(nst) is False, "the runner published no clean step"
        b.ready = True
        b.armed = False
        assert sample_in_graph_ready(nst) is False, "the patch is not installed"
        b.armed = True
        assert sample_in_graph_ready(nst) is True
        b.sample1 = False
        b.pro_graphs = {}

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
        assert b.pending == 0 and b.pending1 is False
    finally:
        b.armed, b.ready, b.key, b.n, b.sample1, b.pro_graphs = saved

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

    # The pre-capture defaults. `_capture_burst_graphs` REPLACES both with whatever FULL
    # decode sizes actually captured, so this pins the seed value, not an invariant.
    assert MAX_BURST_REQS == BURST_SIZES[-1] and BURST_SIZES[0] == 1

    # `__slots__` must be able to hold everything apply() and the burst assign. Omitting a
    # name here is an AttributeError at BOOT, after the model has been loaded -- i.e. the
    # most expensive place to find out.
    for name in ("advance", "bump", "idx_map", "rows", "max_seq_len", "accum", "tok", "step",
                 "num_sampled", "num_rejected", "desc", "pending", "graphs", "runner",
                 "armed", "ready", "key", "n", "mode",
                 "fold", "unroll", "sample1", "pro_graphs", "unroll_graphs",
                 "ones", "zeros", "pending1",
                 "continue_graphs", "rust_steps", "nsampled_k", "nrejected_k"):
        setattr(BURST, name, getattr(BURST, name))
    assert set(_Burst.__slots__) == {
        "armed", "ready", "key", "n", "mode", "runner", "graphs", "accum", "tok", "step",
        "num_sampled", "num_rejected", "desc", "pending",
        "advance", "bump", "idx_map", "rows", "max_seq_len",
        "fold", "unroll", "sample1", "pro_graphs", "unroll_graphs", "ones", "zeros",
        "pending1",
        "continue_graphs", "rust_steps", "nsampled_k", "nrejected_k",
    }, "the slot list above and __slots__ must not drift"

    # The in-graph ladder demotes ONE rung at a time, and `fold` takes `unroll` with it
    # (the unrolled graph CONTAINS the prologue, so it cannot outlive it).
    saved_ladder = (b.fold, b.unroll, b.sample1)
    try:
        log.disabled = True
        b.fold = b.unroll = b.sample1 = True
        b.unroll_graphs = {1: object()}
        b.continue_graphs = {1: object()}
        b.demote("unroll", "test")
        assert not b.unroll and b.unroll_graphs == {}
        assert b.continue_graphs == {}, "the runner launches the unroll exec"
        assert b.fold and b.sample1, "an unroll failure must not cost the prologue"
        b.unroll = True
        b.pro_graphs = {1: object()}
        b.continue_graphs = {1: object()}
        b.demote("fold", "test")
        assert not b.fold and not b.sample1 and not b.unroll
        assert b.pro_graphs == {} and b.unroll_graphs == {} and b.continue_graphs == {}
    finally:
        log.disabled = False
        b.fold, b.unroll, b.sample1 = saved_ladder
        b.pro_graphs, b.unroll_graphs, b.continue_graphs = {}, {}, {}

    assert _int("VTL_NOT_SET_ANYWHERE", 4) == 4
    os.environ["VTL_NOT_SET_ANYWHERE"] = "not-a-number"
    try:
        assert _int("VTL_NOT_SET_ANYWHERE", 4) == 4, "a bad literal falls back, never raises"
        os.environ["VTL_NOT_SET_ANYWHERE"] = "2"
        assert _int("VTL_NOT_SET_ANYWHERE", 4) == 2
    finally:
        del os.environ["VTL_NOT_SET_ANYWHERE"]

    # `_flag` defaults ON (all three ladder knobs ship at "1") and takes an explicit off.
    assert _flag("VTL_NOT_SET_ANYWHERE") is True
    os.environ["VTL_NOT_SET_ANYWHERE"] = "0"
    try:
        assert _flag("VTL_NOT_SET_ANYWHERE") is False
        os.environ["VTL_NOT_SET_ANYWHERE"] = "on"
        assert _flag("VTL_NOT_SET_ANYWHERE") is True
    finally:
        del os.environ["VTL_NOT_SET_ANYWHERE"]

    from vtl.registry import PATCH_REGISTRY

    assert {p.name: p.default for p in PATCH_REGISTRY}.get("nstep_decode") is True
    print("nstep_decode self-check ok")


if __name__ == "__main__":
    _self_check()
