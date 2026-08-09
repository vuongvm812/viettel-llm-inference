"""Pure-decode fast path for the V2 runner's per-step host work (+ pooled pinned staging).

TWO PATCHES, ONE MODULE, because they touch the same three seams and the same step.

--------------------------------------------------------------------------------------
PART 1 -- prepare_inputs / prepare_attn / model_state.prepare_attn on a pure-decode step
--------------------------------------------------------------------------------------
Under ``CUDAGraphMode.FULL`` the ``attn_metadata`` dict that ``MambaHybridModelState.
prepare_attn`` returns is DISCARDED: ``execute_model`` only threads it into
``set_forward_context`` on the PIECEWISE/eager branch, and the FULL branch replays a graph
whose pointers were baked at capture. So on a FULL step the entire metadata build is dead
weight except for its SIDE EFFECTS on persistent buffers, which are exactly three:

  * ``input_buffers.{input_ids, positions, seq_lens, query_start_loc}`` + the block tables
    and slot mappings -- the graph's real inputs;
  * ``FlashAttentionMetadataBuilder.scheduler_metadata[:n]`` -- the FA3 AOT schedule, a
    function of (batch_size, seq_lens, max_seq_len) and nothing else (flash_attn.py:577-595);
  * ``BaseMambaAttentionMetadataBuilder.state_indices_tensor_d`` -- in ``align`` mode the
    per-request current mamba block, ``gather(block_table, (seq_lens-1)//block_size)``
    (utils.py:947-965 -> mamba_attn.py:569-574).

Everything else -- the sort, the two ``np.fromiter``s, six numpy fancy-index gathers, the
``torch.arange``s, the ``InputBatch`` construction, ``CommonAttentionMetadata`` x groups,
``FlashAttentionMetadata``, ``Mamba2AttentionMetadata``, ``split_decodes_and_prefills`` --
is pure host cost that produces objects nobody reads. On a step whose batch is byte-for-byte
the previous step's batch plus one token per request, all of it is recomputable as "the same
as last step", so we recompute the three writes and reuse the cached ``InputBatch``.

WHY NOT REWRITE ``execute_model``. It would be a 150-line copy that has to track
``preprocess_state``, LoRA, EPLB, the cudagraph dispatch and the PP branches forever. The
predicate needs only ``(scheduler_output, batch_desc)``, and both arrive at
``prepare_inputs`` -- so the decision is made there and the other two seams just read the
flag. Stock control flow is untouched.

PREDICATE (all must hold, else the step takes the stock path AND re-primes the cache):
no new/finished/preempted requests, no spec-decode tokens, no encoder inputs, no
new_block_ids_to_zero, no structured output, every request scheduled exactly 1 token, the
same request ids in the same order as the cached step, the same padded
``BatchExecutionDescriptor``, ``cg_mode == FULL``, and no speculator. Freshly allocated
blocks (``scheduled_cached_reqs.new_block_ids``, i.e. a 16-token boundary) do NOT break the
fast path -- they only re-run ``gather_block_tables``.

CACHE COHERENCE. The reused ``InputBatch`` is mutated in place. Its GPU fields are slices of
persistent buffers (already shared across steps by stock code); the two host arrays that
actually advance -- ``num_computed_tokens_np`` and the numpy backing of
``seq_lens_cpu_upper_bound`` -- are refreshed every fast step. ``req_ids`` is handed to
``ModelRunnerOutput`` by reference and outlives the step, but the predicate guarantees its
contents are identical, so the aliasing is inert. ``execute_model_state`` is a single slot
consumed by ``sample_tokens`` within the same step, so no two live InputBatches overlap.

--------------------------------------------------------------------------------------
PART 2 -- ``VTL_UVA_POOL``: kill the two pin_memory-per-step allocations
--------------------------------------------------------------------------------------
``async_copy_to_gpu`` (buffer_utils.py:26-41) calls ``x.pin_memory()``, i.e. a fresh
``cudaHostAlloc`` + free on every call. There are exactly two hot sites per step
(model_runner.py:864 idx_mapping, :910 query_start_loc) and vLLM already ships the fix as a
class it uses elsewhere: ``UvaBufferPool``. We rebind the IMPORT-BOUND NAME
``vllm.v1.worker.gpu.model_runner.async_copy_to_gpu`` -- patching ``buffer_utils`` would do
nothing, the consumers use ``from ... import``.

Pool depth 4, not vLLM's default 2. Depth must cover (uses of one pool per step) x
(concurrent in-flight steps): the H2D is ``non_blocking``, so overwriting a buffer from the
CPU before the stream has drained the copy corrupts it. vLLM's own pools are per-call-site
so depth 2 = max_concurrent_batches suffices; ours are shared by bucket, so two call sites
can land in the same pool in one step. 4 tiny pinned int32 buffers per bucket.

Both parts are ON by default and independently killable:
``VTL_ENABLE_DECODE_FASTPATH=0`` (part 1 + part 2), ``VTL_UVA_POOL=0`` (part 2 only).
"""

from __future__ import annotations

import inspect
import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.decode_fastpath")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------------------
# The predicate. Pure Python, no torch -- so `make check` can exercise it off-box.
# ---------------------------------------------------------------------------------------


def decode_key(so) -> tuple[str, ...] | None:  # noqa: ANN001
    """``tuple(req_ids)`` if ``so`` is an unremarkable 1-token-per-request decode step.

    Order matters and is the dict's insertion order, which is exactly what stock's
    ``sorted(num_tokens_per_req, key=get)`` degenerates to when every value is 1 (Python's
    sort is stable). Two steps with the same key therefore produce the same ``req_ids``,
    the same ``idx_mapping`` and the same ``query_start_loc``.
    """
    if so.scheduled_new_reqs or so.finished_req_ids or so.preempted_req_ids:
        return None
    if so.scheduled_spec_decode_tokens or so.scheduled_encoder_inputs:
        return None
    if so.new_block_ids_to_zero or so.has_structured_output_requests:
        return None
    nst = so.num_scheduled_tokens
    if not nst:
        return None
    for n in nst.values():
        if n != 1:
            return None
    return tuple(nst)


def has_new_blocks(so) -> bool:  # noqa: ANN001
    """True if any request got fresh blocks this step -> ``gather_block_tables`` must re-run."""
    for new_ids in so.scheduled_cached_reqs.new_block_ids:
        if new_ids is not None and any(new_ids):
            return True
    return False


# ---------------------------------------------------------------------------------------
# Per-process step cache. One GPUModelRunner per process, so a module global is the whole
# lifetime story; `runner` is stored so a (hypothetical) second runner invalidates instead
# of silently sharing.
# ---------------------------------------------------------------------------------------


class _Cache:
    __slots__ = ("runner", "key", "ib", "seq_ub_np", "block_tables", "slot_mappings",
                 "fast", "builders", "refresh_bt", "hits", "misses")

    def __init__(self) -> None:
        self.builders = None
        self.reset()
        self.hits = 0
        self.misses = 0

    def reset(self) -> None:
        self.runner = None
        self.key = None
        self.ib = None
        self.seq_ub_np = None
        self.block_tables = None
        self.slot_mappings = None
        self.fast = False
        self.refresh_bt = True
        # Builder classification survives reset(): it is a property of the loaded model, and
        # re-running it is the only thing that could log per step on an unsupported backend.
        self.builders = None if self.builders is not False else False


_C = _Cache()


# ---------------------------------------------------------------------------------------
# Attention-builder classification + the two persistent writes the graph actually consumes.
# ---------------------------------------------------------------------------------------


def _classify(runner):  # noqa: ANN001
    """Split every live metadata builder into (FA3-with-AOT, mamba-align). None/False = bail.

    Classified by capability, not by class name: the FA3 half is identified by the
    persistent ``scheduler_metadata`` buffer + the AOT ``schedule`` inputs, the mamba half by
    ``state_indices_tensor_d``. Anything we cannot place (FlashInfer -- which is what the
    sm_86 dev box picks under fp8 KV -- or a backend swap after a version bump) disables the
    fast path rather than guessing: a wrong metadata buffer is a silent wrong answer, not a
    crash. ``False`` is final, and every bail logs its reason -- a classify that fails at
    ``capture_model`` time costs nstep its one shot at the burst graphs, so "which arm"
    must be readable off the boot log. (``None`` used to mean "retry next step" for the
    unprimed ``aot_sliding_window``; that arm now primes it in place, stock-identically.)
    """
    import vllm.envs as envs
    from vllm.v1.attention.backends import flash_attn as fa_mod

    if not hasattr(fa_mod, "get_scheduler_metadata"):
        log.info("vtl: decode_fastpath off -- no FA varlen in this image")
        return False

    fa, mamba = [], []
    for group_id, groups in enumerate(runner.attn_groups):
        for group in groups:
            b = group.get_metadata_builder(0)
            if hasattr(b, "state_indices_tensor_d"):
                spec = b.kv_cache_spec
                if getattr(spec, "num_speculative_blocks", 0) != 0 or b.num_spec_tokens:
                    log.info("vtl: decode_fastpath off -- mamba builder carries spec-decode state")
                    return False
                if runner.vllm_config.cache_config.mamba_cache_mode != "align":
                    log.info("vtl: decode_fastpath off -- mamba_cache_mode is not align")
                    return False
                # _mamba_write gathers state indices for PADDED rows too (stock tail-fills
                # NULL instead, mamba_attn.py:574). That is safe only while gather_block_tables
                # zero-fills padded rows AND NULL_BLOCK_ID == 0 -- so pin the constant here
                # rather than silently rotating a live block's conv state if it ever moves.
                from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

                if NULL_BLOCK_ID != 0:
                    log.info("vtl: decode_fastpath off -- NULL_BLOCK_ID=%d != 0", NULL_BLOCK_ID)
                    return False
                mamba.append((b, group_id, spec.block_size))
                continue
            if not hasattr(b, "scheduler_metadata"):
                log.info("vtl: decode_fastpath off -- unrecognised builder %s", type(b).__name__)
                return False
            if not (getattr(b, "aot_schedule", False) and getattr(b, "use_full_cuda_graph", False)):
                log.info("vtl: decode_fastpath off -- %s has no AOT/full-cudagraph schedule",
                         type(b).__name__)
                return False
            if getattr(b, "dcp_world_size", 1) != 1 or envs.VLLM_BATCH_INVARIANT:
                log.info("vtl: decode_fastpath off -- DCP or VLLM_BATCH_INVARIANT is set")
                return False
            if b.max_cudagraph_size is None:
                log.info("vtl: decode_fastpath off -- %s has no max_cudagraph_size",
                         type(b).__name__)
                return False
            if b.aot_sliding_window is None:
                # Stock populates this inside the first REAL build() (flash_attn.py:454-468)
                # -- graph capture only runs dummy builds, so at capture_model time it is
                # still None and a "retry next step" answer would cost nstep its one shot at
                # the burst graphs. Replicate the stock block verbatim; idempotent by
                # construction (build() re-derives the identical value).
                b.aot_sliding_window = (-1, -1)
                cfgs = fa_mod._get_sliding_window_configs(b.vllm_config)
                if len(cfgs) == 1:
                    cfg = cfgs.pop()
                    if cfg is not None:
                        b.aot_sliding_window = cfg
                elif len(cfgs) > 1:
                    # Stock's answer too (aot_schedule = False): layers disagree on the
                    # window, so there is no single AOT schedule to precompute.
                    b.aot_schedule = False
                    log.info("vtl: decode_fastpath off -- mixed sliding windows disable "
                             "the AOT schedule")
                    return False
            fa.append((b, group_id))
    # An EMPTY mamba list is legal: a dense (attention-only) model has no SSM group, and
    # every mamba consumer is a loop (`_fast_model_attn`) or an unpack
    # (`nstep_decode._burst_body`), so it no-ops. The mamba-specific refusals above all sit
    # inside the `state_indices_tensor_d` branch and are simply unreachable -- which is why
    # a dense model needs no `--mamba_cache_mode=align`. No FA builder is still fatal.
    if not fa:
        log.info("vtl: decode_fastpath off -- no FA builder in any attention group")
        return False
    return fa, mamba


def _fa_write(b, num_reqs: int, seq_lens, query_start_loc, max_seq_len: int) -> None:  # noqa: ANN001
    """Recompute the FA3 AOT schedule and land it in the builder's persistent buffer.

    Byte-for-byte the ``else`` arm of flash_attn.py's ``schedule()`` plus the
    ``use_full_cuda_graph`` epilogue, with the branch conditions hoisted into ``_classify``
    (no cascade -- ``build_attn_metadata`` hardcodes ``common_prefix_len=0``; no DCP;
    causal=True; ``max_query_len=1`` because every request is scheduled exactly one token).
    """
    from vllm.v1.attention.backends.flash_attn import (
        _maybe_symmetrize_window,
        get_scheduler_metadata,
    )

    sm = get_scheduler_metadata(
        batch_size=num_reqs,
        max_seqlen_q=1,
        max_seqlen_k=max_seq_len,
        num_heads_q=b.num_heads_q * b.dcp_world_size,
        num_heads_kv=b.num_heads_kv,
        headdim=b.headdim,
        cache_seqlens=seq_lens,
        qkv_dtype=b._vtl_qkv_dtype,
        cu_seqlens_q=query_start_loc,
        page_size=b.block_size,
        causal=True,
        window_size=_maybe_symmetrize_window(b.aot_sliding_window, True),
        num_splits=b._vtl_max_num_splits,
    )
    if sm is None:
        raise RuntimeError("vtl: AOT scheduler returned None on the fast path")
    n = sm.shape[0]
    b.scheduler_metadata[:n] = sm
    b.scheduler_metadata[n:] = 0


def _mamba_write(b, block_table, seq_lens, block_size: int, scratch) -> None:  # noqa: ANN001
    """``state_indices_tensor_d[:n] = gather(block_table, clamp((seq_lens-1)//bs, 0))``.

    align mode with zero speculative blocks, so the ``offsets`` arange collapses to ``[0]``
    and the gather is one column -- the whole of ``mamba_get_block_table_tensor`` +
    ``torch.split`` + ``_update_metadata_for_cudagraph_capture``'s copy, minus the six
    allocating ops. ``num_decodes == num_reqs`` here (nothing is prefilling and every row is
    a decode row), so the NULL_BLOCK_ID tail fill of the stock path covers zero rows.
    """
    import torch

    n = seq_lens.shape[0]
    idx = scratch[:n]
    torch.sub(seq_lens, 1, out=idx[:, 0])
    idx.div_(block_size, rounding_mode="floor").clamp_(min=0)
    torch.gather(block_table, 1, idx, out=b.state_indices_tensor_d[:n])


# ---------------------------------------------------------------------------------------
# The three wrapped seams.
# ---------------------------------------------------------------------------------------


def _fast_inputs(runner, c: _Cache):  # noqa: ANN001
    """The must-run subset of ``prepare_inputs``: two triton launches and two host adds."""
    import numpy as np
    from vllm.v1.worker.gpu import model_runner as mr

    ib = c.ib
    rs = runner.req_states
    mr.prepare_pos_seq_lens(
        ib.idx_mapping,
        ib.query_start_loc,
        rs.num_computed_tokens.gpu,
        runner.input_buffers.positions,
        runner.input_buffers.seq_lens,
    )
    ib.logits_indices = mr.combine_sampled_and_draft_tokens(
        runner.input_buffers.input_ids,
        ib.idx_mapping,
        rs.last_sampled_tokens,
        ib.query_start_loc,
        ib.seq_lens,
        rs.prefill_len.gpu,
        rs.draft_tokens,
        ib.cu_num_logits,
        ib.num_reqs,
        runner.model_state.num_new_sampled_tokens_per_step,
    )
    # The only two host arrays that advance. `seq_ub_np` is the numpy backing of
    # `ib.seq_lens_cpu_upper_bound` (shared storage), and every scheduled request adds
    # exactly one token, so the stock `num_computed + num_scheduled` is a +1.
    np.take(rs.num_computed_tokens_np, ib.idx_mapping_np, out=ib.num_computed_tokens_np)
    np.add(ib.num_computed_tokens_np, 1, out=c.seq_ub_np[: ib.num_reqs])
    return ib


def _fast_attn(runner, c: _Cache, refresh_block_tables: bool):  # noqa: ANN001
    ib = c.ib
    if refresh_block_tables:
        c.block_tables = runner.block_tables.gather_block_tables(
            ib.idx_mapping, num_reqs_padded=ib.num_reqs_after_padding
        )
    c.slot_mappings = runner.block_tables.compute_slot_mappings(
        ib.idx_mapping,
        ib.query_start_loc,
        ib.positions,
        num_tokens_padded=ib.num_tokens_after_padding,
    )
    return c.block_tables, c.slot_mappings


def _fast_model_attn(runner, c: _Cache) -> None:  # noqa: ANN001
    fa, mamba = c.builders
    ib = c.ib
    n = ib.num_reqs_after_padding
    seq_lens = runner.input_buffers.seq_lens[:n]
    # max_seq_len is the ONE scalar FA3 needs off the host; it advances +1/step. Max over
    # the REAL rows only -- the padded tail is zero from the slow step's fresh array and
    # stock reads [:num_reqs].max() too (mamba_hybrid.py:251); [:n] would be correct today
    # but couples the max to the tail staying zero.
    max_seq_len = int(c.seq_ub_np[: ib.num_reqs].max())
    for b, _ in fa:
        _fa_write(b, n, seq_lens, ib.query_start_loc, max_seq_len)
    for b, group_id, block_size in mamba:
        _mamba_write(b, c.block_tables[group_id], seq_lens, block_size, runner._vtl_gather_idx)


def _install(cls, name: str, wrapper) -> None:  # noqa: ANN001
    if already_patched(cls, name, patch="decode_fastpath"):
        return
    original = getattr(cls, name)
    bound = wrapper(original)
    mark_patched(bound, original, patch="decode_fastpath")
    setattr(cls, name, bound)


def _patch_runner() -> None:
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    def wrap_prepare_inputs(original):
        def prepare_inputs(self, scheduler_output, batch_desc):  # noqa: ANN001
            c = _C
            c.fast = False
            if _eligible(self, scheduler_output, batch_desc, c):
                try:
                    ib = _fast_inputs(self, c)
                except Exception:
                    log.exception("vtl: decode_fastpath prepare_inputs failed; reverting")
                    c.reset()
                else:
                    c.fast = True
                    c.hits += 1
                    return ib
            c.misses += 1
            ib = original(self, scheduler_output, batch_desc)
            _recache(self, scheduler_output, batch_desc, ib, c)
            return ib

        return prepare_inputs

    def wrap_prepare_attn(original):
        def prepare_attn(self, input_batch):  # noqa: ANN001
            c = _C
            if c.fast:
                return _fast_attn(self, c, c.refresh_bt)
            bt, sm = original(self, input_batch)
            c.block_tables, c.slot_mappings = bt, sm
            return bt, sm

        return prepare_attn

    def wrap_model_prepare_attn(original):
        def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                         attn_groups, kv_cache_config, for_capture=False):  # noqa: ANN001
            c = _C
            if c.fast and not for_capture:
                runner = c.runner
                try:
                    _fast_model_attn(runner, c)
                except Exception:
                    log.exception("vtl: decode_fastpath model prepare_attn failed; reverting")
                    c.reset()
                else:
                    # Discarded by execute_model under FULL. Empty, not the stale dict:
                    # an unexpected consumer must fail loudly, never read last step's
                    # max_seq_len as if it were this step's.
                    return {}
            return original(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                            attn_groups, kv_cache_config, for_capture)

        return prepare_attn

    def _eligible(runner, so, batch_desc, c: _Cache) -> bool:  # noqa: ANN001
        if c.key is None or c.runner is not runner or not c.builders:
            return False
        if batch_desc.cg_mode != CUDAGraphMode.FULL:
            return False
        ib = c.ib
        if (batch_desc.num_tokens != ib.num_tokens_after_padding
                or batch_desc.num_reqs != ib.num_reqs_after_padding):
            return False
        if decode_key(so) != c.key:
            return False
        c.refresh_bt = has_new_blocks(so)
        return True

    def _recache(runner, so, batch_desc, ib, c: _Cache) -> None:  # noqa: ANN001
        """Re-prime from a stock step. Anything unsupported leaves ``key=None`` = always slow."""
        c.runner = runner
        c.ib = ib
        c.key = None
        c.seq_ub_np = None
        if c.builders is False:
            return
        if runner.speculator is not None or runner.vllm_config.num_speculative_tokens:
            return
        if batch_desc.cg_mode != CUDAGraphMode.FULL or batch_desc.num_reqs is None:
            return
        if ib.num_draft_tokens or ib.is_prefilling_np.any() or ib.has_structured_output_reqs:
            return
        if ib.dcp_local_seq_lens is not None or ib.prompt_lens is not None:
            return
        if ib.max_seq_len_np is not None or runner.lora_config is not None:
            return
        if int(ib.num_scheduled_tokens.max()) != 1 or ib.num_reqs != len(ib.req_ids):
            return
        if not c.builders:
            c.builders = _classify(runner)
            if not c.builders:
                return
            fa, _ = c.builders
            for b, _ in fa:
                _prime_fa_consts(runner, b, ib)
            log.info("vtl: decode_fastpath classified %d FA + %d mamba builder(s)",
                     len(fa), len(c.builders[1]))
        if not hasattr(runner, "_vtl_gather_idx"):
            runner._vtl_gather_idx = torch.zeros(
                (runner.max_num_reqs, 1), dtype=torch.int64, device=runner.device
            )
        c.seq_ub_np = ib.seq_lens_cpu_upper_bound.numpy()
        c.key = decode_key(so)

    _install(GPUModelRunner, "prepare_inputs", wrap_prepare_inputs)
    _install(GPUModelRunner, "prepare_attn", wrap_prepare_attn)
    # BOTH model states, because which one is live depends on the model: a hybrid gets
    # MambaHybridModelState, a dense (attention-only) model gets DefaultModelState. The two
    # `prepare_attn` signatures are identical (default.py:132-141 vs mamba_hybrid.py:228-239)
    # and only one class is ever instantiated per boot, so the other install is inert.
    # Patching only the hybrid was what left `_fast_model_attn` -- the biggest of the three
    # seams -- dead on a dense model even with the builders classified.
    _install(MambaHybridModelState, "prepare_attn", wrap_model_prepare_attn)
    _install(DefaultModelState, "prepare_attn", wrap_model_prepare_attn)
    log.info("vtl: decode_fastpath armed")


def _prime_fa_consts(runner, b, ib) -> None:  # noqa: ANN001
    """Freeze the two ``schedule()`` locals that are constant for a fixed capture shape."""
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache

    cache_dtype = b.cache_config.cache_dtype
    b._vtl_qkv_dtype = (
        current_platform.fp8_dtype() if is_quantized_kv_cache(cache_dtype) else b.kv_cache_dtype
    )
    # num_actual_tokens on a fast step is the (capture-static) padded token count, so the
    # `<= max_cudagraph_size` test that selects max_num_splits cannot flip between steps
    # with the same batch_desc -- which the predicate already pins.
    b._vtl_max_num_splits = (
        b.max_num_splits if ib.num_tokens_after_padding <= b.max_cudagraph_size else 0
    )


# ---------------------------------------------------------------------------------------
# Item 4 -- pooled pinned staging.
# ---------------------------------------------------------------------------------------

_POOL_DEPTH = 4
_pools: dict = {}


def _bucket(n: int) -> int:
    """Round up to a power of two so a varying num_reqs shares one pool instead of N."""
    b = 8
    while b < n:
        b <<= 1
    return b


def _patch_uva_pool() -> None:
    import numpy as np
    import torch

    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool, async_copy_to_gpu as stock

    def pooled(x, out=None, device=None):  # noqa: ANN001
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim != 1 or not x.is_cpu or x.is_pinned():
            return stock(x, out=out, device=device)
        key = (x.dtype, _bucket(x.shape[0]))
        pool = _pools.get(key)
        if pool is None:
            try:
                pool = _pools[key] = UvaBufferPool(key[1], key[0], max_concurrency=_POOL_DEPTH)
            except Exception:  # no UVA on this platform -> stock forever
                log.warning("vtl: UVA pool unavailable; keeping stock pin_memory staging")
                _pools[key] = False
                return stock(x, out=out, device=device)
        elif pool is False:
            return stock(x, out=out, device=device)
        if out is None:
            # Allocate the destination exactly as stock does rather than let the pool's
            # `uva.clone()` pick the device -- `device` is an explicit argument here and the
            # UVA view's device is not guaranteed to be it.
            out = torch.empty_like(x, device=device)
        return pool.copy_to_gpu(x, out=out)

    pooled.__vtl_wrapped__ = stock
    n = 0
    for mod_name in ("model_runner", "pp_utils", "structured_outputs"):
        try:
            mod = __import__(f"vllm.v1.worker.gpu.{mod_name}", fromlist=["x"])
        except Exception:
            continue
        if getattr(mod, "async_copy_to_gpu", None) is stock:
            mod.async_copy_to_gpu = pooled
            n += 1
    log.info("vtl: pooled pinned staging rebound in %d namespace(s), depth=%d", n, _POOL_DEPTH)


@register_patch("decode_fastpath", default=True)
def apply() -> None:
    if _flag("VTL_UVA_POOL"):
        _patch_uva_pool()
    _patch_runner()


# ---------------------------------------------------------------------------------------


def _self_check() -> None:
    """No torch, no vLLM: the predicate and the bucketing are the parts that can be wrong."""

    class SO:
        def __init__(self, **kw):
            self.scheduled_new_reqs = []
            self.finished_req_ids = set()
            self.preempted_req_ids = set()
            self.scheduled_spec_decode_tokens = {}
            self.scheduled_encoder_inputs = {}
            self.new_block_ids_to_zero = None
            self.has_structured_output_requests = False
            self.num_scheduled_tokens = {"a": 1, "b": 1}
            self.scheduled_cached_reqs = type("C", (), {"new_block_ids": [None, None]})()
            self.__dict__.update(kw)

    assert decode_key(SO()) == ("a", "b")
    # Order is load-bearing: it becomes idx_mapping, so a permutation must miss the cache.
    assert decode_key(SO(num_scheduled_tokens={"b": 1, "a": 1})) == ("b", "a")
    assert decode_key(SO()) != decode_key(SO(num_scheduled_tokens={"b": 1, "a": 1}))

    # Every disqualifier, one at a time.
    for kw in (
        {"scheduled_new_reqs": [object()]},
        {"finished_req_ids": {"a"}},
        {"preempted_req_ids": {"a"}},
        {"scheduled_spec_decode_tokens": {"a": [7]}},
        {"scheduled_encoder_inputs": {"a": [0]}},
        {"new_block_ids_to_zero": [3]},
        {"has_structured_output_requests": True},
        {"num_scheduled_tokens": {"a": 1, "b": 2}},   # a chunked prefill hiding in the batch
        {"num_scheduled_tokens": {}},
    ):
        assert decode_key(SO(**kw)) is None, kw

    # New blocks are NOT a disqualifier -- they only force a block-table re-gather.
    fresh = SO()
    fresh.scheduled_cached_reqs = type("C", (), {"new_block_ids": [None, ([41],)]})()
    assert decode_key(fresh) == ("a", "b") and has_new_blocks(fresh)
    assert not has_new_blocks(SO())
    empty = SO()
    empty.scheduled_cached_reqs = type("C", (), {"new_block_ids": [([],), ([],)]})()
    assert not has_new_blocks(empty), "an empty per-group list is not a new block"

    assert (_bucket(1), _bucket(8), _bucket(9), _bucket(65)) == (8, 8, 16, 128)

    assert _flag("VTL_NOT_SET_ANYWHERE") is True
    os.environ["VTL_NOT_SET_ANYWHERE"] = "0"
    try:
        assert _flag("VTL_NOT_SET_ANYWHERE") is False
    finally:
        del os.environ["VTL_NOT_SET_ANYWHERE"]

    # A dense (attention-only) model has no mamba builder, so it needs BOTH halves of the
    # dense unlock. `_classify` itself needs torch+vLLM and cannot run here, but the two
    # things that silently re-kill the fast path on dense models are checkable from source:
    #
    #   1. the third seam installed on DefaultModelState as well as MambaHybridModelState --
    #      a hybrid-only install leaves `_fast_model_attn` (the biggest seam) dead;
    #   2. `_classify` not refusing on an empty mamba list.
    # Match the CALL, not the name: `DefaultModelState` also appears in `_patch_runner`'s
    # import line, so a `co_names` membership test still passes with the install deleted.
    installs = inspect.getsource(_patch_runner)
    for cls_name in ("MambaHybridModelState", "DefaultModelState"):
        assert f'_install({cls_name}, "prepare_attn"' in installs, (
            f"decode_fastpath must install prepare_attn on {cls_name}; a dense model "
            "instantiates DefaultModelState and would otherwise take the stock metadata "
            "build every step, leaving the biggest of the three seams dead"
        )
    src = inspect.getsource(_classify)
    assert "if not fa:" in src and "if not fa or not mamba:" not in src, (
        "_classify must accept an empty mamba list (dense model); the mamba-specific "
        "refusals live inside the state_indices_tensor_d branch and are unreachable there"
    )

    from vtl.registry import PATCH_REGISTRY

    assert {p.name: p.default for p in PATCH_REGISTRY}.get("decode_fastpath") is True
    print("decode_fastpath self-check ok")


if __name__ == "__main__":
    _self_check()
