# Full SGLang RadixAttention fork — integration state

Porting SGLang's **token-granular** RadixAttention memory model into the vLLM 0.25.0 fork
(round-1.2, LFM2.5-1.2B). Same discipline as the tree-spec fork: port + unit-test the logic
off-box in `vtl/**.py`, land the on-box wiring via diff patches under
`vtl/vllm_patches/v0.25.0/`, track the frontier here.

## Why this exists (read first)

Measured on the real `trace-round2.jsonl` with vLLM's own prefix-cache simulator
(`bench/trace_stats.py`), token-level sharing (block_size=1) vs vLLM's current block_size=16:

| block_size | granularity | hit rate |
|---|---|---|
| 1 | token (= SGLang page_size=1) | **82.9%** |
| 16 (current) | block | 82.8% |

The token-granular ceiling is **+0.1pp of block hits**, and it does not reach latency
(block_size=1 has much higher per-block kernel/table/cudagraph overhead → E2E worse). This
fork is being built at the user's explicit direction with that number in hand; it is a
fidelity/implementation exercise, **not** an expected speedup. Keep it flag-gated and
default-off so the scored submission is never at risk.

## The key structural insight

vLLM's paged attention already gathers KV per-token when `block_size == 1`: a size-1 block
table row IS a `req_to_token` row (one physical slot per position). So we do **not** need to
rewrite FlashInfer/FlashAttention kernels. The fork = run at `block_size=1` and replace vLLM's
prefix-cache index + eviction with the explicit token-granular radix tree, which then behaves
exactly like SGLang's RadixCache. That collapses the "rewire the attention backend" milestone
from a kernel rewrite to a block-table/`out_cache_loc` wiring task.

## Done — off-box, unit-tested (`python -m vtl.radix.<mod>`)

| Layer | File | SGLang source | Test |
|---|---|---|---|
| Radix tree core | `vtl/radix_tree.py` | `mem_cache/radix_cache.py` (tree) | split / LRU-evict / lock_ref |
| KV slot allocator | `vtl/radix/token_allocator.py` | `mem_cache/allocator/token.py` | alloc/free/reserve-slot-0 |
| Req→token table | `vtl/radix/req_to_token_pool.py` | `mem_cache/memory_pool.py:242` | write/read/alloc/free |
| **Scheduler flow** | `vtl/radix/radix_cache.py` | `radix_cache.py` cache_(un)finished_req | **end-to-end KV reuse + no-leak** |

The end-to-end test drives the real lifecycle: `match_prefix → inc_lock_ref → alloc suffix →
cache_finished_req → dec_lock_ref/evict`, and asserts a second request reuses the first's KV
slots, allocates only the uncached suffix, frees no duplicates twice, and eviction returns
every slot. This is the RadixAttention flow, proven in isolation.

## Frontier — on-box, needs the H200 + serving (ordered)

Each milestone is flag-gated behind `VTL_ENABLE_RADIX_CACHE` and must A/B clean before the next.

- [ ] **M1 — force `block_size=1`.** Compose overlay + confirm LFM2 boots (hybrid: full-attn
      group goes size-1; mamba/short-conv group is unaffected, its state is per-sequence not
      per-token). Baseline the block_size=1 hit rate / TTFT / throughput. This alone delivers
      token-granular sharing natively; the tree below only changes the *index/eviction impl*.
- [ ] **M2 — construct the memory model at startup.** Instantiate `ReqToTokenPool` +
      `TokenToKVPoolAllocator` sized from the KV budget, backed by the real KV pool tensors
      (swap the off-box lists for `torch.int32`/`int64` tensors — same API). Patch site:
      KV-cache-manager / coordinator construction.
- [ ] **M3 — make the tree authoritative.** Route `get_computed_blocks` →
      `TokenRadixCache.match_prefix`, and the cache-write/free paths →
      `cache_unfinished_req`/`cache_finished_req`. Replace `BlockPool`'s
      `cached_block_hash_to_id` + free-queue eviction with the tree + `evict`. Needs a
      `block_pool.patch` (the free-queue is internal to `BlockPool`). Reconcile refcounts:
      the tree must be the *single owner* of slot freeing — no shadow/double free with vLLM.
- [ ] **M4 — write `req_to_token` / `out_cache_loc` into the forward path.** After
      `match_prefix`, write the request's shared+new slots into the block table so paged
      attention gathers reused KV; point new-token writes (`out_cache_loc`) at freshly
      alloc'd slots. At block_size=1 this is block-table population, not a kernel change.
- [ ] **M5 — hybrid mamba clamp (correctness-critical).** LFM2's short-conv group can't
      token-share like attention. Port `mem_cache/mamba_radix_cache.py` (`HybridReqToTokenPool`)
      or clamp every served hit to `min(attn_tree_hit, mamba_group_hit)`. **Getting this wrong
      = silent output corruption** (mamba recurrence reads state for a prefix it never
      computed). Gate M3/M4 behind a byte-identical-output check vs stock before trusting it.
- [ ] **M6 — A/B + correctness gate.** `VTL_ENABLE_RADIX_CACHE=0` vs `=1`: assert identical
      outputs on the grading trace, compare TTFT/throughput. Expected: equal hit rate to M1,
      latency ≤ M1 (tree overhead). If it regresses vs current block_size=16 (likely, per the
      0.1pp/overhead analysis), keep default-off and record the finding.

## Risks (carry forward)

- **Mamba corruption (M5)** — the sharpest; advisory until byte-identical verified.
- **Refcount drift (M3)** — tree vs `BlockPool` double-free/leak; enforce single-owner freeing.
- **block_size=1 overhead** — more blocks, bigger tables, cudagraph capture sizes; may not fit
  the current `cudagraph_capture_sizes`. Re-tune or accept the throughput hit.
- **Submission safety** — all milestones default-off; never make the tree the serving default
  until M6 passes.
