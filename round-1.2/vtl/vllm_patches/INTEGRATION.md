# On-box integration spec — the two kernel-coupled forks

Every algorithm for tree verify is implemented and unit-tested in the `vtl` plugin (see the module
list below). Two forks cannot be finalized or validated off-box because they depend on the H200
buffer/kernel layout (the plan's off-box unknowns #2 #4 #5). This file specifies exactly what each
must do and which tested `vtl` function it calls, so the on-box engineer wires, not designs.

Tested vtl building blocks (all `python vtl/<f>.py` green):
- `tree_runtime.build_carrier` → device-tensor carrier (`md.vtl_tree`).
- `tree_attention.build_full_mask_rows` → per-request boolean mask `[num_nodes, seq_len+num_nodes]`.
- `tree_verify.accept` / `tree_sample.tree_greedy_accept_batch` → greedy tree accept (== oracle).
- `conv_commit.ConvStager` + `conv_commit.commit_conv_states` → conv stage + accepted-path commit.

Already shipped as applies-clean patches: `rejection_sampler.patch` (P4, diverts to `tree_verify.accept`),
`short_conv.patch` (P5, behavior-neutral staging seam calling `_VTL_CONV_STAGE`).

---

## Fork A — `flash_attn.py` (P3): tree custom_mask over the verify forward

**Where:** `FlashAttentionMetadataBuilder.build()` and `FlashAttentionImpl.forward()` (the verify /
`max_query_len>1` decode path).

**What:**
1. In `build()`, when the runner exposes a branching carrier (`md.vtl_tree`, not a chain), call
   `tree_attention.build_full_mask_rows(carrier, seq_lens)` and flatten the per-request rows into the
   backend `custom_mask` buffer alongside the existing `qo_indptr`/`kv_indices`.
   - `qo_indptr = arange(bs+1) * num_nodes` (each request contributes `num_nodes` queries).
   - draft KV is a `block_size=1` region appended after the committed KV; `kv_indices` must index it.
   - **ON-BOX #2:** confirm the flashinfer prefill wrapper's `custom_mask` dtype/packing (bool,
     row-major `[sum_i num_nodes_i * (seq_len_i+num_nodes_i)]`) and the page-index permutation for the
     block_size=1 draft region. SGLang's `USE_FULL_MASK` path (`ngram_worker.py:297-316`,
     `ngram_info.py:62-117`) is the reference.
2. In `forward()`, select the custom-mask prefill wrapper when the tree mask is present; else stock.
3. **ON-BOX #4:** the verify token-count per request is `num_nodes` (padded to `max_nodes`). Confirm
   the `num_decode_draft_tokens` cudagraph family (`gpu_model_runner.py:2203`) captures this static
   shape; pad to fixed `max_nodes` so replay is shape-stable.

**Correctness anchor:** `tree_attention.tree_attention_ref` == cascade merge (unit-tested). On-box,
the greedy-equality master gate catches any masking error.

---

## Fork B — `gpu_model_runner.py` (P2 scheduling + P5 orchestration)

**B1. Tree-node scheduling (P2, `_calc_spec_decode_metadata:2798` + `propose_draft_token_ids:4913`).**
For a branching carrier the per-request verify token count is `num_nodes`, not `draft+1`. Set
`num_scheduled_tokens[req] = num_nodes` and build `cu_num_draft_tokens` / `target_logits_indices`
over the tree node order (the same order `tree_runtime.build_carrier` uses). For a chain this is
identical to stock (node_count == draft+1) → **width-1 stays bit-identical (the P2 gate).**
- **ON-BOX #3:** confirm the `propose` closure (`:4943`) runs before the next step's attn-metadata
  freeze; if not, produce the carrier one step ahead (also what P7 wants).

**B2. Conv-stage activation + commit (P5).** Around the verify forward:
```
from vtl.conv_commit import ConvStager
import vllm.model_executor.layers.mamba.short_conv as sc
stager = ConvStager()
sc._VTL_CONV_STAGE = stager.stage           # activate the short_conv seam (else None = stock)
try:
    ... run target verify forward ...       # short_conv stages Bx, updates scratch clones
finally:
    sc._VTL_CONV_STAGE = None
... sampler returns accepted paths ...       # accept_index -> accepted node ids per request
for key, committed in stager.commit_all(accepted_ids, scratch_by_layer_fn, init_windows_fn, state_len=L_cache-1):
    ... write `committed` into that layer's persistent conv_state ...   # ON-BOX #5 (align write)
stager.reset()
```
- `accepted_ids` come from `tree_verify.accept` / the sampler output (node ids of the accepted path).
- **ON-BOX #5:** the write of `committed` into the persistent conv cache follows `mamba_cache_mode=
  align`'s block-migration contract (length = `eff_len`, supplied by `conv_commit_window`). Confirm
  the mamba cache manager's write path.

**Correctness anchor:** `conv_commit.commit_conv_states` == spike (unit-tested). Master gate:
multi-turn greedy-equality (conv corruption only surfaces after a rejection several turns in — P1.5).

---

## Fork C — P6 corpus upgrade (optional, acceptance-gated)

**Status: deferred by design.** The plan builds P6 "only if acceptance rate justifies it," which is
an on-box measurement (`bench/replay.py` acceptance_rate, now available from P0). The current
`csrc/ngram_tree/ngram_tree.cpp` is a drop-in mirror of the tested Python drafter. Upgrading to a
cross-request LRU-versioned trie / suffix automaton (SGLang `cpp_ngram`) requires a matching Python
reference to preserve the drop-in/testable contract. **Do not build until P4 acceptance data shows
the drafter — not the verify path — is the acceptance ceiling.** If it does: mirror the new matcher
in Python first (extend `vtl/ngram_tree.py` with a self-check), then port to C++ 1:1.

## Fork D — P7 overlap scheduler (profiling-gated)

**Status: largely provided by vLLM.** vLLM's `AsyncScheduler` (`async_scheduler.py`) +
`engine/core.py:519 step_with_batch_queue` already overlap CPU prep with the GPU forward, and
`propose()` already runs one step behind. The only bespoke work is producing the carrier
(`build_carrier`) into **pinned, req-indexed buffers** so the tree CPU work (`flatten_batch_trees` +
`reconstruct_indices`) overlaps the next forward instead of stalling it, plus making the one-step-
ahead placeholder carry tree node count (Fork B1 already threads node count). **Build only if P0/P4
profiling shows a CPU-bound gap between forwards** — measure first (the plan's explicit gate). If it
does: pre-allocate the carrier tensors once (sized `[max_bs, max_nodes]`), have the metadata hook
write into them in place (no per-step alloc), and confirm `AsyncScheduler` is active (default on).

## Master gate (every on-box phase)
Greedy spec output == greedy no-spec output, token-for-token, over a multi-turn trace slice. Any
divergence localizes to the phase just enabled. Run the A/B matrix (`bench/replay.py`, stats-enabled
server) to confirm acceptance_rate rises and TPOT drops without a correctness regression.
