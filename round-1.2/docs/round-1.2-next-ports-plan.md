# Next ports: in-graph sampling (CUDA) + Rust token-bookkeeping move

Investigated 2026-07-30 at score 74.03 (TTFT 32/45, TBT 3ms, 8 failed). Context: the
trace-verified burst math says a working N=4 burst puts TBT at ~1.8ms; the observed 3ms
means the burst is dormant on the judge box, and every off-box-checkable cause has been
eliminated (commit gates, schedule_supported, decisions-dict shape, shadow inertness).
Both ports below are sized AS IF the burst works; each also states its value if it does
not. Neither replaces the burst diagnosis — logs from the next scored run remain gate #1.

## Port 1 — CUDA: lm_head+argmax in-graph, token-1 fold-in, full unroll

### What is ALREADY done (do not re-plan it)

The shipped burst graph (`nstep_decode._burst_body`, captured by `_capture_burst_graphs`)
already contains `compute_logits` (the CUTLASS W4A8 head) and `torch.argmax` **inside the
capture** for iterations 2..N, validated per boot by `_graph_matches_eager` (token-level
compare, drops to eager on mismatch). The plan's "hard part #1 — W4A8 lm_head in-graph
unproven" is therefore already implemented; what is unproven is only whether **capture
succeeds on the H200** — probe A in `bench/test_nstep_capture_probe.py` answers that, and
the mode ladder (graph→eager→off) makes a FAIL safe.

### Remaining work, ranked

1. **Token-1 fold-in.** `_run_burst:468-470` still host-launches `compute_logits` +
   `argmax` once per burst for the first token. Both read
   `cudagraph_manager.hidden_states[:rows]` — a persistent buffer (verified:
   `cudagraph_utils.run_fullgraph` returns a slice of `self.hidden_states`), so the pair
   is capturable as a burst-graph prologue. Combined with (2), `_run_burst` collapses to:
   idx_map copy, step zero, ONE `graph.replay()`, accum fill. Saves ~2 dispatches +
   slice allocs per burst.
2. **Full N-unroll.** 3 replays → 1 graph containing the prologue + (N-1) bodies.
   Probe C prices exactly this (expected ~0.05-0.1ms per burst for the 3 extra replays,
   i.e. ~0.01-0.02ms/token). Build only if probe C on the H200 comes in high; the
   descriptor cost is +1 graph per burst size (4 sizes), same memory pool.
3. **N=1 in-graph greedy sampling (the generalization that pays even if bursts stay
   rare).** Capture forward+logits+argmax per decode size and commit it from the
   scheduler exactly like the burst (the per-request greedy checks in
   `burst_request_blocked` are already computed every step; an N=1 commit needs no align
   gate and no queue-empty gate). Every plain greedy decode step then skips the V2
   sampler entirely (its 7 numpy gathers + `new_ones` alloc + SamplerOutput build) and
   replaces 3 host dispatch points with one replay. Estimated 0.03-0.1ms/step — the only
   port here that is robust to the burst staying dormant. Interacts with nothing: the
   step-0 EOS ban lives on prefill steps, which keep the stock sampler.

Expected combined: ~0.1-0.2ms TPOT (≈1-1.7pt) if bursts stay dormant; ~0.03-0.05ms
(≈0.3-0.4pt) if bursts fire (items 1-2 then mostly overlap with what the burst already
amortizes). ALL items gate on probes A/B passing — run `make test-kernel` on the H200
before writing any of it.

## Port 2 — Rust: full token-bookkeeping move

### The decisive fact (2026-07-30 consumer sweep, full inventory in the session)

In the deployed config (V2 runner, Rust frontend, R8, UFO, Rust hasher, no PP / LoRA /
spec / connector / structured output), the live consumers of
`Request._all_token_ids` / `_output_token_ids` reduce to:

- **Actual ints, 2 consumers only:**
  1. `NewRequestData.prefill_token_ids` (rust_sched.py:1937 → model_runner:807 →
     StagedWriteTensor) — full list, once per **admission**; for a RESUMED request the
     tail beyond the prompt is the regenerated output (the preemption-resume re-prefill).
  2. The block hasher (rust_sched.py:2044-2061) — block-aligned `ConstantList` slices,
     per step, but early-returns `[]` on most steps.
- **Counts only, everything else:** `num_tokens` / `num_output_tokens` at ~10 sites
  (pack_req, burst_commit, `_update_after_schedule:1255`, n_out/n_tok for update_step,
  CachedRequestData.num_output_tokens whose only reader is config-gated off).
- **Sole writer:** the single bulk `append_output_token_ids` at rust_sched.py:1582.
- `output_token_ids` (the ConstantList view) has **zero** live readers. Stop-checking
  reads the token tail only in Rust (`update.rs`).

### Design

Rust `update.rs` gains a per-slot token store: `update_step_pack` (which already receives
every sampled token every step) appends to the store, runs the block-hash catch-up itself
(`hash.rs` has the exact ported hasher), and maintains `num_tokens`/`num_output_tokens`
as slot state instead of per-call inputs. The crate already carries the `numpy` dep
(python feature), so the same change accepts the sampled ids as `PyReadonlyArray2<i64>` +
per-row counts — deleting the two `tolist()` calls in `AsyncOutput.get_output` (vtl
monkeypatch, no fork rebuild) and the per-request `toks.extend` loop in `decide()`.

Python `Request` degrades to counters after registration: `num_tokens` /
`num_output_tokens` become plain ints bumped by the (single) writer call;
`_all_token_ids` stops growing. The two int consumers are handled by materialization:

- **Admission**: prompt list is still fully present in Python — nothing changes.
- **Preemption-resume / any pack-refusal fallback**: a `slot_tokens(slot)` FFI rebuilds
  `_all_token_ids` before delegating to the stock path. Rare by construction (the scored
  trace never preempts; refusal reasons are the existing 6-clause gate).
- **np.int64 hazard (the reason tolist() couldn't die alone)**: block-hash input is
  pickled; np.int64 pickles differently from int and silently corrupts cache keys. With
  hashing moved fully into Rust this hazard disappears — ints never round-trip through
  numpy into the hasher. Any Python materialization MUST go through `int()` coercion
  (slot_tokens returns Python ints from Rust, which is automatic via PyO3).

### Gates and effort

Ship exactly like R8: golden vectors shared Python/Rust, a `_SHADOW` flag that keeps
Python authoritative and byte-compares (hashes + counts + record) per step, cargo tests
for the store + resume-materialization, `make check` self-checks for the facade. Est.
2-3 days including the soak. Expected: 0.05-0.15ms TPOT (≈0.4-1.3pt) — it deletes the
tolist()s, the decide() token loop, the per-step hasher invocation, and the counter
appends, and it shrinks the once-per-burst residue that caps the burst's payoff.

## Order of operations

1. Next scored run's logs answer the burst question (engagement counters are already in).
2. `make test-kernel` on the H200 → probes A/B/C.
3. If bursts dormant for a graph/capture reason → fix per probe results (slab fallback
   etc.) BEFORE either port; the burst is ~9pt, these are ~1pt each.
4. Then Port 2 (works either way), then Port 1 item 3, then items 1-2 only if probe C
   says replays are expensive.
