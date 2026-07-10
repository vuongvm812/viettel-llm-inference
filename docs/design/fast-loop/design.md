# Design — Core 2: The Fast Loop (Scheduler + Dynamic Batcher + KV Monitor)

Pinned to **Core 2**. `BusySpin`. Owns the mutable `LlamaContext` (KV cache). This is the
ultra-low-latency loop that drives the GPU — the reason we accept burning a whole core.

## Responsibilities

- Admit tokenized requests (R2) into a running set, subject to KV memory.
- Set up shared-prefix KV, prefill, and run **continuous (iteration-level) batching**.
- Each iteration: build one `llama_batch` over active seqs → `decode()` (GPU) → sample per seq
  → emit tokens on R3 → retire finished seqs.
- Be the **GPU/KV monitor**: track free KV cells and gate admission (prevents OOM = the
  concrete job of the user's "GPU Monitor").

## "No blocking/lock" clarification

`decode()` blocks Core 2 while the GPU runs — that's *the work*, not lock contention. The
"no blocking / no lock" rule targets inter-core coordination, which the Disruptor rings
satisfy (no mutex between cores). Core 2 never waits on a peer; it waits on the GPU.

## Scheduler state

```rust
struct Scheduler {
    pending:   VecDeque<u32>,        // slots tokenized but not yet admitted (KV-limited)
    active:    Vec<ActiveSeq>,       // currently decoding
    prefix:    PrefixCache,          // shared-prefix → cached seq_id + len
    ctx:       LlamaContext,         // KV cache; single-threaded here
    free_seq:  SeqIdPool,            // reusable llama seq ids
    kv:        KvMonitor,            // free cells / capacity accounting
}

struct ActiveSeq {
    slot: u32, seq_id: i32,
    n_prompt: u32, n_generated: u32, max_tokens: u32,
    sampler: SamplerState,           // per-seq (seed/temp) for determinism
}
```

## The loop

```rust
core_affinity::set_for_current(core_2);          // no-op on macOS
loop {
    // 1. Ingest newly tokenized requests (non-blocking drain of R2)
    if let Ok(mut g) = r2_poller.take(ADMIT_BATCH) {
        for ev in &mut g { pending.push_back(ev.slot); }
    }

    // 2. Admit from pending while KV budget allows
    while let Some(slot) = kv.can_admit(pending.front()) {
        admit(slot);                 // prefix-share + prefill (below); pending.pop_front()
    }

    // 3. One decode step over ALL active seqs (continuous batching)
    if !active.is_empty() {
        batch.clear();
        for a in &active { batch.add(a.next_input_token, a.pos, a.seq_id, /*logits=*/true); }
        ctx.decode(&mut batch);                    // GPU forward pass (the blocking work)

        // 4. Sample one token per seq, emit, retire
        let mut burst = SmallVec::new();
        for a in &mut active {
            let tok = a.sampler.sample(ctx.logits_for(a.seq_id));
            a.n_generated += 1; a.pos += 1;
            burst.push(RingEvent{slot: a.slot, kind: Token(tok)});
            if tok == eos || a.n_generated >= a.max_tokens {
                r3_producer.publish(|e| *e = RingEvent{slot: a.slot, kind: Finish(reason)});
                ctx.kv_cache_seq_rm(a.seq_id);     // free KV cells
                free_seq.release(a.seq_id); kv.on_release(a);
                mark_retired(a);
            }
        }
        r3_producer.batch_publish(burst.len(), |it| for (e,ev) in it.zip(burst) { *e = ev; });
        compact_active();            // drop retired
    }
    // idle → busy spin (crate BusySpin semantics if using a managed handler)
}
```

- **Continuous batching**: new seqs join the very next `decode()`; finished seqs leave
  immediately. No static batch window — this is the vLLM-style throughput win (P3).
- **Dynamic batcher**: cap the batch by `MAX_BATCH_SEQS`; prefill and decode share one
  `decode()` batch (**chunked prefill**, P7 — below).

## Chunked prefill / unified batching (P7)

P3 prefilled a whole prompt inside `admit()` before the sequence joined the running set, so a
long 40K-token prompt head-of-line blocked every active decode — the trace stayed at 1 seq per
`decode()`. P7 splits a sequence into two phases:

- **Prefilling** (`prefill_pos < n_prompt`): `admit()` reserves KV + copies any shared-prefix
  cells but does **not** run the prompt. Each `step()` prefills a chunk of `tokens[prefill_pos..]`.
- **Decoding** (`prefill_pos == n_prompt`): generates one token per step.

Each `step()` builds **one** `llama_batch` mixing one decode token per Decoding seq with up to
`step_prefill_budget(n_decode, n_batch, max_batch_tokens)` prompt tokens drawn from Prefilling
seqs, then a single `decode()`. `max_batch_tokens` is the **per-step prefill budget** (not an
admission gate). A Prefilling seq that completes its prompt this step samples its first token and
becomes Decoding next step. Decode never stalls behind a long prefill, and many prompts prefill
concurrently — the throughput win the long-prompt trace needed.

## Admission = GPU/KV monitor

`KvMonitor` tracks free KV cells vs `n_ctx` capacity. `can_admit(slot)` returns the slot only
if the request's prompt (minus shared prefix) + its `max_tokens` fits the remaining KV budget.
This is admission control: never start a seq we can't finish → no mid-generation OOM. `n_ctx`
and `MAX_IN_FLIGHT`/`MAX_BATCH_SEQS` are the tuning knobs (KV memory bounds batch size, exactly
as in vLLM).

## Shared-prefix radix cache (P7, generalizes P4)

A **token radix trie** (`backend/prefix_trie.rs`) caches arbitrary, variable-length, nested
shared prefixes — the generalization of P4's single fixed-K prefix. On `admit(slot)`:

1. `trie.insert_structural(tokens)` records the request's path so future requests can fork
   against it. `trie.longest_match(tokens)` returns the deepest *ready* materialized prefix node
   → `reused` cells; `trie.deepest_unmaterialized_fork(tokens)` is the shared-prefix boundary
   (a trie fork of ≥ 2 requests) worth caching next.
2. Share: `ctx.copy_kv_cache_seq(match_node.seq, new_seq, 0..reused)` — copy the matched prefix's
   cells into the new sequence (no recompute of the shared system prompt).
3. Establish: if a fork deeper than `reused` is worth caching (≥ `shared_prefix_tokens` and the
   cache has room), materialize it; when this request's prefill completes it donates its
   `0..depth` cells to the fork's prefix seq and marks the node `ready` (deferred matching
   requests then share it — the readiness gate prevents copying an unmaterialized prefix).
4. Prefill only the **suffix** `tokens[reused..]` (chunked, above).

Bounded to `MAX_PREFIX_SEQS` materialized nodes; idle LRU eviction reclaims cold prefixes. For the
trace, all requests share one system prompt → one fork node, prefilled once. Ceiling: llama's KV is
copy-based (not block-ref-counted), so nested nodes don't physically dedup — see the module doc.
See `inference-backend/design.md` for the exact llama.cpp KV ops.

## Determinism

Greedy sampling with `temp=0` (argmax) reproduces vLLM's deterministic decoding; `seed=42`
carried per seq for any stochastic path. Per-seq `SamplerState` avoids cross-request coupling.

## Ordering / correctness

- Tokens for a given slot are emitted on R3 **in generation order** (single producer, one seq
  advances one token/iter) → Core 1/Core 0 preserve order without sorting.
- `kv_cache_seq_rm` must run before the seq id is reused (`free_seq.release` after rm).
- `demo()` self-check (mock decode): admit 3 seqs sharing a prefix, assert prefix prefilled
  once (token counter), each seq emits `max_tokens` then `Finish`, all KV released.
