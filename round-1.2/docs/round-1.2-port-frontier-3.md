# Port frontier #3 — what's left after tokstore + in-graph ladder (74.55)

Investigated 2026-07-30 at score 74.55 (TTFT 31/44, TBT 3ms, failed 8). Everything in
`round-1.2-next-ports-plan.md` has SHIPPED and is gated on in compose (tokstore, rust
hasher, fold_t1, unroll, N=1 in-graph sampling, step0 EOS ban). No shadow arm costs
time. Engagement counters were removed in 741d7e6, so burst engagement is only
inferable from TBT.

Structural fact that ranks everything: async scheduling means step k+1's
schedule+execute+sample are issued before step k's result — the ONLY host work
serialized behind the GPU is what runs after `copy_event.synchronize()`
(core.py:590 → rust_sched.py:1536). Everything else is overlapped, but still matters
because decode (batch mean 2) is host-bound.

## Ranked next moves

### 0. Not ports, but the biggest quantified points

- **8 failures (+1.4pt)**: step0_eos_ban (empty stream from step-0 `<|im_end|>` under
  the int4 lm_head) shipped in 8015b81, but the 74.55 run still shows failed=8 —
  either the scored image predates the ban or a residual mode fired (different stop
  token first / `is_prefilling_np` gap). Gate #1: next scored run's logs. If failures
  persist: revert `VTL_LM_HEAD_QUANT` int4 → bf16/fp8 (root cause; trades a little
  TPOT for +1.4pt).
- **Keep-alive 60→600**: `VLLM_HTTP_TIMEOUT_KEEP_ALIVE: "60"` (compose:154), trace
  inter-turn gaps tail past 30s; any gap >60s reproduces the old idle-close race
  (the previous 5-failure mode). One-line literal edit, zero risk.
- **Graph coverage 82→99% (declined in 741d7e6, reconsider)**: 17% of decoded tokens
  land at batch 3/5/6, pad out of graph-mode bursts AND out of in-graph sampling,
  falling back to the numpy V2 sampler (nstep_decode.py:112 BURST_SIZES + compose
  cudagraph_capture_sizes). Risk is unvalidated FULL captures at new sizes — needs a
  run that can afford to fail, but it's the largest TBT item that needs no new code.

### 1. Rust port — critical-path update loop collapse (items 4.4+4.5)

`update_from_output` runs TWO per-request Python loops over the same
`req_id_to_index` after the blocking sync: decide() packability loop
(rust_sched.py:1713-1800, ~20 ops/req incl. the 6-clause packability check and a
`_Row` alloc) then r8_apply() (rust_sched.py:1865-1911, another `_Row` alloc +
`_urwo_inner` + **one `cache_blocks` FFI per request** — the last per-request FFI on
the critical path). All the per-request facts checked are admission-immutable; the
interning pattern already exists (`mirror._stops`, rust_sched.py:1767-1772).
Port: intern packability + tok_on at admission into the resident table, fold
cache_blocks into `update_step_pack_np` (it already receives everything needed), and
collapse both loops into the single existing FFI crossing. Est 0.03-0.1ms TPOT.

### 2. Rust port — output publish from Rust (items 5.1+5.2+4.7)

The finished shm record bytes are ALREADY produced inside the crate
(update_step_pack_np). They then travel: Python return → `output_queue.put_nowait`
(queue.Queue lock + notify, core.py:1306) → GIL handoff to the Python shm thread
(shm_ipc.py:588) → Python loan/memmove/send/notify (shm_ipc.py:601-620). Port: the
crate publishes the record directly (iceoryx2 publisher in Rust, or from the
busy-loop thread without the queue hop). Kills a GIL-dependent thread hop on the
critical path; the 200µs GIL switch tune exists solely to soften this hop.
Est 0.05-0.2ms TBT (the hop's latency is bounded below by GIL switch interval).

### 3. Rust port — schedule marshalling round trip (items 1.4/1.7/1.8/1.9/2.3)

Rust decides, then Python re-shapes: flat arrays → per-request `RustBlocks(tuple)`
(rust_sched.py:2474-2487) → `_make_cached_request_data` (6 lists + set + dataclass,
scheduler.py:1256-1313) → SchedulerOutput → runner `update_requests` re-iterates it
all (model_runner.py:813-825) + `_update_after_schedule` per-request arithmetic that
already exists in Rust (`sched.rs::advance`). Overlapped, not serialized — but the
single biggest surviving pure-Python block per step. Port both ends together: runner
reads the flat decision buffers directly, skipping the object round trip. Bigger diff
than #1/#2; do it after them.

### 4. CUDA — prepare_attn iteration-1 into the graph (item 2.12)

`_fast_model_attn` (decode_fastpath.py:332-345) is the densest remaining launch
cluster (~7 launches + a fresh tensor from `get_scheduler_metadata` per step). Burst
iterations 2..N already capture exactly this (`_burst_body`); iteration 1 runs it
host-side because it precedes the FULL replay. Fold it + the replay into one capture
(extend the existing prologue). Only pays on non-burst steps and burst step 1.

### 5. One-liners (each trivial, do opportunistically)

- nstep `_emit` rebuilds `{req_id: i}` every step (nstep_decode.py:943) — the
  `_vtl_r2i` identity cache from hotpath_microopt exists but is not wired here.
- `model_inputs` dict built then ignored on every FULL-graph step
  (model_runner.py:1256-1264) — guard it like microopt #2.
- commit_burst per-request 14-attribute gate (rust_sched.py:1101-1155) — intern into
  the resident table.
- `AsyncOutput.__init__` allocates a `torch.cuda.Event` per step — pool it.

## Admission/TTFT side (low priority: 1ms TTFT = 0.0021 ERS vs TPOT 0.0487)

- msgpack ADD decode materializes ~2400 token ids as a Python list, then THREE more
  full copies (`prompt_token_ids.copy()`, hasher `tokens[:end]` slice, runner
  `stage_write` extend). A raw ADD record / numpy handoff (mirror of TAG_RAW)
  collapses all of them.
- Rust frontend re-encodes the full growing context (~2250-4400 tok) from scratch
  every turn on 2 worker threads; `Prompt::TokenIds` pass-through already exists —
  a per-conversation prefix→ids cache would encode only the ~150-token suffix.

## What is NOT worth re-planning

Dead/done: Python detokenizer & OutputProcessor (Rust frontend owns them),
msgspec_stream/msgspec_json patches (inert under rust frontend), V2 sampler on
graph-covered steps, tolist()s, Python hasher, pin_memory, ZMQ/msgpack output on the
R8 path. NOTHING per-token remains in Python — per-token work lives only in the Rust
frontend (incremental detok + per-token SSE).
