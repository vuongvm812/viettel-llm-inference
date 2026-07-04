# P2 Plan — llama.cpp FFI, single sequence

Ref: `docs/ROADMAP.md` §P2, `docs/design/inference-backend/design.md`,
`docs/design/text-processing/design.md`, `docs/design/fast-loop/design.md`.

## Scope

Real GPU inference, **one request at a time**. Acceleration gated by OS:
**Metal on macOS, CUDA on Linux** (`llama-cpp-2` target-specific features).
Continuous batching (P3), shared-prefix KV (P4) stay out — single seq,
`clear_kv_cache()` between requests.

## Environment reality

No `cmake`, no GGUF model, no GPU in the dev sandbox → the real backend
(`--features llama`) **compiles/runs only on the target box**. So the backend
is behind a `llama` Cargo feature; the **mock is the default build** and stays
green in CI/macOS-without-model. This mirrors the P0 `lossy` feature-gate.

## Verifiable-now vs target-only

| Part | Verifiable here | How |
|---|---|---|
| Multi-byte detokenize handoff (slab `out_bytes`) | ✅ | mock emits multi-byte UTF-8, TDD |
| Partial-UTF-8-safe streaming (Core 0) | ✅ | canned reply carries a split code point |
| llama.cpp tokenize/detok/prefill/greedy-decode | ❌ target only | written vs `llama-cpp-2` 0.1.150 API, `#[cfg(feature="llama")]` |

**P2 exit criterion (determinism vs `llama-cli`)** is a checked-in target-only
artifact: `script/verify-determinism.sh` builds `--features llama`, runs one fixed
request through the runtime, runs `llama-cli` greedy on the same GGUF, and
byte-compares. Run it on the GPU box (`ROADMAP.md` §P2).

## Design

**Egress handoff (mandatory: a multi-byte piece can't ride a `u32`).**
- `RequestState.out_bytes: Vec<u8>` (pre-reserved) accumulates detok pieces.
  `out_committed: usize` marks bytes that form complete UTF-8.
- R3 `Token(id)` = token id (Core 2 → Core 1). R4 `Piece(delta)` = count of
  newly-committed complete-UTF-8 bytes (Core 1 → Core 0). Distinct variants so
  the two ring meanings can't be confused.
- Core 0 reads the piece via `Slab::read_committed(slot, cursor, delta)`, which
  builds the slice from the buffer's fixed heap base (captured once at
  construction) — not from a `&RequestState` — so the read forms no reference
  into the cell Core 1 may hold `&mut` on. Sound because `out_bytes` is
  pre-reserved and never reallocates (base stable); Core 1 guards against
  exceeding capacity. Core 0 resets its per-slot cursor on claim.

**Backend seam (static dispatch, no `dyn`).** Two cfg-selected concrete types:
- `TextBackend` (Core 1): mock = `id as u8`; llama = `str_to_token` /
  `token_to_bytes` over `&'static LlamaModel`.
- `Decoder` (Core 2): mock = canned bytes; llama = prefill + greedy argmax
  loop over an owned `LlamaContext`, `clear_kv_cache()` on finish.
- Model + backend are `Box::leak`'d to `'static` (process-lifetime singleton),
  so `LlamaContext<'static>` is owned by Core 2 with no self-reference. Shared
  read-only to Core 1. `Send/Sync` asserted via audited newtype.

## Steps (TDD)

1. Slab: add `out_bytes` + `out_committed`; reset clears. **RED** a
   round-trip/commit test.
2. Core 1: unify detok to `append + commit-complete-UTF-8 + emit Token(delta)`;
   mock `id→byte`. Multi-byte canned reply exercises partial UTF-8.
3. Core 0: egress reads `out_bytes[cursor..+delta]` via saved base ptr.
4. Update pipeline + HTTP tests for the new handoff. All mock tests green.
5. `llama.rs` (`#[cfg(feature="llama")]`): backend/model load, `TextBackend`
   + `Decoder` real impls. Target-compiled.
6. Cargo.toml: optional `llama-cpp-2`, OS-gated `metal`/`cuda`.
