# Future work: chunk-parallel GDN scan on H200

Status: **scaffold implemented, perf body pending.** The op, wiring, flags, and parity test now
exist as a default-off sequential baseline (see "Wiring & gates" below); the chunk-parallel
tensor-core body — the only part that can actually win — is NOT written. Opt-in, profile-gated:
attempt the parallel body only if an on-box profile shows the GDN prefill scan is a top cost AND
you have time to beat a hand-tuned incumbent. High risk of no win.

**What ships today (fail-closed, default OFF):**
- `vllm_cuda::gdn_chunk_scan` — a SEQUENTIAL per-token recurrence (`vtl/csrc/gdn/chunk_scan.cu`).
  Correctness-first; will NOT beat the chunked incumbent on long prefills. It is the parity
  reference and the base to build the WY body on.
- `vtl/patches/gdn_prefill_backend.py` — routes the prefill scan to that op behind
  `VTL_ENABLE_GDN_KERNELS` + `VTL_GDN_CHUNK_SCAN`, and pins the stock backend for A/B via
  `VTL_GDN_PREFILL_BACKEND ∈ {stock,flashinfer,cutedsl,triton}`.
- `bench/test_gdn_chunk_scan.py` — parity vs a pure-torch oracle (wired into `make test-kernel`).
- Both compose files expose `VTL_GDN_PREFILL_BACKEND` + `VTL_GDN_CHUNK_SCAN` at their defaults.

**Remaining (the actual win):** on-box Phase-0 profile; A/B the stock backends; then the
CUTLASS/CuteDSL chunk-parallel WY body below (raw `wmma`/`mma.sync` fallback in
`docs/improvements/gdn-scan-raw-mma.md`).

## Why this is hard (read first)

The GDN prefill scan on the served qwen3_5 (18/24 layers) is already handled by a mature,
SM90-tuned kernel: vLLM selects **FlashInfer `chunk_gated_delta_rule`** on H200 (device
capability 9.0), with **FLA Triton** and **CuteDSL** as alternates, chosen by
`_resolve_gdn_prefill_backend`. Decode uses FLA Triton `fused_recurrent_gated_delta_rule_packed_decode`;
the conv is Triton `causal_conv1d`. These use tensor-core GEMMs and the chunk-parallel
WY-representation algorithm.

The reverted "gemm gdn optimization" (commit `c8ebc58`) did NOT do this. Its
`chunk_gated_delta.cu` was a **sequential per-token fp32 scalar recurrence** (grid `(S,H)`,
128-thread blocks, state streamed from HBM every token, no tensor cores), and it was wired
**inert** (a pass-through TODO) so it never even ran. A sequential scan cannot beat a
chunk-parallel tensor-core kernel on long prefills. Do not resurrect it.

To have any chance, a custom scan must be **chunk-parallel and tensor-core-based**, i.e. a real
reimplementation of the gated-delta-rule chunked algorithm — not a faster loop.

## The algorithm to implement (gated delta rule, chunked / WY form)

Per (sequence, head), state `S ∈ R^{Dk×Dv}` (Dk=Dv=head_dim=128). Split the sequence into
chunks of `C` tokens (e.g. C=64). Two levels:

1. **Intra-chunk (parallel across the C tokens), tensor-core GEMMs:**
   - Apply the per-token log-decay `g_t` cumulatively within the chunk (`exp` of a prefix sum).
   - Build the strictly-lower-triangular `A = tril(diag(β) · K Kᵀ, -1)` (a `C×C` GEMM).
   - Solve `T = (I − A)^{-1}` (unit-lower-triangular inverse — `solve_tril`, blocked forward
     substitution; can be done with small GEMMs).
   - Form the WY factors `W = T·diag(β)·K`, `U = T·diag(β)·V`, then the intra-chunk output
     `O_intra = tril(Q Kᵀ) · V'` and the chunk's state contribution `ΔS = Kᵀ·U` — all GEMMs.
2. **Inter-chunk (sequential over chunks):** carry `S` across chunks with the chunk decay,
   `O += Q · S_carried`, `S = decay·S + ΔS`. Only this level is serial, and it is over
   `L/C` chunks, not `L` tokens.

Reference math to match exactly: vLLM's FLA `chunk_gated_delta_rule` (Triton) and the
`fused_recurrent` decode step — reconcile the l2norm-of-q/k, the gate convention (per-step vs
cumulative log-decay), and the `beta` placement on the box before trusting parity.

## H200 implementation notes

- **Tensor cores / MMA:** use bf16 `wmma`/`mma.sync` (or CUTLASS/CuteDSL) for the `C×C` and
  `C×Dk`/`C×Dv` GEMMs. This is the whole point — scalar FMA loops will lose.
- **State resident in shared memory / registers** across the chunk's GEMMs; only spill `S` to
  HBM at chunk boundaries. The reverted kernel's fatal flaw was re-reading `S` from HBM every
  token.
- **One block per (sequence, head)** (or per (sequence, head, chunk-stage)); pick `C` so a
  chunk's tiles fit in shared memory at high occupancy on 132 SMs.
- **Coalesced varlen loads** via `query_start_loc`; handle the initial-state seed and final-state
  writeback like the incumbent.
- Consider just calling **CuteDSL** `chunk_gated_delta_rule_cutedsl` or tuning the existing
  FlashInfer path (autotune configs) before writing MMA by hand — often a better ROI.

## Wiring & gates (when/if implemented)

- Op `vllm_cuda::gdn_chunk_scan(...)`, own op (parity-testable side-by-side) —
  `vtl/csrc/gdn/chunk_scan.cu`, registered in `vtl/csrc/torch_bindings.cpp`. **Done** (sequential
  body). The WY-parallel version replaces the body under the same op name (the aspirational
  `_parallel` suffix was dropped — one op, evolving body). `Dk=Dv=128`, `[L,H,D]` packed varlen.
- Routed via `ChunkGatedDeltaRule.forward_cuda` behind `VTL_ENABLE_GDN_KERNELS` +
  `VTL_GDN_CHUNK_SCAN` (fail-closed: stock unless armed) — `vtl/patches/gdn_prefill_backend.py`.
  **Done.** The shim bails to stock on any layout/convention mismatch (esp. the `g` gate
  convention, which must be reconciled on the box).
- Bench gate: `bench/test_gdn_chunk_scan.py` (in `make test-kernel`/`bench-kernel`) — parity vs a
  pure-torch oracle **done**; the definitive cross-check vs FLA/FlashInfer + the perf win are the
  on-box A/B. Only enable after it BOTH passes parity AND wins the micro-bench + trace replay.

## Bar

Beat FlashInfer `chunk_gated_delta_rule` on H200 at the served shapes. If a prototype does not
clearly win in `make bench-kernel`, keep stock — a correct-but-slower custom scan is worse than
nothing (it adds maintenance and risk for zero latency benefit).
