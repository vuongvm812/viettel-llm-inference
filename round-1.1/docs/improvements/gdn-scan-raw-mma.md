# Fallback vehicle: raw `wmma`/`mma.sync` for the GDN chunk-parallel scan

Status: **not implemented.** This is the escape hatch if the primary vehicle
(CUTLASS collective MMA / extending vLLM's `chunk_gated_delta_rule_cutedsl`,
per `docs/plans/gdn-parallel-scan.md`) cannot hit the served shapes at high enough
occupancy. Reach for it only after the CUTLASS/CuteDSL prototype has been tried and
measured — hand-written PTX is the most code and the most risk, for a win the A/B
must still prove.

## Where it plugs in

Same op, same wiring — only the kernel *body* changes:

- Op `vllm_cuda::gdn_chunk_scan` (`vtl/csrc/gdn/chunk_scan.cu`), routed by
  `vtl/patches/gdn_prefill_backend.py` behind `VTL_ENABLE_GDN_KERNELS` +
  `VTL_GDN_CHUNK_SCAN`. Signature and parity test (`bench/test_gdn_chunk_scan.py`)
  are unchanged; the raw-MMA version just replaces the sequential recurrence with the
  chunk-parallel WY body under the same op identity.

## What the body has to compute (chunked WY, `docs/plans/gdn-parallel-scan.md:25-43`)

Per (sequence, head), state `S ∈ R^{Dk×Dv}` with `Dk=Dv=128`. Chunk the sequence into
`C` tokens (start with `C=64`):

1. **Intra-chunk (parallel across the C tokens), tensor-core GEMMs:**
   - cumulative log-decay `g_t` within the chunk (`exp` of a prefix sum),
   - `A = tril(diag(β)·K Kᵀ, -1)` — a `C×C` GEMM,
   - `T = (I − A)^{-1}` — unit-lower-triangular inverse (`solve_tril`, blocked forward
     substitution via small GEMMs),
   - WY factors `W = T·diag(β)·K`, `U = T·diag(β)·V`,
   - intra-chunk output `O_intra = tril(Q Kᵀ)·V'` and state delta `ΔS = Kᵀ·U`.
2. **Inter-chunk (serial over `L/C` chunks):** carry `S` with the chunk decay,
   `O += Q·S_carried`, `S = decay·S + ΔS`.

## H200 raw-MMA notes (`gdn-parallel-scan.md:46-57`)

- **MMA:** bf16 `wmma`/`mma.sync` for the `C×C` and `C×Dk`/`C×Dv` GEMMs. Scalar FMA
  loops lose — that is the entire reason to leave the sequential baseline.
- **State resident in shared memory / registers** across the chunk's GEMMs; spill `S`
  to HBM only at chunk boundaries. The reverted per-token kernel's fatal flaw was
  re-reading `S` from HBM every token — do not repeat it.
- **One block per (sequence, head)** (or per (sequence, head, chunk-stage)); pick `C`
  so a chunk's tiles fit in shared memory at high occupancy across the 132 SMs.
- **Coalesced varlen loads** via `query_start_loc`; seed the initial state and write the
  final state back exactly like the incumbent.
- Reconcile against vLLM v0.25.0 before trusting parity: l2norm-of-q/k, the gate
  convention (per-step vs cumulative log-decay), and `beta` placement.

## Decision rule

Ship the raw-MMA body only if it BOTH passes `make test-kernel` parity AND wins
`make bench-kernel` + trace replay against the CUTLASS/CuteDSL prototype and stock
FlashInfer. Otherwise keep the primary vehicle (or stock) — a slower hand-written
kernel is pure maintenance debt.
