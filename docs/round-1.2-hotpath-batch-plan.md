# Round-1.2: 6-item optimization batch (warmup, Rust hot-path, CUDA kernels)

## Context

Round-1.2 (LFM2.5-1.2B on H200 MIG 1g.18gb: 16 SMs, ~600 GB/s, 3 vCPU) is at TTFT p95 44ms / TPOT 3ms. ERS gradient: 1ms TPOT ≈ 0.086 ERS ≈ 37x 1ms TTFT. The user selected 6 items from the optimization survey and (per explicit decision) the megakernel is **built to completion** — no fail-fast kill gates; interim measurements inform tuning only.

1. Warmup healthcheck: concurrent + shape-covering (TTFT tail)
2. `prepare_attn` pure-decode metadata fast path (~33% of host/step on dev box)
3. vtl-sched R6: `update_from_output` port + per-step marshalling removal
4. `prepare_inputs` pin_memory-twice-per-step fix via existing `UvaBufferPool`
5. CUDA fused conv+align kernel (`VTL_V2_FUSED_CONV_ALIGN`)
6. CUDA scoped decode megakernel (`VTL_SHORTCONV_MEGA`), built to completion

## Execution model (per user decisions)

- **Step 0:** write this plan verbatim to `docs/round-1.2-hotpath-batch-plan.md` in the repo (same convention as `docs/round-1.2-latency-optimization-plan.md`).
- **Step 1:** spawn **one implementation sub-agent (model: Opus 5)**, hand it the docs plan file, and let it implement **all 6 items end-to-end in one pass — no measurement pauses, no stop-losses, no "continue after A/B" checkpoints**. All phases of every item (including megakernel P0–P3 code and tests) are implemented to completion; on-box measurement happens whenever the user chooses, afterwards, without blocking anything. One agent (not parallel) because the items share files: `vtl/patches/__init__.py`, `setup.py`, `torch_bindings.cpp`, `docker-compose.yaml`, `short_conv.patch`.
- **Step 2:** orchestrator reviews the diff (latency lens: no per-step heap allocation on hot paths, no graph breaks, no busy-spin threads on 3 vCPUs, compose literals only), fixes findings, runs `make check` + the off-box self-checks, and reports.
- **Env policy (user decision): every new env var lands in `docker-compose.yaml` as a literal, ON by default** — same discipline as the existing `opt/round12-5ws` block: each line documents its one-line revert, and the block comment notes the flags are inert until the image carries this code (`make vllm-fork PUSH=1` → re-pin → `make build`). Runtime safety gates (megakernel probe GO + per-kernel occupancy check, conv-layout check, `hasattr` op checks) stay — a gate failing means fail-closed to the stock path at boot, never a crash.

### Repo law (applies to every item)
- Served stack = v0.25.0. Reference tree: `vllm-v0.25.0-edited/` (post-fork-patch). NEVER trust or edit `vllm/` (drifted; only its `tests/kernels/mamba/test_precopy_mamba_align.py` is borrowed as a portable parity reference).
- Delivery paths: plugin monkeypatches → `round-1.2/vtl/patches/*.py` (registry: `register_patch`, env `VTL_ENABLE_<NAME>`, loggers = children of `vllm.vtl`, imports inside `apply()`); fork edits → `round-1.2/vtl/vllm_patches/v0.25.0/*.patch` (applied `patch -p1` vs pristine v0.25.0, Dockerfile.vllm-fork:173-186; needs `make vllm-fork PUSH=1` + digest re-pin, done by the user); CUDA → `round-1.2/vtl/csrc/` + `setup.py` sources + `torch_bindings.cpp` (`TORCH_LIBRARY(vllm_cuda)` pattern); Rust → `round-1.2/vtl-sched/` (maturin, Dockerfile:44-72).
- V2 runner is live: patch `vllm.v1.worker.gpu.model_runner.GPUModelRunner` (NOT the V1 class in `gpu_model_runner.py`).
- Everything env-gated with a one-line revert; compose literals only; no `${VAR}`. (User decision: this batch ships ON by default, overriding the default-off convention; kill-switches must still work.)
- A/B: 3 boots/arm, ~0.5ms boot-to-boot noise floor, ≥0.22ms TPOT to prove a knob.
- Off-box gates: `make check` (runs every patch module as a script → each needs a `_self_check`/selfcheck path), Docker build compiles CUDA+Rust, `make test-kernel` globs `bench/test_*.py` on GPU.

---

## Item 1 — Warmup healthcheck: concurrent + shape coverage

**File:** `round-1.2/vtl/warmup_healthcheck.py` only.

Change `_prime()`:
- POST trace line 0 **synchronously first** (warms the shared ~6.4k system prefix without racing it — the exact serialization the docstring documents).
- Fire remaining lines via `concurrent.futures.ThreadPoolExecutor(max_workers=8)` so concurrent decode batches exercise cudagraph capture sizes 2/4/8 and the multi-seq kernels.
- Keep: sentinel-before-prime one-shot, best-effort exception swallowing per request, `VTL_WARMUP_POST_TIMEOUT`, stderr summary. New env `VTL_WARMUP_CONCURRENCY=8` (default 8; `1` restores sequential as the A/B control).
- Update `_selfcheck()`: fake `_post` records ordering; assert line 0 completes before any burst request starts, and all lines primed.

Stdlib only. No compose/Dockerfile change.

---

## Item 2 — prepare_attn: pure-decode metadata fast path (plugin patch)

**New:** `round-1.2/vtl/patches/decode_fastpath.py`, `register_patch("decode_fastpath", default=True)`; compose literal `VTL_ENABLE_DECODE_FASTPATH: "1"` (kill-switch: set to 0).

**Verified basis:** under FULL cudagraph the attn_metadata dict returned by `model_state.prepare_attn` is DISCARDED (`model_runner.py:1287-1293` skips `set_forward_context`). Load-bearing per-step effects are only:
- persistent buffer writes: `input_buffers.{input_ids,positions,seq_lens,query_start_loc}`, `block_tables.input_block_tables`, `block_tables.slot_mappings`
- `FlashAttentionMetadataBuilder`: FA3 AOT `get_scheduler_metadata` → `self.scheduler_metadata[:n]` write (flash_attn.py:578-595) — function of seq_lens
- `BaseMambaAttentionMetadataBuilder`: align-mode state-indices gather `(seq_lens-1)//block_size` → `self.state_indices_tensor_d[:num_decodes]` write (mamba_attn.py:570-574)

**Design:** wrap `GPUModelRunner.execute_model`. Pure-decode predicate on `scheduler_output`: `scheduled_new_reqs` empty, `finished_req_ids` empty, `preempted_req_ids` empty, `scheduled_spec_decode_tokens` empty, `new_block_ids_to_zero` empty, `has_structured_output_requests` False, all `num_scheduled_tokens == 1`, same req-id order + same padded `BatchExecutionDescriptor` as previous step. New blocks in `scheduled_cached_reqs.new_block_ids` (16-token boundary) do NOT break the fast path — they trigger a `gather_block_tables` re-run only.

Fast path replaces `prepare_inputs` + `prepare_attn` + `model_state.prepare_attn` with the must-run set:
1. existing `update_requests` bookkeeping (num_computed_tokens += 1) stays untouched
2. `prepare_pos_seq_lens` triton (positions/seq_lens) + `combine_sampled_and_draft_tokens` (input_ids/logits_indices)
3. `compute_slot_mappings` triton every step; `gather_block_tables` only when new blocks appeared
4. FA3 `get_scheduler_metadata` + persistent write (reuse the builder's own code via a narrow helper, not a copy)
5. mamba align state-indices gather + `state_indices_tensor_d` write
6. reuse the cached `InputBatch`/metadata objects from the last slow step, with `seq_lens_cpu_upper_bound`/`max_seq_len` scalars refreshed (they advance +1/step and feed FA3 + mamba builders)

Slow path = originals, and re-caches all reused objects. Bail to slow path on anything unrecognized.

**Seams (all plugin-patchable class methods, no fork patch):** `GPUModelRunner.{execute_model,prepare_inputs,prepare_attn}`, `MambaHybridModelState.prepare_attn`, `FlashAttentionMetadataBuilder`, `BaseMambaAttentionMetadataBuilder`, `BlockTables`. (Module-level `build_attn_metadata`/`build_slot_mappings_by_layer` never need rebinding — we bypass them on the fast path.)

**Correctness gate:** `VTL_DECODE_FASTPATH_SHADOW` (compose "0" — shadow runs both paths and is a debug arm, not a production default): compare persistent-buffer writes (seq_lens, slot_mappings, scheduler_metadata, state_indices) per step, log divergence via `vllm.vtl.decode_fastpath`. Sizing honesty: dev-box `prepare_attn` is 2.0ms/step but ~0.85ms is FlashInfer-only code the judge (FLASH_ATTN) never runs — expect ~1ms-class win on the slice, re-profile there.

**Tests:** `bench/test_decode_fastpath.py` — predicate unit tests on synthetic SchedulerOutputs (no GPU); buffer-parity test under `make test-kernel`. Module `_self_check` for `make check`.

---

## Item 4 — prepare_inputs: pooled pinned staging (same patch module as Item 2)

**Verified basis:** `async_copy_to_gpu` (`buffer_utils.py:26-44`) does `x.pin_memory()` = fresh `cudaHostAlloc` per call; exactly 2 hot sites/step: `model_runner.py:864` (idx_mapping int32[num_reqs], also `empty_like` out) and `:910` (query_start_loc int32[max_num_reqs+1], persistent out). `UvaBufferPool` (round-robin, depth=`max_concurrent_batches`=2) already exists at `buffer_utils.py:52-86` — reuse, don't rewrite.

**Design:** in `decode_fastpath.py` `apply()` (own sub-gate `VTL_UVA_POOL`, compose "1", independent kill-switch), rebind the **import-bound name** `vllm.v1.worker.gpu.model_runner.async_copy_to_gpu` (patching `buffer_utils` does nothing — `from` imports) to a pooled version: dict of `UvaBufferPool` keyed by (dtype, capacity bucket), capacity `max_num_reqs+1`, depth 2. Slice to `x.shape`, `copy_to_gpu`. Depth-2 safety = the same invariant `StagedWriteTensor`'s own pools rely on (`max_concurrent_batches=2`); document in header. Also rebind in `pp_utils`/`structured_outputs` namespaces for uniformity. On Item-2's fast path the idx_mapping copy is skipped entirely (unchanged on pure-decode steps).

Prize ~0.2ms/step (measured `aten::is_pinned` 0.167 + `_pin_memory` 0.031); below the solo A/B bar — if a later bisect is ever needed, flip this sub-gate independently.

---

## Item 3 — vtl-sched R6: update_from_output port + marshalling removal

**Verified boundaries:** `SchedulerOutput` is consumed in-process by the V2 runner; `NewRequestData` holds live Python refs → full Rust construction is off the table; the async batch queue retains a `SchedulerOutput` across later `schedule()` calls → no Rust-owned reused arenas may back its fields; `finished_req_ids`/`preempted_req_ids` aliasing is load-bearing (rebound not cleared, scheduler.py:1207-1211). `update_from_output` (scheduler.py:1499-1851) live subset here: token append + `check_stop` (:1626-1632), stop → `_free_request` (:1704-1716), `EngineCoreOutput` build (:1730-1754), queue removal (:1759-1764), outputs/stats (:1820-1849); all spec/structured/connector/pooling branches are dead. AsyncScheduler overrides only `_update_after_schedule` + `_update_request_with_output`.

**R6a — Rust update core.** New `vtl-sched/src/update.rs` + `python.rs` method `KvManager.update_step(slots: Vec<u32>, new_token_ids: Vec<i64>) -> finished mask/slots`: per request, output-token bookkeeping and the stop decision (port `check_stop` from `sched/utils.py:94-130`: max_tokens, EOS, stop-token ids; per-slot stop params pushed once at intern time — extend `Manager` slot state). Python wrapper (in `rust_sched.py`, gate `VTL_RUST_SCHED_UFO`, requires `VTL_RUST_SCHED_FULL`) wraps `Scheduler.update_from_output`: Rust decides stops; Python keeps `request.append_output_token_ids` (objects still needed for detokenize + block hashes), `EngineCoreOutput` build, queue removal, `_free_request` (which already routes `kv.free` → `mirror.drop`). Block hashing stays in Python (already parity-tested via the bytes push; moving it is scope creep — noted as future flag).

**R6b — kill per-step marshalling.** Add a Rust-side persistent slot-indexed request table (extend `manager.rs`): fields of `SchedReq` stored in Rust, updated by deltas — admissions/preemptions/resumes (events the Python wrapper already applies at steps 5-8 of the current flow, rust_sched.py:938-989) and per-step token counts from R6a. `Scheduler.schedule` gains a no-args variant reading its own table; `pack_req`/`mirror.slot` per-step loop (rust_sched.py:894-921) reduces to: push new hashes for requests whose `block_hashes` grew (unavoidable — Python owns the hasher), plus bail-condition checks. Fallback ladder preserved: any bail → stock `wrapped()` with a full-table resync on next Rust step (table rebuilt from `pack_req` — keep that code as the resync path).

**Parity:** extend `bench/test_rust_sched_parity.py` with an update-tier: replay recorded trace (`data/input/trace-round2.jsonl` through the existing harness), assert Rust stop decisions/finished sets/table state == Python oracle per step. Shadow env `VTL_RUST_SCHED_UFO_SHADOW` mirrors the existing ShadowState pattern. `cargo test` units in `update.rs` (stop matrix: max_tokens hit, EOS, stop-id, none).

**Files:** `vtl-sched/src/{update.rs,manager.rs,sched.rs,python.rs,lib.rs}`, `vtl/patches/rust_sched.py`, `bench/test_rust_sched_parity.py`, compose literals (`VTL_RUST_SCHED_UFO: 1`, `VTL_RUST_SCHED_UFO_SHADOW: 0`; revert ladder: UFO→0 falls back to the Python update_from_output, all existing rust-sched flags untouched). Sizing honesty: scheduler ≈0.3ms/step total — smallest prize of the batch.

---

## Item 5 — CUDA fused conv+align (`VTL_V2_FUSED_CONV_ALIGN`)

**Verified basis + correction:** BOTH stock Triton launches live in `mamba_hybrid.py` (`preprocess_mamba_align_fused_kernel` at :206-219, `ctx.run_fused_precopy` at :220), outside the cudagraph, every step; `mamba_utils.py` needs no patch. Copies fire only on mamba-block-boundary crossings; per (req,layer) up to 8KiB of `[2,2048]` bf16 conv state; metadata is prebuilt persistent GPU tensors. `_mamba_src_col_gpu`/`_mamba_src_off_gpu` have no other consumers → the fused kernel skips materializing them.

**Kernel — one launch, one block per request** (`vtl/csrc/conv_align_fused.cu`, added to existing `vtl._C`):
- Grid `(num_reqs,)`, 256 threads. Thread 0: load pre-mutation `state_idx`/`num_accepted`/`num_computed`/`query_start_loc` via `idx_mapping` (exit on -1), compute `src_col`/`token_bias`/`new_state_idx` (port of mamba_utils.py:284-330 arithmetic), perform the mutation exactly once (store `new_state_idx`; reset `num_accepted=1` on boundary cross), publish to smem. `__syncthreads()`. If `src_col >= 0 && src_col != dst_col`: block serially copies the 10 conv states (port of `_copy_mamba_state_block` conv branch), `uint4` 16B-vectorized. No inter-block race by construction (distinct req slots per block).
- Worst case 10×8KiB from one block ≈ 5µs, only on boundary steps; fast path ≈ 6 scalar loads. Support only LFM2's SD conv layout; temporal states / DS layout → patch-level fallback to stock Triton. `# ponytail:` comment the ceiling.
- Op `vllm_cuda::conv_align_fused(...)` with `mutates_args=("state_idx","num_accepted")`, no-op fake, `VTL_KERNEL_SYNC` hook, `TORCH_CHECK` guards — `bcx_conv_gate_quant.cu` style.

**Integration:** new fork patch `vtl/vllm_patches/v0.25.0/mamba_align_precopy.patch` on `mamba_hybrid.py` only (hunks at ~:179-226 — no overlap with `mamba_hybrid_postprocess.patch` hunks at :1/:76/:300; context vs pristine v0.25.0; glob order fine). In-file env read `VTL_V2_FUSED_CONV_ALIGN` + `hasattr(torch.ops.vllm_cuda, "conv_align_fused")` + layout check, fail-closed to the stock two launches. Compose comment already reserves the flag; set the literal `VTL_V2_FUSED_CONV_ALIGN: "1"` (kill-switch: 0).

**Parity:** `bench/test_precopy_conv_align.py` — port `vllm/tests/kernels/mamba/test_precopy_mamba_align.py`; reference = the two stock Triton kernels on cloned inputs; bit-exact conv states + `state_idx` + `num_accepted`. Cases: fresh (src=-1), same-block no-op, boundary cross ± token_bias, mixed batch with -1 holes, num_reqs ∈ {1,2,8}.

**Honesty:** ≤0.05ms/step — unprovable by TPOT A/B; acceptance = parity green. New patch module `vtl/patches/conv_align_fused.py` (fake registration + `_MODULES` entry, `default=True`).

---

## Item 6 — CUDA scoped decode megakernel (`VTL_SHORTCONV_MEGA`) — BUILD TO COMPLETION

**Verified basis:** chain = `rms_norm_quant → in_proj W4A8 GEMM (N=6144,K=2048) → bcx_conv_gate_quant → out_proj W4A8 GEMM (N=2048,K=2048)`, per layer, 10 layers/step, M=1..8 decode, inside the FULL cudagraph (`torch.ops.vllm.short_conv`). Cooperative launches ARE stream-capturable (probe-verified doc claim; machine-verify in P0); MPS is the only disqualifier; probe GO needs ≥2048 co-resident threads. Weights per linear: `weight_packed` (int4), `weight_group_scale` (group-128), `weight_chan_scale` (see `quant_w4a8.py`); out_proj decode already flows through the fork's `_vtl_out_proj_quantized` staging seam (short_conv.patch:140-176, y_fp8/y_scale staging from `mul_quant`).

**Kernel** (`vtl/csrc/shortconv_decode_mega.cu` + `w4a8_gemv.cuh`): grid **32 blocks × 256 threads**, `__launch_bounds__(256, 2)` (2 blocks/SM × 16 SMs = 8192 threads, clears the 2048 floor). Persistent global scratch allocated once at patch import (stable addresses for graph capture): `x_fp8[8,2048]+x_scale[8]`, `bxc[8,6144]` bf16, `y_fp8[8,2048]+y_scale[8]` (~150KB). Phases with device-wide barrier between:
1. rms_norm+quant — blocks 0..M-1, one per token (port of `rms_norm_quant.cu` body)
2. in_proj W4A8 GEMV — all 32 blocks, **N-split** (192 rows/block, full K per row; zero inter-block reduction), `x_fp8` staged in smem, int4 group-128 dequant in-register, fp32 FMA, bf16 out. Ideal ≈ 10µs (6MB @ 600GB/s). Split-K fallback only if M=1 microbench shows N-split leaving bandwidth
3. bcx_conv_gate+quant — blocks 0..M-1 (port of `bcx_conv_gate_quant.cu` body incl. null-block zero-fill and state rotation)
4. out_proj GEMV — all 32 blocks, 64 rows each, semantics matching `_vtl_out_proj_quantized`

Barrier: decide in P0 between `cg::this_grid().sync()` (needs `cudaLaunchCooperativeKernel`; check the `-rdc` requirement) and a hand-rolled global arrive/wait barrier (atomics + `__threadfence()`, no rdc, no cooperative launch, same co-residency requirement) — take whichever compiles clean in the Docker image and captures; hand-rolled is the likely winner. Smem ≤ ~32KB/block; watch `-Xptxas -v` for spills in the build log.

**Phases (ALL implemented in one pass — no measurement pauses; benches ship as runnable artifacts for later on-box use):**
- **P0:** standalone `w4a8_gemv` parity test + microbench vs `ops.cutlass_w4a8_mm` at M=1..8 × both shapes; barrier microbench (30 syncs/step budget) + graph-capture smoke for both barrier variants — all as `bench/test_*.py`, runnable later on the GPU box.
- **P1:** full 4-phase single-layer kernel; parity vs the real 4-op chain (output AND post-kernel conv state, tolerance = `test_bcx_conv_gate_quant.py`); graph-replayed chain-vs-mega microbench artifact.
- **P2:** integration — extend `short_conv.patch` decode branch: `if VTL_SHORTCONV_MEGA && probe GO && num_tokens <= 8: torch.ops.vllm_cuda.shortconv_decode_mega(...)` else existing chain (capture sizes 16/32 statically keep the stock chain — confirm `num_tokens` is capture-time-static under piecewise capture; if dynamic, mega op handles M≤32 with global-staged activations). New `vtl/patches/shortconv_mega.py`: scratch alloc, opaque-op fake, boot gate extending the probe with the real kernel's `cudaOccupancyMaxActiveBlocksPerMultiProcessor` (≥2 blocks/SM with actual smem/regs — the probe's ceiling is the device's, not the kernel's), fail-closed to stock chain otherwise. Compose literal `VTL_SHORTCONV_MEGA: "1"` (runtime gates keep it safe; kill-switch: 0).
- **P3:** ship the on-box validation recipe (`eval_quality.py` + `replay.py` + 3-boot A/B commands) as comments/Make targets — run by the user whenever, blocking nothing.

**Design risks (retire early, don't block):** rdc/device-link for `cg::grid.sync` (retire day 1 with a hello-world cooperative build; fall back to hand-rolled barrier); capture claim machine-verified in P0; capture-size 16/32 static-branch check in P2 (15 min).

---

## File inventory (new / modified)

| Item | New | Modified |
|---|---|---|
| 1 | — | `vtl/warmup_healthcheck.py` |
| 2+4 | `vtl/patches/decode_fastpath.py`, `bench/test_decode_fastpath.py` | `vtl/patches/__init__.py`, `docker-compose.yaml` |
| 3 | `vtl-sched/src/update.rs`, update-tier in `bench/test_rust_sched_parity.py` | `vtl-sched/src/{manager.rs,sched.rs,python.rs,lib.rs}`, `vtl/patches/rust_sched.py`, `docker-compose.yaml` |
| 5 | `vtl/csrc/conv_align_fused.cu`, `vtl/patches/conv_align_fused.py`, `vtl/vllm_patches/v0.25.0/mamba_align_precopy.patch`, `bench/test_precopy_conv_align.py` | `setup.py`, `vtl/csrc/torch_bindings.cpp`, `vtl/patches/__init__.py`, `docker-compose.yaml` |
| 6 | `vtl/csrc/{w4a8_gemv.cuh,shortconv_decode_mega.cu}`, `vtl/patches/shortconv_mega.py`, `bench/{test_w4a8_gemv.py,test_shortconv_decode_mega.py}` | `vtl/vllm_patches/v0.25.0/short_conv.patch` (regenerated vs pristine), `setup.py`, `vtl/csrc/torch_bindings.cpp`, `vtl/patches/__init__.py`, `docker-compose.yaml` |

All compose additions are ON-by-default literals (user decision), each with a one-line revert comment, grouped in a new block noting: inert until the image carries this code (rebuild + re-pin), and bisect by flipping flags to 0 one at a time if TPOT/TTFT regresses.

Plus: **`docs/round-1.2-hotpath-batch-plan.md`** (this plan, written in Step 0).

## Verification (all off-box, nothing blocks on measurement)

1. `python3 round-1.2/vtl/warmup_healthcheck.py --selfcheck` (Item 1).
2. `make check` — every patch module self-check incl. the new `decode_fastpath` and `conv_align_fused` modules.
3. `cargo test` in `vtl-sched/` (Item 3 update-tier units); Docker build compiles the wheel.
4. Docker build = CUDA compile gate for Items 5+6 (`cuobjdump` asserts extended; watch `-Xptxas -v` for spills).
5. GPU-side artifacts shipped, run later by the user: `make test-kernel` (all new parity tests), `make vllm-fork PUSH=1` → re-pin → `make build`, shadow soaks, 3-boot A/Bs (`bench/replay.py`). None of these gates implementation.
