# GDN decode megakernel — design note

The full fusion of one GDN block into a single cooperative-grid launch: `in_proj_qkvz` +
`in_proj_ba` → conv1d + gated delta rule → gated RMSNorm + group-128 fp8 quant → `out_proj`.
Precedent is `reference/lfm2/vtl/csrc/shortconv_decode_mega.cu`, which did exactly this for
LFM2.5's short-conv block. **Verdict up front: buildable, and much smaller than it looks — size
the launch gap with nsys before writing a line of it (§6).**

One option is off the table immediately. *One launch spanning all 36 GDN layers* is not
available here: `mlp_only_layers: []`, so all 48 layers carry a 256-expert MoE and the GDN blocks
are separated by MoE blocks on the sequential trunk. **Per-layer is the only shape.**

## 1. What is actually left to win

After the Phase-B epilogue fusion folds the gated RMSNorm + group-128 quant into
`gdn_decode_step`'s tail, one GDN layer is four launches — `in_proj_qkvz` (fp8 block GEMM,
62.9 MB of weights), `in_proj_ba` (bf16, 0.79 MB), `gdn_decode_step` (NVRTC; conv + delta rule +
epilogue), `out_proj` (fp8 block GEMM, 25.2 MB). 4 × 36 = **144 launches per decode step**; a
per-layer megakernel makes it 36. Per layer at the served `--max-num-seqs=5` (M=5) that moves
**89.0 MB of weights** (batch-independent) plus **43.4 MB of fp32 state** (4.34 MB/seq, read and
written exactly once, §2) — both irreducible — against **~0.63 MB of intermediates**, which is
everything a megakernel could keep on chip.

**That is the finding.** Intermediate HBM traffic is **0.5%** of the layer's 133 MB. The argument
that carried the LFM2.5 megakernel — three round trips of the activations — does not carry here,
because this model's projections are 20–60× wider than its activations. The residual prize is
**launch and tail overhead only**: 108 launches/step removed, at a measured inter-kernel gap of
*g*. At a plausible g ≈ 1 µs that is **~0.11 ms of TPOT**, and under the round-2 scoring band
(`RUST-RUNNER.md` §6: TPOT ≤ ~0.011 ERS/ms) **~0.0012 ERS**. Against a GDN-stack floor of
133 MB × 36 / 4.8 TB/s ≈ **1.0 ms/step**, this is a ~10% shave on a term that is itself a
fraction of the step.

## 2. Geometry (Qwen3.5-122B-A10B-FP8, TP=1, 36 GDN layers)

`hidden 3072`; `key_dim = 16 × 128 = 2048`; `value_dim = 64 × 128 = 8192`;
`conv_dim = 2·key_dim + value_dim = 12288`; conv width 4.

| tensor | shape | dtype | bytes |
|---|---|---|---|
| `in_proj_qkvz.weight` | `[20480, 3072]` = `[q 2048 \| k 2048 \| v 8192 \| z 8192]` | fp8 | 62.9 MB |
| `in_proj_ba.weight` | `[128, 3072]` = `[b 64 \| a 64]` | **bf16** (`modules_to_not_convert`) | 0.79 MB |
| `conv1d.weight` (+bias) | `[12288, 1, 4]` | bf16 | 98 KB |
| `out_proj.weight` | `[3072, 8192]` | fp8 | 25.2 MB |
| ssm state | `[64, 128, 128]` /seq | fp32 | 4.19 MB |
| conv state | `[12288, 3]` /seq | fp32 | 147 KB |

Per step: **3.20 GB** of GDN weights, **156 MB/seq** of state (matches MODEL.md's ~154 MB/seq).

**Scales.** Both axes are 128-wide, which is what makes an in-kernel GEMV tractable: activations
are dynamic per-`(token, 128-group)` fp8 (`xs[M, K/128]`, from the preceding fused RMSNorm+quant
— so the megakernel's first phase is quant-only or absent), weights block-quant `[128, 128]`
(`ws[N/128, K/128]` fp32: `[160, 24]` and `[24, 64]`, 21 KB total). A lane owning 64 consecutive
k values — `w4a8_gemv.cuh`'s `kLaneK` — lies entirely inside one activation group **and** one
weight block, so it reads exactly one `xs` and one `ws` per pass. K = 3072 (24 groups) and
K = 8192 (64) are both multiples of 128. `in_proj_ba` has no scales at all (bf16): a plain GEMV.

**fp32 state residency.** Nothing makes 4.34 MB/seq/layer resident: H200 has 228 KB smem/SM
(30 MB device-wide) and 60 MB of L2, and the next touch of a layer's state is 36 layers and ~5 GB
of traffic later. It also constrains phase mapping — the delta-rule phase **must** keep one block
per key head per token (grid `(HK, T)` = 128 blocks at T=8), because the q/k conv channels and
their rotating conv state are shared by the `HV/HK = 4` sibling value heads and there is no
inter-block ordering to protect a split.

## 3. Hard part 1 — an in-kernel block-fp8 GEMV

`reference/lfm2/vtl/csrc/w4a8_gemv.cuh` is the device-callable precedent, and exists for
precisely our reason: *"Not to beat CUTLASS at GEMM — to be callable from inside another
kernel."* Carrying over verbatim: `w4a8_gemv_row`'s skeleton — one warp per output row, the
**token loop inside the weight load** (token-outer would stream 62.9 MB eight times), `acc[m]`
unrolled over a compile-time `kMaxM` so accumulators stay register-indexed, a shuffle ladder
closing the row; `kMaxM = 8`, which Phase A's `[1,2,4,8]` capture set makes exactly the padded
ceiling rather than a guess; **N-split, not split-K**, so no inter-block reduction and no extra
barrier per GEMV phase; and the `K > kWarpK` loop we need at K = 3072 and K = 8192.

What changes:

- **The int4 arm becomes the fp8 arm.** Swap `nib_to_int4` for a second `fp8_to_float`, and the
  1-D `gs_row[k/G]` for the 2-D `ws[(n>>7)*(K>>7) + (k>>7)]`. Rows in one 128-row n-block share
  `ws`, so it broadcasts.
- **The second weight copy disappears — the reason to build the fp8 arm first.**
  `w4a8_gemv.cuh` spends twelve lines on it: `quant_w4a8.py` stores CUTLASS-reordered
  `weight_packed`/`weight_group_scale` that hand-written code cannot address, so LFM2 paid an
  84 MB re-derived plain copy. vLLM's block-fp8 path stores `weight` + `weight_scale_inv`
  **plain**, so an fp8 megakernel addresses the live tensors. The int4-from-fp8 arm inherits the
  LFM2 problem at 36 × 88 MB = **3.2 GB of duplicate weights** against a ~6.7 GB non-weight
  budget — a non-starter unless the plain layout *replaces* the CUTLASS one, prefill GEMM and all.
- **Numerics are not bit-exact vs CUTLASS and cannot be** (per-lane fp32 half-group sums vs an
  MMA's tile order). Bound against a torch reference built from the same tensors, as
  `bench/test_w4a8_gemv.py` does — not against the stock kernel.

## 4. Hard part 2 — cudagraph capture and the Rust runner

`shortconv_decode_mega.cu` supplies the structure, and all four of its design notes are
load-bearing here:

1. **The grid barrier is hand-rolled, deliberately.** A sense-reversing counter (atomics +
   `__threadfence` + `__nanosleep` backoff), *not* `cg::this_grid().sync()`:
   `cudaLaunchCooperativeKernel` is unreachable from torch's op dispatch and drags `-rdc=true`
   into a build `BuildExtension` cannot device-link. The hand-rolled one launches on the current
   stream like any kernel and **is stream-capturable** — mandatory under `FULL_AND_PIECEWISE` —
   and its generation counter is monotonic *across launches*, which is what makes repeated replay
   of one baked graph safe.
2. **Co-residency is a correctness precondition, not a tuning knob**: a block that never gets
   scheduled never arrives and the barrier hangs the device. The grid comes from
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` *on this kernel* × SM count, computed in
   Python and passed in; below a `kMinBlocks` floor the Python gate refuses and stock runs.
3. **Scratch is caller-owned, allocated once at patch import**, because a cudagraph bakes
   pointers — and `num_blocks` likewise must be capture-time constant, not batch-derived.
4. **Nothing else may be co-resident**; a sibling graph branch holding SMs deadlocks the barrier.
   Safe only because the GDN block sits on the single sequential trunk — assert, don't assume.

Integration with the Rust runner is, happily, **nothing**: `RUST-RUNNER.md` §2 and §3/Phase 1
establish that Rust replays `raw_cuda_graph_exec()` handles via `cuGraphLaunch` and never sees
individual kernels, and hazard 8's resolution (`VTL_RUST_RUNNER_COMMIT="update"`) is about commit
ordering, not kernel identity — so the megakernel is invisible to it **provided capture
succeeds**. Test the other direction: a capture failure demotes `nstep_decode` down the
`VTL_NSTEP_MODE` ladder to eager and silently costs far more than the fusion ever paid. Boot with
`VTL_NSTEP_MODE=graph` and assert the "captured N burst body graph(s)" line still reports sizes
`[1, 2, 4, 8]`.

## 5. Staged build plan

Each stage is independently shippable, independently measurable, and degrades to the one below.
Numerics stay `gdn_decode_step.cu`'s contract throughout: `__fmul_rn`/`__fadd_rn`, no fast-math,
the two tolerated fp32 divergences and no third.

- **Stage 1 — `fp8_block_gemv.cuh` as a standalone op.** Port `w4a8_gemv_row` to the fp8×fp8
  block-scale arm behind a thin `torch.ops` wrapper. No barrier, no megakernel. Test: parity
  against a torch reference at M ∈ {1,2,4,5,8}, K ∈ {3072, 8192}, plus a wall-clock A/B against
  the stock block-fp8 GEMM. **If it does not match the stock GEMM at M ≤ 8, stop here** — a
  megakernel cannot rescue a slow GEMV.
- **Stage 2 — fold `in_proj_ba` into `gdn_decode_step`.** One launch removed per layer, no
  barrier and no new numerics, but *not* free: each `(h, t)` block computes its own 8 of 128 `ba`
  rows, re-reading a 49 KB weight slice per token → ~5.5 MB/layer extra (~200 MB/step ≈ 42 µs).
  Ship only if the 36 launches beat the 42 µs.
- **Stage 3 — the barrier, two phases.** `in_proj_qkvz` GEMV → barrier → the existing
  `gdn_decode_step` body, one launch. This is where co-residency, the occupancy query, the Python
  `kMinBlocks` gate and cudagraph capture get proven, on the smaller half of the risk. Registry
  gate `gdn_decode_mega`, `default=False` until parity passes on the box.
- **Stage 4 — absorb `out_proj`.** Second barrier, second GEMV phase, full four-phase kernel.

## 6. Go / no-go

`vtl/patches/megakernel_probe.py` ships enabled and already answers the three questions that can
kill the design, at boot, in microseconds: cooperative-launch support, MPS in use, and
`max_grid_blocks = blocks_per_mp × sm_count`. Read its `GO`/`NO-GO` line from a real boot, with
one adjustment — `verdict()` defaults to `hidden_size=2048`, but the width one GDN token needs is
**`conv_dim = 12288`**, so call `verdict(info, hidden_size=12288)`. At the 128 threads/block that
`gdn_decode_step.cu`'s `THREADS == DK == DV == 128` forces, that is 96 co-resident blocks, under
1 block/SM on 132 SMs: it will say GO. A GO is *permission to write the kernel and measure
again* — the real number is `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the finished
kernel, and it can only be lower.

**The deciding number is not the probe's.** Run nsys on a steady decode step, read the mean
inter-kernel gap across the GDN chain, and build only if all three hold: (1) gap × 108
launches/step ≥ **0.15 ms** (≈ 0.0017 ERS at the round-2 TPOT band) — below that, Stages 1–2
capture most of it for a fraction of the risk; (2) Stage 1's GEMV is at or under the stock
block-fp8 GEMM at M ≤ 8; (3) the finished kernel's occupancy query clears `kMinBlocks` *with* the
delta-rule phase's register pressure included. Fail any one and the honest answer is that this
model's GDN decode is weight- and state-bandwidth-bound (§1), and the launches are not where its
time goes.
