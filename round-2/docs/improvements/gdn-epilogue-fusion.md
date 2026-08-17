# GDN epilogue fusion — what was built, why it was reverted, and the version worth building

Phase B (commits `a22c31d`, `fac50cd`, reverted 2026-08-17) folded the gated-RMSNorm +
group-128 fp8 quant epilogue into `gdn_decode_step.cu`, so the norm+quant ran where
`o = h * q` was still in registers. The kernel work was correct and is preserved in git; the
**handshake** that delivered its result to `gdn_kernels` was not, and the measured economics
were negative. Both halves of that verdict matter, and the second is the one that decides
what to build next.

## What was built

`FUSED_EPILOGUE=1` variant of the decode-step kernel emitting fp8 + per-group scales
directly, plus a Python-side **claim book** in `gdn_decode_step.py` (`_claims`,
`_publish_claim`, `take_epilogue`, `note_epilogue_consumer`, `EPILOGUE_CONSUMERS`) and a
per-layer three-state latch (`probe` → `fused` after two observed consumptions → `plain`
permanently on any unconsumed claim). `gdn_kernels._decode_step_epilogue` sat at tier 0 of
the gated-RMSNorm ladder and, on a claim hit, `copy_`d the staged fp8 into the caller's
output instead of computing it.

## The four soundness findings (all Critical, all independent)

1. **Claims are keyed on `data_ptr()` with no layer or step identity.** Activation buffers
   are pooled and recycled, so two layers in one step — or one layer across two steps — can
   present the same address. A claim published by layer *i* can be consumed by layer *j*,
   substituting one layer's epilogue for another's. Nothing in the key distinguishes them.
2. **The fused state disables its own detector.** The latch demotes to `plain` on an
   unconsumed claim, but once latched to `fused` the kernel no longer writes the bf16
   superset — so "unconsumed" stops being observable and the failure mode becomes zeros in,
   zeros out, silently, for the rest of the run.
3. **Piecewise cudagraph replay bakes the consumer while the producer keeps deciding.** The
   consumer's `copy_` is captured into the replayed graph; the producer re-evaluates the
   latch in Python every step. Any divergence serves a stale fp8 buffer from a previous
   step's contents. This is the dominant path in the shipped config.
4. **`take_epilogue` credits the probe counter before the consumer commits.** A claim that
   is looked up but whose consumer then bails still counts toward the two-consumption
   promotion, so the latch can promote to `fused` on evidence it never actually had.

## The measured cost/benefit — it was net-negative as shipped

| per token per layer | bf16 round trip | fp8 + scales | total |
|---|---|---|---|
| unfused (stock ladder) | 16.0 KB w + 16.0 KB r | 8.25 KB w | **56.5 KB** |
| fused, probe state | 16.0 KB w + 16.0 KB r | 8.25 KB w | **56.5 KB** (superset — no saving) |
| fused, latched | — | 8.25 KB w + `copy_` 8.25 KB r/w | **40.5 KB** |

The saving is 16 KB/token/layer → at 5 rows × 36 layers ≈ 2.9 MB/step ≈ **0.8 µs** of HBM at
4.8 TB/s. Against that, the fused arm raised the launch's `pack_args` from 9 to 31 arguments
and added a `_fused_plan` decision with pybind calls, per layer, per step — ~0.25 ms/step of
**host** time across 36 layers, on the eager PIECEWISE path that dominates this workload. The
consumer's `copy_` eats roughly half of the theoretical win before it is banked, and adds
+36 graph nodes per step. A 300× loss, and the probe state — which is where a conservative
latch spends most of its life — saves exactly nothing.

## The right version: the out_proj seam

The handshake exists only because the fp8 the kernel already produced had to be handed back
to a *torch-level* consumer that then handed it to the GEMM. Delete that hop: **have
`out_proj`'s block-fp8 GEMM read the staged fp8 + scales directly.**

- It **doubles the win**: the `copy_` disappears (it was ~50% of the saving) and so does the
  bf16 the GEMM would otherwise quantize itself.
- It **deletes the handshake entirely**: no claim book, no pointer keys, no latch, no probe
  state — the staging buffer is an explicit, caller-owned argument with the layer's identity
  built into it, not a value recovered by address at runtime. Every one of the four findings
  above is a property of the handshake, not of the kernel.
- It is a **static** decision made at patch/arm time, so cudagraph capture and replay see one
  shape; finding 3 cannot exist.

Staged plan, each stage shippable and reversible on its own:

1. **Staging buffer, caller-owned.** Allocate the `[max_rows, 8192]` fp8 + `[max_rows, 64]`
   scale buffers once at arm time (cudagraphs bake pointers — this must not be per-step), and
   have the unfused kernel write bf16 as today. Pure plumbing, no numerics change.
2. **`FUSED_EPILOGUE` writes into the staging buffer**, bf16 output retained. Ship behind the
   existing geometry envelope; parity is the same fp8-bit-equality test Phase B already had.
3. **Teach the `out_proj` linear method to accept pre-quantized activations** — a wrapper
   that, when the staging buffer is armed for this layer, skips its own quant and calls the
   block-fp8 GEMM with the staged pair. Fall back to stock on any mismatch. This is the stage
   that banks the win; it is also the only one that touches the quant path, so A/B it alone.
4. **Drop the bf16 write** once (3) is proven on the box.

Do not restart at stage 2 without stage 1: the pointer-recovery shortcut *is* the bug class.

See also `gdn-decode-megakernel.md` §1 — a per-layer megakernel subsumes this seam entirely
(the GEMV reads the registers, never HBM), and its §6 go/no-go should be settled before
stage 3 duplicates work the megakernel would delete.
