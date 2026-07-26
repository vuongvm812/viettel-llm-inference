# Patches that are NOT applied by the build

Nothing here is copied into the image. `Dockerfile.vllm-fork` only applies `../v0.26.0/*.patch`.
This is a parking lot for work that is real but not wired in.

## Why this directory exists

During the v0.25.0 -> v0.26.0 upgrade, the local (gitignored) `vllm/` working tree was found to
contain **eight** modified files, only three of which any committed `*.patch` reproduced.
`gen.sh` had a hardcoded file list and was never re-run, so the rest were never captured and
**never shipped in any built image** — despite notes elsewhere claiming one of them had landed.

`gen.sh` now fails if a modified file is not captured, so this cannot recur silently. The
never-shipped work is preserved below rather than deleted.

## ~~`shortconv_specdecode_rollback.patch`~~ — LANDED 2026-07-26, no longer parked

Folded into the applied set as `v0.26.0/{short_conv,lfm2,mamba_utils}.patch`. It is the fix for
the "truncation / dual-path" flag (short-conv `conv_state` committed rejected drafts, so the
long-context probe returned garbage). Inert at `num_spec=0`, so shipping it does not change the
non-spec path; `bench/test_shortconv_spec_rollback.py` is the contract test. `mamba_utils.py`
now has its own `gen` line — all three files must move together or boot raises
`TypeError: short_conv_state_shape() got an unexpected keyword argument 'num_spec'` /
`assert page_size_padded >= page_size`. Still **not runtime-verified**: it has never been booted.

## ~~`hybrid_draft_kv_groups.v0.25.0.patch`~~ — LANDED 2026-07-26, no longer parked

Superseded by the applied set. The hybrid separate draft model (LFM2.5-350M drafting for the
1.2B target, vLLM issue #49112) now ships as `v0.26.0/{gpu_model_runner,
llm_base_proposer_multigroup,mamba_groups_hybrid_draft}.patch`. Two of the parked hunks survived
essentially verbatim (the `worker/mamba_utils.py` shapes-excluded spec comparison, and the
`ShortConvAttentionMetadataBuilder` runner gate — which turned out to be upstream PR #44296's
missing *third file*, not a ubatch allowlist entry, and its absence had made the whole shipped
short-conv rollback inert). Two were dropped as unnecessary:

- **`kv_cache_utils.py` drafter-layer isolation** (upstream #35062 / #49138) is a **no-op for this
  model pair**. Draft and target have identical attention specs (8 KV heads x 64 head_size, same
  fp8_e4m3, same block size), so `FullAttentionSpec` is `__eq__`-equal and their 12 attention
  layers already share one `same_type_layers` key; the two conv specs already differ in `shapes`
  and are already separate keys. The natural grouping is `[attn 12, conv_target 10, conv_draft 10]`
  — every group spec-uniform, no patch needed. Sharing one attention group is also *desirable*:
  one block table, one prefix-cache hash chain.
- **the `llm_base_proposer.py` rewrite** was strictly weaker than what v0.26.0 already ships.
  `Step3p5MTPProposer` (`v1/spec_decode/step3p5.py`) is a complete multi-group drafter; the applied
  patch lifts its five members into `SpecDecodeBaseProposer` instead of hand-porting #49138, which
  only resolved the metadata *builder* per layer and still fed one block table to every group.

## Re-checking

`bash round-1.2/vtl/vllm_patches/gen.sh` dry-runs everything here on top of the applied
`v0.26.0/` set, so rot shows up as a failed run rather than a stale claim in this file.
