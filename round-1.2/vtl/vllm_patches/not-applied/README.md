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

## `hybrid_draft_kv_groups.v0.25.0.patch` (4 files)

Unfinished work toward letting a *hybrid* draft model (e.g. LFM2.5-350M drafting for the 1.2B
target) survive vLLM's KV-cache-group assertions: isolates the drafter's layers into their own
group (`kv_cache_utils.py`), relaxes `validate_same_kv_cache_group` (`llm_base_proposer.py`),
compares mamba specs on everything except `shapes` (`worker/mamba_utils.py`), and allows
`ShortConvAttentionMetadataBuilder` in the ubatch allowlist (`gpu_model_runner.py`).

**Status:** generated against **v0.25.0**. It happens to still apply to v0.26.0, but all four
files drifted substantially upstream (+355/-87 lines), so a clean `patch` result here means very
little — treat this as a design record, not a working patch. Never booted.

## Prerequisite for what is left here

`hybrid_draft_kv_groups` only matters for a *separate draft model*, which needs speculative
decoding AND a draft whose layers land in one KV-cache group. `docker-compose.yaml` no longer
forbids spec-decode (`--speculative-config=${VTL_SPEC:-None}`, plus `VTL_V2_RUNNER=0` because
v0.26.0's V2 runner rejects ngram/suffix), but the no-model drafters — ngram / ngram_gpu /
suffix — need none of this patch.

## Re-checking

`bash round-1.2/vtl/vllm_patches/gen.sh` dry-runs everything here on top of the applied
`v0.26.0/` set, so rot shows up as a failed run rather than a stale claim in this file.
