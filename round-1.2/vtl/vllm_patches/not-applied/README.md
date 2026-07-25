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

## `shortconv_specdecode_rollback.patch` (3 files)

Short-conv state rollback for chain speculative decode (ngram / ngram_gpu / suffix).

`short_conv.py` passes `num_accepted_tokens` / `query_start_loc` / `max_query_len` to
`causal_conv1d_update` so the conv state rolls back to the accepted prefix instead of committing
rejected drafts, captures `num_speculative_tokens`, widens `get_state_shape`, and bypasses the
fused `bcx_conv_gate_quant` while spec-decode is active (that kernel has no `num_accepted` path).
`mamba_utils.py` adds the `num_spec` parameter to `short_conv_state_shape` that the widening
calls. `lfm2.py` threads the same `num_spec` through
`get_mamba_state_shape_from_config`, so `Platform._align_hybrid_block_size` and
`MambaSpec.page_size_bytes` agree — without it boot trips `assert page_size_padded >= page_size`.

All three must land together. An earlier revision of this patch omitted the `mamba_utils.py`
hunk; it applied cleanly and then raised
`TypeError: short_conv_state_shape() got an unexpected keyword argument 'num_spec'`
at the first `get_state_shape()`.

**Status:** applies cleanly on top of `v0.26.0/*.patch` and byte-compiles (re-checked by
`gen.sh`). **Not runtime-verified on v0.26.0** — it has never been booted on any version.

## `hybrid_draft_kv_groups.v0.25.0.patch` (4 files)

Unfinished work toward letting a *hybrid* draft model (e.g. LFM2.5-350M drafting for the 1.2B
target) survive vLLM's KV-cache-group assertions: isolates the drafter's layers into their own
group (`kv_cache_utils.py`), relaxes `validate_same_kv_cache_group` (`llm_base_proposer.py`),
compares mamba specs on everything except `shapes` (`worker/mamba_utils.py`), and allows
`ShortConvAttentionMetadataBuilder` in the ubatch allowlist (`gpu_model_runner.py`).

**Status:** generated against **v0.25.0**. It happens to still apply to v0.26.0, but all four
files drifted substantially upstream (+355/-87 lines), so a clean `patch` result here means very
little — treat this as a design record, not a working patch. Never booted.

## Prerequisite for either

Speculative decoding, which `docker-compose.yaml` currently forbids
(`DO NOT enable --speculative-config`). Without spec-decode both diffs are inert: `num_spec` is 0
and the extra kernel args are `None`.

## Re-checking

`bash round-1.2/vtl/vllm_patches/gen.sh` dry-runs everything here on top of the applied
`v0.26.0/` set, so rot shows up as a failed run rather than a stale claim in this file.
