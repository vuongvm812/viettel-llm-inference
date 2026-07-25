# Patches that are NOT applied by the build

Nothing in this directory is copied into the image. `Dockerfile.vllm-fork` only
applies `../v0.26.0/*.patch`. This is a parking lot for work that is real and
verified-to-apply, but not wired in.

## `shortconv_specdecode_rollback.patch`

Short-conv state rollback for chain speculative decode (ngram / ngram_gpu / suffix),
plus the paired `num_spec` threading in `lfm2.py`.

**Why it is here and not in `v0.26.0/`:** it was found only in the local (gitignored)
`vllm/` working tree during the v0.25.0 -> v0.26.0 upgrade. It was never captured by
`gen.sh` into a `*.patch`, so **it never shipped in any built image** — despite notes
elsewhere claiming the fix had landed. It is preserved here rather than deleted.

**What it does:** `ShortConv.forward_cuda` passes `num_accepted_tokens` /
`query_start_loc` / `max_query_len` to `causal_conv1d_update`, so the conv state rolls
back to the accepted prefix instead of committing rejected drafts. `__init__` captures
`num_speculative_tokens` and `get_state_shape` widens the conv state by `num_spec`;
`Lfm2ForCausalLM.get_mamba_state_shape_from_config` passes the same `num_spec` so
`Platform._align_hybrid_block_size` and `MambaSpec.page_size_bytes` agree (otherwise
boot trips `assert page_size_padded >= page_size`). The fused conv-gate kernel has no
`num_accepted` path, so it is bypassed while spec-decode is active.

**Prerequisite for enabling:** speculative decoding, which `docker-compose.yaml`
currently forbids (`DO NOT enable --speculative-config`). Without spec-decode the whole
diff is inert — `num_spec` is 0 and the extra args are `None`.

Verified to apply cleanly on top of `v0.26.0/*.patch` (2026-07-25):

    patch -p1 --dry-run < not-applied/shortconv_specdecode_rollback.patch
