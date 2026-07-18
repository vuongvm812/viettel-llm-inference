# vLLM source patches (tree-spec fork)

The tree-verify path needs edits to vLLM internals that change tensor shape / read new state and so
cannot be runtime monkeypatches (see `docs`/the plan). We keep them as **version-pinned unified
diffs** here and bake them into a forked base image (`round-1.2/Dockerfile.vllm-fork`), rather than
vendoring the whole vLLM tree.

## Layout

- `v0.25.0/*.patch` — one diff per forked vLLM file, generated against the pinned tag.

## Editing workflow (source of truth = the patches, edit surface = the gitignored `vllm/` checkout)

The repo-root `vllm/` checkout is **byte-identical to v0.25.0** for the forked files (verified). It is
gitignored (like `sglang/`); we edit it only to regenerate diffs.

```bash
# 1. edit the real files under vllm/vllm/... (IDE/LSP work normally)
# 2. regenerate the patch for a file:
git -C vllm diff -- vllm/v1/sample/rejection_sampler.py \
    > round-1.2/vtl/vllm_patches/v0.25.0/rejection_sampler.patch
# 3. rebuild the fork image (applies patch -p1 at site-packages) and push; pin the main image by digest.
```

`Dockerfile.vllm-fork` asserts `vllm.__version__=='0.25.0'` and `patch --dry-run`s every diff before
applying, so drift fails the build loudly instead of mis-applying.

## The forked files (why each is a source fork, not a plugin hook)

| Patch | vLLM file | Why forked |
|---|---|---|
| `gpu_model_runner.patch` | `v1/worker/gpu_model_runner.py` | tree-node scheduling (node count ≠ draft+1) + post-sample conv-commit call site |
| `rejection_sampler.patch` | `v1/sample/rejection_sampler.py` | tree accept in `RejectionSampler.forward` (the flat-chain kernels can't express a tree) |
| `flash_attn.patch` | `v1/attention/backends/flash_attn.py` | boolean tree `custom_mask` over the block_size=1 draft KV |
| `short_conv.patch` | `model_executor/layers/mamba/short_conv.py` | stage per-node Bx to scratch + accepted-path commit/rollback |

All algorithmic logic lives in the `vtl` plugin (`vtl/tree_spec.py`, `vtl/tree_sample.py`,
`vtl/tree_attention.py`, `vtl/conv_commit.py`); the patches are **thin shims** that call into it, so
the diffs stay small and the logic stays unit-tested off-box.

## On-box completion markers

Grep the patches (and the vtl modules they call) for `ON-BOX:` — those are the hardware-coupled
specifics (FlashInfer paged buffer layout, cudagraph capture of tree buffers, `mamba_cache_mode=align`
block-migration write contract) that must be confirmed on the H200. Each is guarded by a loud
`assert`/log so a wrong assumption fails fast rather than corrupting output silently.
