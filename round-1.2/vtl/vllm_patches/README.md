# vLLM source patches

Two of the three ways this repo modifies vLLM live here. (The third — runtime monkeypatches —
is `../patches/`, loaded via the `vllm.general_plugins` entry point.)

| Dir | Applies to | Applied by |
|---|---|---|
| `v$VLLM_VER/` | the **installed Python package** in site-packages | `Dockerfile.vllm-fork` stage 2 (and `Dockerfile` for the rust-frontend one, so it lands on a stock base too) |
| `rust-frontend/` | the **vLLM Rust workspace source** (`rust/`) | `Dockerfile.vllm-fork` stage `rust-builder`, before `cargo build` |
| `not-applied/` | nothing — parking lot | never; see its README |

`gen.sh` regenerates `v$VLLM_VER/` from the local (gitignored) `vllm/` checkout, which is a real
git clone pinned to the tag. **Run it after every edit to that checkout** — the committed
`*.patch` files are the source of truth, and an edit that is not captured never ships.

## Why patch source at all, instead of monkeypatching

Anything the `vtl` plugin can wrap at runtime belongs in `../patches/`. Source patches are for
what a plugin cannot reach: changing a function's *signature* (`short_conv.patch` hoists
`in_proj` out of a `torch.ops` custom op), editing code inside a `torch.compile` region, or
adding a fast path to a class the runtime instantiates before plugins load.

## Upgrading to a new vLLM tag

This is the whole checklist. It is ordered; steps 6-9 need the H200.

1. **Retag the checkout and 3-way merge the old patches.**
   ```
   git -C vllm fetch --depth 1 origin tag vX.Y.Z && git -C vllm checkout vX.Y.Z
   for p in round-1.2/vtl/vllm_patches/v<OLD>/*.patch; do git -C vllm apply -3 "$p"; done
   ```
   `git apply -3` gives real conflict markers instead of `patch` fuzz. Resolve any, and note
   that it **stages** what it merges — that is why `gen.sh` diffs `HEAD`, not the index.

2. **Check upstream drift for every file the copy-style monkeypatches reimplement.** These fail
   silently (wrong tokens, or a lost fusion), so a clean build proves nothing:
   ```
   git -C vllm diff --stat <OLD> vX.Y.Z -- \
     vllm/model_executor/models/lfm2.py \
     vllm/model_executor/layers/mamba/short_conv.py \
     vllm/v1/sample/metadata.py \
     vllm/compilation/passes/fusion/act_quant_fusion.py \
     vllm/model_executor/layers/vocab_parallel_embedding.py \
     vllm/v1/core/sched/scheduler.py vllm/v1/core/kv_cache_coordinator.py
   ```
   Empty output is the good case. Anything non-empty must be read, not skimmed. Pay particular
   attention to **return arity and signatures** the plugin unpacks: `find_longest_cache_hit` went
   2-tuple -> 3-tuple in v0.26.0, and because both call sites are `try/except`-wrapped, the only
   symptom was cache-aware scheduling silently degrading to shortest-prompt-first.

3. **Re-diff the two `_C` op schemas** that `../csrc/torch_bindings.cpp` hijacks via
   `TORCH_LIBRARY_IMPL(_C, CUDA, m)` — `rms_norm_dynamic_per_token_quant` and
   `dynamic_per_token_scaled_fp8_quant`, in `csrc/libtorch_stable/torch_bindings.cpp`. They must
   match byte-for-byte **including `Tensor!` / `Tensor?` markers**. A mismatch is a mid-request
   `cudaErrorNoKernelImageForDevice`, not a boot failure.

4. **Regenerate:** `VER=vX.Y.Z bash round-1.2/vtl/vllm_patches/gen.sh`. It fails if any modified
   file is uncaptured, if any patch came out empty, or if anything under `not-applied/` stopped
   applying. Forward-port or drop whatever it flags.

5. **Bump the pin — one place:** `Makefile`'s `VLLM_STOCK`. `VLLM_VER` derives from it and is
   passed to both Dockerfiles, which derive the assert and the `v$VLLM_VER/` paths in turn.
   Then re-verify the `rust-frontend/` patches still apply (see that dir's README) and that
   `vllm/rust-toolchain.toml` still matches `RUST_TOOLCHAIN`.

6. `make check` — off-box, catches import-level breakage.
7. `make vllm-fork PUSH=1`, then set `VLLM_FORK_DIGEST=@sha256:<digest>` in the Makefile.
8. `make build && make up && make verify && make test-kernel`.
9. **Greedy-output parity** — the one check that catches every silent-wrong-output risk from
   step 2 at once. `bench/eval_quality.py` captures against the OLD image, then the new one, and
   compares; sampling is greedy at temp 0, so any divergence shows as differing tokens.
   Only after this passes should a v-specific serve flag land in `docker-compose.yaml` —
   and it must land in the **same commit** that re-pins the compose digest, or the judge gets a
   `SystemExit(2)` from argparse and scores zero.
