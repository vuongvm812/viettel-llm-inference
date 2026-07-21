# Rust frontend source patches

Unlike the patches in `../v0.25.0/` (which apply to the **installed Python package**
in `site-packages` at fork-image build time), these apply to the **vLLM Rust
workspace source** (`vllm/rust/`) inside the `rust-builder` stage of
`round-1.2/Dockerfile.vllm-fork`, before `cargo build` compiles the `vllm-rs`
frontend binary. They carry our frontend optimizations so a clean checkout (where
`vllm/` is gitignored) still builds the optimized binary rather than stock.

## `frontend_optimize.patch`
- `Cargo.toml` — tuned `[profile.release]` (fat LTO, `codegen-units=1`).
- `src/server/Cargo.toml` — add the `simd-json` dependency.
- `src/server/.../validated_json.rs` — parse request bodies with `simd_json`
  instead of axum's serde_json `Json` extractor. The body `Bytes` is *moved*
  into simd_json's mutable buffer (zero-copy when uniquely owned, not a
  per-request `to_vec()`), and a non-JSON `Content-Type` is rejected up front —
  restoring the guard axum's `Json` gave us.

Applied idempotently: the stage skips the patch if the checkout already carries it
(grep-guard on `simd_json`), so building from a locally-modified checkout is a no-op.

## Regenerating
Paths are workspace-relative (`patch -p1 -d <rust-workspace>`). To regenerate after
editing the checkout, diff the modified files against pristine v0.25.0 sources, e.g.
`git diff --no-index` between a pristine tree and the modified one, with the `a/`/`b/`
prefixes normalized to the workspace root.
