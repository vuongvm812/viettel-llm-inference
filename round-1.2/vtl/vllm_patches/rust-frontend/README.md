# Rust frontend source patches

Unlike the patches in `../v0.25.0/` (which apply to the **installed Python package**
in `site-packages` at fork-image build time), these apply to the **vLLM Rust
workspace source** (`vllm/rust/`) inside the `rust-builder` stage of
`round-1.2/Dockerfile.vllm-fork`, before `cargo build` compiles the `vllm-rs`
frontend binary. They carry our frontend optimizations so a clean checkout (where
`vllm/` is gitignored) still builds the optimized binary rather than stock.

## `frontend_optimize.patch`
- `Cargo.toml` — tuned `[profile.release]` (fat LTO, `codegen-units=1`).
- `src/server/Cargo.toml` — add the `sonic-rs` dependency.
- `src/server/.../validated_json.rs` — parse request bodies with `sonic_rs`
  instead of axum's serde_json `Json` extractor. sonic_rs parses from an
  immutable slice, so the body `Bytes` is borrowed directly (no per-request
  copy), and a non-JSON `Content-Type` is rejected up front — restoring the
  guard axum's `Json` gave us.

Applied idempotently: the stage skips the patch if the checkout already carries it
(grep-guard on `sonic_rs`), so building from a locally-modified checkout is a no-op.

## Regenerating
Paths are workspace-relative (`patch -p1 -d <rust-workspace>`). To regenerate after
editing the checkout, diff the modified files against pristine v0.25.0 sources, e.g.
`git diff --no-index` between a pristine tree and the modified one, with the `a/`/`b/`
prefixes normalized to the workspace root.
