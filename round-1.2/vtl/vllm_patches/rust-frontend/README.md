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

## `mock_engine_pgo_pacing.patch`
- `src/mock-engine/...` — adds a `--decode-step-delay-ms` flag (default `0` = unchanged)
  to the mock engine and paces its decode loop by that delay. `pgo_train.sh` passes
  `~TPOT` (4 ms) so the frontend's streaming/detokenize loop is profiled at production
  cadence instead of memory speed — otherwise PGO trains on the wrong hot/cold split and
  can ship a binary slower than stock.

Applied idempotently: the stage skips **both** patches if the checkout already carries the
frontend optimization (grep-guard on `sonic_rs`), so building from a locally-modified
checkout that already has both is a no-op.

## Regenerating
Paths are workspace-relative (`patch -p1 -d <rust-workspace>`). To regenerate after
editing the checkout, diff the modified files against pristine v0.25.0 sources, e.g.
`git diff --no-index` between a pristine tree and the modified one, with the `a/`/`b/`
prefixes normalized to the workspace root.
