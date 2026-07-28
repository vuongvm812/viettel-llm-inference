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

## `http_trace_toggle.patch`
- `src/server/src/routes.rs` — gate the per-request `TraceLayer` behind
  `VLLM_RS_DISABLE_HTTP_TRACE` (default off = layer stays on). Set the env to `1` to drop
  the layer entirely — a latency A/B knob. The optimized compose defaults it to `1`.

Applied idempotently: the stage skips **all** patches if the checkout already carries the
frontend optimization (grep-guard on `sonic_rs`), so building from a locally-modified
checkout that already has both is a no-op.

## `shm_ipc.patch`
- `Cargo.toml` / `src/engine-core-client/Cargo.toml` — add the `iceoryx2` (pinned **0.9.3**)
  and `sha2` dependencies. `Cargo.lock` is deliberately **not** patched: the in-image
  `cargo build --release` is not `--locked`, so it resolves the new dep itself and the patch
  stays free of a 1000-line lockfile hunk that every other patch would conflict with.
- `src/engine-core-client/src/shm_ipc.rs` (new) — the whole opt-in iceoryx2 data plane:
  service naming, the `InputSink` enum, the output reader thread and the Phase-B fixed-layout
  record decoder, plus unit tests over golden vectors shared with the Python packer.
- `src/engine-core-client/src/{lib.rs,client.rs,client/imp.rs}` — the three-line seam:
  `mod shm_ipc`, build the sink + spawn the shm output reader in `from_connected`, and try the
  shm publish before the ZMQ `transport::send_message` in `send_to_engine`.

Gated on **`VTL_SHM_IPC=1`** (read by the Rust frontend *and* by `vtl/patches/shm_ipc.py` in
the EngineCore process). Unset — the default — nothing in the module runs and the transport is
byte-for-byte stock. `VTL_SHM_IPC_RAW=1` (Python-side only; requires `VTL_SHM_IPC=1`) switches
the output records from msgpack to the fixed layout, with automatic per-message fallback.
ZMQ is never torn down: it carries the handshake/registration, stays as the request-path
degrade path, and its output loop remains the `ENGINE_CORE_DEAD` detector.

The pinned crate version must equal the `iceoryx2` PyPI wheel version installed in
`round-1.2/Dockerfile` — the two halves share a shm data-segment layout.

## Regenerating
Paths are workspace-relative (`patch -p1 -d <rust-workspace>`). To regenerate after
editing the checkout, diff the modified files against pristine v0.25.0 sources, e.g.
`git diff --no-index` between a pristine tree and the modified one, with the `a/`/`b/`
prefixes normalized to the workspace root.
