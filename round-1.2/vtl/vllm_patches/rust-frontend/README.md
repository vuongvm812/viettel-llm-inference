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

## `fxhash_hot_paths.patch`
- `src/engine-core-client/src/client/state.rs` — replace `HashMap` with `FxHashMap` for
  `RequestRegistry.requests` (the hot-path request lookup used per decode step).
- `src/engine-core-client/src/protocol/sampling.rs` — `logit_bias`, `extra_args`.
- `src/server/src/config.rs` — `default_chat_template_kwargs`.
- `{engine-core-client,text,server,chat,cmd}/Cargo.toml` — add `rustc-hash.workspace = true`.
  The workspace root only declares the *version* (`[workspace.dependencies]`); each member
  crate still has to opt in, and only `src/tokenizer` did.

The last three are deserialized DTO fields, so the hasher is part of the public type and
the change propagates to every producer and consumer — 23 files across five crates
(`vllm-text` request/lowering, the OpenAI + gRPC + inference route DTOs, `merge_kv_transfer_params`
/ `convert_logit_bias`, and the chat-template renderer chain down to `TemplateContext`).
Most sites resolve by inference (`.collect()`); only the declared types needed editing.

Two boundaries are foreign types that cannot be converted, so the patch rehashes there:
- `src/server/src/grpc/convert.rs` — prost generates `HashMap` for the protobuf `logit_bias`.
- `src/chat/src/lib.rs` — `ReasoningParserKwargs` comes from the external `reasoning-parser`
  crate.

Note `TemplateContext.template_kwargs` is `#[serde(flatten)]`, so the hasher changes
chat-template kwarg iteration order. Jinja resolves by key, and the `expect_test` snapshot
suite passes, but that is the one place where order is observable.

## `sse_static_strings.patch`
- `src/server/.../chat_completions/types.rs` — replace `String` with `Arc<str>` for
  `ChatCompletionStreamResponse.id`/`.model` (ref-counted, no per-chunk allocation)
  and `&'static str` for `.object` (always `"chat.completion.chunk"`). Change
  `ChatCompletionStreamChoice.finish_reason` from `String` to `&'static str`.
- `src/server/.../chat_completions.rs` — convert `request_id` and `response_model`
  `String` → `Arc<str>` once in `chat_completion_chunk_stream`, then pass by
  `Arc::clone()` (cheap ref-count bump, zero allocation) through all chunk
  builders. Removes `finish_reason.to_string()` in `final_chunk`.
- Eliminates 3 `String` allocations per SSE chunk (`id`, `object`, `model`):
  ~378K allocs avoided across the 70-conversation workload.

## `http_trace_toggle.patch`
- `src/server/src/routes.rs` — gate the per-request `TraceLayer` behind
  `VLLM_RS_DISABLE_HTTP_TRACE` (default off = layer stays on). Set the env to `1` to drop
  the layer entirely — a latency A/B knob. The optimized compose defaults it to `1`.

Applied idempotently: the stage skips **all** patches if the checkout already carries the
frontend optimization (grep-guard on `sonic_rs`), so building from a locally-modified
checkout that already has both is a no-op.

## Static dispatch (not patched)

The per-token decode hot path has several `dyn` dispatches (`IncrementalDecoder`,
`Tokenizer::decode`, `UnifiedParser::parse_into`). Eliminating any requires making
`TextLlm` generic over `T: Tokenizer`, which cascades to `ChatLlm` and `AppState`
(the axum router state stored in `Arc<AppState>`). A full rewrite — not feasible as
a targeted patch. The inner code is already generic (`DecodeStream<T>`), but erased
at the `create_decode_stream → Box<dyn IncrementalDecoder>` boundary for runtime
tokenizer selection.

## Regenerating
Paths are workspace-relative (`patch -p1 -d <rust-workspace>`). To regenerate after
editing the checkout, diff the modified files against pristine v0.25.0 sources, e.g.
`git diff --no-index` between a pristine tree and the modified one, with the `a/`/`b/`
prefixes normalized to the workspace root.
