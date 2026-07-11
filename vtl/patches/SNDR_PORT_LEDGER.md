# sndr_core_engine patch port — ledger

Record of which Genesis/SNDR (`sndr_core_engine`) patches were evaluated for our
`vtl` overlay and why. **Stack:** stock `vllm/vllm-openai:v0.22.1` + `vtl`, serving
`hf-model/` = `Qwen2ForCausalLM` — a **dense** GQA transformer (no MoE, no GDN/linear
attention, no MTP). `--served-model-name=Qwen3.5-2B` is only a label. Round-1 trace
(`data/input/trace-round1.jsonl`) is plain multi-turn chat — no tool-calls, no reasoning.
**Round 2 adds JSON / structured-output requests**, which re-opens the structured-output
family that round 1 let us skip (see the `xgrammar_jump_forward` +
`reject_degenerate_structured_outputs` rows below and `data/input/trace-json.jsonl`).

The sndr suite targets Sander's **custom vLLM fork** running a **Qwen3.5-MoE+GDN+MTP**
model. Its patches *wire into* backends they don't ship (e.g. `turboquant_attn` is
imported, never defined in sndr), so on stock vLLM they self-skip.

## Ported

| vtl patch | from | effect |
|---|---|---|
| `inputs_embeds_optional` | **PN35** (`worker/pn35_inputs_embeds_optional.py`, upstream vllm#35975) | Drop the dead `(max_num_tokens, hidden)` `inputs_embeds` buffer on text-only models (~32 MiB GPU + 32 MiB pinned). Guard preserved exactly: keep iff `supports_mm_inputs or enable_prompt_embeds`. Verified: anchor + attrs present in v0.22.1; wrap binds to real `GPUModelRunner`. Marginal on a 141 GB H200 — a zero-risk freebie, not a metric mover. |
| `xgrammar_jump_forward` | **new** (SGLang `xgrammar_backend` jump-forward; not an sndr patch) | Skip decode steps for grammar-forced tokens: post-pass wrap of `Scheduler.update_from_output` queries `matcher.find_jump_forward_string()`, appends the forced tokens (strict safe-seam retokenization → byte-identical greedy output), advances the FSM; vLLM's chunked-prefill computes their KV in one forward. **`default=False`** until byte-parity + async accounting validated on the H200. Jumps only under the **sync** `Scheduler` for now — `AsyncScheduler` (default-on here) pipelines `schedule()` ahead of `update_from_output()`, so a naive append reorders sampled-vs-jumped tokens; async-safe coordination is the GPU-spike item. Only helps round-2 JSON, not plain chat. |
| `reject_degenerate_structured_outputs` | **PN387** (vllm#45346) **+ PN523**, consolidated | Reject `json_object=False` / empty-string `json`/`grammar`/`regex`/`structural_tag` / empty `choice` at `SamplingParams._validate_structured_outputs` (request admission) → clean 400 instead of a compiler raise inside the `EngineCore` step loop (`EngineDead`, which is unscoreable on the judge box). `default=True`; well-formed structured requests pass through untouched. Signature-agnostic wrap. Re-scoped from "Skipped" now that round 2 sends structured outputs. |

## Skipped (no-op / irrelevant on this stack)

| Family (listed IDs) | Reason |
|---|---|
| TurboQuant 18 + `SNDR_WORKSPACE_001` (P3,P18B_TEXT,P22,P26,P38,P44,P67,P98,P99,P101,PN116,PN118,PN119,PN399,PN401,G4_61,G4_62) | Wire into `vllm.v1.attention.backends.turboquant_attn`, absent in stock vLLM → import fails → self-skip. |
| GDN/FLA 7 (P28,P39a,P46,P60,P60b,P103,PN11,PN59,PN106) | Dense Qwen2 never instantiates a linear-attention layer. |
| MoE/Marlin (P17,P24,P31,P37,PN96b,PN368,PN377,G4_84) | Dense model, not Marlin-quantized. |
| Spec-decode/MTP (P70,P82,P107,P108,PN33,PN348,PN390,PN402) | No MTP heads, no spec-decode configured. |
| Flash-attn (PN17,PN286) | PN286 = sm86 (we're sm90); PN17 = FA2 (we run FA3). |
| cudagraph/compile (P66,P72,P74,PN25,PN125,PN128,PN130,PN367,PN60,PN63) | P66/PN367 MTP-specific; P72 MoE `moe_align_block_size`; P74 protects TQ/GDN prealloc pools; PN130 warms TQ, PN128 warms EAGLE; PN25 OOM-pool (not OOM-bound on H200); PN60/PN63 marker-only. |
| Quant (P1,P81,P91,PN77) | P81 tunes *block-scaled* fp8; our path is *per-channel* W8A8. P91 autoround (unused). PN77 fp8 lm_head conflicts with our bf16 tied lm_head. P1 marker-only. |
| KV/memory (P5,P14,PN95,PN96,PN346,PN346B,PN201) | P5 hybrid page-size (uniform for us); PN95/96/201 OOM defrag (not OOM-bound); PN346/B mamba-mtp; P14 micro. |
| Scheduler/worker (P34,P58) | P34 mamba; P58 spec-token `[-1]*num_spec_tokens` placeholder (no spec-decode). |
| Serving/tool/reasoning (P68,P69,P109,PN16,PN71,PN73,PN91,PN92,PN127,PN525,P15,P29) | No tools/reasoning even in round 2 (JSON via `response_format`, not tool-calls); these target tool-call/reasoning paths. P62 (structured-output + spec-decode reasoning-end timing) still N/A: no spec-decode, no reasoning in this stack. PN523 folded into `reject_degenerate_structured_outputs` above. |
| PN389 grammar-compile timeout / PN526 threadsafe SO tokenizer | **Deferred, not skipped.** Real robustness value under round-2 JSON (compile-DoS guard; tokenizer thread-safety) but both need GPU/vLLM to validate the hook — port after the jump-forward spike if a compile stall or SO-tokenizer race actually shows up. |
| Marker-only stubs | P18b,P32,P51,P17,P1,P29,PN60,PN63 — no code exists to port. |

Note: our existing `sched_policy.py` is already the `vtl` realization of sndr's
cache-aware scheduling idea.

## Deferred (GPU-box check, not ported)

Generic JIT warmup (PN126/PN129) — only helps if kernels still JIT on request 1
*despite* our shipped warmed caches (`docker/cache/`, `make warm`, `VLLM_USE_AOT_COMPILE=1`).
Verify on the H200: bring the server up, send 1 request, grep logs for post-warmup Triton
JIT compilation. Port a minimal synthetic-decode warmup only if some fire. PN128/PN130 warm
EAGLE/TurboQuant kernels we don't have — skip outright.
