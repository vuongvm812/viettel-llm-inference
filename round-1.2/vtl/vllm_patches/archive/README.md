# Archived vLLM source work — retired, not parked

Nothing here is applied by any build, and nothing here is checked by `gen.sh`.

This is deliberately **not** `not-applied/`. That directory is a parking lot for work we still
intend to land, and `gen.sh` dry-runs every `not-applied/*.patch` on top of the applied set on
every run — which is the right cost for live work and the wrong cost for retired work. Files here
carry a `.patch.txt` suffix so the `not-applied/*.patch` glob cannot pick them up. They are a
provenance record, nothing more. Treat them as design notes; do not expect them to apply.

## `spec-decode-wip.v0.26.0.patch.txt` (9 files, +1238 lines)

The uncommitted contents of the gitignored `vllm/` checkout as of 2026-07-27, covering everything
`gen.sh` did **not** capture:

    vllm/config/vllm.py                              vllm/v1/spec_decode/dflash.py
    vllm/model_executor/model_loader/weight_utils.py vllm/v1/spec_decode/llm_base_proposer.py
    vllm/v1/attention/backends/mamba_attn.py         vllm/v1/spec_decode/step3p5.py
    vllm/v1/cudagraph_dispatcher.py                  vllm/v1/worker/gpu_model_runner.py
                                                     vllm/v1/worker/mamba_utils.py

All of it is speculative-decoding / hybrid-draft-model work: the `ShortConvAttentionMetadataBuilder`
entry in `gpu_model_runner.py`'s `extra_attn_metadata_args` tuple (upstream PR #44296's third file),
`mamba_attn.py`'s `vtl_capture_query_len`, a PIECEWISE fallthrough in `cudagraph_dispatcher.py` where
stock asserts, and a large `llm_base_proposer.py` build-out toward a hybrid LFM2.5-350M drafter.

**Why it was retired.** Spec decode was measured to be a dead end on this workload, not merely
unfinished:

- `bench/build_trace_round2.py` synthesises the trace by cycling a bank of 26 fixed notes, so ~79%
  of a turn-6 prompt is verbatim padding. That makes prompt-side n-gram self-match look excellent
  (50-55%) and it is **the wrong statistic** — vLLM's ngram proposer scores on what the model
  *writes*. Prose-side n-gram coverage against prior context is 0-1.7%, and a simulated proposer
  yields ~1.28 accepted tokens per forward pass, i.e. no meaningful speedup. The system prompt
  actively suppresses the copying behaviour ngram needs ("cite the relevant part of the material in
  your own words rather than quoting it at length", `build_trace_round2.py:204-205`).
- `ngram` sets `async_scheduling = False` (`config/vllm.py:1101-1112`) — it disables host/device
  overlap on a decode loop that is host-bound. `ngram_gpu` keeps async scheduling but allocates
  `max_num_seqs × max_model_len × 4B` = **9.17 GB** at our settings (`gpu_model_runner.py:617-622`).
- vLLM v0.26.0's V2 model runner rejects `ngram`, `ngram_gpu` **and** `suffix`
  (`config/vllm.py:2185-2196`), and `_validate_v2_model_runner` raises rather than falling back
  because our compose sets `VLLM_USE_V2_MODEL_RUNNER` explicitly. Every V2-allowed method
  (eagle/eagle3/mtp/dflash/dspark) needs a trained head or an architecture LFM2 does not have.
- The vtl greedy fast path goes dead on every spec step: `rejection_sampler.py:136-147` calls the
  sampler with `predict_bonus_token=True` and `max_num_logprobs=-1`, both of which fail
  `_should_fastpath` (`vtl/patches/greedy_sampler.py:32-42`).

## `short-conv-spec-wip.v0.26.0.patch.txt` (1 file, 600 lines)

The **full** `short_conv.py` diff as the checkout held it — a strict superset of the shipped
`v0.26.0/short_conv.patch`. Unlike the nine files above, this one *is* captured by `gen.sh`, so the
extra work would have been swept into the image on the next `make vllm-fork` without anyone
choosing it. Two additions beyond the shipped patch:

- `_VTL_BCX_CONV_GATE_SPEC` / `VTL_BCX_SPEC` and a one-shot fused-decode-miss log line — both only
  meaningful with a drafter running.
- `_vtl_flag(name)` gains `default=True` for `VTL_ENABLE_MUL_QUANT` and `VTL_ENABLE_BCX_CONV_GATE`,
  where the shipped patch defaults them off. Our compose sets both explicitly, so this is inert
  *here*, but it silently changes what any environment that omits them serves.

The checkout was reset to the committed patch, which `gen.sh`'s header names as the source of
truth, so `git diff HEAD` in `vllm/` now reproduces the shipped patch set exactly.

**The trap this file documents.** The shipped `short_conv.patch` rollback fix gates on
`attn_metadata.num_accepted_tokens is not None`, but stock `gpu_model_runner.py:2470-2478` populates
that only for `(Mamba2, GDN, BailingLinear)` builders — `ShortConvAttentionMetadataBuilder` is absent.
The fix for that is in this archive and was never turned into a `.patch`, so **the rollback fix is
inert in every image ever built**, silently. Anyone re-opening spec decode must land that file first
or the 0% long-context probe returns. This is the second time unshipped checkout state was mistaken
for landed work; see `../not-applied/README.md` for the first.
