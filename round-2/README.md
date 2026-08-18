# viettel-llm-inference (round 2)

An OpenAI-compatible inference server, scored by **ERS** (a per-request blend of TTFT and
TPOT — round-2 band: TTFT 200/6,000 ms, TPOT 8/100 ms, γ=2, w=0.5) on the official round-2
workload: an **aiperf AgentX replay of the SemiAnalysis Weka corpus** — real multi-turn
Claude Code sessions with subagents and recorded think time, 900 s, concurrency 5, context
cap 204,800. Full spec, commands, and what it means for tuning: [`HANDOFF.md`
§6](HANDOFF.md#6-official-grading-workload--scoring-round-2-btc-spec).

Rather than reimplementing an engine, we run stock `vllm/vllm-openai` and patch it in-process
through vLLM's [plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/). The
`vtl` package registers a `vllm.general_plugins` entry point; vLLM calls `vtl.plugin:register`
in every process before it does any work.

## Know the workload before tuning it

Two bench paths, with different jobs:

- `make bench-aiperf` — the grading workload itself (aiperf, Weka corpus, H200 only).
  **Every shipping decision is justified on these runs.**
- `make bench` — the in-repo synthetic 420-request trace (`bench/replay.py`). Fast and runs
  anywhere, but its arrival/token/prefix statistics predate the round-2 spec: a *relative*
  regression signal only.

The flag rationale that used to live here was derived from the round-1 trace statistics
(prefill-bound 101:1, 82% prefix-hit) and no longer holds under the grading workload —
`ignore_eos` full-length decodes shift the balance heavily toward decode. See `HANDOFF.md`
§6.4 for what has to be re-measured.

## Layout

| Path | What |
|---|---|
| `vtl/` | The plugin. `plugin.py` is the entry point, `registry.py` the patch registry, `patches/` the patches. |
| `bench/` | `trace_stats.py` (workload characterization), `replay.py` (open/closed-loop replay), `aiperf_adapter.py` (aiperf → repo schema), `metrics.py`, `compare.py`. |
| `Dockerfile` | Bakes the plugin **wheel** into the vLLM image. A bind-mount would not register the entry point. |
| `docker-compose.yaml` | **The submission.** Registry-only: no build context, judge provides `/model`, serves `:8000`. Single source of truth for every serve flag and env var. |
| `docker-compose-optimized.yaml` | Local-dev overlay — swaps the pinned digest for the `:dev` tag. Stacks on top of `docker-compose.yaml`; not standalone. |
| `docker-compose.localtest.yaml` | Local overlay — builds the image and mounts `hf-model/`. Not submitted. |

## Patches

Each patch registers into `PATCH_REGISTRY` under a name and is gated by `VTL_ENABLE_<NAME>`.
`register()` never raises: a patch that fails is logged and skipped, degrading to stock vLLM.
`VTL_DISABLE=1` turns the whole overlay off without rebuilding.

## Develop

```
make check      # self-checks; no GPU, no vLLM, no server needed
make up         # build + run locally against hf-model/ (Linux + GPU)
make trace-weka # build the grading-shaped replay trace (needed once, for warm + PGO)
make warm       # warm the torch.compile/Triton caches on a GPU, bake into the image
make bench      # synthetic trace: open-loop replay + closed-loop sweep at 1/8/32/128
make bench-aiperf  # grading workload (aiperf AgentX / Weka corpus) + ERS report
make push       # push and print the digest to pin in the compose file
```

Everything except `make check` needs a Linux box with the H200.

**Build on the GPU box.** `vllm/vllm-openai` is multi-arch, so an unpinned `docker build` on an
arm64 Mac silently produces an arm64 image that the amd64 H200 host refuses to run. The
Dockerfile, compose, and `make build/push` all pin `linux/amd64`, so a Mac build is *correct* —
but it runs every `RUN` under qemu emulation and pulls a ~10 GB foreign base. Native is faster.
