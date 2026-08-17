# Round-2 official spec update: new ERS Floor/Ceiling + AgentX/aiperf grading workload

## Context

The organizers (BTC) published the official round-2 spec, which changes two things the repo currently hard-codes to round-1 values:

1. **New ERS scoring band** (formula unchanged: `S = w·s_ttft + (1−w)·s_tpot`, `s_x = clamp((C−x)/(C−F),0,1)^γ`, failures score 0):

   | Param | Round-1 (current in repo) | Round-2 (new) |
   |---|---|---|
   | F_ttft / C_ttft | 10 / 400 ms | **200 / 6,000 ms** |
   | F_tpot / C_tpot | 1 / 10 ms | **8 / 100 ms** |
   | γ | 2.0 | 2.0 |
   | w | 0.5 | 0.5 |

2. **New grading workload (spec §3.1)**: no longer anything like the repo's synthetic 420-request trace. Grading is NVIDIA **aiperf** `profile --scenario inferencex-agentx-mvp` — an AgentX replay of the SemiAnalysis Weka corpus (real multi-turn Claude Code sessions, with subagents, recorded inter-turn delays, KV-cache reuse) for **900 s**. Confirmed parameters: `--concurrency 5`, `--max-context-length 204800`, `--public-dataset semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126`, pinned, auto-downloaded), `--random-seed` hidden. Scenario-locked flags: `--streaming`, `--extra-inputs ignore_eos:true`, `--cache-bust first_turn_prefix`, `--system-idle-gap-cap-seconds 10`, recorded delays preserved, chat endpoint, `--use-server-token-count`. Output carries `submission_valid` (false on >1% context overflow, cancellation, unsafe override).

Target model per spec is unchanged (Qwen3.5-122B-A10B-FP8 on 1× H200) — but note the compose is still pinned to the LFM2.5 placeholder (see Part D).

**Agreed scope** (per user): update ERS constants + rewrite round-2 workload/scoring docs + integrate aiperf as a new grading-fidelity bench path with an adapter into the existing record schema, so `_ci_report.py` ERS reporting keeps working. Keep the synthetic `replay.py` path alive (fast iteration + rtx3060 CI). `round-1.2/**` and `round-2/reference/**` stay untouched (historical).

---

## Part A — ERS constants + derived comments (do first, ~30 min)

**A1. `round-2/bench/_ci_report.py:19-23`** — the only live computation:

```python
# ERS constants — round-2 official spec (BTC, 2026-08).
F_TTFT, C_TTFT = 200.0, 6000.0  # ms
F_TPOT, C_TPOT = 8.0, 100.0     # ms
GAMMA = 2.0
W = 0.5
```

`_ers()` needs no change (failure-counts-as-zero already correct). Consumers (`sweep_report.py`, `::VTL_BENCH::` JSON → `bench.yml`) pick it up automatically.

**A2. Refresh stale derived figures.** New band math: TTFT band 5,800 ms, TPOT band 92 ms → at equal normalized headroom **1 ms TPOT ≈ 63 ms TTFT** (was "~28"); ceiling ratio 60×; marginal dERS/dTPOT ≈ 0.011·u ERS/ms (~10× less than round-1 — TPOT micro-opts devalued an order of magnitude).

- `round-2/bench/sweep_report.py:19`, `:81`, `:153` — "28 ms" → "63 ms"
- `round-2/docker-compose.yaml` "~23x" ERS-gradient comment (~line 190) → "~63x" (comment-only)
- `round-2/vtl/patches/quant_w4a8.py:212` — same "~23x" comment → "~63x"
- `round-2/RUST-RUNNER.md:169` — rewrite "TPOT worth ~0.05 ERS/ms": now ≤ ~0.011 ERS/ms; the 0.22 ms/knob A/B floor now gates only ~0.002 ERS — re-derive thresholds

## Part B — Docs (parallel with C)

**B1. `round-2/HANDOFF.md`** — add "## 6. Official grading workload & scoring (round-2 BTC spec)" after §5, plus a pointer line in the intro:
- Scoring: formula, new parameter table, failure=0 rule, 63 ms exchange rate + "TPOT devalued ~10×" note, pointer to `bench/_ci_report.py` as reference impl.
- Workload: the full aiperf/AgentX description above (all confirmed flags, hidden seed, `submission_valid` semantics, dataset auto-download from HF).
- Two bench paths: `make bench` (synthetic, fast/CI) vs `make bench-aiperf` (grading fidelity, H200 only) — state explicitly the synthetic trace's arrival/token statistics no longer match grading and must not drive final tuning.
- Add `aiperf_adapter.py` to the §1.6 bench file list (line 136).

**B2. `round-2/README.md`** — replace stale round-1 workload text (lines ~3-45: "120-request trace", 18,707-token medians, prefill-bound rationale) with a short round-2 summary linking HANDOFF §6; keep the plugin-architecture paragraph.

**B3. `round-2/bench/README.md`** — rewrite around the two bench paths (currently describes round-1 trace + a port-8001 mock).

## Part C — aiperf integration (main build, ~half day + H200 session)

**C1. Install/pin**: new `round-2/bench/requirements-aiperf.txt` with pinned `aiperf==<latest at install>` (pip, repo `ai-dynamo/aiperf`). Separate from `requirements.txt` — heavy dep tree, only needed on the H200 host, never in CI or the serving image.

**C2. Adapter — new `round-2/bench/aiperf_adapter.py`** (pure stdlib). Reads the **per-request** `profile_export.jsonl` (one MetricRecordInfo/line — the aggregate `profile_export_aiperf.json` lacks per-request records), emits the repo run schema consumed via `metrics.aggregate` by `_ci_report.py:59-76` / `sweep_report.py:78-79` / `compare.py:33-35`:

| repo field | aiperf source | conversion |
|---|---|---|
| `ttft` (s) | `metrics.time_to_first_token` (ms) | /1000 |
| `itl_mean` (s) | `metrics.inter_token_latency`; fallback mean of `inter_chunk_latency` | /1000 |
| `e2e` (s) | `metrics.request_latency` (ms) | /1000 |
| `output_tokens` | `metrics.output_token_count` (fallback `output_sequence_length`) | int |
| `success` | `error == null` and not `metadata.was_cancelled` | bool |
| `send_time` / `wall_time` | `metadata.request_start_ns` / span of start→end ns | /1e9 |
| `mode` | `--mode-label`, default `aiperf-agentx-c5` | |

Defensive handling (documented in module docstring; field names re-verified at first real run): accept plain numbers or `{"value","unit"}` MetricValue objects; skip warmup-phase records if flagged; missing-TTFT-with-no-error → failure, never silently dropped. Also parse `submission_valid` from `profile_export_aiperf.json` — loud banner + non-zero exit when false. Include `--selfcheck` with a synthetic JSONL fixture (mirrors `sweep_report.py --selfcheck` convention) asserting mapping, unit conversion, cancelled→failure, and failures-in-denominator ERS.

**C3. Reporting hookup**: `_ci_report.py` `main()` — add `glob("bench-aiperf*.json")` to the run set (schema identical, tables/JSON work unchanged). `compare.py`/`sweep_report.py` need nothing.

**C4. Make target** — root `Makefile`, vars near `TRACE ?=` (:79), target near `bench:` (:380), `.PHONY` updated:

```make
AIPERF ?= aiperf
AIPERF_SEED ?= 0            # BTC seed hidden; sweep ours
AIPERF_DURATION ?= 900
AIPERF_CONCURRENCY ?= 5
AIPERF_DATASET ?= semianalysis_cc_traces_weka_062126
AIPERF_MODEL ?= <served-model-name from compose>
AIPERF_LIMIT ?=             # N ⇒ --num-dataset-entries N (smoke)
AIPERF_ART ?= bench-aiperf

bench-aiperf:  ## grading-fidelity bench; mirrors BTC cmd except hidden seed. H200 only.
	$(IN) $(AIPERF) profile --scenario inferencex-agentx-mvp \
	  --model $(AIPERF_MODEL) --url $(TARGET) --endpoint-type chat \
	  --public-dataset $(AIPERF_DATASET) \
	  --concurrency $(AIPERF_CONCURRENCY) --max-context-length 204800 \
	  --benchmark-duration $(AIPERF_DURATION) --random-seed $(AIPERF_SEED) \
	  --streaming --extra-inputs ignore_eos:true --cache-bust first_turn_prefix \
	  --system-idle-gap-cap-seconds 10 --use-server-token-count \
	  $(if $(AIPERF_LIMIT),--num-dataset-entries $(AIPERF_LIMIT)) \
	  --artifact-dir $(AIPERF_ART)
	$(IN) python3 bench/aiperf_adapter.py --artifact-dir $$(ls -td $(AIPERF_ART)/*/ | head -1) --out bench-aiperf.json
	$(IN) python3 bench/_ci_report.py
```

Two verify-at-first-run unknowns, encoded as comments: (a) the scenario preset may own some explicit flags (drop duplicates if rejected; check `--help`); (b) exact artifact-dir layout.

**C5. `make check`**: add `python3 bench/aiperf_adapter.py --selfcheck` to the check loop (adapter imports cleanly without aiperf installed — it only parses files).

## Part D — Serving-config findings (document in HANDOFF §6; fixes are a separate task)

- **Blocking precondition**: `round-2/docker-compose.yaml` is still pinned to the LFM2.5-1.2B placeholder (its own header, lines 7-19, lists 4 model literals to revisit) with `--max-model-len=32768` (:47) and `--max-num-seqs=70` (:61). 32768 < 204800 guarantees >1% context overflow → `submission_valid: false`. A valid `bench-aiperf` run needs the Qwen3.5-122B literals landed first.
- `--max-num-seqs` comment "peak trace concurrency ~6" refers to the dead synthetic trace; under 5 session trees + subagent fan-out, re-sweep against measured aiperf concurrency (also `cudagraph_capture_sizes`).
- `ignore_eos:true` ⇒ full-length decodes: EOS-dependent early-stop logic (e.g. `VTL_ENABLE_STEP0_EOS_BAN`) becomes inert; decode volume ≫ the prefill-biased synthetic trace, so prefill-heavy tuning rationale in README/compose comments no longer holds.

## Verification

1. `python3 round-2/bench/sweep_report.py --selfcheck` and existing self-checks still pass (no schema change).
2. ERS band sanity one-liner: score(TTFT=200, TPOT=8) = 1.0; score(≥6000 / ≥100) component = 0; score(3100, 54) = 0.25; run with 1 ok + 1 failed record halves ERS.
3. `python3 round-2/bench/aiperf_adapter.py --selfcheck`.
4. H200 smoke: `make bench-aiperf AIPERF_LIMIT=8 AIPERF_DURATION=120` → confirm flag acceptance, artifact layout, JSONL field names/units, `submission_valid` location; update the adapter's mapping-table comment with observations.
5. Round-trip: `bench-aiperf.json` → `_ci_report.py` prints an ERS row; cross-check TTFT/TPOT p50s against aiperf's own aggregate JSON (validates unit conversion).
6. Full run after the model-literal update, seed swept over 2-3 values to bound seed sensitivity.

## Order

1. Part A (unblocks all downstream numbers) → 2. Parts B & C2/C5 in parallel (adapter selfcheck is off-box) → 3. C3/C4 → 4. H200 smoke (resolves the two unknowns) → full run gated on the compose model update (separate task).

Per repo convention, save this plan as `docs/round-2-spec-update-plan.md` at implementation time.
