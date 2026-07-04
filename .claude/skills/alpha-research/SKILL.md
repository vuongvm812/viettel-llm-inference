---
name: alpha-research
description: Use when the user runs `/alpha-research <topic>` or asks you to research, design, and validate a new quantitative trading signal/alpha from scratch — running literature search, hypothesis design, a test notebook, adversarial verification, and robustness checks. Outputs a research report under docs/research/strategies/ and a test notebook under experiments/notebooks/<topic>/. Keywords: alpha research, find alpha, new signal, new strategy, quant research, hypothesis, backtest a signal, factor research, agentic research.
---

# Alpha Research — Instrumented Multi-Role Pipeline

Ports Jonathan Kinlay's *"Agentic Workflows for Alpha Research"* architecture. The
**architecture is the load-bearing piece**, not any single prompt. A single agent told to
"find alpha" is a *specification-gaming machine* — it returns in-sample Sharpe 2.4 built on a
look-ahead bug. Role separation, typed handoffs, and human gates are what kill that.

**Core invariant:** No role sees its own prior outputs as ground truth. Each handoff is a
fresh subagent context that receives only a schema-typed artifact and nothing else. You (the
orchestrator) enforce this by controlling exactly what each `Agent` call receives.

## What this is / is not

Autonomous **only inside pre-specified rails**: a controlled pipeline of LLM roles with two
human gates. It does **not** choose its own data permissions, change validation criteria,
redefine the promotion threshold, or promote its own results. That separation is the design.

## Inputs

- `<topic>` — kebab-case slug from the user's argument (e.g. `order-flow-imbalance`,
  `funding-rate-carry`). Becomes the notebook folder and research-doc filename.
- Data source for the notebook: **CCXT live + `experiments/utils/`** (matches existing
  notebooks). Generated notebooks are **generate-only** — you write the code, the user runs it.

## Setup (once per run)

```bash
TOPIC="<topic>"                                  # from the user's argument
RUN_ID=$(python3 -c "import uuid;print(uuid.uuid4().hex[:8])")
NB_DIR="experiments/notebooks/$TOPIC"
mkdir -p "$NB_DIR/research-log"
echo "run_id=$RUN_ID  topic=$TOPIC"
```

Read `references/roles.md` (the role prompts + JSON schemas), `references/objective.md`
(the U objective + multiple-testing guidance), and `references/guardrails.md` (anti-patterns +
the point-in-time data-guard snippet) **before dispatching any role**.

## The pipeline

Kinlay's pipeline is **four roles** — Proposer, Implementer, Critic, Replicator. We add a
**Scout literature prepass** (orchestrator-side) in front of the Proposer; it is a prepass, not
a fifth peer role. Dispatch each role as a **separate `Agent` call**
(`subagent_type: general-purpose`) using the matching prompt from `references/roles.md`. Each
typed output is written to `$NB_DIR/research-log/` as JSON and is the *only* thing passed
downstream.

**Two phases, because the notebook is generate-only.** Phase A (this invocation) builds the
notebook and statically reviews it, ending at a **"ready for execution" checkpoint** — no
backtest numbers exist yet. Phase B runs **after you (the user) execute the notebook and return
its outputs**: the orchestrator fills results, runs the Critic's output pass, computes U, and
runs the promotion gate. Gate 2 cannot be evaluated without executed outputs — do not attempt it
in Phase A.

### Phase A — build (this run)

| # | Step | Dispatch receives | Emits |
|---|------|-------------------|-------|
| 0 | **Scout** (prepass) | `topic` only. WebSearch + WebFetch. | `literature.json` |
| 1 | **Proposer** | `literature.json` + sanitized prior-trial summary (no results) | `hypothesis.json` |
| — | **GATE 1** | *you* present literature + hypothesis | — |
| 2 | **Implementer** | `hypothesis.json` **only** (no results, no critique, no U) | notebook cells 1–6 |
| 3 | **Critic (static)** | notebook **source** only | `critique.json` (`static_findings`) |
| 4 | **Replicator** | `hypothesis.json` + frozen data contract + notebook path + public-output interface + `critique.json` | reimpl + panel cell *source* (orchestrator appends), `robustness.json` (schema, results blank) |
| — | **CHECKPOINT** | *you* write report w/ placeholders + tell the user to run the notebook | report + notebook |

### Phase B — promote (after the user runs the notebook and returns outputs)

| # | Step | Dispatch receives | Emits |
|---|------|-------------------|-------|
| 5 | **Critic (output pass)** | executed cell **outputs** + notebook | appends `output_findings` to `critique.json` |
| 6 | **Synthesis** | *you* fill results, compute U | finalized report |
| — | **GATE 2** | *you* present results + U + criteria | promotion decision |

### 0. Scout (literature prepass)
Dispatch with the Scout prompt + `topic`. It picks the ~3 references that matter and extracts
each one's core method/equations/parameters. Write to `research-log/literature.json`.

### 1. Proposer
Build a **sanitized prior-trial summary** first: scan `experiments/notebooks/*/research-log/hypothesis.json`
and emit, for each, only `{dependent_variable, predictor (one line), date}` — **no results**.
Also compute the `N_trials` tally per `dependent_variable`. Dispatch with the Proposer prompt +
`literature.json` + this summary, so it avoids re-proposing recently tested hypotheses. It emits
**ONE** falsifiable hypothesis in the fixed schema and applies its own rejection criteria. No
code, no data access. Write to `research-log/hypothesis.json` (record `N_trials` in the envelope).

### GATE 1 — does this hypothesis deserve compute?
`AskUserQuestion`: show the literature summary + rendered hypothesis JSON + the prior-trial
summary (so the user sees N_trials). Options: **Approve** / **Reject** / **Send back with
comment**. On send-back, re-dispatch the Proposer with the comment appended. Do not proceed
until approved.

### 2. Implementer
Dispatch with the Implementer prompt + `hypothesis.json` **and nothing else** — no prior
results, no critique, and it is **not** told to maximize or even see U. It writes
`$NB_DIR/<strategy>.ipynb` (cells 1–6) following `templates/notebook-cells.md`: the universe,
timeframe, date range, and OOS split are **derived from the hypothesis `sample` field**, not
hardcoded; if the sample is ambiguous it stops and asks. `<strategy>` is a kebab slug it derives
from the predictor.

### 3. Critic — static pre-run pass
Dispatch with the Critic (static) prompt + the **notebook source only**. It catches defects
visible in code: look-ahead/feature contamination, sample-boundary drift in the code, cost-model
omission, unstable parameters (taxonomy in `references/guardrails.md`). Every finding **quotes
the offending cell source verbatim** with cell index + line. It cannot validate reported numbers
— that is the output pass (step 5). It fixes nothing. Write `static_findings` to
`research-log/critique.json`.

### 4. Replicator
Dispatch with the Replicator prompt + `hypothesis.json` + the frozen data contract + the
**notebook path** + a **public-output interface contract** (the variable names it may compare
against, e.g. `ir_oos`, the signal series name) + `critique.json`. It does **not** see the
Implementer's feature code (cell 5). It returns, as text, (a) an **independent reimplementation**
cell built from the schema alone + an agreement-check against the public interface, and (b) an
8-row robustness-panel cell. **The orchestrator appends these cells** to the notebook (cells 7–8b)
— the Replicator has no write access. Write `robustness.json` (schema; result fields blank until
Phase B).

### CHECKPOINT — notebook ready for execution
Write the report to `docs/research/strategies/<topic>.md` from `templates/research-doc.md` with
all result/U fields as `<to fill after running>`. Then tell the user: run the notebook and return
the cell outputs (or paths). Run the Phase-A validity checks below. **Do not run Gate 2 yet.**

### 5. Critic — output pass (Phase B)
Once the user returns executed outputs, dispatch the Critic (output) prompt + the **cell outputs**
+ notebook. Its mitigation is **quote-the-cell-output verbatim**: validate reported IR, flag any
summary that contradicts the numbers, sample drift visible in outputs, and cost sensitivity.
Append `output_findings` to `critique.json`.

### 6. Synthesis + GATE 2 — promote to candidate?
Fill the report's result tables and the U scorecard (`references/objective.md`) from the returned
outputs. Then `AskUserQuestion`: show Critic findings (static + output, by severity), the
robustness panel, and U. Candidate criteria (all must hold):
1. Positive net-of-cost OOS IR over the full Proposer-defined sample.
2. No unresolved severity-1 finding (severity-1 fixed & re-run; severity-2 explicitly waived
   in writing with reasoning).
3. Stable IR sign in ≥6 of 8 robustness-panel rows.
4. No single regime contributes >40% of total backtest P&L.
5. Independent reimplementation IR within ±15% of the original.
6. A one-paragraph economic rationale (yours) written **before** viewing the final U score.

Options: **Promote to candidate** / **Reject** / **Send back with comment**. Record the verdict in
the report.

## Verification (must pass before reporting done)

```bash
# Notebook is well-formed JSON and the Python parses (generate-only — no execution):
jupyter nbconvert --to script --stdout "$NB_DIR/<strategy>.ipynb" >/dev/null && echo "notebook OK"
# Artifacts are valid JSON:
for f in literature hypothesis critique robustness; do
  python3 -c "import json;json.load(open('$NB_DIR/research-log/$f.json'))" && echo "$f.json OK"
done
```

If `jupyter` is unavailable, fall back to
`python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$NB_DIR/<strategy>.ipynb"` to at
least confirm the `.ipynb` is valid JSON.

## Hard rules

- **Never** let the Implementer see results, the critique, or the U objective (U does not appear
  in any Implementer cell — it is computed by the orchestrator in Phase B).
- **Never** let the Replicator reuse the Implementer's feature-construction code; it gets only the
  hypothesis, data contract, and the public-output interface, and returns cell *source* that the
  orchestrator appends.
- **Never** run Gate 2 in Phase A — it requires executed outputs the user has returned.
- **Never** skip a gate or auto-promote — gates are `AskUserQuestion`, decided by the user.
- **Never** fabricate backtest numbers. The notebook is generate-only; numbers come from the
  user running it. The report's results/U tables stay as `<to fill after running>` placeholders
  until the user provides outputs.
- **Never** hardcode the universe/timeframe/sample/OOS split — derive them from the hypothesis
  `sample`; if ambiguous, stop and ask.
- One hypothesis per run. Re-invoke the skill for more (and increment `N_trials` —
  `references/objective.md`).
