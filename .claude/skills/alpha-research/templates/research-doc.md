# Research Report Template → `docs/research/strategies/<topic>.md`

Fill every section from the research-log artifacts. Keep `<to fill after running>` placeholders
for any number that comes from the user executing the notebook — **never fabricate results**.
Match the rigor of `docs/research/strategies/market-making-stochastic-approximation.md`.

---

```markdown
# Strategies — {Human-Readable Topic Title}

> **Topic:** {topic-slug}  ·  **Strategy:** {strategy-slug}
> **Notebook:** ../../../experiments/notebooks/{topic}/{strategy}.ipynb
> **Pipeline:** alpha-research (Scout · Proposer · Implementer · Critic · Replicator)
> **Run:** {run_id}  ·  **Generated:** {YYYY-MM-DD}  ·  **Status:** {gate-2 verdict}

## TL;DR

- {Economic claim in one line.}
- {The predictor / signal in one line.}
- {Sample + OOS split.}
- {Verdict: promoted-to-candidate / rejected / sent-back — and the headline reason.}

## 1. Hypothesis (Proposer)

| Field | Value |
|-------|-------|
| Economic claim | {…} |
| Dependent variable | {…} |
| Predictor | {…} |
| Sample (incl. OOS) | {…} |
| Null (falsifier) | {…} |

## 2. Literature & References (Scout)

For each of the ~3 sources: mechanism, key equation(s), parameters.

| # | Source | Mechanism | Key equation(s) | Parameters |
|---|--------|-----------|-----------------|------------|
| 1 | {title, authors, venue/year} [url] | {…} | {…} | {…} |

**Synthesis:** {what the literature jointly implies}
**Open questions:** {gaps}

## 3. Method / Signal Construction

{How the predictor is built from data, step by step. Point-in-time discipline noted. Reference
`experiments/utils/metrics.py` / `brownian.py` functions reused.}

| Symbol | Meaning | Value / range |
|--------|---------|---------------|
| {…} | {…} | {per source, or "tuned (k_eff)"} |

## 4. Backtest Design

- **Universe & sample:** {…}  · **OOS split:** {fixed in hypothesis schema}
- **Cost model:** {bps assumed, slippage, fill assumption}
- **Point-in-time guard:** PointInTimeView — any access by `t` returns only data ≤ `t`.
- **k_eff (effective tuned knobs):** {count, declared by Implementer}

## 5. Critic Findings (static + output passes)

Severity 1 = must fix & re-run · 2 = waivable with written reasoning · 3 = minor. Static-pass
findings quote the offending cell **source**; output-pass findings quote the executed cell
**output** verbatim.

| Sev | Defect class | Cell | Quoted source | Why spurious | Status |
|-----|--------------|------|---------------|--------------|--------|
| {1} | {…} | {idx} | `{verbatim}` | {…} | {fixed/waived/open} |

## 6. Robustness Panel (Replicator)

**Independent reimplementation IR delta:** {<to fill after running>} (criterion: ≤15%)

| Row | Perturbation | IR | Sign stable? |
|-----|--------------|----|--------------|
| 1 | {exclude 2022} | {<fill>} | {<fill>} |
| … | … | … | … |

**Max single-regime P&L share:** {<to fill>} (criterion: ≤40%)

## 7. Objective-U Scorecard

{Paste the filled scorecard table from references/objective.md. N_trials tally and Deflated
Sharpe noted.}

## 8. Verdict (Gate 2)

**Economic rationale (written before viewing U):** {one paragraph}

Candidate criteria:
- [ ] Positive net-of-cost OOS IR
- [ ] No unresolved severity-1 finding
- [ ] IR sign stable in ≥6/8 robustness rows
- [ ] No regime >40% of P&L
- [ ] Independent reimpl IR within ±15%
- [ ] Economic rationale written pre-U

**Decision:** {promote-to-candidate / reject / send-back} — {reasoning}. A candidate is *not* a
deployed strategy; it has earned a further round of paper trading / live-data review.

## Appendix A — Typed Artifacts

```json
{hypothesis.json}
```
```json
{critique.json}
```
```json
{robustness.json}
```

## Provenance Notes

- Literature transcribed faithfully; "not specified" where a source omitted a value.
- All results in §6–7 marked `<to fill after running>` until the user executes the notebook.
- {Any ambiguity resolved, any role sent back and why.}
```
