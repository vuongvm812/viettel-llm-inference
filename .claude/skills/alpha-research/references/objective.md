# The Objective Function & Multiple-Testing Discipline

A single Sharpe number is **not** the objective. The system optimizes a composite that writes
your taste — penalize overfitting, fragility, and data-snooping — explicitly into the score
rather than trusting the optimizer to share it.

## The composite

```
U = IR_oos − λ1·|IR_is − IR_oos| − λ2·k_eff − λ3·S_tc − λ4·log(1 + N_trials) − λ5·C_frag
```

| Term | Meaning |
|------|---------|
| `IR_oos` | Information ratio on out-of-sample data the Implementer never saw. The sample boundary is fixed by the Proposer's hypothesis schema, not chosen by the Implementer. |
| `\|IR_is − IR_oos\|` | **Overfitting drift.** A 2.0/0.4 in/out strategy is worse than a 0.9/0.7 one. |
| `k_eff` | **Effective parameters.** Counts every knob set after seeing data (lookbacks, thresholds, feature inclusions, regime switches). Declared by the Implementer at submission, not estimated post hoc. |
| `S_tc` | **Transaction-cost sensitivity.** Slope of net returns w.r.t. a 1 bp shift in assumed cost. From 0.8 IR @ 2bps to 0.0 @ 3bps = fragile to a part of the world we don't know well. |
| `log(1 + N_trials)` | **Search-intensity penalty.** Effective number of related hypotheses run on the same dependent variable in this workstream. Links directly to the data-snooping literature. |
| `C_frag` | **Fragility.** Max proportional loss in IR when any single dimension (one date range, one currency, one regime, one cost assumption, one feature family) is ablated. |

### λ weights (calibrated once, frozen)
Default starting weights — recalibrate offline against a held-out historical set of strategies
whose ex-post outcomes are known, then **freeze before evaluating any candidate** and never
re-tune during a run:

```
λ1 = 0.5   λ2 = 0.05   λ3 = 0.3   λ4 = 0.2   λ5 = 0.5
```

These are a starting point, not magic. The point is that ranking by `U` should correlate with
realized forward performance on the calibration set.

## Who sees U, and who does not

- **Implementer: does NOT see U and is NOT told to maximize it.** Telling it to maximize U
  re-introduces specification-gaming. It is told only to *test the hypothesis*.
- **Critic:** uses U to flag any result where any term contributes negatively beyond a fixed
  threshold.
- **Human gate 2:** uses U to compare candidates — **after** writing the economic rationale.

## N_trials tracking

`N_trials` is the count of hypotheses touching the **same `dependent_variable`** across runs.
Increment it every time the Implementer commits a notebook for that dependent variable,
regardless of whether the result was positive. Maintain a per-dependent-variable tally by
scanning prior `research-log/hypothesis.json` files under `experiments/notebooks/*/`. Without
this term, a stack that runs 38 hypotheses will mechanically look better than one that runs 11,
even when the marginal hypothesis is no better.

## U scorecard (for the report & gate 2)

| Term | Value | Contribution to U |
|------|-------|-------------------|
| IR_oos | `<fill after running>` | + |
| λ1·\|IR_is−IR_oos\| | `<fill>` | − |
| λ2·k_eff | `<fill>` (k_eff = `<from notebook>`) | − |
| λ3·S_tc | `<fill>` | − |
| λ4·log(1+N_trials) | `<fill>` (N_trials = `<tally>`) | − |
| λ5·C_frag | `<fill>` | − |
| **U** | **`<fill>`** | |

## References (multiple-testing & backtest overfitting)

These anchor the search-intensity and overfitting terms — cite them in the report:

- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5), 1097–1126.
- Bailey, Borwein, López de Prado, Zhu (2016). "The Probability of Backtest Overfitting."
  *J. Computational Finance* 20(4), 39–69. (Gives the usable **Deflated Sharpe Ratio**.)
- Harvey, Liu, Zhu (2016). "…and the Cross-Section of Expected Returns." *RFS* 29(1), 5–68.

Report `IR_oos` alongside a **Deflated Sharpe Ratio** that accounts for `N_trials` whenever the
notebook can compute it.
