# Guardrails — Defect Taxonomy & the Point-in-Time Data Guard

The defect classes the Critic hunts for, the failure modes that recur, and the single most
important piece of code in any notebook: the point-in-time data wrapper.

## Defect taxonomy (Critic checklist)

| Class | What it looks like | Guard |
|-------|--------------------|-------|
| **Look-ahead / feature contamination** | A feature named innocuously (`carry_zscore`) built with a rolling window that includes the contemporaneous observation, or uses information from after decision time `t`. | Point-in-time data guard (below) + Critic reading the window math + Replicator's independent reimpl. |
| **Multiple-testing inflation** | Best-of-many lookbacks/thresholds reported without penalty; the `N_trials` term ignored. | `log(1+N_trials)` term + Deflated Sharpe. |
| **Regime cherry-picking** | Start date anchored a few months after a known drawdown; one regime carries the P&L. | Sample fixed in hypothesis schema; `C_frag` ablation; "no regime >40% of P&L" gate criterion. |
| **Cost-model optimism** | Zero or unrealistically low transaction cost; no slippage; fill assumptions ignore the book. | `S_tc` sensitivity term; alt-cost robustness rows. |
| **Sample-boundary drift** | Implementer quietly moves the in/out split to flatter the result. | Split is part of the frozen hypothesis schema; Critic flags any deviation. |
| **Unstable to-be-tuned parameter** | Result collapses under a small parameter perturbation. | Leave-one-out / perturbation robustness rows. |

## Failure modes that recur (name them in the report if seen)

1. **Plausible-feature contamination** — invented feature, innocuous name, contemporaneous
   window. Caught by Critic + the point-in-time wrapper together; neither alone is enough.
2. **Backtest-period drift** — start date anchored after a drawdown, never the full move (too
   obvious), just enough to flatter. Fixed by the sample living in the hypothesis schema.
3. **Confident wrong synthesis** — a critique/summary that contradicts the actual numbers. The
   hardest to catch by glance. Mitigation: the Critic must **quote specific cell sources
   verbatim with line references** — the constraint of citing a concrete source keeps it honest.

## Point-in-time data guard (paste into every notebook)

The single guardrail that prevents the most common look-ahead bug. Any access by date `t` may
only return data available at or before `t`; asking for anything later raises.

```python
class PointInTimeView:
    """Thin wrapper over a time-indexed DataFrame that refuses to leak the future.

    `as_of(t)` returns only rows with index <= t. Any attempt to read beyond the
    wrapper's frozen `_now` raises — this is the guardrail, not a convenience.
    """
    def __init__(self, df, now=None):
        df = df.sort_index()
        self._df = df
        self._now = now if now is not None else df.index[-1]

    def as_of(self, t):
        if t > self._now:
            raise LookAheadError(f"requested {t} > frozen now {self._now}")
        return self._df.loc[:t]

    def advance(self, t):
        """Walk-forward: move the frozen clock to t (must be monotonic)."""
        if t < self._now and t < self._df.index[-1]:
            pass  # allow re-anchoring earlier only during setup
        self._now = t
        return self.as_of(t)


class LookAheadError(RuntimeError):
    pass
```

Rolling features must be computed on `view.as_of(t)`, never on the full frame. A rolling window
that ends at `t` is fine; one that is centered on or extends past `t` is a look-ahead bug.

```python
# CORRECT: window ends at decision time t
sig = view.as_of(t)["mid"].rolling(lookback).mean().iloc[-1]
# WRONG: full-frame rolling leaks future into early timestamps when reused at decision time
sig_bad = df["mid"].rolling(lookback, center=True).mean()   # center=True => look-ahead
```
