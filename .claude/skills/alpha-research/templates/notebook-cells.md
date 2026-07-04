# Notebook Skeleton → `experiments/notebooks/<topic>/<strategy>.ipynb`

Generate-only: the user runs it. Build the `.ipynb` with these cells in order.
**Ownership matters** (it enforces the no-anchoring invariant):
- Cells 1–6 are the **Implementer's** — signal + backtest. **No U, no λ weights.**
- Cells 7–8 are the **Replicator's** source, **appended by the orchestrator** (the Replicator
  never sees cell 5).
- Cell 8b (Objective-U) is added by the **orchestrator in Phase B**, never by the Implementer.
- Cell 9 is shared plotting.

Match the style of `experiments/notebooks/avellaneda-stoikov/avellaneda-stoikov.ipynb` (CCXT,
plotly/matplotlib). Note the path depth: from `experiments/notebooks/<topic>/`, `experiments/utils`
is **two** levels up, so use `sys.path.append("../..")`.

**Cell 1 — Markdown header**
> Hypothesis (the 5 fields), run_id, the ~3 references with links. State the OOS split and the
> point-in-time rule up front.

**Cell 2 — Imports**
```python
import sys; sys.path.append("../..")   # experiments/ — where utils/ lives (two levels up)
import math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ccxt
import utils.metrics as metrics          # realized-vol, OFI, WAP — reuse, don't reinvent
import utils.brownian as bm              # synthetic paths / null-distribution checks
```

**Cell 3 — Point-in-time data guard** (paste `PointInTimeView` + `LookAheadError` from
`references/guardrails.md`). All later data access goes through `view.as_of(t)`.

**Cell 4 — Frozen data contract (from the hypothesis) + CCXT load**
The contract is **derived from the hypothesis `sample` field — never hardcoded, never from the
fetched-row count** (deriving OOS from row count recreates sample-boundary drift). If the sample
is ambiguous, the Implementer stops and asks rather than guessing.
```python
import datetime as dt
# FROZEN DATA CONTRACT — transcribed from hypothesis.json `sample`; this is ALL the Replicator gets:
UNIVERSE    = "<symbol from hypothesis sample>"          # e.g. "BTC/USDT"
TIMEFRAME   = "<timeframe from hypothesis sample>"       # e.g. "1m"
DATE_RANGE  = (dt.datetime.fromisoformat("<start>"), dt.datetime.fromisoformat("<end>"))
OOS_SPLIT   = dt.datetime.fromisoformat("<oos boundary from hypothesis>")   # fixed by Proposer, not by us
assert DATE_RANGE[0] < OOS_SPLIT < DATE_RANGE[1], "OOS split must lie inside the sample"

exchange = ccxt.binance()
since = int(DATE_RANGE[0].timestamp() * 1000)
ohlcv = []                       # loop with `since` until DATE_RANGE[1], as in avellaneda-stoikov.ipynb
while True:
    batch = exchange.fetch_ohlcv(UNIVERSE, TIMEFRAME, since=since, limit=1000)
    if not batch: break
    ohlcv += batch; since = batch[-1][0] + 1
    if pd.to_datetime(since, unit="ms") >= DATE_RANGE[1]: break
df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"]).set_index("ts")
df.index = pd.to_datetime(df.index, unit="ms")
df = df.loc[DATE_RANGE[0]:DATE_RANGE[1]]
view = PointInTimeView(df)
```

**Cell 5 — Signal construction (Implementer)**
> Build the predictor from the hypothesis. Compute every rolling feature on `view.as_of(t)`,
> window ending at `t`. NO `center=True`, NO full-frame rolling reused at decision time.
> Comment each tuned knob and keep a running `k_eff` count.
```python
K_EFF = 0   # increment for every knob set after seeing data
# ... signal logic ...
```

**Cell 6 — Backtest + cost model → IR_is / IR_oos**
```python
COST_BPS = 2.0     # assumed transaction cost; vary in robustness
# walk-forward using view.advance(t); split returns at OOS_SPLIT
# ir = mean(ret)/std(ret)*sqrt(periods_per_year) on net-of-cost returns
ir_is  = ...   # in-sample
ir_oos = ...   # out-of-sample (data the signal logic never saw during construction)
print(f"IR_is={ir_is:.3f}  IR_oos={ir_oos:.3f}  K_EFF={K_EFF}")
```

> **Note:** the Implementer stops here. There is deliberately **no objective-U cell among the
> Implementer's cells** — seeing/optimizing U is what reintroduces specification-gaming. U is
> computed later (cell 8b), added by the orchestrator in Phase B.

**Cell 7 — Independent reimplementation (Replicator)**
> Reimplement the predictor **from the hypothesis text + frozen data contract only** — do not
> copy cell 5. Then check agreement.
```python
sig_reimpl = ...   # independent construction
ir_reimpl  = ...
delta = abs(ir_reimpl - ir_oos) / abs(ir_oos)
print(f"reimpl IR={ir_reimpl:.3f}  delta={delta:.1%}  (criterion: <=15%)")
```

**Cell 8 — Robustness panel (Replicator)** → 8-row comparison table
```python
rows = []
for label, perturb in [
    ("exclude 2022", ...), ("+1bp cost", ...), ("-1bp cost", ...),
    ("drop feature A", ...), ("drop feature B", ...),
    ("alt lookback", ...), ("alt universe", ...), ("regime-only OOS", ...),
]:
    ir = ...                       # rerun backtest under perturbation
    rows.append({"perturbation": label, "ir": ir, "sign_stable": np.sign(ir)==np.sign(ir_oos)})
panel = pd.DataFrame(rows); print(panel)
max_regime_pnl_share = ...         # feeds the >40% gate criterion
```

**Cell 8b — Objective-U scoring (orchestrator, Phase B — NOT the Implementer)**
Added only after the user has run cells 1–8. Reads the public outputs and the panel; the
Implementer never sees this cell. λ from `references/objective.md`.
```python
L1,L2,L3,L4,L5 = 0.5,0.05,0.3,0.2,0.5
N_TRIALS = 1   # tally of hypotheses on this dependent_variable (orchestrator-supplied; see objective.md)
S_tc   = ...   # d(net return)/d(1bp cost), from the +/-1bp panel rows
C_frag = ...   # max proportional IR loss across the panel's single-dimension ablations
U = ir_oos - L1*abs(ir_is-ir_oos) - L2*K_EFF - L3*S_tc - L4*math.log(1+N_TRIALS) - L5*C_frag
print(f"U = {U:.3f}")
```

**Cell 9 — Plots** (matplotlib for distributions, plotly Scattergl for >10k points, per existing
notebooks): cumulative net PnL (IS vs OOS shaded), IR-by-perturbation bar, signal vs forward
return scatter.

---

**Discipline reminders baked into the cells:**
- Sample boundary (`OOS_SPLIT`) is fixed by the hypothesis, never moved to flatter results.
- Every rolling feature ends at decision time `t` via `view.as_of(t)`.
- `K_EFF` (cell 5) is declared honestly; `N_TRIALS` is orchestrator-supplied in cell 8b — both
  cost U points, and that's the point. The Implementer never sees U.
- Cell 7's reimplementation must not import or copy cell 5's logic.
