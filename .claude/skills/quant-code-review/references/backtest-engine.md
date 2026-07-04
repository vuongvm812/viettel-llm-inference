# backtest-engine Review Checklist

Binary: `services/backtest-engine/` — historical simulation using ClickHouse data via `ch-client`.

## Lookahead Bias (Critical)

- [ ] Signals are computed using only data available at the bar's open timestamp — no peeking at close/future bars
- [ ] Any resampling or rolling window aligns to the **left** edge (no `min_periods` tricks that expose future data)
- [ ] ClickHouse queries filter strictly `WHERE timestamp < signal_time` — no `<=` on the current bar
- [ ] `ORDER BY` on time columns is explicit — never rely on ClickHouse insertion order
- [ ] Indicator warm-up periods are handled: the first N bars where the indicator is undefined are skipped, not filled with stale values

## Fill Simulation

- [ ] Fill model matches the live `hl-client` fee schedule — maker vs taker, tiered fees, funding
- [ ] Slippage model accounts for bid-ask spread; market orders do not fill at mid-price
- [ ] Partial fills are handled — orders are not assumed fully filled in a single bar
- [ ] Fill timestamps advance bar time — no fills at a price from a bar the order could not have seen

## PnL Accounting

- [ ] PnL uses the same price representation as the live system (e.g., both in USD, same precision)
- [ ] Unrealised PnL and realised PnL are tracked separately
- [ ] Fees are subtracted from PnL — gross PnL figures without fees are explicitly labelled
- [ ] Funding payments are included for perpetual futures strategies
- [ ] Splits/resets in position (e.g., from risk limits) are correctly reflected in cost basis

## ClickHouse Queries

- [ ] Queries use parameterised values — no string interpolation of user/config inputs into SQL
- [ ] Time range filters use `DateTime64` or appropriate ClickHouse time type consistently
- [ ] Large result sets are streamed or paginated — no `SELECT *` with unbounded row counts
- [ ] Query errors propagate as `Result` — not swallowed or defaulting to empty data silently

## Reproducibility

- [ ] Random seeds are fixed and logged for any stochastic fill models
- [ ] Backtest output includes the git commit hash and config snapshot used for that run
- [ ] Results are deterministic across runs with the same config and data
