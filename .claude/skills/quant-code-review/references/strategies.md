# strategies Review Checklist

Crate: `services/crates/strategies/` — trading strategy implementations depending on `domain` types.

## Signal Logic

- [ ] Signals are purely functional where possible — no hidden mutable state that persists between bars
- [ ] Signal inputs are validated: NaN/Inf checks on price/volume inputs before computation
- [ ] Division operations check for zero denominator (e.g., VWAP, normalisation)
- [ ] Indicator calculations match their documented definitions — verify formula against reference
- [ ] Signals return a typed enum or newtype, not bare `f64` — prevents accidental comparison of incompatible signals

## Domain Type Correctness

- [ ] Price, quantity, and notional values use the correct `domain` types — not bare `f64` fields
- [ ] Comparisons between prices use domain-aware logic (e.g., tick-size snapping, not exact `==`)
- [ ] Time comparisons use `DateTime<Utc>`, not `NaiveDateTime` — no silent DST assumptions
- [ ] Symbol/instrument identifiers are typed (newtype or enum), not bare `String` — prevents cross-symbol mixing

## Portfolio Construction

- [ ] Position sizing is bounded by explicit max position and notional limits from config
- [ ] Rebalancing logic handles the case where current position == target (no spurious orders)
- [ ] Short positions are explicitly modelled — no assumption that positions are always long
- [ ] Leverage is computed and checked against risk limits before order generation

## State Management

- [ ] Strategy state that must survive between ticks is clearly separated from per-tick computation
- [ ] Shared strategy state accessed from multiple tasks is wrapped in `Arc<Mutex<T>>` or `Arc<RwLock<T>>`
- [ ] No `static mut` for strategy state — use proper synchronisation primitives
- [ ] State is initialised from config at startup, not hardcoded defaults that differ from config

## Testability

- [ ] Signal functions are unit-testable without running the full engine
- [ ] Edge cases tested: empty input, single-bar input, flat market, position at max limit
- [ ] Tests use fixed timestamps and prices — no `Utc::now()` in test data
