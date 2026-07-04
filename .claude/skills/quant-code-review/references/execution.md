# execution Review Checklist

Crate: `services/crates/services/` — core order submission and lifecycle logic (not strategy signal generation).

## Order Lifecycle

- [ ] Order state machine covers all transitions: `Pending → Submitted → PartiallyFilled → Filled | Cancelled | Rejected`
- [ ] Invalid transitions (e.g., `Filled → Submitted`) are rejected at the type level or with an explicit error
- [ ] `order_id` from the exchange is stored and used for all subsequent cancel/amend operations
- [ ] Duplicate submission guard: a pending order for the same symbol/side is not re-submitted without explicit cancellation

## Risk Gates

- [ ] Every order submission path passes through risk checks before reaching `hl-client`
- [ ] Risk checks cover: max notional per order, max position per symbol, max daily loss, max open orders
- [ ] Risk gate failures return a typed `RiskRejected` error — not a generic `anyhow::Error` that gets swallowed
- [ ] There is no code path (e.g., emergency close, manual override) that bypasses risk checks silently
- [ ] Risk state is updated atomically with order submission — no TOCTOU between check and send

## Fill Handling

- [ ] Fills from `hl-client` update position state atomically — partial fills accumulate correctly
- [ ] Position updates are not lost on reconnect — fill replay or state reconciliation on startup
- [ ] Fill price and filled quantity are used for PnL, not the order's limit price
- [ ] Slippage (fill price vs order price) is recorded for post-trade analysis

## Error Handling & Retries

- [ ] Transient errors (network timeout, rate limit) trigger retry with backoff — not immediate failure
- [ ] Non-retryable errors (margin insufficient, invalid symbol) fail immediately and alert
- [ ] Retry logic has a maximum attempt count and circuit-breaker — not infinite retry
- [ ] All `Result` errors from `hl-client` are explicitly handled — no `let _ =` on order responses
- [ ] Errors are logged with order context (symbol, side, size, price) for post-mortem

## Concurrency

- [ ] Shared order book / position state is protected by `RwLock` or `Mutex` — no data races
- [ ] `Mutex` is not held across `.await` points — deadlock risk
- [ ] Order submission tasks do not block the main Tokio executor (no `std::thread::sleep` in async path)
