# data-ingestion Review Checklist

Binary: `services/data-ingestion/` — real-time WebSocket data ingestion to ClickHouse via `ch-client`.

## WebSocket Reliability

- [ ] Connection has a heartbeat / ping-pong with a timeout — dead connections are detected, not silently stalled
- [ ] Disconnect triggers reconnection with exponential backoff (not immediate tight-loop retry)
- [ ] Maximum reconnection attempts or a circuit-breaker is configured — not infinite retry on persistent failure
- [ ] On reconnect, all subscriptions are re-established — subscription state is not lost
- [ ] Reconnection events are logged with timestamp and reason for gap analysis

## Missed Tick Detection

- [ ] Exchange messages include a sequence number or update ID — gaps are detected and logged
- [ ] On sequence gap: the current data window is flagged as incomplete, and downstream consumers are notified
- [ ] ClickHouse rows include the exchange-provided timestamp, not just the local ingestion timestamp
- [ ] A monotonically increasing sequence is stored per symbol to enable gap detection in backtest queries

## ClickHouse Write Path

- [ ] Writes are batched — not one INSERT per message (ClickHouse is optimised for bulk inserts)
- [ ] Batch flush is triggered by both size (e.g., 1000 rows) and time (e.g., every 500ms) — neither alone is sufficient
- [ ] Write errors are retried with backoff — a ClickHouse hiccup does not drop data silently
- [ ] Buffer between WS receive and ClickHouse write is bounded — backpressure is applied when ClickHouse is slow
- [ ] Schema matches `domain` types exactly — no silent field truncation or type coercion at insert time

## Backpressure & Memory

- [ ] The ingestion channel between the WS task and the write task is bounded (`mpsc::channel(N)`)
- [ ] When the channel is full, the policy is explicit: drop oldest, drop newest, or apply backpressure to the WS reader
- [ ] Memory usage is bounded — no unbounded accumulation of unwritten rows
- [ ] Buffer metrics (queue depth, flush latency, drop count) are logged or exposed for monitoring

## Data Quality

- [ ] Prices and sizes are validated on receipt — zero size, negative price, or NaN triggers a warning, not a write
- [ ] Symbol normalisation is applied before writing — Hyperliquid symbol format is mapped to internal canonical form
- [ ] Duplicate rows (same exchange timestamp, same symbol) are detected or handled at the ClickHouse schema level (`ReplacingMergeTree` or dedup key)
