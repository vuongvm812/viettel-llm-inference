# hl-client Review Checklist

Crates: `services/crates/hl-client/` (thin wrapper) + `services/exchanges/hyperliquid/` (vendored SDK).

## Authentication & Secrets

- [ ] Private key / API secret is never logged, serialised, or included in error messages
- [ ] Key material is zeroed on drop (`zeroize` crate or manual `Drop` impl)
- [ ] Auth headers/signatures are generated per-request — not cached and reused across reconnects
- [ ] Wallet address derivation from private key is tested against known vectors

## Order Submission

- [ ] Order parameters (price, size, side) are validated before submission — no negative size, price in valid tick range
- [ ] Market orders are explicitly typed separately from limit orders — no accidental market order from a malformed limit price
- [ ] Order responses are checked: `order_id` is present and non-empty before treating as success
- [ ] Rejected orders return a structured error, not just an HTTP 200 with error body
- [ ] Duplicate order detection: client-side order ID (`cloid`) is used and checked to prevent double-submission on retry

## Rate Limits

- [ ] Request rate is tracked and throttled against Hyperliquid's documented limits
- [ ] Rate limit errors (HTTP 429) trigger backoff + retry, not immediate crash
- [ ] WebSocket subscription limits are respected — not subscribing to unlimited symbols
- [ ] Order placement rate (orders/second) is bounded in config, not unlimited

## WebSocket Management

- [ ] WebSocket connection has a heartbeat / ping-pong mechanism with timeout detection
- [ ] Reconnection on disconnect uses exponential backoff with a maximum retry limit
- [ ] Reconnection re-subscribes to all active subscriptions — subscription state is not lost
- [ ] Messages received during reconnection window are not silently dropped (buffer or gap detection)
- [ ] The WS receive loop does not block the Tokio executor — `tokio::spawn` or async I/O throughout

## Order Normalisation

- [ ] Hyperliquid-specific fields (e.g., `asset` index, `sz_decimals`) are encapsulated inside `hl-client` — the rest of the codebase uses `domain` types
- [ ] Price and size are correctly converted between Hyperliquid wire format and internal `domain` types
- [ ] Order status mapping covers all Hyperliquid states: filled, partially filled, cancelled, rejected, resting
