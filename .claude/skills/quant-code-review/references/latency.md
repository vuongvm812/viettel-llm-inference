# latency Review Checklist

Focus: `services/crates/protocol/` (fixed-capacity message types) and the critical path in `trading-core`.

## protocol Crate — Fixed-Capacity Types

- [ ] Message types use fixed-capacity arrays (e.g., `arrayvec::ArrayVec`, `heapless::Vec`, or `[T; N]`) — no `Vec<T>` on the hot path
- [ ] No heap allocation in the message encoding/decoding path — `Box<T>` or `String` fields are a red flag
- [ ] `protocol` types implement `Copy` where the size is small (< 64 bytes) to avoid move overhead
- [ ] Struct layout is `#[repr(C)]` or `#[repr(packed)]` where cache-line alignment matters — document the choice
- [ ] No `Clone`-heavy paths (e.g., `msg.clone()` in a tight loop) — prefer references or `Copy`

## Critical Path Analysis

- [ ] Tick-to-trade path is identifiable: WS message receipt → parse → signal → order → `hl-client` submission
- [ ] No blocking I/O on the critical path: no MongoDB (`pkg`), no HTTP calls (`reqwest`) before order submission
- [ ] No `Mutex` or `RwLock` on the hot path — use lock-free structures (`orderbook` SeqLock, `std::sync::atomic`)
- [ ] No unbounded channel (`mpsc::channel()`) between critical-path tasks — use bounded channels with drop-on-full policy
- [ ] `tokio::task::yield_now()` or `tokio::time::sleep(0)` is not called in the hot path (forces task switch)

## Async Runtime

- [ ] CPU-bound work (signal computation, book aggregation) is offloaded to `tokio::task::spawn_blocking` — not blocking the Tokio worker threads
- [ ] `tokio::select!` branches in the hot path do not include slow futures (e.g., DB writes) alongside fast market data processing
- [ ] Task priorities are considered — `trading-core` critical path tasks should not be starved by `data-ingestion` tasks

## Measurement

- [ ] Timestamps are recorded at: WS message receipt, signal generation, order dispatch — enabling latency breakdown
- [ ] Latency logging uses `std::time::Instant`, not `chrono::Utc::now()` (wall clock, higher overhead)
- [ ] P99/P999 latency is tracked, not just average — tail latency matters for trading systems
