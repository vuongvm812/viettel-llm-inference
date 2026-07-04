# orderbook Review Checklist

Crate: `services/crates/orderbook/` — L2 OrderBook with SeqLock for lock-free reads. Contains `unsafe` code.

## SeqLock Correctness (Critical)

- [ ] Writer increments sequence to odd BEFORE writing, then back to even AFTER writing (standard SeqLock protocol)
- [ ] Consumer reads: `seq1 = load(seq)` → read data → `seq2 = load(seq)` → retry if `seq1 != seq2` or `seq1` is odd
- [ ] Sequence loads use `Acquire` ordering; sequence stores use `Release` ordering — not `Relaxed`
- [ ] Data reads inside the SeqLock critical section use `read_volatile` or `AtomicUsize::load` with appropriate ordering, not plain field reads that the compiler can reorder
- [ ] No consumer skips the retry loop (e.g., reads data once without checking the sequence)
- [ ] Writer holds the SeqLock exclusively — no two writers can race (enforced by `&mut self` or a `Mutex` around writes)

## Unsafe Code

- [ ] Every `unsafe` block has a `// SAFETY:` comment explaining why the invariant holds
- [ ] No raw pointer arithmetic without bounds checks documented in SAFETY comment
- [ ] `unsafe impl Send` / `unsafe impl Sync` has a documented justification
- [ ] `transmute` is absent or justified — prefer `bytemuck` or explicit layout types
- [ ] All unsafe code has been reviewed with `unsafe-checker` skill

## L2 Book Correctness

- [ ] Bid levels are sorted descending by price; ask levels ascending — violated sort is a logic bug
- [ ] Best bid < best ask at all times — crossed book is flagged as an error, not silently used
- [ ] Level updates correctly handle: new level insertion, existing level size update, level deletion (size → 0)
- [ ] Sequence number gaps in feed updates are detected and logged — stale/partial book state is not used
- [ ] Book is reset / invalidated on reconnect — stale levels from previous connection are not carried forward

## Performance

- [ ] Hot path (best bid/ask access) is O(1) — not iterating the entire price level array
- [ ] No heap allocation in the update path — `Vec::push` inside the SeqLock writer is a latency spike
- [ ] `protocol` fixed-capacity arrays are used for price levels where applicable
- [ ] Benchmark exists for update throughput and read latency under contention
