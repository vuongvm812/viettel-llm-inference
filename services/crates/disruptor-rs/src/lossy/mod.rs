//! Single-producer, **N-independent-consumer broadcast** ring for the
//! market-data hot path.
//!
//! Unlike [`spsc`], where the producer *moves* each item to exactly one
//! consumer, here the producer publishes each item **once** and *every* consumer
//! reads the whole stream through its own cursor.
//!
//! # Backpressure: lossy / overrun-tolerant
//!
//! The producer never blocks. A consumer that falls more than `N - 1` items
//! behind is *lapped*: it detects the overrun via the **global `tail`
//! counter** (`tail - cursor >= N` ⇒ the slot at `idx(cursor)` is either
//! already overwritten by position `cursor + N` or is currently being
//! overwritten as the producer's next write target), resynchronises to the
//! oldest still-live position `tail - (N - 1)`, and counts the gap
//! ([`Consumer::lagged`]). This matches the existing `queue_overflow` /
//! `mark_data_loss` philosophy — producer isolation is preserved at the cost
//! of dropping data for a stalled consumer (the right trade for an HFT
//! feed).
//!
//! Effective capacity is therefore `N - 1`, not `N` — slot `idx(tail)` is
//! reserved for the producer's in-progress write under the Phase 2
//! tail-versioned protocol. Size buffers with this in mind.
//!
//! ## Deliberate divergence from LMAX Disruptor
//!
//! The canonical LMAX Disruptor (`SingleProducerSequencer::next`) **gates the
//! producer on the slowest consumer's `Sequence`** — the producer waits when
//! `next - bufferSize > min(gatingSequences)`, making fan-out lossless at the
//! cost of stalling the upstream when any consumer falls behind. This ring
//! inverts that choice: the producer never reads consumer cursors and **never
//! blocks**, so a slow regime/mm consumer cannot back-pressure the parser and
//! starve the latency-critical book stage co-fed by the same upstream. Roles
//! that cannot tolerate loss (the spot/perp order books) ride dedicated SPSC
//! queues instead — see `hot_path::adaptive_bus` — leaving this ring's lossy
//! semantics correct for the loss-tolerant streams it does carry.
//!
//! # Soundness: tail-versioned memcpy payload
//!
//! Each slot holds a cheap-filter `header` (atomic) plus payload bytes. There
//! is **no per-slot sequence number** — correctness is enforced by the single
//! global `tail` counter and the rule that position `p` is live iff
//! `tail - p ∈ 1..N`. The producer copies the payload with
//! [`ptr::copy_nonoverlapping`], stores the header with `Relaxed`, then
//! publishes by storing `tail.wrapping_add(1)` with `Release`. A consumer
//! copies the payload with [`ptr::read`] into a `MaybeUninit<T>`, then
//! re-reads `tail` with an `Acquire`-fenced load and only `assume_init`s the
//! copy if `tail_after - position < N` (the slot has not been lapped, and
//! the producer is not currently overwriting it).
//!
//! ## Inter-publish Release fence
//!
//! The producer also inserts a `fence(Release)` between consecutive publishes
//! (before each payload write). The fence forces the **previous** publish's
//! `tail.store(Release)` to drain globally before the **next** payload write
//! can be observed. On x86 TSO this is a compiler-only barrier (store-store
//! is already preserved); on AArch64 it lowers to `DMB ISH` and closes a
//! gap: `STLR` is a one-way release barrier and does NOT prevent the next
//! regular `STR` (payload) from being observed before it. Without the fence,
//! a consumer could see the overwrite bytes while still observing a stale
//! `tail`, and incorrectly accept the read. See `Producer::publish_with_header`
//! for the full rationale.
//!
//! Reserving one slot (effective capacity **`N - 1`**, not `N`) is structural:
//! `idx(tail)` shares an index with the producer's next write target, so a
//! consumer reading position `tail - N` would race the producer's in-progress
//! payload write at `idx(tail)` with no version field to disambiguate. The
//! Phase 2 protocol caps the read window at `[tail - (N - 1), tail - 1]`.
//!
//! This is the classical tail-versioned ring (cyclotrace; Linux kernel
//! seqlock without the per-slot version field — the global `tail` is the
//! version, traded against losing one slot of capacity). A producer
//! overwriting a slot mid-read races a consumer's payload memcpy — strictly a
//! data race under the C/C++/Rust abstract memory model, but the post-read
//! `tail` check always catches the overlap and the torn `MaybeUninit` is
//! discarded before `assume_init`, so no value with torn bytes is ever
//! exposed to user code.
//!
//! To keep the verification story honest the atomic-word implementation is
//! retained under `cfg(any(loom, miri))`: loom continues to model-check the
//! tail-version protocol on instrumented atomics, and Miri keeps validating
//! the single-threaded ownership story. The non-`loom`, non-`miri` build
//! uses the memcpy path, which on x86 TSO and AArch64 vectorises the payload
//! move and is the dominant throughput win over a per-word `AtomicUsize` loop
//! (≈3–5× fewer instructions on x86, ≈10× fewer barriers on ARM).
//!
//! The payload bound is [`bytemuck::NoUninit`]: every byte of `T` must be
//! initialised (no padding), so a torn copy still has well-defined bits —
//! even though the consumer never delivers it, Miri's tree-borrows model
//! still tracks reads of the underlying storage. `bytemuck`'s derive checks
//! this at compile time.

#![allow(clippy::result_large_err)]

mod ring;
mod sink;

pub use ring::{Consumer, Producer, TryRecvError, channel};
pub use sink::SpmcSink;

#[cfg(test)]
mod tests;
