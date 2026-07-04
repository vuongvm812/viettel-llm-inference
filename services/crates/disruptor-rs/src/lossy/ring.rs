// Conditional imports so loom can instrument atomics for model checking.
// Production builds use the standard-library versions.
#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};

#[cfg(not(loom))]
use std::cell::UnsafeCell;

use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Bound, RangeBounds};

use bytemuck::NoUninit;
use ring_core::CachePadded;

use super::sink::SpmcSink;
#[cfg(not(loom))]
use ring_core::HugepageBox;

// ── Error type ─────────────────────────────────────────────────────────────────

/// Result of a non-blocking broadcast receive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    /// No new item for this consumer's cursor yet.
    Empty,
    /// The consumer fell more than `N` items behind and was *lapped*: `n` items
    /// were overwritten before it could read them. The cursor has resynchronised
    /// to the oldest still-live position; the next receive returns live data.
    Lagged(u64),
    /// The producer has been dropped and this consumer has drained every item.
    Disconnected,
}

// ── Slot ───────────────────────────────────────────────────────────────────────

/// One ring slot: a cheap-filter header and the payload stored as atomic
/// words.
///
/// There is no per-slot sequence number in the Phase 2 tail-versioned
/// protocol — correctness is enforced by the global `tail` counter alone.
/// A consumer reads the payload then re-reads `tail` Acquire and accepts the
/// copy iff `tail_after - position` is strictly less than `N` (i.e. the slot
/// has not been lapped, and the producer is not currently overwriting it via
/// the in-progress write at `idx(tail)`). See the lib.rs module docs for the
/// full soundness argument.
///
/// `#[repr(align(64))]` starts every slot on a 64 B cache-line boundary (and
/// rounds its size up to a multiple of 64), so a producer's payload write
/// never shares a line with an adjacent slot a consumer is reading.
/// `CachePadded` only aligns the whole `slots` array, which would otherwise
/// let 80 B slots straddle lines and false-share across slot boundaries.
#[cfg(not(loom))]
#[repr(align(64))]
struct Slot<T> {
    header: AtomicU64,
    /// Payload bytes. Accessed under the tail-versioned racy-memcpy contract;
    /// see the module docs in `lib.rs`.
    val: UnsafeCell<MaybeUninit<T>>,
}

/// Loom models a single-word payload (`size_of::<T>() == size_of::<usize>()`),
/// which is sufficient to verify the tail-versioning protocol — the
/// multi-word case is just a loop of the same atomic accesses.
#[cfg(loom)]
#[repr(align(64))]
struct Slot<T> {
    header: AtomicU64,
    val: AtomicUsize,
    _marker: PhantomData<T>,
}

// ── Ring ───────────────────────────────────────────────────────────────────────

/// Shared broadcast state. Hot single-writer fields (`tail`, liveness flags)
/// are cache-padded so the producer's writes do not evict consumers' reads.
pub struct Ring<T, const N: usize, const M: usize> {
    /// Count of items published so far; the next write position. Written only by
    /// the producer.
    tail: CachePadded<AtomicUsize>,
    /// Set to `false` by the producer's `Drop`.
    producer_alive: CachePadded<AtomicBool>,
    /// Set to `false` by each consumer's `Drop`.
    consumer_alive: [CachePadded<AtomicBool>; M],
    slots: CachePadded<[Slot<T>; N]>,
}

// SAFETY:
// * `tail`, `producer_alive`, `consumer_alive` are atomics — safe to share.
// * `slots`: each slot's `header` is `AtomicU64`. The payload `val`
//   (`UnsafeCell<MaybeUninit<T>>` in production, `AtomicUsize` under loom)
//   is accessed under the tail-versioned racy-memcpy contract — the
//   producer's `Release` store to `tail` synchronises-with the consumer's
//   `Acquire` load, and the post-read `tail` re-check rejects any read that
//   raced an overwrite. No value with torn bytes is ever exposed to user
//   code.
unsafe impl<T: Send, const N: usize, const M: usize> Send for Ring<T, N, M> {}
unsafe impl<T: Send, const N: usize, const M: usize> Sync for Ring<T, N, M> {}

#[inline(always)]
const fn idx<const N: usize>(position: usize) -> usize {
    position & (N - 1)
}

// ── Tail-versioning protocol ────────────────────────────────────────────────────
//
// The global `tail` counter is the *only* version field; there is no per-slot
// sequence number. Position `p` is **live** (its payload occupies `slot[idx(p)]`)
// iff `tail - p` is strictly in `1..N`. The producer's next write target is
// position `tail`, which shares `idx(tail) == idx(tail - N)` with the oldest
// reachable position; restricting the read window to `[tail - (N - 1), tail - 1]`
// keeps consumers one slot ahead of the producer's in-progress write — the
// effective capacity drops by one slot to `N - 1`.
//
// Wrapping: positions are unsigned counters and assumed not to wrap within any
// realistic runtime. The `has_item` / `resync` distance checks are
// wrapping-aware (a stale-low `tail` lands the distance in the upper half and
// is rejected as "no item / not safe to resync there") — the cursor only ever
// advances.

// ── Payload transfer helpers ───────────────────────────────────────────────────
//
// Production (`cfg(not(any(loom, miri)))`): one vectorisable memcpy per move
// (the classical racy-memcpy pattern — Linux seqlock / LMAX Disruptor /
// cyclotrace / AtomicRing — version-checked by the global `tail` here instead
// of a per-slot seqlock).
// Verification (`cfg(any(loom, miri))`): per-word atomic accesses so loom can
// model-check the tail-version protocol and Miri keeps tracking the
// single-threaded ownership story without flagging the racy memcpy. Both bound
// `T: NoUninit`.
//
// `channel` asserts `size_of::<T>()` is a multiple of `size_of::<usize>()` and
// `align_of::<T>() >= align_of::<usize>()`, which the loom/miri loop relies on.

#[cfg(not(any(loom, miri)))]
#[inline]
unsafe fn store_value<T: NoUninit>(slot: &Slot<T>, value: &T) {
    // SAFETY (tail-version): the producer is the sole writer of this slot;
    // any consumer mid-read of the previous occupant will observe the global
    // `tail` advance past the safe horizon via the caller's post-read check
    // and discard the torn `MaybeUninit` before `assume_init`, so no value
    // with torn bytes is ever exposed to user code. `T: NoUninit` ⇒
    // `Copy + Sized + 'static` and zero padding, so the source bytes are all
    // initialised; `slot.val.get()` is a properly-aligned
    // `*mut MaybeUninit<T>` (the slot's only payload storage).
    unsafe {
        std::ptr::copy_nonoverlapping(value as *const T, slot.val.get().cast::<T>(), 1);
    }
}

#[cfg(not(any(loom, miri)))]
#[inline]
unsafe fn load_value<T: NoUninit>(slot: &Slot<T>) -> MaybeUninit<T> {
    // SAFETY (tail-version): returns `MaybeUninit<T>` so a torn read carries
    // no type-system claim; the caller's post-read `tail.load(Acquire)`
    // decides `assume_init` vs. discard. The source is
    // `*mut MaybeUninit<T>`, so reading it as `MaybeUninit<T>` never requires
    // the bytes to be a valid `T`. Alignment matches `T`.
    unsafe { std::ptr::read(slot.val.get().cast::<MaybeUninit<T>>()) }
}

#[cfg(all(miri, not(loom)))]
#[inline]
#[allow(clippy::cast_ptr_alignment)] // payload is asserted to be usize-aligned in `channel`
unsafe fn store_value<T: NoUninit>(slot: &Slot<T>, value: &T) {
    let words = size_of::<T>() / size_of::<usize>();
    let src = (value as *const T).cast::<usize>();
    let dst = slot.val.get().cast::<usize>();
    for i in 0..words {
        // SAFETY: `i < words` keeps `src.add(i)` within `value`; `T: NoUninit`
        // means the read bytes are all initialised; `value` is usize-aligned.
        let word = unsafe { src.add(i).read() };
        // SAFETY: `dst.add(i)` is the i-th usize of this slot's payload storage
        // (`i < words` covers exactly `size_of::<T>()` bytes), usize-aligned, and
        // only ever accessed atomically under Miri.
        let cell = unsafe { AtomicUsize::from_ptr(dst.add(i)) };
        cell.store(word, Ordering::Relaxed);
    }
}

#[cfg(all(miri, not(loom)))]
#[inline]
#[allow(clippy::cast_ptr_alignment)] // payload is asserted to be usize-aligned in `channel`
unsafe fn load_value<T: NoUninit>(slot: &Slot<T>) -> MaybeUninit<T> {
    let words = size_of::<T>() / size_of::<usize>();
    let src = slot.val.get().cast::<usize>();
    let mut out = MaybeUninit::<T>::uninit();
    let dst = out.as_mut_ptr().cast::<usize>();
    for i in 0..words {
        // SAFETY: i-th payload word, in bounds, accessed atomically.
        let word = unsafe { AtomicUsize::from_ptr(src.add(i)) }.load(Ordering::Relaxed);
        // SAFETY: `dst.add(i)` is the i-th usize of `out`, in bounds, usize-aligned.
        unsafe { dst.add(i).write(word) };
    }
    // The bytes are only a coherent, valid `T` if the caller's post-read
    // `tail` check confirms the producer did not overwrite the slot during
    // the copy.
    out
}

#[cfg(loom)]
#[inline]
unsafe fn store_value<T: NoUninit>(slot: &Slot<T>, value: &T) {
    // SAFETY: under loom `size_of::<T>() == size_of::<usize>()` (asserted in
    // `channel`); `T: NoUninit` means every byte is initialised.
    //
    // Loom over-synchronises (`Release`) compared to production (raw memcpy)
    // so loom can verify the tail-version protocol against a strict C11
    // model. Production relies on hardware (x86 TSO load-load ordering, ARM
    // with explicit fences in the producer's tail `Release` store) to make
    // the racy raw memcpy observable in the right order; loom can't model
    // TSO, so the protocol is proved instead with explicit Acquire/Release
    // on the slot. This mirrors cyclotrace's loom adaptation and is the
    // standard idiom for verifying seqlock-style rings.
    let word: usize = unsafe { core::mem::transmute_copy::<T, usize>(value) };
    slot.val.store(word, Ordering::Release);
}

#[cfg(loom)]
#[inline]
unsafe fn load_value<T: NoUninit>(slot: &Slot<T>) -> MaybeUninit<T> {
    // SAFETY: see `store_value` for the loom/production divergence. The
    // `Acquire` load synchronises-with the producer's `Release` store so
    // the post-read `tail` check correctly rejects lapped reads.
    let word = slot.val.load(Ordering::Acquire);
    MaybeUninit::new(unsafe { core::mem::transmute_copy::<usize, T>(&word) })
}

// ── Producer ───────────────────────────────────────────────────────────────────

/// Sending half of the broadcast channel. There is exactly one producer
/// (`!Clone`) and it may only be used from one thread (`!Sync`).
pub struct Producer<T, const N: usize, const M: usize> {
    #[cfg(not(loom))]
    ring: Arc<HugepageBox<Ring<T, N, M>>>,
    #[cfg(loom)]
    ring: Arc<Ring<T, N, M>>,
    not_sync: PhantomData<Cell<()>>,
}

impl<T: NoUninit, const N: usize, const M: usize> Producer<T, N, M> {
    /// Publish `value` to every consumer. Never blocks.
    #[inline]
    pub fn publish(&self, value: T) {
        self.publish_with_header(value, 0);
    }

    /// Publish `value` together with a cheap-filter `header`.
    ///
    /// Consumers can inspect `header` via [`Consumer::try_recv_filter`] and
    /// skip irrelevant items without copying the (potentially large) payload.
    ///
    /// Phase 2: drops the per-slot `seq` two-step but keeps the inter-publish
    /// `fence(Release)` for portability — see the inline comment.
    #[inline]
    pub fn publish_with_header(&self, value: T, header: u64) {
        let position = self.ring.tail.load(Ordering::Relaxed);
        let slot = &self.ring.slots[idx::<N>(position)];

        // Inter-publish Release fence — orders the **previous** publish's
        // `tail.store(Release)` strictly before the payload write below.
        //
        // This closes a portability gap the tail-only protocol has on
        // weakly-ordered architectures. The producer's per-publish program
        // order is:
        //
        //   ... → tail.store(N, Release)      // previous publish
        //       → payload write at idx(N)     // overwrites idx(0)  ← (*)
        //       → header.store
        //       → tail.store(N+1, Release)
        //
        // On x86 TSO, store-store is preserved, so `(*)` cannot be observed
        // before the prior `tail.store(N, Release)`. On AArch64 / ARMv8,
        // however, `STLR` is a **one-way** release barrier: it forces prior
        // stores to drain before itself, but does **not** prevent subsequent
        // regular `STR`s from being observed before it. Without the fence, a
        // consumer at `cursor == 0` could observe the new payload bytes at
        // `idx(0)` while `tail` is still `< N`, get `post_distance < N`, and
        // falsely deliver the overwrite as `cursor`'s value.
        //
        // `fence(Release)` lowers to `DMB ISH` on AArch64 (a full data memory
        // barrier) and emits no instruction on x86 (compiler-only barrier),
        // so this is free on the production target. Mirrors the role the
        // pre-Phase-2 seqlock's `fence(Release)` played between the
        // in-progress marker and the payload write.
        fence(Ordering::Release);

        // SAFETY: the producer is the sole writer of this slot. A consumer
        // mid-copy of the previous occupant (position `position - N`) will
        // observe the global `tail` advance past the safe horizon
        // (`tail_after - cursor >= N`) via its post-read check and discard
        // the torn `MaybeUninit` before `assume_init`.
        unsafe { store_value(slot, &value) };
        slot.header.store(header, Ordering::Relaxed);

        // Release store: payload + header become observable to any consumer
        // whose `tail.load(Acquire)` observes `position + 1` or later, and
        // the cached "is anything new?" fast-path advances.
        self.ring
            .tail
            .store(position.wrapping_add(1), Ordering::Release);
    }

    /// Returns `true` while at least one consumer is still alive.
    #[must_use]
    pub fn is_any_consumer_alive(&self) -> bool {
        self.ring
            .consumer_alive
            .iter()
            .any(|alive| alive.load(Ordering::Acquire))
    }
}

impl<T, const N: usize, const M: usize> Drop for Producer<T, N, M> {
    fn drop(&mut self) {
        self.ring.producer_alive.store(false, Ordering::Release);
    }
}

// ── Consumer ───────────────────────────────────────────────────────────────────

/// Receiving half of the broadcast channel. Each consumer owns an independent
/// cursor and sees every published item (until lapped). `!Clone`, `!Sync`.
pub struct Consumer<T, const N: usize, const M: usize> {
    #[cfg(not(loom))]
    ring: Arc<HugepageBox<Ring<T, N, M>>>,
    #[cfg(loom)]
    ring: Arc<Ring<T, N, M>>,
    /// This consumer's index into `consumer_alive`.
    id: usize,
    /// Next position to read.
    cursor: Cell<usize>,
    /// Cached snapshot of `tail` to skip an atomic load when items are pending.
    cached_tail: Cell<usize>,
    /// Cumulative count of items dropped because this consumer was lapped.
    lagged: Cell<u64>,
    not_sync: PhantomData<Cell<()>>,
}

impl<T: NoUninit, const N: usize, const M: usize> Consumer<T, N, M> {
    /// Receive the next item for this consumer's cursor.
    #[inline]
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let cursor = self.cursor.get();
        self.ensure_published(cursor)?;

        // Pre-check via the (possibly stale) cached tail: if `cursor` is
        // already at or beyond the live horizon, resync without paying for
        // the payload copy. A fresh tail load happens inside `resync`.
        let initial_distance = self.cached_tail.get().wrapping_sub(cursor);
        if initial_distance >= N || initial_distance > isize::MAX as usize {
            return Err(self.resync(cursor));
        }

        let slot = &self.ring.slots[idx::<N>(cursor)];

        // SAFETY: `ensure_published` observed `tail > cursor` via Acquire,
        // synchronising-with the producer's publishing Release at `cursor`,
        // so the payload words for `cursor` are visible. The copy is trusted
        // only if the post-read tail check confirms the slot was not lapped.
        let raw = unsafe { load_value(slot) };
        // Acquire fence: the payload loads above are ordered before the tail
        // re-read, so any producer write that has bumped `tail` past
        // `cursor + N - 1` (i.e. has started or finished overwriting
        // `idx(cursor)`) is observed here.
        fence(Ordering::Acquire);
        let tail_after = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail_after);
        let post_distance = tail_after.wrapping_sub(cursor);
        if post_distance < N && post_distance <= isize::MAX as usize {
            self.cursor.set(cursor.wrapping_add(1));
            // SAFETY: `tail_after - cursor < N` ⇒ the producer has not begun
            // overwriting `idx(cursor)` for position `cursor + N`, so the
            // payload bytes belong to position `cursor`'s publish.
            Ok(unsafe { raw.assume_init() })
        } else {
            // Slot lapped during the copy: discard `raw` without `assume_init`.
            Err(self.resync(cursor))
        }
    }

    /// Like [`Self::try_recv`], but skip items whose `header` fails `keep`,
    /// without copying their payload. A filtered skip is **not** counted as a
    /// drop ([`Self::lagged`] is unchanged); only an actual lapping is.
    #[inline]
    pub fn try_recv_filter(&self, keep: impl Fn(u64) -> bool) -> Result<T, TryRecvError> {
        loop {
            let cursor = self.cursor.get();
            self.ensure_published(cursor)?;

            let initial_distance = self.cached_tail.get().wrapping_sub(cursor);
            if initial_distance >= N || initial_distance > isize::MAX as usize {
                return Err(self.resync(cursor));
            }

            let slot = &self.ring.slots[idx::<N>(cursor)];

            // Read header (Acquire — synchronises-with the producer's tail
            // Release that published this position) and validate via the
            // post-read tail check. If the header read raced a producer
            // overwrite, the tail-after check rejects it before we either
            // skip or copy the payload.
            let header = slot.header.load(Ordering::Acquire);
            fence(Ordering::Acquire);
            let tail_after_header = self.ring.tail.load(Ordering::Acquire);
            self.cached_tail.set(tail_after_header);
            let header_distance = tail_after_header.wrapping_sub(cursor);
            if header_distance >= N || header_distance > isize::MAX as usize {
                return Err(self.resync(cursor));
            }

            if !keep(header) {
                // Header is confirmed-valid (above check) and doesn't match —
                // skip without paying for the payload copy. Not a lag.
                self.cursor.set(cursor.wrapping_add(1));
                continue;
            }

            // SAFETY: see `try_recv`.
            let raw = unsafe { load_value(slot) };
            fence(Ordering::Acquire);
            let tail_after_payload = self.ring.tail.load(Ordering::Acquire);
            self.cached_tail.set(tail_after_payload);
            let payload_distance = tail_after_payload.wrapping_sub(cursor);
            if payload_distance < N && payload_distance <= isize::MAX as usize {
                self.cursor.set(cursor.wrapping_add(1));
                // SAFETY: see `try_recv`.
                return Ok(unsafe { raw.assume_init() });
            }
            return Err(self.resync(cursor));
        }
    }

    /// Drain up to `out.len()` items in FIFO order, returning the count written.
    /// Stops at the first `Empty`/`Lagged`/`Disconnected`; a mid-batch lapping is
    /// reflected in [`Self::lagged`].
    pub fn pop_batch(&self, out: &mut [MaybeUninit<T>]) -> usize {
        let mut count = 0;
        while count < out.len() {
            match self.try_recv() {
                Ok(value) => {
                    out[count] = MaybeUninit::new(value);
                    count += 1;
                }
                Err(_) => break,
            }
        }
        count
    }

    /// Published-distance from this consumer's cursor, clamped to `N - 1`
    /// (the safe read horizon in the tail-versioned protocol).
    ///
    /// Loads `tail` once with `Acquire` and updates the cached snapshot. A
    /// return of `0` means caught up at the observed `tail` (the producer
    /// may still be alive). A return in `1..N - 1` is a fully-live batch —
    /// the next `try_recv`/`drain_available` will deliver every one of them.
    /// A return of `N - 1` is ambiguous: the consumer may be exactly `N - 1`
    /// items behind (still fully live, no lap) or arbitrarily further behind
    /// (lapped-to-full); in the lapped-to-full case the next read returns
    /// `Lagged(_)` and resyncs. Use this as a single-load "is there a batch
    /// to process?" check, not as a guarantee that that many `Ok` deliveries
    /// are queued.
    #[inline]
    #[must_use]
    pub fn available(&self) -> usize {
        let cursor = self.cursor.get();
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);
        let distance = tail.wrapping_sub(cursor);
        // Same wrapping-aware filter as `has_item`: a stale-low `tail`
        // observation lands `distance` in the upper half and reads as 0.
        if distance == 0 || distance > isize::MAX as usize {
            0
        } else {
            distance.min(N - 1)
        }
    }

    /// Drain up to `max` consecutive items, invoking `f(&item, end_of_batch)` on
    /// each. Returns the count delivered. `end_of_batch` is `true` on the final
    /// successfully-delivered call of this invocation, mirroring LMAX
    /// `BatchEventProcessor::onEvent` — a handler may use it to flush buffered
    /// work once per batch instead of once per event.
    ///
    /// Phase 2: each slot is validated by a `tail` re-load (rather than a
    /// per-slot `seq` re-load). The per-item cost in steady state is one
    /// payload copy + one `Acquire` re-load of `tail`. Snapshot the live
    /// horizon `min(distance, N - 1, max)` once at entry; subsequent slots
    /// re-confirm against `tail` only.
    ///
    /// Mid-batch lap detection (the producer's `tail` has overrun the safe
    /// horizon `cursor + N - 1` after the payload copy) flushes the
    /// preceding successful delivery with `end_of_batch = true`, advances
    /// the consumer cursor via [`Self::resync`] (counting the skipped gap
    /// into [`Self::lagged`]), and returns — exactly as `try_recv` would on
    /// the next call.
    ///
    /// # Re-entrancy
    ///
    /// `&mut self` is exclusive for the duration of the call, so the closure
    /// `f` cannot re-enter any method on this `Consumer` (the borrow checker
    /// rejects `rx.try_recv()` / `rx.drain_available(...)` from within `f`).
    /// This is intentional: re-entry would observe the partially-advanced
    /// cursor and could duplicate already-delivered items.
    ///
    /// # Panic semantics
    ///
    /// `self.cursor` is advanced **past** the in-flight item *before* `f` is
    /// invoked. If `f` panics, the in-flight item is treated as consumed (lost
    /// from this consumer's stream) and the next call resumes at the next
    /// position — analogous to how LMAX `BatchEventProcessor` advances its
    /// sequence after delivering each event to the `ExceptionHandler`. Callers
    /// that need at-least-once delivery must catch the panic inside `f` (e.g.
    /// `std::panic::catch_unwind`) and re-process the item themselves.
    #[inline]
    pub fn drain_available<F>(&mut self, max: usize, mut f: F) -> usize
    where
        F: FnMut(&T, bool),
    {
        if max == 0 {
            return 0;
        }

        // Snapshot tail *once*. The cached value also primes subsequent
        // `ensure_published` fast-paths on the next `try_recv`.
        let cursor_start = self.cursor.get();
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);

        let distance = tail.wrapping_sub(cursor_start);
        // See `has_item` — guard the wrapping-half boundary.
        if distance == 0 || distance > isize::MAX as usize {
            return 0;
        }
        // Already past the live horizon at entry → resync without entering
        // the per-slot loop. (`distance > N - 1` ≡ `distance >= N`.)
        if distance >= N {
            let _ = self.resync(cursor_start);
            return 0;
        }
        // Clamp to safe live items and to the caller's batch budget.
        let limit = distance.min(N - 1).min(max);

        // Deferred-flush by one slot: hold the most recent successfully-
        // validated item in `pending` so that whichever delivery turns out to
        // be the last (either we hit `limit`, or the next slot laps us) is
        // the one tagged `end_of_batch = true`. `T: NoUninit` ⇒ `T: Copy`, so
        // the by-value owner-handoff has no `Drop` concerns and a panic
        // unwinding past `pending` cannot double-free.
        //
        // Crucially: `self.cursor` is advanced *past* `pending`'s position
        // *before* `f` is invoked. That makes a panicking `f` lose the
        // in-flight item rather than re-deliver it on retry (see the
        // method docs).
        let mut pending: Option<T> = None;
        let mut count = 0usize;
        let mut cursor = cursor_start;

        for _ in 0..limit {
            let slot = &self.ring.slots[idx::<N>(cursor)];

            // SAFETY: at loop entry the snapshot guaranteed `cursor < tail`
            // with `tail - cursor < N`. The post-read tail check below
            // verifies the slot was not lapped during the copy.
            let raw = unsafe { load_value(slot) };
            // Order the payload reads before the tail re-read so that any
            // producer write that has bumped `tail` past `cursor + N - 1`
            // (i.e. has started or finished overwriting `idx(cursor)`) is
            // observed here.
            fence(Ordering::Acquire);
            let tail_after = self.ring.tail.load(Ordering::Acquire);
            let post_distance = tail_after.wrapping_sub(cursor);
            if post_distance >= N || post_distance > isize::MAX as usize {
                // Overwritten mid-copy: drop `raw` without `assume_init`,
                // then flush+resync as above.
                self.cached_tail.set(tail_after);
                if let Some(value) = pending.take() {
                    self.cursor.set(cursor);
                    f(&value, true);
                    count += 1;
                }
                let _ = self.resync(cursor);
                return count;
            }

            // Slot validated. Flush the previous pending as non-EOB with the
            // cursor committed past it; then promote this one. The very last
            // pending is flushed below with EOB = true.
            if let Some(value) = pending.take() {
                // `cursor` here is the position of the slot we just validated,
                // which is one past `pending`'s position — exactly where the
                // next read should resume if `f` panics.
                self.cursor.set(cursor);
                f(&value, false);
                count += 1;
            }
            // SAFETY: `tail_after - cursor < N` ⇒ the producer has not begun
            // overwriting `idx(cursor)` for position `cursor + N`, so the
            // payload bytes belong to position `cursor`'s publish.
            pending = Some(unsafe { raw.assume_init() });
            cursor = cursor.wrapping_add(1);
        }

        if let Some(value) = pending {
            // Commit the cursor past `pending`'s position *before* the final
            // delivery — `cursor` here is already one past it.
            self.cursor.set(cursor);
            f(&value, true);
            count += 1;
        }
        // Prime the cached tail for the next `try_recv` fast-path.
        self.cached_tail.set(self.ring.tail.load(Ordering::Acquire));
        count
    }

    /// Cumulative number of items dropped because this consumer was lapped.
    #[must_use]
    pub fn lagged(&self) -> u64 {
        self.lagged.get()
    }

    /// Returns `true` while the producer is still alive.
    #[must_use]
    pub fn is_producer_alive(&self) -> bool {
        self.ring.producer_alive.load(Ordering::Acquire)
    }

    /// Peek the **freshest** published item without advancing the cursor.
    ///
    /// Returns `None` when the ring is empty or the slot was overwritten
    /// mid-read (torn). Wait-free: payload copy + a single `tail` re-load,
    /// no loops, no retries.
    ///
    /// `peek_latest` ignores [`Self::cursor`](Consumer): repeated calls can
    /// return the same item, and items published between two calls are **not**
    /// observed via this path. Use [`Self::try_recv`] / [`Self::drain_available`]
    /// for stream delivery semantics. This is the register-style "latest state"
    /// accessor (mirroring `cyclotrace::BufReader::get_latest`) for callers
    /// whose product semantics are "freshest sample wins" — e.g. an MM hot path
    /// where a stale `BookMid` is worse than useless once a newer one lands.
    #[inline]
    #[must_use]
    pub fn peek_latest(&self) -> Option<T> {
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);
        let position = tail.checked_sub(1)?;
        self.peek_at_position(position)
    }

    /// Peek the `n`-th item back from the latest (`n == 0` ≡ [`Self::peek_latest`]).
    ///
    /// Returns `None` when `n >= N - 1` (the slot at `idx(tail)` is the
    /// producer's next write target and cannot be peeked safely under the
    /// tail-versioning protocol), the ring is too shallow (`tail <= n`),
    /// or the slot was overwritten mid-read.
    ///
    /// Does not advance the cursor — see [`Self::peek_latest`].
    #[inline]
    #[must_use]
    pub fn peek(&self, n: usize) -> Option<T> {
        if n >= N - 1 {
            return None;
        }
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);
        let position = tail.checked_sub(n + 1)?;
        self.peek_at_position(position)
    }

    /// Atomic snapshot of a range of past items into `out`, oldest first.
    ///
    /// `range` is interpreted as **offsets back from latest** with standard
    /// Rust half-open semantics: `0..K` covers offsets `0..K` (the K most
    /// recent items, written into `out` in publication order); `..` covers
    /// offsets `0..N - 1` (the maximum safe peek window). The highest reachable
    /// offset is `N - 2`; an end offset beyond that yields `None`.
    ///
    /// Returns `None` when the range is empty/invalid, the ring has fewer
    /// items than the range needs, or the snapshot tore (any slot in the range
    /// was overwritten while the snapshot was being assembled). On tear, `out`
    /// is rolled back to its entry length so the caller never observes a
    /// mixed-snapshot buffer. Does not advance the cursor — see
    /// [`Self::peek_latest`].
    ///
    /// # Wait-freedom contract — caller must pre-reserve
    ///
    /// The **read protocol** (per-slot tail-version re-check + post-walk
    /// global-`tail` re-check) is wait-free. The **sink push** is wait-free
    /// only if the sink reports `out.remain() >= range_len` on entry: when
    /// it does, `peek_range` skips its [`SpmcSink::reserve`] call and never
    /// asks the sink to allocate. When `out.remain() < range_len`,
    /// `peek_range` invokes [`SpmcSink::reserve`] to make room — for
    /// heap-backed sinks like [`Vec`] this can call the global allocator
    /// (not wait-free), and for fixed-capacity sinks it may panic per that
    /// sink's contract.
    ///
    /// **Latency-critical callers** (hot-path consumers, real-time threads)
    /// must pre-size `out` so `out.remain() >= range_len` before every call;
    /// then `peek_range` is end-to-end wait-free and allocation-free.
    #[inline]
    pub fn peek_range<S, R>(&self, range: R, out: &mut S) -> Option<()>
    where
        S: SpmcSink<Item = T> + ?Sized,
        R: RangeBounds<usize>,
    {
        let start_offset = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1)?,
            Bound::Unbounded => 0,
        };
        let end_offset = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1)?,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => N - 1,
        };

        // Empty range, or highest offset (`end_offset - 1`) would land on the
        // producer's next write target (`idx(tail)` ≡ offset `N - 1`). Both
        // reject without touching `out`.
        if start_offset >= end_offset || end_offset > N - 1 {
            return None;
        }

        let count = end_offset - start_offset;
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);

        // Highest offset in range is `end_offset - 1`; corresponding lowest
        // position is `tail - end_offset`. Underflow ⇒ fewer items published
        // than the range needs.
        let oldest_pos = tail.checked_sub(end_offset)?;

        let entry_len = out.len();
        if let Some(lack) = count.checked_sub(out.remain()) {
            out.reserve(lack);
        }

        // Walk positions oldest → newest, validating each via the per-slot
        // seqlock. A torn slot short-circuits and rolls `out` back.
        for i in 0..count {
            match self.peek_at_position(oldest_pos.wrapping_add(i)) {
                Some(value) => out.push(value, i),
                None => {
                    out.truncate(entry_len);
                    return None;
                }
            }
        }

        // Post-walk atomicity guard. The per-slot tail-version validation
        // inside `peek_at_position` catches a slot lapped during its own
        // copy, but two slots in the range could be lapped consecutively if
        // the producer raced through them. A single `tail` re-load confirms
        // that the oldest position we read is still strictly inside the
        // live window (`tail_after - oldest_pos < N`).
        let tail_after = self.ring.tail.load(Ordering::Acquire);
        let distance = tail_after.wrapping_sub(oldest_pos);
        // Wrapping-aware (cf. `has_item`): a stale-low `tail_after` lands the
        // distance in the upper half and is treated as "no longer atomic".
        if distance < N && distance <= isize::MAX as usize {
            Some(())
        } else {
            out.truncate(entry_len);
            None
        }
    }

    /// Tail-versioned read of an absolute position without advancing the
    /// cursor. Shared between [`Self::peek_latest`], [`Self::peek`], and
    /// [`Self::peek_range`]. Mirrors the read half of [`Self::try_recv`]
    /// but omits the `self.cursor.set` step and returns `None` (not
    /// `Lagged`) on tear — peek is a snapshot, not a stream.
    ///
    /// Caller invariant: `position` must be inside the live window of the
    /// `tail` already observed by the public peek method (each public peek
    /// loads `tail` Acquire before computing `position`). This helper does
    /// not pre-check; it copies first and validates via the post-read `tail`
    /// re-load (the same protocol the cursor-stream `try_recv` uses).
    #[inline]
    fn peek_at_position(&self, position: usize) -> Option<T> {
        let slot = &self.ring.slots[idx::<N>(position)];

        // SAFETY: the public peek caller observed `tail > position` via
        // Acquire, synchronising-with the producer's Release at `position`,
        // so the payload words are visible. The copy is trusted only if
        // the post-read `tail` re-load confirms the slot was not lapped.
        let raw = unsafe { load_value(slot) };
        // Acquire fence: order the payload loads before the `tail` re-read.
        fence(Ordering::Acquire);
        let tail_after = self.ring.tail.load(Ordering::Acquire);
        let distance = tail_after.wrapping_sub(position);
        if distance != 0 && distance < N && distance <= isize::MAX as usize {
            // SAFETY: `tail_after - position < N` ⇒ the producer has not
            // begun overwriting `idx(position)` for position + N, so the
            // bytes belong to `position`'s publish.
            Some(unsafe { raw.assume_init() })
        } else {
            None
        }
    }

    /// Ensure an item exists at `cursor` (i.e. `cursor` is strictly behind
    /// `tail`). Reloads `tail` only when the cached value does not already
    /// show one.
    ///
    /// `cursor <= real_tail` always holds, but the *observed* tail can lag
    /// the cursor under the wrapping arithmetic. "Caught up" is checked via
    /// [`Self::has_item`] (wrapping-aware), not raw equality, so a stale
    /// `tail < cursor` observation is treated as "no item" rather than
    /// misread as an available one. Lap detection (`tail - cursor >= N`)
    /// happens in the callers — `ensure_published` only certifies that a
    /// position has been *published*, not that it is still in the safe
    /// horizon.
    #[inline]
    fn ensure_published(&self, cursor: usize) -> Result<(), TryRecvError> {
        if Self::has_item(cursor, self.cached_tail.get()) {
            return Ok(());
        }
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);
        if Self::has_item(cursor, tail) {
            return Ok(());
        }
        if self.is_producer_alive() {
            return Err(TryRecvError::Empty);
        }
        // Producer gone: re-read tail (Acquire synchronises-with its drop
        // Release, after the last publish) to drain any late items.
        let tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(tail);
        if Self::has_item(cursor, tail) {
            Ok(())
        } else {
            Err(TryRecvError::Disconnected)
        }
    }

    /// `true` iff position `cursor` has been published given an observed `tail`,
    /// i.e. `tail > cursor`. Wrapping-aware: a `tail` that has wrapped below
    /// `cursor` (a stale-low observation) yields a large `available` and reads as
    /// "no item".
    #[inline]
    fn has_item(cursor: usize, tail: usize) -> bool {
        let available = tail.wrapping_sub(cursor);
        available != 0 && available <= isize::MAX as usize
    }

    /// Resynchronise a lapped cursor to the **oldest still-live** position
    /// and record the skipped gap as dropped items.
    ///
    /// Phase 2: the safe horizon is `tail - (N - 1)` (Phase 1 was `tail - N`).
    /// The slot at `idx(tail)` shares its index with the producer's
    /// in-progress write target, so position `tail - N` cannot be read
    /// safely; the oldest peek/recv-able position is `tail - (N - 1)`.
    ///
    /// A freshly loaded `tail` can be observed *stale-low*, and a blind
    /// `tail - (N - 1)` could then sit at or behind `cursor` and drag it
    /// backward. Wrapping-aware comparisons reject that case and fall back
    /// to `tail` itself (the cursor catches up to "nothing pending"; the
    /// next `try_recv` returns `Empty` until a new publish). Both targets
    /// are `>= cursor + 1` under the lap precondition, so the cursor only
    /// ever advances and `delivered + lagged` accounting stays exact.
    #[inline]
    fn resync(&self, cursor: usize) -> TryRecvError {
        let observed_tail = self.ring.tail.load(Ordering::Acquire);
        self.cached_tail.set(observed_tail);

        let oldest_live = observed_tail.wrapping_sub(N - 1);
        // Wrapping-aware (cf. `has_item`): a forward distance lands in
        // `0..=isize::MAX`; a stale-low `observed_tail` would land
        // `oldest_live` behind `cursor`, which we reject and fall back to
        // `observed_tail`.
        let oldest_ahead = {
            let d = oldest_live.wrapping_sub(cursor);
            d != 0 && d <= isize::MAX as usize
        };
        let target = if oldest_ahead {
            oldest_live
        } else {
            observed_tail
        };
        let skipped = target.wrapping_sub(cursor) as u64;
        self.cursor.set(target);
        self.lagged.set(self.lagged.get().wrapping_add(skipped));
        TryRecvError::Lagged(skipped)
    }
}

impl<T, const N: usize, const M: usize> Drop for Consumer<T, N, M> {
    fn drop(&mut self) {
        self.ring.consumer_alive[self.id].store(false, Ordering::Release);
    }
}

// ── channel constructor ─────────────────────────────────────────────────────────

/// Create a broadcast channel: one [`Producer`] and `M` independent
/// [`Consumer`]s over a ring of `N` slots.
///
/// `N` must be a power of two; `M >= 1`. The payload `T` must be
/// [`bytemuck::NoUninit`] with a size that is a multiple of `size_of::<usize>()`
/// and alignment `>= align_of::<usize>()` — all checked at compile time.
#[cfg(not(loom))]
pub fn channel<T: NoUninit + Send, const N: usize, const M: usize>()
-> (Producer<T, N, M>, [Consumer<T, N, M>; M]) {
    const {
        assert!(N >= 2, "spmc ring capacity must be at least 2");
        assert!(
            N.is_power_of_two(),
            "spmc ring capacity must be a power of two"
        );
        assert!(M > 0, "spmc must have at least one consumer");
        assert!(size_of::<T>() > 0, "spmc payload must be non-zero-sized");
        assert!(
            size_of::<T>().is_multiple_of(size_of::<usize>()),
            "spmc payload size must be a multiple of usize"
        );
        assert!(
            align_of::<T>() >= align_of::<usize>(),
            "spmc payload alignment must be >= usize"
        );
    }

    let ring: Arc<HugepageBox<Ring<T, N, M>>> = {
        let mut storage = HugepageBox::<MaybeUninit<Ring<T, N, M>>>::new_zeroed();
        let ptr = storage.as_mut_ptr();

        // SAFETY: the zeroed storage is uniquely owned. Each field is written
        // in place before `assume_init`. The zeroed bytes are already valid
        // for every field except the liveness flags (must be `true`); those
        // are written below. Slot `header` words stay zeroed (valid for the
        // first atomic load); payload words stay zeroed and no consumer reads
        // a slot before the producer publishes it (the `tail == 0` initial
        // value gates every consumer path until the first publish).
        unsafe {
            std::ptr::addr_of_mut!((*ptr).tail).write(CachePadded::new(AtomicUsize::new(0)));
            std::ptr::addr_of_mut!((*ptr).producer_alive)
                .write(CachePadded::new(AtomicBool::new(true)));
            std::ptr::addr_of_mut!((*ptr).consumer_alive).write(std::array::from_fn(|_| {
                CachePadded::new(AtomicBool::new(true))
            }));
        }

        // SAFETY: all fields initialised above.
        Arc::new(unsafe { storage.assume_init() })
    };

    let producer = Producer {
        ring: Arc::clone(&ring),
        not_sync: PhantomData,
    };
    let consumers = std::array::from_fn(|id| Consumer {
        ring: Arc::clone(&ring),
        id,
        cursor: Cell::new(0),
        cached_tail: Cell::new(0),
        lagged: Cell::new(0),
        not_sync: PhantomData,
    });
    (producer, consumers)
}

/// Loom variant: explicit field-by-field construction with loom atomics and a
/// single-word payload (`size_of::<T>() == size_of::<usize>()`).
#[cfg(loom)]
pub fn channel<T: NoUninit + Send, const N: usize, const M: usize>()
-> (Producer<T, N, M>, [Consumer<T, N, M>; M]) {
    assert!(N >= 2 && N.is_power_of_two());
    assert!(M > 0);
    assert!(size_of::<T>() == size_of::<usize>());

    let ring = Arc::new(Ring {
        tail: CachePadded::new(AtomicUsize::new(0)),
        producer_alive: CachePadded::new(AtomicBool::new(true)),
        consumer_alive: std::array::from_fn(|_| CachePadded::new(AtomicBool::new(true))),
        slots: CachePadded::new(std::array::from_fn(|_i| Slot {
            header: AtomicU64::new(0),
            val: AtomicUsize::new(0),
            _marker: PhantomData,
        })),
    });

    let producer = Producer {
        ring: Arc::clone(&ring),
        not_sync: PhantomData,
    };
    let consumers = std::array::from_fn(|id| Consumer {
        ring: Arc::clone(&ring),
        id,
        cursor: Cell::new(0),
        cached_tail: Cell::new(0),
        lagged: Cell::new(0),
        not_sync: PhantomData,
    });
    (producer, consumers)
}
