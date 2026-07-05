//! RequestSlab — pre-allocated, zero-copy request storage (P1).
//!
//! `Box<[UnsafeCell<RequestState>]>` sized to max in-flight requests. Rings
//! reference entries by `slot` index; the ~78K-char prompt is written once and
//! never copied through a ring. See `design/disruptor-pipeline/`.
//!
//! ## Soundness
//! Each slot is written by exactly one core at a time as its handle advances
//! through the pipeline (Core 0 → 1 → 2 → 1 → 0). The Disruptor publish/consume
//! pair is a release/acquire, so the ring hand-off *is* the synchronization edge
//! — that single-writer-per-stage invariant is what makes `UnsafeCell` sound
//! without locks. Callers of [`Slab::slot_mut`] must uphold it.
//!
//! The slab holds no async-runtime types: the egress connection handle lives on
//! the Core-0 side (like the free-list), so cores 1/2 never pull `tokio` into
//! their storage layout.

use std::cell::UnsafeCell;

/// One in-flight request. Buffers own their storage with pre-reserved capacity
/// so steady-state append never reallocates.
pub struct RequestState {
    // Written by Core 0 (ingress), read by Core 1 (tokenize) / Core 2 (max_tokens).
    pub prompt: String,
    pub max_tokens: u32,
    // Written by Core 1 (tokenize).
    pub tokens: Vec<i32>,
    /// Detokenized output bytes, appended by Core 1 (egress). A real vocab piece
    /// is multi-byte, so it can't ride the `u32` ring event — it lands here and
    /// Core 0 reads the committed range (see [`Slab::out_addr`]). **Never
    /// reallocates**: pre-reserved and capped at `capacity()`, so its heap base
    /// stays fixed for the process — the invariant `out_addr` relies on.
    pub out_bytes: Vec<u8>,
    /// Bytes of `out_bytes` that already form complete UTF-8 and were published
    /// to Core 0. Core-1-local detok cursor (multi-byte pieces span tokens).
    pub out_committed: usize,
    /// Set by Core 1 if detokenization failed or the output buffer overflowed for
    /// this request. Converts the terminal `Finish` into `Finish(Error)` so a
    /// corrupt/truncated stream is reported as failure, not success.
    pub out_error: bool,
}

impl RequestState {
    fn with_capacity(prompt_cap: usize, tokens_cap: usize, out_cap: usize) -> Self {
        RequestState {
            prompt: String::with_capacity(prompt_cap),
            max_tokens: 0,
            tokens: Vec::with_capacity(tokens_cap),
            out_bytes: Vec::with_capacity(out_cap),
            out_committed: 0,
            out_error: false,
        }
    }

    /// Clear for reuse, keeping buffer capacity. Called by Core 0 on claim.
    pub fn reset(&mut self) {
        self.prompt.clear();
        self.max_tokens = 0;
        self.tokens.clear();
        self.out_bytes.clear();
        self.out_committed = 0;
        self.out_error = false;
    }
}

pub struct Slab {
    slots: Box<[UnsafeCell<RequestState>]>,
    /// Heap base address of each slot's `out_bytes`, captured once at
    /// construction. Core 0 reads output through these raw addresses instead of
    /// a `&RequestState`, so it forms no reference into a cell that Core 1 may
    /// hold `&mut` on — the two only ever touch disjoint byte ranges of the same
    /// (never-reallocated) heap buffer, synchronized by the R4 ring hand-off.
    out_base: Box<[usize]>,
    /// Per-slot `out_bytes` capacity — the read bound for [`Slab::read_committed`].
    out_cap: usize,
}

// SAFETY: `Slab` hands out `&mut RequestState` from `&self`, but the pipeline
// guarantees only one core writes a given slot at a time (single-writer-per-
// stage, synchronized by ring hand-offs — see module docs). No two threads ever
// alias the same slot concurrently.
unsafe impl Sync for Slab {}

impl Slab {
    /// Allocate `capacity` slots once, pre-reserving each buffer.
    pub fn new(capacity: usize, prompt_cap: usize, tokens_cap: usize, out_cap: usize) -> Self {
        let slots: Box<[UnsafeCell<RequestState>]> = (0..capacity)
            .map(|_| UnsafeCell::new(RequestState::with_capacity(prompt_cap, tokens_cap, out_cap)))
            .collect();
        // Capture each output buffer's stable heap base now, while we are the sole
        // owner (no thread has a handle yet). `with_capacity` allocated the buffer,
        // and it never reallocates afterward, so these addresses stay valid.
        let out_base = slots
            .iter()
            .map(|cell| unsafe { (*cell.get()).out_bytes.as_ptr() as usize })
            .collect();
        Slab { slots, out_base, out_cap }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Read `[start, start+len)` of `slot`'s committed `out_bytes` output.
    ///
    /// Builds the slice from the buffer's fixed heap base (captured at
    /// construction), *not* from a `&RequestState`/`&Vec` — so the read forms no
    /// reference into the cell that Core 1 may hold `&mut` on. The pointer math and
    /// the bounds live here, in the one place that owns the never-realloc
    /// invariant. Bounds (`slot`, `start+len` vs capacity) are checked in **all**
    /// builds — a bad index returns [`SlabError`] instead of an out-of-bounds read,
    /// so a logic bug degrades to a dropped chunk rather than UB in release.
    ///
    /// # Safety
    /// `[start, start+len)` must be a range Core 1 has appended to `out_bytes` and
    /// published on R4 before this read (the ring hand-off is the happens-before
    /// edge). Reading an unpublished span is a data race. (In-bounds is enforced
    /// here; happens-before is the caller's obligation.)
    pub unsafe fn read_committed(&self, slot: u32, start: usize, len: usize) -> Result<&[u8], SlabError> {
        let idx = slot as usize;
        if idx >= self.slots.len() {
            return Err(SlabError::SlotOutOfRange(slot));
        }
        let end = start.checked_add(len).ok_or(SlabError::RangeOutOfBounds { start, len })?;
        if end > self.out_cap {
            return Err(SlabError::RangeOutOfBounds { start, len });
        }
        let base = self.out_base[idx] as *const u8;
        Ok(std::slice::from_raw_parts(base.add(start), len))
    }

    /// Mutable access to a slot's state.
    ///
    /// # Safety
    /// The caller must own `slot` at its current pipeline stage — i.e. hold it
    /// via a ring hand-off and not share it with another core. Violating the
    /// single-writer-per-stage invariant is undefined behavior.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slot_mut(&self, slot: u32) -> &mut RequestState {
        &mut *self.slots[slot as usize].get()
    }

    /// Prompt-token count of `slot` (its `tokens`, written by Core 1 during
    /// tokenize). Read-only accessor for Core 2's admission token budget, so the
    /// scheduler doesn't reach through `slot_mut` for a length.
    ///
    /// # Safety
    /// Same as [`slot_mut`](Self::slot_mut): the caller must own `slot` at its
    /// current pipeline stage. Core 2 does while the slot sits in its `pending`
    /// queue (arrived via R2, no R3 publish for it yet).
    pub unsafe fn prompt_len(&self, slot: u32) -> usize {
        debug_assert!((slot as usize) < self.slots.len(), "prompt_len: slot {slot} out of range");
        (*self.slots[slot as usize].get()).tokens.len()
    }
}

/// Bounds failure from [`Slab::read_committed`] — a logic bug, returned rather
/// than allowed to become an out-of-bounds read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabError {
    SlotOutOfRange(u32),
    RangeOutOfBounds { start: usize, len: usize },
}

impl std::fmt::Display for SlabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlabError::SlotOutOfRange(s) => write!(f, "slot {s} out of range"),
            SlabError::RangeOutOfBounds { start, len } => {
                write!(f, "out_bytes read [{start}..{}] out of bounds", start.saturating_add(*len))
            }
        }
    }
}

impl std::error::Error for SlabError {}
