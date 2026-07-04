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
}

impl RequestState {
    fn with_capacity(prompt_cap: usize, tokens_cap: usize) -> Self {
        RequestState {
            prompt: String::with_capacity(prompt_cap),
            max_tokens: 0,
            tokens: Vec::with_capacity(tokens_cap),
        }
    }

    /// Clear for reuse, keeping buffer capacity. Called by Core 0 on claim.
    pub fn reset(&mut self) {
        self.prompt.clear();
        self.max_tokens = 0;
        self.tokens.clear();
    }
}

pub struct Slab {
    slots: Box<[UnsafeCell<RequestState>]>,
}

// SAFETY: `Slab` hands out `&mut RequestState` from `&self`, but the pipeline
// guarantees only one core writes a given slot at a time (single-writer-per-
// stage, synchronized by ring hand-offs — see module docs). No two threads ever
// alias the same slot concurrently.
unsafe impl Sync for Slab {}

impl Slab {
    /// Allocate `capacity` slots once, pre-reserving each buffer.
    pub fn new(capacity: usize, prompt_cap: usize, tokens_cap: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| UnsafeCell::new(RequestState::with_capacity(prompt_cap, tokens_cap)))
            .collect();
        Slab { slots }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
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
}
