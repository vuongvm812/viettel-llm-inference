//! Ring topology — 4 one-directional Disruptor rings (P1).
//!
//! R1 Core0→Core1 (MPSC), R2 Core1→Core2, R3 Core2→Core1, R4 Core1→Core0 (SPSC).
//! Rings carry only the `Copy` `RingEvent { slot, kind }` handle, never payload.
//! See `design/disruptor-pipeline/`.

/// The only thing that travels through a ring: an 8-byte `Copy` handle.
///
/// The derived `Default` is `{ slot: 0, kind: New }` (via `EventKind`'s default),
/// which is the factory value pre-allocated into every ring slot at startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RingEvent {
    /// Index into the `Slab`.
    pub slot: u32,
    pub kind: EventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EventKind {
    /// R1, R2: a fresh request to process.
    #[default]
    New,
    /// R3, R4: one generated token id / detokenized piece for `slot`.
    Token(u32),
    /// Stream complete.
    Finish(FinishReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    MaxTokens,
    /// Emitted on backend failure — first surfaced when llama.cpp can fail (P2).
    #[allow(dead_code)]
    Error,
}
