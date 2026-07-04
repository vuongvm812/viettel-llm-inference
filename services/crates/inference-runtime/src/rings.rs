//! Ring topology — 4 one-directional Disruptor rings (P1).
//!
//! R1 Core0→Core1 (MPSC), R2 Core1→Core2, R3 Core2→Core1, R4 Core1→Core0 (SPSC).
//! Rings carry only the `Copy` `RingEvent { slot, kind }` handle, never payload.
//! See `design/disruptor-pipeline/`.
