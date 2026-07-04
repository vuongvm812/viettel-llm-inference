//! RequestSlab — pre-allocated, zero-copy request storage (P1).
//!
//! `Vec<RequestState>` sized to max in-flight requests, with a lock-free
//! free-list. Rings reference entries by `slot` index; the ~40K-char prompt is
//! written once and never copied through a ring. See `design/disruptor-pipeline/`.
