//! Core 0 — Web I/O & streaming (P1).
//!
//! Owns the `tokio` runtime, sockets, and the slot↔connection map. Accepts
//! OpenAI requests, claims a slab slot, publishes on R1, and streams SSE egress
//! drained from R4. Event-driven (epoll), not busy-spin. See `design/web-io/`.
