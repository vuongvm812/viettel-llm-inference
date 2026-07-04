//! Core 2 — The fast loop (P1–P4).
//!
//! Scheduler + batcher + KV monitor. Owns the mutable `LlamaContext` (KV cache).
//! Admits sequences (R2), batches active seqs into one `llama_batch`, `decode()`
//! on GPU, samples, and publishes tokens/finish on R3. `BusySpin`. See
//! `design/fast-loop/`.
