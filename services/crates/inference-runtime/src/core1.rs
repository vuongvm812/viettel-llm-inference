//! Core 1 — Text processing (P1/P2).
//!
//! Tokenizes prompts (R1→R2) and detokenizes generated tokens (R3→R4). Holds
//! `Arc<LlamaModel>` (const vocab). `BusySpinWithSpinLoopHint`. See
//! `design/text-processing/`.
