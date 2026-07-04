//! Disruptor-based LLM inference runtime.
//!
//! Serving/scheduling/batching fabric over 3 pinned cores + 1 GPU. See
//! `docs/GENERAL_ARCHITECTURE.md`. P0 is scaffolding only — each module below is
//! a stub filled in by later roadmap phases.

mod affinity; // Thread → core pinning (no-op on macOS)
mod backend; // Model backend seam: mock byte-codec | real llama.cpp (P2)
mod config; // Runtime config, loaded from YAML
mod core0; // Web I/O & SSE streaming     (P1)
mod core1; // Tokenize / detokenize       (P1/P2)
mod core2; // Fast loop: schedule + decode (P1–P4)
mod pipeline; // Ring wiring + worker spawn (P1)
mod rings; // 4 one-directional Disruptor rings + RingEvent (P1)
mod slab; // Pre-allocated zero-copy RequestSlab            (P1)

use std::sync::Arc;

/// Prompt/token/output buffer capacities pre-reserved per slot. Sized for the
/// ~78K-char trace prompts so steady-state append never reallocates.
const PROMPT_CAP: usize = 96 * 1024;
const TOKENS_CAP: usize = 96 * 1024;
/// Detokenized output buffer per slot. Never reallocates (Core 0 reads it via a
/// fixed base address), so size it above the largest expected output — max_tokens
/// (capped 8192) × a few UTF-8 bytes each. Overflow drops trailing bytes.
const OUT_CAP: usize = 64 * 1024;

fn main() {
    // Default anchors to the repo-root `config/` at compile time so it resolves
    // from any cwd in dev; deployment passes an explicit `inference-runtime <config.yaml>`.
    // ponytail: baked dev path — the arg is the real knob once this ships.
    const DEFAULT_CONFIG: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../config/default-config.yaml");
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG.to_string());
    let cfg = config::Config::from_yaml_file(&path).unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });

    let slab = Arc::new(slab::Slab::new(
        cfg.runtime.max_inflight as usize,
        PROMPT_CAP,
        TOKENS_CAP,
        OUT_CAP,
    ));
    let pipe = pipeline::spawn(&cfg, Arc::clone(&slab));
    if let Err(e) = core0::serve(&cfg, slab, pipe) {
        eprintln!("core0 error: {e}");
        std::process::exit(1);
    }
}
