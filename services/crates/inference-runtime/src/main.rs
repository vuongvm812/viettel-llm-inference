//! Disruptor-based LLM inference runtime.
//!
//! Serving/scheduling/batching fabric over 3 pinned cores + 1 GPU. See
//! `docs/GENERAL_ARCHITECTURE.md`. P0 is scaffolding only — each module below is
//! a stub filled in by later roadmap phases.

// ponytail: fields land in P1+ when the pipeline reads them; drop the allow then.
#[allow(dead_code)]
mod config; // Runtime config, loaded from YAML
mod core0; // Web I/O & SSE streaming     (P1)
mod core1; // Tokenize / detokenize       (P1/P2)
mod core2; // Fast loop: schedule + decode (P1–P4)
mod rings; // 4 one-directional Disruptor rings + RingEvent (P1)
mod slab; // Pre-allocated zero-copy RequestSlab            (P1)

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
    println!("inference-runtime: P0 scaffold — pipeline not yet wired");
    println!("loaded {path}: {cfg:?}");
}
