//! `vtl_sched` — logic-preserving Rust port of vLLM v0.25.0's KV-cache block metadata,
//! prefix cache, hybrid coordinator and scheduler decision loop.
//!
//! The pure logic (everything except `python`) has NO PyO3 dependency, so `cargo test`
//! runs without libpython — which is what lets the Docker builder stage gate the wheel on
//! `cargo test --release` inside an image that ships no python dev libraries.
//!
//! The `python` feature adds the PyO3 extension module consumed by
//! `vtl/patches/rust_sched.py`. maturin enables it via `pyproject.toml`.

pub mod block_pool;
pub mod config;
pub mod coordinator;
pub mod hash;
/// Crate-internal: the journal is an implementation detail of `spec.rs`, and every type
/// in it (`Block`, the index snapshots) is one callers must not be able to forge.
pub(crate) mod journal;
pub mod manager;
pub mod radix;
pub mod sched;
pub mod single_type;
pub mod spec;
pub mod update;

#[cfg(feature = "python")]
mod python;
