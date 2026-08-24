//! Construction-time configuration + the feature gate.
//!
//! This port covers the configuration the round-2 stack actually serves. Anything
//! outside it is REJECTED here rather than approximated, so an unported feature can
//! never silently produce a wrong scheduling decision — the Python patch logs the
//! rejection and leaves vLLM's own manager in place.

use crate::single_type::Kind;

#[derive(Clone, Debug)]
pub struct GroupConfig {
    pub kind: Kind,
    pub block_size: usize,
    /// `isinstance(spec, FullAttentionSpec)` — decided on the Python side so subclass
    /// relationships stay upstream's.
    pub is_full_attention: bool,
    /// Canonical rendering of the spec dataclass; equal signatures == equal specs
    /// (`kv_cache_coordinator.py:573` `group.spec == spec`).
    pub spec_signature: String,
    pub mamba_align: bool,
    pub num_speculative_blocks: usize,
    pub use_eagle: bool,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub num_blocks: usize,
    pub enable_caching: bool,
    pub max_model_len: usize,
    pub scheduler_block_size: usize,
    pub hash_block_size: usize,
    pub log_stats: bool,
    pub watermark: f64,
    /// `VTL_RUST_SCHED_RADIX=1` — alternate index implementation, same answers.
    pub radix: bool,
    /// A KV connector is live on this engine. The ONLY thing it changes inside the crate
    /// is which cache-hit walk `Manager::get_computed_blocks` dispatches to: the per-group
    /// one (kv_cache_coordinator.py:742) that scheduler.py:698 uses when a connector is
    /// attached to a mamba-hybrid model, instead of the converging fixed point. Every
    /// other connector-shaped argument (external tokens, delayed caching, reserved blocks)
    /// is passed per call, so `connector == false` leaves every existing path bit-identical.
    pub connector: bool,
    pub groups: Vec<GroupConfig>,
}

impl Config {
    /// Everything this port refuses to guess at. Called before any state is built.
    pub fn validate(&self) -> Result<(), String> {
        if self.groups.is_empty() {
            return Err("no kv cache groups".into());
        }
        if self.num_blocks < 2 {
            return Err(format!("num_blocks={} is too small", self.num_blocks));
        }
        if self.watermark < 0.0 {
            return Err("watermark must be non-negative".into());
        }
        if self.hash_block_size == 0 || self.scheduler_block_size == 0 {
            return Err("block sizes must be positive".into());
        }
        if self.scheduler_block_size % self.hash_block_size != 0 {
            return Err(format!(
                "scheduler_block_size ({}) must be a multiple of hash_block_size ({})",
                self.scheduler_block_size, self.hash_block_size
            ));
        }
        for (i, g) in self.groups.iter().enumerate() {
            if g.block_size == 0 || self.scheduler_block_size % g.block_size != 0 {
                return Err(format!(
                    "group {i}: block_size {} does not divide scheduler_block_size {}",
                    g.block_size, self.scheduler_block_size
                ));
            }
            if g.block_size % self.hash_block_size != 0 {
                return Err(format!(
                    "group {i}: block_size {} is not a multiple of hash_block_size {}",
                    g.block_size, self.hash_block_size
                ));
            }
            if g.kind == Kind::Mamba && !g.mamba_align {
                // Non-align mamba is ported (`single_type.rs` handles it) but never
                // exercised on this stack, so it stays behind the gate rather than
                // shipping untested hybrid semantics.
                return Err(format!(
                    "group {i}: mamba_cache_mode != 'align' is not enabled in this port"
                ));
            }
        }
        Ok(())
    }
}
