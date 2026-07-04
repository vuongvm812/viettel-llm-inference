//! Runtime configuration, loaded from a YAML file.
//!
//! Fields transcribe the knobs already named in `docs/GENERAL_ARCHITECTURE.md`
//! and the roadmap; later phases read them as the pipeline is wired. The
//! canonical file is `config/default-config.yaml` at the repo root.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: Server,
    // Read by Core 2 once llama.cpp lands (P2); parsed now for config parity.
    #[allow(dead_code)]
    pub model: Model,
    pub runtime: Runtime,
}

#[derive(Debug, Deserialize)]
pub struct Server {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // read once llama.cpp lands (P2)
pub struct Model {
    /// Path to the GGUF model file (Qwen3.5-2B dense transformer, BF16).
    pub gguf_path: String,
    /// KV context length.
    pub n_ctx: u32,
    /// CPU threads for llama.cpp; 1 with full GPU offload.
    pub n_threads: u32,
    /// Layers offloaded to GPU; -1 = all.
    pub n_gpu_layers: i32,
}

#[derive(Debug, Deserialize)]
pub struct Runtime {
    /// `RequestSlab` size = max in-flight requests.
    pub max_inflight: u32,
    /// Ring buffer size; power of two, R1 (multi-producer) needs >= 64.
    pub ring_size: u32,
    /// Pinned core ids (no-op on macOS).
    pub cores: Cores,
}

#[derive(Debug, Deserialize)]
pub struct Cores {
    pub web_io: usize,
    pub text: usize,
    pub fast_loop: usize,
}

impl Config {
    /// Load, parse, and validate a YAML config file.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Read(path.display().to_string(), e))?;
        let cfg: Config = serde_yaml::from_str(&raw).map_err(ConfigError::Parse)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check the invariants the pipeline relies on, at the config trust boundary,
    /// so a bad value fails with a clear error instead of panicking deep in the
    /// disruptor builder (or, worse, deadlocking ingress at runtime).
    pub fn validate(&self) -> Result<(), ConfigError> {
        let r = &self.runtime;
        if r.max_inflight == 0 {
            return Err(ConfigError::Invalid("runtime.max_inflight must be > 0".into()));
        }
        if !r.ring_size.is_power_of_two() {
            return Err(ConfigError::Invalid(format!(
                "runtime.ring_size ({}) must be a power of two",
                r.ring_size
            )));
        }
        // R1 is multi-producer (crate requires >= 64), and a ring smaller than the
        // slab could fill from ingress alone → the single Core-0 thread would spin
        // on a blocked publish and starve egress. Keep ring_size >= max_inflight.
        if r.ring_size < 64 {
            return Err(ConfigError::Invalid(format!(
                "runtime.ring_size ({}) must be >= 64 (multi-producer R1)",
                r.ring_size
            )));
        }
        if r.ring_size < r.max_inflight {
            return Err(ConfigError::Invalid(format!(
                "runtime.ring_size ({}) must be >= runtime.max_inflight ({})",
                r.ring_size, r.max_inflight
            )));
        }

        // Model invariants — P2 mandates full GPU offload + a single CPU thread, and
        // the context builder needs a non-zero n_ctx (it becomes a NonZeroU32, which
        // would otherwise silently clamp to the default). Enforce them here so a bad
        // value fails at the config boundary, not deep in llama.cpp.
        let m = &self.model;
        if m.n_ctx == 0 {
            return Err(ConfigError::Invalid("model.n_ctx must be > 0".into()));
        }
        if m.n_threads != 1 {
            return Err(ConfigError::Invalid(format!(
                "model.n_threads ({}) must be 1 — full GPU offload keeps CPU for the 3 pinned cores",
                m.n_threads
            )));
        }
        if m.n_gpu_layers != -1 {
            return Err(ConfigError::Invalid(format!(
                "model.n_gpu_layers ({}) must be -1 (offload all layers to the GPU; P2 requires full offload)",
                m.n_gpu_layers
            )));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading config `{0}`: {1}")]
    Read(String, #[source] std::io::Error),
    #[error("parsing config: {0}")]
    Parse(#[source] serde_yaml::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Guards the struct↔YAML shape: a rename on either side fails here.
    #[test]
    fn parses_default_shaped_yaml() {
        let yaml = r#"
server:
  host: "0.0.0.0"
  port: 8001
model:
  gguf_path: "models/qwen3.5-2b.gguf"
  n_ctx: 262144
  n_threads: 1
  n_gpu_layers: -1
runtime:
  max_inflight: 256
  ring_size: 1024
  cores:
    web_io: 0
    text: 1
    fast_loop: 2
"#;
        let cfg: Config = serde_yaml::from_str(yaml).expect("parse");
        assert_eq!(cfg.server.port, 8001);
        assert_eq!(cfg.model.n_threads, 1);
        assert_eq!(cfg.runtime.max_inflight, 256);
        assert_eq!(cfg.runtime.cores.fast_loop, 2);
        cfg.validate().expect("default shape is valid");
    }

    // The canonical repo config must load and validate — catches drift between
    // the struct and the shipped file, and any regression in the load path.
    #[test]
    fn canonical_config_file_loads_and_validates() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../config/default-config.yaml");
        let cfg = Config::from_yaml_file(path).expect("load canonical config");
        assert_eq!(cfg.runtime.cores.web_io, 0);
    }

    #[test]
    fn validate_rejects_bad_runtime() {
        let base = |mi: u32, rs: u32| Runtime {
            max_inflight: mi,
            ring_size: rs,
            cores: Cores { web_io: 0, text: 1, fast_loop: 2 },
        };
        let with = |rt| Config {
            server: Server { host: "h".into(), port: 1 },
            model: Model { gguf_path: "g".into(), n_ctx: 1, n_threads: 1, n_gpu_layers: -1 },
            runtime: rt,
        };
        assert!(with(base(0, 1024)).validate().is_err(), "max_inflight 0");
        assert!(with(base(256, 1000)).validate().is_err(), "ring_size not power of two");
        assert!(with(base(256, 32)).validate().is_err(), "ring_size < 64");
        assert!(with(base(2048, 1024)).validate().is_err(), "ring_size < max_inflight");
        assert!(with(base(256, 1024)).validate().is_ok(), "valid");
    }

    #[test]
    fn validate_rejects_bad_model() {
        let rt = || Runtime {
            max_inflight: 256,
            ring_size: 1024,
            cores: Cores { web_io: 0, text: 1, fast_loop: 2 },
        };
        let with = |n_ctx, n_threads, n_gpu_layers| Config {
            server: Server { host: "h".into(), port: 1 },
            model: Model { gguf_path: "g".into(), n_ctx, n_threads, n_gpu_layers },
            runtime: rt(),
        };
        assert!(with(0, 1, -1).validate().is_err(), "n_ctx 0");
        assert!(with(1024, 4, -1).validate().is_err(), "n_threads != 1");
        assert!(with(1024, 1, 20).validate().is_err(), "partial GPU offload");
        assert!(with(1024, 1, -1).validate().is_ok(), "full offload, 1 thread, n_ctx>0");
    }
}
