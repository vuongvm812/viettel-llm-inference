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
    pub model: Model,
    pub runtime: Runtime,
}

#[derive(Debug, Deserialize)]
pub struct Server {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct Model {
    /// Path to the GGUF-quantized model.
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
    /// Load and parse a YAML config file.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Read(path.display().to_string(), e))?;
        serde_yaml::from_str(&raw).map_err(ConfigError::Parse)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading config `{0}`: {1}")]
    Read(String, #[source] std::io::Error),
    #[error("parsing config: {0}")]
    Parse(#[source] serde_yaml::Error),
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
    }
}
