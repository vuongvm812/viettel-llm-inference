//! Real llama.cpp backend (`--features llama`). **Compiled on the target box**
//! (needs `cmake` + `libllama` toolchain + a GGUF model); not buildable in a
//! CPU-only/macOS-without-cmake dev sandbox. Acceleration is chosen by the
//! `metal`/`cuda` cargo features wired per-OS in `Cargo.toml`.
//!
//! Written against `llama-cpp-2` 0.1.150. The crate occasionally renames C-API
//! wrappers between versions — if a method name below doesn't resolve, check the
//! installed crate's docs; the *mechanisms* (tokenize, prefill, greedy sample,
//! KV clear) are stable. Confirm on target.
//!
//! Scope: **single sequence, greedy (temp=0)**. Continuous batching (P3) and
//! shared-prefix KV (P4) are later phases; here each request prefills, decodes to
//! EOS/`max_tokens`, then `clear_kv_cache()` reclaims all KV for the next one.

use super::{argmax, BackendError};
use crate::config::Config;
use crate::rings::{EventKind, FinishReason, RingEvent};
use crate::slab::Slab;
use disruptor::Producer;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;

/// Far above any real model's layer count → offload every layer to the GPU
/// (Metal/CUDA). Avoids the `u32::MAX`→`i32` wrap the C API would see.
const ALL_GPU_LAYERS: u32 = 1_000_000;

/// `&'static` handle to a process-lifetime llama object, made `Send + Sync` for
/// our pinned worker threads. The `Send`/`Sync` assertions are scoped to the two
/// concrete types we actually share (below), not blanket over all `T`, so the
/// unsafe claim stays auditable and can't silently override a crate type that
/// deliberately withholds `Sync`.
#[derive(Clone, Copy)]
pub struct StaticRef<T: 'static>(pub &'static T);

// SAFETY / threading contract (llama.cpp):
// - `llama_model` is **read-only after load**. llama.cpp's own contract is that a
//   single loaded model is shared across many `llama_context`s, including from
//   multiple threads — this is exactly how `llama-server` / the `parallel` example
//   serve concurrent requests (one model, N contexts). All mutable inference state
//   (KV cache, sampling) lives in the per-thread `LlamaContext`, never in the model.
//   Here Core 1 only calls vocab methods (`str_to_token` / `token_to_bytes`) and
//   Core 2 only `new_context` on the shared `&LlamaModel` — all `&self`, no mutation.
// - `llama_backend` is process-global init-once state (`llama_backend_init`), held
//   for the whole run and never mutated through this handle.
// If a future `llama-cpp-2` already marks these `Send + Sync`, drop this newtype and
// share `&'static T` directly; the impls below just make the current guarantee
// explicit and version-independent. (Re-confirm the crate's markers on the target.)
unsafe impl Send for StaticRef<LlamaModel> {}
unsafe impl Sync for StaticRef<LlamaModel> {}
unsafe impl Send for StaticRef<LlamaBackend> {}
unsafe impl Sync for StaticRef<LlamaBackend> {}

/// Load the backend + model once and leak them to `'static`. The leak is
/// **deliberate**: backend + model are process-lifetime singletons (one model, one
/// backend for the whole run), so `Box::leak` is the honest owner — it avoids `Arc`
/// refcount atomics for something never freed, hands out `Copy` `&'static` refs, and
/// lets `LlamaContext<'static>` be owned by Core 2 with no self-reference. `load` is
/// called exactly once (from `pipeline::spawn`); a second call would leak another
/// multi-GB model. Panics on failure — a missing/corrupt GGUF is a fatal startup
/// error, like a bad config.
pub fn load(cfg: &Config) -> (TextBackend, DecoderInit) {
    let backend = LlamaBackend::init().expect("llama backend init");
    let backend: &'static LlamaBackend = Box::leak(Box::new(backend));

    let gpu_layers = if cfg.model.n_gpu_layers < 0 {
        ALL_GPU_LAYERS
    } else {
        cfg.model.n_gpu_layers as u32
    };
    let mparams = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
    let model = LlamaModel::load_from_file(backend, &cfg.model.gguf_path, &mparams)
        .unwrap_or_else(|e| panic!("load GGUF `{}`: {e}", cfg.model.gguf_path));
    let model: &'static LlamaModel = Box::leak(Box::new(model));

    let text = TextBackend { model: StaticRef(model) };
    let init = DecoderInit {
        model: StaticRef(model),
        backend: StaticRef(backend),
        n_ctx: cfg.model.n_ctx,
        // Bound tokens per decode() call. Full context per batch is fine for
        // single-seq prefill chunking; tune on the GPU.
        n_batch: cfg.model.n_ctx.min(2048),
    };
    (text, init)
}

/// Core 1 text codec (real vocab).
#[derive(Clone, Copy)]
pub struct TextBackend {
    model: StaticRef<LlamaModel>,
}

impl TextBackend {
    /// Tokenize the prompt with the model's own vocab (must match the GGUF or
    /// token ids won't line up). BOS is prepended, matching a plain `llama-cli`
    /// run for the determinism check. A failure is returned (not swallowed): Core 1
    /// turns it into a `Finish(Error)` for the client instead of decoding an empty
    /// prompt.
    pub fn tokenize(&self, prompt: &str, out: &mut Vec<i32>) -> Result<(), BackendError> {
        let tokens = self
            .model
            .0
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| BackendError::Tokenize(e.to_string()))?;
        out.extend(tokens.into_iter().map(|t| t.0));
        Ok(())
    }

    /// Detokenize one generated token id into `piece` (may be multi-byte, or a
    /// partial code point completed by a later token — Core 1's UTF-8 gate handles
    /// that). A failure is returned so Core 1 flags the stream as errored rather
    /// than silently dropping output.
    ///
    /// ponytail: `token_to_bytes` allocates a `Vec` per token inside the crate —
    /// llama-cpp-2 0.1.150 exposes no buffer-writing detok variant, so this copy
    /// is unavoidable without a lower-level binding; P6 target if the profile
    /// flags it. The `extend_from_slice` into `piece` is the one copy we control
    /// and is needed for Core 1's capacity guard.
    pub fn token_bytes(&self, id: u32, piece: &mut Vec<u8>) -> Result<(), BackendError> {
        piece.clear();
        // Special::Plaintext → don't render control tokens as literal text.
        let bytes = self
            .model
            .0
            .token_to_bytes(LlamaToken(id as i32), Special::Plaintext)
            .map_err(|e| BackendError::Detokenize(e.to_string()))?;
        piece.extend_from_slice(&bytes);
        Ok(())
    }
}

/// Startup handles for the real decoder. `Send` via [`StaticRef`] so it can move
/// into the Core 2 thread, where [`Decoder::new`] builds the (thread-affine,
/// single-threaded) `LlamaContext`.
pub struct DecoderInit {
    model: StaticRef<LlamaModel>,
    backend: StaticRef<LlamaBackend>,
    n_ctx: u32,
    n_batch: u32,
}

/// Core 2 decoder (real). Owns the mutable `LlamaContext` (KV cache) and a reusable
/// `LlamaBatch` (allocated once, cleared between chunks/steps — no per-request or
/// per-token batch allocation on the fast loop).
pub struct Decoder {
    model: StaticRef<LlamaModel>,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch,
    n_batch: usize,
    /// KV context budget; a request whose prompt + generation exceeds it is
    /// rejected before any decode (no partial streaming).
    n_ctx: usize,
}

impl Decoder {
    pub fn new(init: DecoderInit) -> Self {
        let cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(init.n_ctx))
            .with_n_batch(init.n_batch)
            .with_n_threads(1); // full GPU offload → 1 CPU thread (don't fight the 3 cores)
        let ctx = init
            .model
            .0
            .new_context(init.backend.0, cparams)
            .expect("new llama context");
        Decoder {
            model: init.model,
            ctx,
            batch: LlamaBatch::new(init.n_batch as usize, 1),
            n_batch: init.n_batch as usize,
            n_ctx: init.n_ctx as usize,
        }
    }

    /// Prefill + greedy-decode one sequence to completion, emitting each token on
    /// R3 and a terminal `Finish`. KV is cleared afterward so the next request
    /// starts from an empty cache (shared-prefix reuse is P4).
    pub fn run_sequence<P: Producer<RingEvent>>(&mut self, slot: u32, slab: &Slab, r3: &mut P) {
        let reason = match self.generate(slot, slab, r3) {
            Ok(reason) => reason,
            Err(e) => {
                eprintln!("core2 decode error on slot {slot}: {e}");
                FinishReason::Error
            }
        };
        r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(reason) });
        self.ctx.clear_kv_cache();
    }

    fn generate<P: Producer<RingEvent>>(
        &mut self,
        slot: u32,
        slab: &Slab,
        r3: &mut P,
    ) -> Result<FinishReason, BackendError> {
        // Phase 1 — prefill, reading prompt tokens straight from the slab (no copy).
        // Core 2 still exclusively owns the slot here: we publish nothing on R3 until
        // Phase 2, so Core 1 cannot touch this slot's output, and holding the `&mut`
        // across prefill is sound. The borrow ends with this block, before any R3
        // publish. Returns (max_tokens, n_prompt, logit_idx of the last prompt token).
        // SAFETY: slot arrived via R2 → Core 2 owns it until the first R3 publish below.
        let (max_tokens, n_prompt, logit_idx) = {
            let s = unsafe { slab.slot_mut(slot) };
            let n_prompt = s.tokens.len();
            let max_tokens = s.max_tokens;
            if n_prompt == 0 || max_tokens == 0 || self.n_batch == 0 {
                return Ok(FinishReason::MaxTokens);
            }
            // Preflight the context budget: reject *before* prefill so a too-large
            // request never streams a partial reply it can't finish.
            let needed = n_prompt.saturating_add(max_tokens as usize);
            if needed > self.n_ctx {
                return Err(BackendError::ContextOverflow { needed, n_ctx: self.n_ctx });
            }

            // Prefill in n_batch-sized chunks (a 40K-token prompt exceeds one decode).
            // Only the final prompt token needs logits (that's what we sample from).
            let mut logit_idx = 0i32;
            let mut chunk_start = 0usize;
            while chunk_start < n_prompt {
                let chunk_end = (chunk_start + self.n_batch).min(n_prompt);
                self.batch.clear();
                for pos in chunk_start..chunk_end {
                    let is_last = pos == n_prompt - 1;
                    self.batch
                        .add(LlamaToken(s.tokens[pos]), pos as i32, &[0], is_last)
                        .map_err(|e| BackendError::Decode(e.to_string()))?;
                }
                self.ctx
                    .decode(&mut self.batch)
                    .map_err(|e| BackendError::Decode(e.to_string()))?;
                if chunk_end == n_prompt {
                    logit_idx = (chunk_end - chunk_start - 1) as i32;
                }
                chunk_start = chunk_end;
            }
            (max_tokens, n_prompt, logit_idx)
        };

        // Phase 2 — greedy decode loop. Publishes on R3, touches no slab state.
        // Determinism is pure argmax (temp=0), so the request's `seed` is a no-op
        // here — the spec's "seed=42" is satisfied vacuously; it would matter only
        // on a stochastic sampler (P-general).
        let mut logit_idx = logit_idx;
        let mut pos = n_prompt as i32;
        for _ in 0..max_tokens {
            let next = LlamaToken(argmax(self.ctx.get_logits_ith(logit_idx)));
            // Stop on EOS/EOG *before* publishing — the terminal token is a control
            // token, not user-visible text; emitting it could leak a stray sentinel.
            if self.model.0.is_eog_token(next) {
                return Ok(FinishReason::Eos);
            }
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Token(next.0 as u32) });
            self.batch.clear();
            self.batch
                .add(next, pos, &[0], true)
                .map_err(|e| BackendError::Decode(e.to_string()))?;
            self.ctx
                .decode(&mut self.batch)
                .map_err(|e| BackendError::Decode(e.to_string()))?;
            pos += 1;
            logit_idx = 0; // single-token batch
        }
        Ok(FinishReason::MaxTokens)
    }
}
