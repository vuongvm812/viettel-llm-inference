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
//! Scope: **continuous batching (P3) + shared-prefix KV caching (P4), greedy
//! (temp=0)**. The [`Decoder`] runs a running set of sequences: each
//! [`step`](Decoder::step) batches one token per active seq into a single
//! `decode()`, samples per seq, emits, and retires finished seqs (freeing their KV
//! so the seq id can be reused). Admission reserves `suffix + max_tokens` KV cells
//! per request plus the shared prefix's K cells once, so the ~39K-token system
//! prompt is prefilled a single time and `kv_cache_seq_cp`'d into each request
//! ([`admit`](Decoder::admit) / [`prefill`](Decoder::prefill)). The per-target
//! prefill-once/TTFT verification is deferred to the GPU box (see ROADMAP P4).

use super::{admission_plan, argmax, shared_reuse_len, Admit, BackendError, Verdict};
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

    // Continuous batching (P3): cap concurrent sequences (clamped to the slab).
    let max_batch_seqs = cfg.runtime.effective_max_batch_seqs();
    // Bound tokens per decode() call. Must hold both a prefill chunk and a full
    // decode step (one token per active seq), so keep it >= max_batch_seqs.
    let n_batch = cfg.model.n_ctx.min(2048).max(max_batch_seqs);
    let text = TextBackend { model: StaticRef(model) };
    let init = DecoderInit {
        model: StaticRef(model),
        backend: StaticRef(backend),
        n_ctx: cfg.model.n_ctx,
        n_batch,
        max_batch_seqs,
        shared_prefix_tokens: cfg.runtime.shared_prefix_tokens,
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
    /// Max sequences decoded together per iteration (continuous batching, P3).
    max_batch_seqs: u32,
    /// Shared-prefix window K (P4); `0` disables (P3 behavior).
    shared_prefix_tokens: u32,
}

/// One sequence in the running set: its llama.cpp seq id, KV position, and the
/// already-sampled token to feed on the next decode step (see [`Decoder::step`]).
struct ActiveSeq {
    slot: u32,
    seq_id: i32,
    /// KV position where `next_token` will be placed.
    pos: i32,
    /// Tokens emitted so far (bounded by `max_tokens`).
    n_generated: u32,
    max_tokens: u32,
    /// Next token to feed — already sampled and confirmed non-EOG (so it will be
    /// emitted). Seeded from the prefill sample in [`Decoder::admit`].
    next_token: i32,
    /// KV cells reserved for this seq at admit (`n_prompt + max_tokens`), released
    /// back to the shared budget when it retires.
    reserve: usize,
}

/// Result of prefilling a request's first token.
enum FirstToken {
    /// First generated token (non-EOG) — the seq joins the running set.
    Token(i32),
    /// Nothing to generate (`max_tokens == 0`) or an immediate EOS — finish now.
    Done(FinishReason),
}

/// Core 2 decoder (real, P3 continuous batching). Owns the mutable `LlamaContext`
/// (KV cache) and a reusable `LlamaBatch` (allocated once, cleared between
/// chunks/steps — no per-request or per-token batch allocation on the fast loop).
/// Runs a *running set* of sequences: each [`step`](Self::step) batches one token
/// per active seq into a single `decode()`.
pub struct Decoder {
    model: StaticRef<LlamaModel>,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch,
    n_batch: usize,
    /// KV context budget; a request whose prompt + generation exceeds it is
    /// rejected before any decode (no partial streaming).
    n_ctx: usize,
    /// Running set of decoding sequences (the batch).
    active: Vec<ActiveSeq>,
    /// Free llama.cpp sequence ids (0..max_batch_seqs), used as a stack.
    seq_pool: Vec<i32>,
    max_batch_seqs: usize,
    /// KV cells currently reserved across the running set (per-seq suffix+generation
    /// plus the shared prefix's K cells, counted once). Admission keeps
    /// `reserved <= n_ctx` so a sequence is never started that can't finish.
    reserved: usize,
    /// Shared-prefix window K (P4); `0` disables sharing (P3 behavior).
    prefix_k: usize,
    /// The established shared-prefix tokens (first `prefix_k`), or `None` until the
    /// first eligible request prefills them into `prefix_seq_id`. Single entry (v1):
    /// a non-matching / short prompt falls back to a full prefill. Multi-prefix +
    /// eviction (radix tree) is P7. Cleared on a full KV wipe so it re-establishes.
    prefix: Option<Vec<i32>>,
    /// Reserved llama seq id holding the shared prefix's KV cells (id
    /// `max_batch_seqs`, outside the request `seq_pool`). Never retired while the
    /// prefix is established. Unused when `prefix_k == 0`.
    prefix_seq_id: i32,
}

impl Decoder {
    pub fn new(init: DecoderInit) -> Self {
        let max_batch_seqs = init.max_batch_seqs as usize;
        let prefix_k = init.shared_prefix_tokens as usize;
        // Shared-prefix caching (P4) needs one extra sequence id beyond the request
        // pool to hold the prefix's KV cells (id `max_batch_seqs`).
        let prefix_extra = if prefix_k > 0 { 1 } else { 0 };
        let cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(init.n_ctx))
            .with_n_batch(init.n_batch)
            // Allow up to `max_batch_seqs` concurrent request sequences plus the
            // reserved prefix seq in the KV cache. (Confirm the method name against
            // llama-cpp-2 on the target — the crate occasionally renames setters.)
            .with_n_seq_max(init.max_batch_seqs + prefix_extra)
            .with_n_threads(1); // full GPU offload → 1 CPU thread (don't fight the 3 cores)
        let ctx = init
            .model
            .0
            .new_context(init.backend.0, cparams)
            .expect("new llama context");
        // Free-id stack: pop yields 0, 1, 2, … for readable seq ids.
        let seq_pool: Vec<i32> = (0..max_batch_seqs as i32).rev().collect();
        Decoder {
            model: init.model,
            ctx,
            // Second arg is `n_seq_max` = max sequence ids a *single token* may belong
            // to, not the batch's sequence count. Every `add` here passes `&[seq_id]`
            // (one seq per token), and P4 shared-prefix reuses cells via
            // `kv_cache_seq_cp` rather than multi-seq tokens — so this stays 1. Token
            // capacity (holding up to `max_batch_seqs` tokens per decode step) is the
            // first arg, `n_batch`.
            batch: LlamaBatch::new(init.n_batch as usize, 1),
            n_batch: init.n_batch as usize,
            n_ctx: init.n_ctx as usize,
            active: Vec::with_capacity(max_batch_seqs),
            seq_pool,
            max_batch_seqs,
            reserved: 0,
            prefix_k,
            prefix: None,
            // The prefix seq id sits just past the request ids (`0..max_batch_seqs`).
            prefix_seq_id: max_batch_seqs as i32,
        }
    }

    /// Room in the running set for another sequence *and* a free seq id to give it.
    /// Gating on `seq_pool` too keeps [`Admit::Deferred`]'s contract: without it, an
    /// empty pool (seq ids leaked by an "impossible" KV-clear failure) would make
    /// `admit` return `Deferred` while idle, and Core 2 — seeing pending work but an
    /// idle decoder — would spin forever. With this gate, admit is only called when a
    /// seq id is available, so an idle pipeline never defers.
    pub fn has_capacity(&self) -> bool {
        self.active.len() < self.max_batch_seqs && !self.seq_pool.is_empty()
    }

    /// No active sequences → nothing to step.
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    /// Tokens [`admit`](Self::admit) would actually decode for `slot`: the full prompt,
    /// unless it *shares an already-established* prefix, in which case only the unshared
    /// suffix (the K prefix cells are `kv_cache_seq_cp`'d, not re-decoded). Read-only;
    /// the establishing request still reports the full prompt (it pays the prefix cost).
    /// Core 2 peeks this for the per-iteration prefill-token budget, so shared-prefix
    /// requests batch instead of serializing behind the HOL guard.
    pub fn effective_prefill_len(&self, slot: u32, slab: &Slab) -> usize {
        // SAFETY: `slot` sits in Core 2's `pending` → Core 2 solely owns it (read only).
        let tokens = unsafe { slab.slot_tokens(slot) };
        tokens.len() - shared_reuse_len(self.prefix_k, self.prefix.as_deref(), tokens)
    }

    /// Wipe the entire KV cache and reset the decoder to its baseline (empty running
    /// set, full seq-id pool, zero reservation, no resident prefix). The single place
    /// that resets `prefix = None` on a full clear, so no recovery path can forget it.
    fn wipe_all_kv(&mut self) {
        self.ctx.clear_kv_cache();
        self.seq_pool.clear();
        self.seq_pool.extend((0..self.max_batch_seqs as i32).rev());
        self.active.clear();
        self.reserved = 0;
        self.prefix = None;
    }

    /// Drop the resident shared prefix, freeing its K reserved cells (and the prefix
    /// seq's KV). Called only when the decoder is idle and a fallback request can't fit
    /// alongside the prefix — v1 has no eviction, so without this the never-reclaimed
    /// prefix would wedge the fittable request. No-op if none is resident.
    fn evict_prefix(&mut self) {
        if self.prefix.take().is_none() {
            return;
        }
        self.reserved = self.reserved.saturating_sub(self.prefix_k);
        // Idle here (the sole caller guarantees `active.is_empty()`), so on the
        // "impossible" in-range-id clear failure a full wipe is safe and reclaims all.
        if let Err(e) = self.free_seq_kv(self.prefix_seq_id) {
            eprintln!("core2 evict_prefix: KV clear failed for prefix seq ({e}); full wipe");
            self.wipe_all_kv();
        }
    }

    /// Admit a tokenized slot into the running set, honouring the shared KV budget.
    /// With shared-prefix caching (P4) the first K tokens are shared: the prefix is
    /// prefilled once and `kv_cache_seq_cp`'d into each matching request, so only the
    /// suffix + generation is reserved per request (the K prefix cells are counted
    /// once). Returns [`Admit::Deferred`] when the seq won't currently fit (Core 2
    /// retries after a retire), [`Admit::Rejected`] (having emitted `Finish(Error)`)
    /// when it can't fit `n_ctx` even alone, else [`Admit::Admitted`] with the
    /// *effective* prefill-token count (suffix only when the prefix was shared).
    ///
    /// The per-seq reservation is still conservative (books `max_tokens` even though
    /// most seqs stop early at EOS); shared-prefix lowers effective per-request KV.
    pub fn admit<P: Producer<RingEvent>>(&mut self, slot: u32, slab: &Slab, r3: &mut P) -> Admit {
        // SAFETY: slot is in Core 2's `pending` (arrived via R2, no R3 publish for it
        // yet) → Core 2 solely owns it.
        let (n_prompt, max_tokens) = {
            let s = unsafe { slab.slot_mut(slot) };
            (s.tokens.len(), s.max_tokens)
        };
        // Nothing to generate → finish immediately without a seq id or KV.
        if n_prompt == 0 || max_tokens == 0 || self.n_batch == 0 {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::MaxTokens) });
            return Admit::Admitted(0);
        }
        // The KV/share accounting is the shared `admission_plan` oracle (identical to the
        // mock, which is its sandbox test); this fn owns only the real KV side effects.
        let plan = {
            // SAFETY: slot owned by Core 2 here (read-only peek of the prompt tokens).
            let tokens = unsafe { slab.slot_tokens(slot) };
            admission_plan(
                self.prefix_k,
                self.prefix.as_deref(),
                self.reserved,
                self.n_ctx,
                self.active.is_empty(),
                tokens,
                max_tokens,
            )
        };
        match plan.verdict {
            Verdict::Reject => {
                eprintln!("core2 admit rejected slot {slot}: needs {} > n_ctx {}", n_prompt + max_tokens as usize, self.n_ctx);
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
                return Admit::Rejected;
            }
            Verdict::Defer => return Admit::Deferred,
            Verdict::Admit => {}
        }
        if plan.evict_prefix {
            self.evict_prefix();
        }

        // `has_capacity()` guarantees a free id, but don't rely on the caller: if the
        // pool is somehow empty, defer (retry after a retire) rather than panic.
        let Some(seq_id) = self.seq_pool.pop() else {
            return Admit::Deferred;
        };
        match self.prefill(slot, slab, seq_id, n_prompt, plan.reused, plan.establishing) {
            Ok(FirstToken::Token(tok)) => {
                if plan.establishing {
                    // Prefill already donated this seq's prefix cells to `prefix_seq`
                    // (only reached on a non-EOG first token), so record the prefix now.
                    // SAFETY: slot owned by Core 2 here (no R3 publish in prefill).
                    let tokens = unsafe { slab.slot_tokens(slot) };
                    self.prefix = Some(tokens[..self.prefix_k].to_vec());
                }
                self.reserved += plan.inc;
                self.active.push(ActiveSeq {
                    slot,
                    seq_id,
                    pos: n_prompt as i32,
                    n_generated: 0,
                    max_tokens,
                    next_token: tok,
                    reserve: plan.reserve,
                });
                Admit::Admitted(plan.effective)
            }
            Ok(FirstToken::Done(reason)) => {
                self.retire_seq(seq_id); // never reserved
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(reason) });
                Admit::Admitted(plan.effective)
            }
            Err(e) => {
                eprintln!("core2 prefill error on slot {slot}: {e}");
                self.retire_seq(seq_id);
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
                Admit::Rejected
            }
        }
    }

    /// Prefill `slot`'s prompt into `seq_id`'s KV and sample its first token, reusing
    /// `reused` leading cells from the shared prefix when `reused > 0` (P4):
    ///
    /// - **sharing** (`reused > 0`): `kv_cache_seq_cp` the prefix's `0..reused` cells
    ///   into `seq_id` (no recompute of the 39K system prompt), then decode only the
    ///   suffix `tokens[reused..n_prompt]` at `pos = reused`.
    /// - **establishing** (`reused == 0`, first eligible request): decode the full
    ///   prompt, then donate `seq_id`'s `0..K` cells to the reserved prefix seq so later
    ///   requests can copy them — the prefix is computed exactly once, here.
    /// - **fallback** (`reused == 0`, not establishing): decode the full prompt as in P3.
    ///
    /// `reused`/`establishing` come from [`admission_plan`] (`reused` is already capped
    /// so ≥1 token is decoded, and is 0 whenever establishing). Reads prompt tokens
    /// straight from the slab (no copy); publishes nothing on R3, so holding the `&mut`
    /// slot across prefill is sound. P7 adds chunked prefill (split one prompt across
    /// iterations) to also interleave a long establisher with decode.
    /// SAFETY: slot arrived via R2 → Core 2 owns it here (no R3 publish in this fn).
    fn prefill(
        &mut self,
        slot: u32,
        slab: &Slab,
        seq_id: i32,
        n_prompt: usize,
        reused: usize,
        establishing: bool,
    ) -> Result<FirstToken, BackendError> {
        let s = unsafe { slab.slot_mut(slot) };
        // Sharing an established prefix: copy its `0..reused` cells into this seq and skip
        // straight to the suffix (`reused > 0` only when sharing — 0 for establish /
        // fallback). Confirm the exact llama-cpp-2 spelling on target — recent llama.cpp
        // renamed `llama_kv_cache_seq_cp` toward a `kv_self`/memory API; the copy/remove
        // semantics are what we depend on (see module note).
        let start = if reused > 0 {
            self.ctx
                .copy_kv_cache_seq(self.prefix_seq_id, seq_id, 0, reused as i32)
                .map_err(|e| BackendError::Decode(e.to_string()))?;
            reused
        } else {
            0
        };
        // Decode `tokens[start..n_prompt]` in n_batch-sized chunks (a 40K-token prompt
        // exceeds one decode). Only the final prompt token needs logits.
        let mut logit_idx = 0i32;
        let mut chunk_start = start;
        while chunk_start < n_prompt {
            let chunk_end = (chunk_start + self.n_batch).min(n_prompt);
            self.batch.clear();
            for pos in chunk_start..chunk_end {
                let is_last = pos == n_prompt - 1;
                self.batch
                    .add(LlamaToken(s.tokens[pos]), pos as i32, &[seq_id], is_last)
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

        // Sample the first generated token from the last prompt position. Determinism
        // is pure argmax (temp=0), so the request's `seed` is a no-op — "seed=42" is
        // satisfied vacuously; it would matter only on a stochastic sampler.
        let first = argmax(self.ctx.get_logits_ith(logit_idx));
        // Stop before emitting an EOG sentinel (never user-visible text).
        if self.model.0.is_eog_token(LlamaToken(first)) {
            // Immediate EOS: the seq won't join and `admit` won't record the prefix, so
            // donate NOTHING to `prefix_seq` — otherwise it would hold orphaned cells and
            // the next request would re-establish on top of them (KV corruption).
            Ok(FirstToken::Done(FinishReason::Eos))
        } else {
            // The seq will join. If establishing, donate its first K cells to the
            // reserved prefix seq now (once) so future matching requests copy them
            // instead of recomputing. The cells are shared, so this seq retiring later
            // (`seq_rm`) leaves the prefix resident. Paired with `admit` recording
            // `self.prefix` on this same `Token` arm — donation and bookkeeping stay
            // atomic (nothing is left behind on the Done/Err paths).
            if establishing {
                self.ctx
                    .copy_kv_cache_seq(seq_id, self.prefix_seq_id, 0, self.prefix_k as i32)
                    .map_err(|e| BackendError::Decode(e.to_string()))?;
            }
            Ok(FirstToken::Token(first))
        }
    }

    /// One decode step over the whole running set: feed each active seq's pending
    /// token into a single batch, `decode()` once (the GPU forward for every sequence
    /// at once — where continuous batching pays off), then per seq emit that token,
    /// sample the next, and retire on `max_tokens`/EOS.
    pub fn step<P: Producer<RingEvent>>(&mut self, _slab: &Slab, r3: &mut P) {
        // Build one batch: one token per active seq, each at its own position/seq id.
        // Batch index i == active[i], so get_logits_ith(i) reads seq i's logits below.
        self.batch.clear();
        let mut build_err: Option<String> = None;
        for seq in &self.active {
            if let Err(e) = self.batch.add(LlamaToken(seq.next_token), seq.pos, &[seq.seq_id], true) {
                build_err = Some(e.to_string());
                break;
            }
        }
        if let Some(e) = build_err {
            return self.fail_all(r3, &e);
        }
        if let Err(e) = self.ctx.decode(&mut self.batch) {
            return self.fail_all(r3, &e.to_string());
        }

        // Emit the fed token for every active seq in ONE R3 burst — the design's
        // batch-publish throughput path. Each token was confirmed non-EOG when sampled.
        // The burst is `active.len()` events, which is <= max_batch_seqs <= max_inflight
        // <= ring_size (config-enforced), so it always fits: never a batch larger than
        // the ring, and Core 1 drains R3 independently, so the spin is bounded.
        let n = self.active.len();
        r3.batch_publish(n, |iter| {
            for (e, seq) in iter.zip(self.active.iter()) {
                *e = RingEvent { slot: seq.slot, kind: EventKind::Token(seq.next_token as u32) };
            }
        });

        // Sample the next token per seq, retire finished ones, compact survivors in one
        // pass. Fields (`active`, `ctx`, `model`, `seq_pool`) are disjoint, so the
        // indexed borrows below don't overlap. Finishes (a minority) stay individual.
        let mut w = 0;
        // Set if a retiring seq's KV clear fails: its id can't be safely reused, so we
        // wipe the whole cache and fail the survivors after this pass (below) rather
        // than decode the next step on top of KV we couldn't clean.
        let mut kv_clear_failed = false;
        for r in 0..self.active.len() {
            self.active[r].n_generated += 1;
            self.active[r].pos += 1;

            let finished = if self.active[r].n_generated >= self.active[r].max_tokens {
                Some(FinishReason::MaxTokens)
            } else {
                let next = argmax(self.ctx.get_logits_ith(r as i32));
                if self.model.0.is_eog_token(LlamaToken(next)) {
                    Some(FinishReason::Eos) // stop before emitting the EOG sentinel
                } else {
                    self.active[r].next_token = next;
                    None
                }
            };

            match finished {
                None => {
                    // Keep: compact toward the front (order among seqs is irrelevant —
                    // each slot's tokens stay in generation order across steps).
                    if w != r {
                        self.active.swap(w, r);
                    }
                    w += 1;
                }
                Some(reason) => {
                    let slot = self.active[r].slot;
                    let seq_id = self.active[r].seq_id;
                    r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(reason) });
                    self.reserved -= self.active[r].reserve; // give the KV budget back
                    // Only hand the seq id back to the pool once its KV is actually
                    // gone — a reused id on stale KV corrupts generation.
                    match self.free_seq_kv(seq_id) {
                        Ok(()) => self.seq_pool.push(seq_id),
                        Err(e) => {
                            eprintln!("core2 step: KV clear failed for seq {seq_id}: {e}");
                            kv_clear_failed = true;
                        }
                    }
                }
            }
        }
        self.active.truncate(w);
        // A retire couldn't free its KV: the cache is in an unknown state, so the
        // survivors can't be trusted to keep decoding. Fail them, wipe the whole cache,
        // and rebuild the id pool from scratch (baseline) — conservative but sound.
        if kv_clear_failed {
            self.fail_all(r3, "seq KV clear failed during retire; wiping cache");
        }
    }

    /// Drop a sequence's KV cells so its id can be reused. `(None, None)` must map to
    /// the FULL position range (llama.cpp `seq_rm(seq, -1, -1)`) — any cells left
    /// behind would let a reused id prefill on top of stale KV → corrupt output and
    /// eventual cache exhaustion. Seq ids are `0..max_batch_seqs` (always `>= 0` and
    /// `< i32::MAX`), so the `u32` the crate wants is an exact cast and this *should*
    /// never fail — but the result is propagated, not discarded, so callers gate seq-id
    /// reuse on it. On the target, also assert free cells return to baseline after a
    /// retire. (Confirm the exact `clear_kv_cache_seq` return type against the crate
    /// version; propagated here as a `Result`.)
    #[must_use = "a failed KV clear means the seq id is NOT safe to reuse"]
    fn free_seq_kv(&mut self, seq_id: i32) -> Result<(), BackendError> {
        self.ctx
            .clear_kv_cache_seq(Some(seq_id as u32), None, None)
            .map_err(|e| BackendError::Decode(e.to_string()))
    }

    /// Return a seq id to the free pool and drop its KV cells. Used on admit-time
    /// finish/error, before the seq ever joins the running set (so it holds no
    /// reservation to release). On a KV-clear failure (the "impossible" in-range-id
    /// path) the id must not be handed back dirty — a reused id on stale KV corrupts
    /// output. Recovery depends on what's decoding:
    /// - **Cache idle** (no active seqs): safe to wipe the whole cache and rebuild the
    ///   id pool from scratch — this recovers the just-failed id *and* any previously
    ///   leaked one, so repeated admit-time failures while idle can't exhaust the pool.
    /// - **Active seqs present**: can't wipe their live KV mid-flight, so leak just this
    ///   one id (degrades capacity, never corrupts) until the running set drains.
    fn retire_seq(&mut self, seq_id: i32) {
        match self.free_seq_kv(seq_id) {
            Ok(()) => self.seq_pool.push(seq_id),
            Err(e) if self.active.is_empty() => {
                eprintln!("core2 retire: KV clear failed for seq {seq_id} ({e}); cache idle → full wipe + pool reset");
                self.wipe_all_kv();
            }
            Err(e) => eprintln!("core2 retire: KV clear failed for seq {seq_id}, leaking id (active seqs present): {e}"),
        }
    }

    /// A batch build/decode failure kills every active sequence (the forward pass is
    /// shared): report `Error` for each, wipe the KV cache, and restore all seq ids +
    /// the whole KV budget.
    fn fail_all<P: Producer<RingEvent>>(&mut self, r3: &mut P, err: &str) {
        eprintln!("core2 batch decode error ({} seqs): {err}", self.active.len());
        for seq in &self.active {
            let slot = seq.slot;
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
        }
        // All active seqs die → reclaim everything (KV, seq ids, reservation, prefix).
        self.wipe_all_kv();
    }
}
