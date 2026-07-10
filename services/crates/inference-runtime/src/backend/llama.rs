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

use super::{
    admission_plan, argmax, step_prefill_budget, Admit, BackendError, NodeId, PrefixTrie, Verdict,
    MAX_PREFIX_SEQS, MAX_TRIE_NODES,
};
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

// `llama.cpp`'s `llama_flash_attn_type` is a bindgen alias to `c_int`; these are its two
// explicit values (the crate's `with_flash_attention_policy` takes that type). Naming them
// here avoids a whole `llama-cpp-sys-2` dependency just for two integers.
// ponytail: hard-coded ABI enum values — if upstream ever renumbers them, `make parity-flash`
// (which forces the policy on) fails loudly rather than silently mis-setting it.
const FLASH_ATTN_DISABLED: i32 = 0;
const FLASH_ATTN_ENABLED: i32 = 1;

/// `&'static` handle to a process-lifetime llama object, made `Send + Sync` for
/// our pinned worker threads. The `Send`/`Sync` assertions are scoped to the two
/// concrete types we actually share (below), not blanket over all `T`, so the
/// unsafe claim stays auditable and can't silently override a crate type that
/// deliberately withholds `Sync`.
pub struct StaticRef<T: 'static>(pub &'static T);

// Manual Clone/Copy: `&'static T` is always `Copy` regardless of `T`, but a `#[derive]`
// wrongly adds a `T: Clone`/`T: Copy` bound (which `LlamaModel` doesn't satisfy).
impl<T: 'static> Clone for StaticRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: 'static> Copy for StaticRef<T> {}

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
    // `0` = one ubatch per decode step (= n_batch); a smaller configured value splits the
    // forward pass. llama.cpp requires n_ubatch <= n_batch, so clamp.
    let n_ubatch = if cfg.runtime.n_ubatch == 0 {
        n_batch
    } else {
        cfg.runtime.n_ubatch.min(n_batch)
    };
    let text = TextBackend { model: StaticRef(model) };
    let init = DecoderInit {
        model: StaticRef(model),
        backend: StaticRef(backend),
        n_ctx: cfg.model.n_ctx,
        n_batch,
        n_ubatch,
        kv_defrag_thold: cfg.runtime.kv_defrag_thold,
        flash_attention: cfg.runtime.flash_attn(),
        max_batch_seqs,
        shared_prefix_tokens: cfg.runtime.shared_prefix_tokens,
        max_batch_tokens: cfg.runtime.max_batch_tokens as usize,
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
    /// Physical micro-batch (`with_n_ubatch`); already resolved (`0` → `n_batch`) and
    /// clamped to `n_batch` in [`load`].
    n_ubatch: u32,
    /// KV-cache defragmentation threshold (`with_defrag_thold`); `<= 0` disables.
    kv_defrag_thold: f32,
    /// Flash-attention policy; `Auto` leaves llama.cpp's default (byte-parity preserved).
    flash_attention: crate::config::FlashAttn,
    /// Max sequences decoded together per iteration (continuous batching, P3).
    max_batch_seqs: u32,
    /// Shared-prefix window K (P4); `0` disables (P3 behavior).
    shared_prefix_tokens: u32,
    /// Per-step prefill-chunk budget (P7 chunked prefill).
    max_batch_tokens: usize,
}

/// One sequence in the running set: its llama.cpp seq id, KV position, and (once
/// Decoding) the already-sampled token to feed on the next decode step (see
/// [`Decoder::step`]).
///
/// P7 chunked prefill splits the lifecycle into two phases keyed off `prefill_pos`:
/// - **Prefilling** (`prefill_pos < n_prompt`): `step` decodes a chunk of the prompt
///   `tokens[prefill_pos..]` per iteration, interleaved with the decode of active seqs.
///   `next_token` is not yet valid.
/// - **Decoding** (`prefill_pos == n_prompt`): prefill is complete, `next_token` holds
///   the confirmed non-EOG token to feed next, and the seq emits one token per step.
struct ActiveSeq {
    slot: u32,
    seq_id: i32,
    /// KV position where the next token (prompt or generated) will be placed.
    pos: i32,
    /// Total prompt length; prefill is done when `prefill_pos == n_prompt`.
    n_prompt: usize,
    /// Next prompt index to prefill. `< n_prompt` ⇒ Prefilling; `== n_prompt` ⇒ Decoding.
    prefill_pos: usize,
    /// This seq establishes a radix-cache fork (P7): the trie node to donate its cells to
    /// and mark `ready` once prefill completes (`None` otherwise).
    establishes: Option<NodeId>,
    /// Tokens emitted so far (bounded by `max_tokens`).
    n_generated: u32,
    max_tokens: u32,
    /// Next token to feed once Decoding — sampled at prefill completion, confirmed
    /// non-EOG. Meaningless while Prefilling.
    next_token: i32,
    /// KV cells reserved for this seq at admit (`n_prompt + max_tokens`), released
    /// back to the shared budget when it retires.
    reserve: usize,
}

/// Core 2 decoder (real, P3 continuous batching). Owns the mutable `LlamaContext`
/// (KV cache) and a reusable `LlamaBatch` (allocated once, cleared between
/// chunks/steps — no per-request or per-token batch allocation on the fast loop).
/// Runs a *running set* of sequences: each [`step`](Self::step) batches one token
/// per active seq into a single `decode()`.
pub struct Decoder {
    model: StaticRef<LlamaModel>,
    ctx: LlamaContext<'static>,
    // `'static`: `LlamaBatch`'s lifetime only binds `get_one` (unused here); we build via
    // `new` + `add`, so any lifetime works.
    batch: LlamaBatch<'static>,
    n_batch: usize,
    /// KV context budget; a request whose prompt + generation exceeds it is
    /// rejected before any decode (no partial streaming).
    n_ctx: usize,
    /// Running set of decoding sequences (the batch).
    active: Vec<ActiveSeq>,
    /// Scratch reused across `step()` calls so the fast loop allocates nothing per iteration
    /// (the design's zero-alloc contract): decode seqs' `(active_idx, batch_logit_idx)`, and a
    /// per-seq retire mask. Cleared/resized at the top of each `step`, never reallocated once
    /// grown to `max_batch_seqs`.
    decode_idx: Vec<(usize, i32)>,
    retire_mask: Vec<bool>,
    /// Free llama.cpp sequence ids (0..max_batch_seqs), used as a stack.
    seq_pool: Vec<i32>,
    max_batch_seqs: usize,
    /// KV cells currently reserved across the running set (per-seq suffix+generation
    /// plus the shared prefix's K cells, counted once). Admission keeps
    /// `reserved <= n_ctx` so a sequence is never started that can't finish.
    reserved: usize,
    /// Per-step prefill-chunk budget (P7): prompt tokens prefilled per unified batch,
    /// on top of the one decode token per active Decoding seq.
    max_batch_tokens: usize,
    /// Radix cache of shared prefixes (P7) — variable-length longest match, nested forks,
    /// LRU eviction. Generalizes P4's single fixed-K prefix. Each materialized node's cells
    /// live in a prefix seq id drawn from `prefix_seq_pool`.
    trie: PrefixTrie,
    /// Free llama seq ids reserved for materialized prefixes (ids `max_batch_seqs ..
    /// max_batch_seqs + MAX_PREFIX_SEQS`, outside the request `seq_pool`). A materialized
    /// trie node holds one; it returns here on eviction / full wipe.
    prefix_seq_pool: Vec<i32>,
}

impl Decoder {
    pub fn new(init: DecoderInit) -> Self {
        let max_batch_seqs = init.max_batch_seqs as usize;
        let prefix_k = init.shared_prefix_tokens as usize;
        // The radix cache (P7) reserves up to `MAX_PREFIX_SEQS` extra sequence ids beyond
        // the request pool to hold materialized prefixes' KV cells (ids `max_batch_seqs ..
        // max_batch_seqs + MAX_PREFIX_SEQS`). None are reserved when sharing is off.
        let prefix_extra = if prefix_k > 0 { MAX_PREFIX_SEQS } else { 0 };
        let cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(init.n_ctx))
            .with_n_batch(init.n_batch)
            // Physical micro-batch: how many tokens one forward pass computes. A larger
            // value prefills long prompts in fewer passes (lower TTFT) and keeps the batch
            // shape stable for CUDA-graph capture. Resolved/clamped to n_batch in load().
            .with_n_ubatch(init.n_ubatch)
            // Allow up to `max_batch_seqs` concurrent request sequences plus the reserved
            // prefix seqs in the KV cache. (Confirm the method name against llama-cpp-2 on
            // the target — the crate occasionally renames setters.)
            .with_n_seq_max(init.max_batch_seqs + prefix_extra as u32)
            .with_n_threads(1) // full GPU offload → 1 CPU thread (don't fight the 3 cores)
            // Compact the copy-based KV cache once fragmentation passes this fraction, so a
            // long-running server keeps reclaiming cells retired by continuous batching.
            .with_defrag_thold(init.kv_defrag_thold);
        // Flash attention: `Auto` leaves llama.cpp's own default untouched (byte-parity
        // preserved). `On`/`Off` force the policy — validate byte-parity on the GPU target
        // (`make parity-flash`) before forcing it on; some hybrid/recurrent layers don't
        // support it and it can change the attention reduction order.
        let cparams = match init.flash_attention {
            crate::config::FlashAttn::Auto => cparams,
            crate::config::FlashAttn::On => cparams.with_flash_attention_policy(FLASH_ATTN_ENABLED),
            crate::config::FlashAttn::Off => {
                cparams.with_flash_attention_policy(FLASH_ATTN_DISABLED)
            }
        };
        let ctx = init
            .model
            .0
            .new_context(init.backend.0, cparams)
            .expect("new llama context");
        // Free-id stack: pop yields 0, 1, 2, … for readable seq ids.
        let seq_pool: Vec<i32> = (0..max_batch_seqs as i32).rev().collect();
        // Prefix seq ids sit just past the request ids.
        let prefix_seq_pool: Vec<i32> =
            (max_batch_seqs as i32..(max_batch_seqs + prefix_extra) as i32).rev().collect();
        Decoder {
            model: init.model,
            ctx,
            // Second arg is `n_seq_max` = max sequence ids a *single token* may belong
            // to, not the batch's sequence count. Every `add` here passes `&[seq_id]`
            // (one seq per token), and shared-prefix reuse copies cells via
            // `copy_kv_cache_seq` rather than multi-seq tokens — so this stays 1. Token
            // capacity (holding up to `max_batch_seqs` tokens per decode step) is the
            // first arg, `n_batch`.
            batch: LlamaBatch::new(init.n_batch as usize, 1),
            n_batch: init.n_batch as usize,
            n_ctx: init.n_ctx as usize,
            active: Vec::with_capacity(max_batch_seqs),
            decode_idx: Vec::with_capacity(max_batch_seqs),
            retire_mask: Vec::with_capacity(max_batch_seqs),
            seq_pool,
            max_batch_seqs,
            reserved: 0,
            max_batch_tokens: init.max_batch_tokens,
            trie: PrefixTrie::new(MAX_PREFIX_SEQS, prefix_k),
            prefix_seq_pool,
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

    /// Wipe the entire KV cache and reset the decoder to its baseline (empty running
    /// set, full seq-id pool, zero reservation, no resident prefix). The single place
    /// that resets `prefix = None` on a full clear, so no recovery path can forget it.
    fn wipe_all_kv(&mut self) {
        self.ctx.clear_kv_cache();
        self.seq_pool.clear();
        self.seq_pool.extend((0..self.max_batch_seqs as i32).rev());
        self.active.clear();
        self.reserved = 0;
        // Reset the radix cache and reclaim all prefix seq ids.
        let prefix_extra = if self.trie.enabled() { MAX_PREFIX_SEQS } else { 0 };
        self.trie.clear();
        self.prefix_seq_pool.clear();
        self.prefix_seq_pool
            .extend((self.max_batch_seqs as i32..(self.max_batch_seqs + prefix_extra) as i32).rev());
    }

    /// Evict resident prefix cells (idle only) until the request's `footprint` fits `n_ctx`,
    /// freeing each evicted prefix's KV + returning its seq id and subtracting its cells from
    /// `reserved` in lock-step. Generalizes P4's single-prefix eviction to the radix cache.
    fn evict_to_fit(&mut self, footprint: usize) {
        while self.reserved.saturating_add(footprint) > self.n_ctx {
            match self.trie.evict_lru_leaf() {
                Some((seq_id, cells)) => {
                    self.reserved = self.reserved.saturating_sub(cells);
                    if let Err(e) = self.free_seq_kv(seq_id) {
                        // Idle (sole caller), so a full wipe on the "impossible" clear failure
                        // is safe and reclaims everything.
                        eprintln!("core2 evict: KV clear failed for prefix seq {seq_id} ({e}); full wipe");
                        self.wipe_all_kv();
                        return;
                    }
                    self.prefix_seq_pool.push(seq_id);
                }
                None => break, // nothing left to evict
            }
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
    ///
    /// P7 chunked prefill: the seq joins as **Prefilling** (`prefill_pos = plan.reused`)
    /// — this fn reserves KV and copies any shared-prefix cells, but does **not** run the
    /// prompt. [`step`](Self::step) prefills a chunk of `tokens[prefill_pos..]` per
    /// iteration, interleaved with decode, so a long prompt never blocks the running set.
    /// The readiness gate defers a request matching a prefix whose establisher is still
    /// Prefilling (its `0..K` cells aren't donated yet) so it can't copy an empty prefix.
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
        // Reject a request that can't fit `n_ctx` even alone BEFORE any trie work: the trie
        // walk + structural insert (a multi-MB edge Vec for a 40K-token prompt) are wasted on
        // the latency-critical Core 2 for a request `admission_plan` would reject anyway, and
        // an uncapped prompt would accrete a huge edge into the arena. Same bound as the plan.
        if n_prompt.saturating_add(max_tokens as usize) > self.n_ctx {
            eprintln!("core2 admit rejected slot {slot}: needs {} > n_ctx {}", n_prompt + max_tokens as usize, self.n_ctx);
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
            return Admit::Rejected;
        }
        // Record this request's path in the radix trie so future requests can fork against
        // it (discovers variable-length shared prefixes). Structural nodes only, no KV.
        // SAFETY: slot owned by Core 2 here (read-only peek).
        {
            let tokens = unsafe { slab.slot_tokens(slot) };
            self.trie.insert_structural(tokens, MAX_TRIE_NODES);
        }
        // The full reuse/establish decision (incl. the readiness gate) is the shared
        // `PrefixTrie::plan_reuse` oracle — one trie walk, identical to the mock.
        let reuse = {
            let tokens = unsafe { slab.slot_tokens(slot) };
            self.trie.plan_reuse(tokens, n_prompt)
        };
        // Readiness gate: matching a prefix whose establisher is still Prefilling must wait —
        // its cells aren't donated yet, so a `copy_kv_cache_seq` would read empty KV. The
        // establisher always makes prefill progress (per-step budget >= 1) → bounded defer.
        if reuse.defer_unready {
            return Admit::Deferred;
        }
        let plan = admission_plan(
            reuse.reused,
            reuse.establish_new_cells,
            self.reserved,
            self.n_ctx,
            self.active.is_empty(),
            n_prompt,
            max_tokens,
        );
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
            self.evict_to_fit(n_prompt.saturating_add(max_tokens as usize));
        }

        // `has_capacity()` guarantees a free id, but don't rely on the caller: if the
        // pool is somehow empty, defer (retry after a retire) rather than panic.
        let Some(seq_id) = self.seq_pool.pop() else {
            return Admit::Deferred;
        };
        // Sharing a resident prefix: copy the matched node's `0..reused` cells into this seq
        // now (cheap — no forward pass), so `step` prefills only the suffix at `pos = reused`.
        // `plan.reused > 0` only on a real share (0 for establish / fallback / post-evict).
        // Confirm the exact llama-cpp-2 spelling on target (see module note).
        if plan.reused > 0 {
            // `plan.reused > 0` ⇒ `plan_reuse` found a ready materialized `match_node`, so its
            // prefix seq id must exist. Treat a missing source as an error (don't silently skip
            // the copy → prefill over empty cells / corrupt output).
            let src = reuse.match_node.and_then(|n| self.trie.seq_id(n));
            let Some(src_seq) = src else {
                eprintln!("core2 admit: reused {} but no source prefix seq for slot {slot}", plan.reused);
                self.retire_seq(seq_id); // never reserved
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
                return Admit::Rejected;
            };
            if let Err(e) = self.ctx.copy_kv_cache_seq(src_seq, seq_id, Some(0), Some(plan.reused as u32)) {
                eprintln!("core2 admit: prefix KV copy failed for slot {slot}: {e}");
                self.retire_seq(seq_id); // never reserved
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
                return Admit::Rejected;
            }
            self.trie.touch(reuse.match_node.expect("reused > 0 ⇒ match_node set"));
        }
        // Caching a newly-discovered fork: materialize it now (assign a prefix seq id);
        // `step` donates this request's cells + marks it ready when its prefill completes.
        // `plan_reuse` already checked `can_materialize()`, and the prefix seq pool stays in
        // lockstep with the trie's materialized count (pop on establish, push on evict/demote/
        // wipe), so a free id + successful materialize is invariant here — like the mock's
        // `debug_assert!(ok)`. If that invariant ever broke, the plan's `reserve = inc -
        // new_cells` would leak the fork cells; assert rather than silently drift.
        let establishes = if plan.establishing {
            let f = reuse.fork.expect("establishing ⇒ a fork was found");
            let pseq = self.prefix_seq_pool.pop().expect("free prefix seq id (pool ↔ trie invariant)");
            let ok = self.trie.materialize(f, pseq);
            debug_assert!(ok, "can_materialize checked in plan_reuse before establish");
            Some(f)
        } else {
            None
        };
        self.reserved += plan.inc;
        self.active.push(ActiveSeq {
            slot,
            seq_id,
            pos: plan.reused as i32,
            n_prompt,
            prefill_pos: plan.reused,
            establishes,
            n_generated: 0,
            max_tokens,
            next_token: 0, // unused until Decoding
            reserve: plan.reserve,
        });
        Admit::Admitted(plan.effective)
    }

    /// Prefill up to `budget` prompt tokens for the Prefilling seq at `active[idx]`,
    /// appending them to `batch` at successive positions (`logits` only on the token that
    /// completes the prompt). Returns `(tokens_added, Some(batch_logit_idx))` when this
    /// chunk completes the prompt (the caller samples the first token from that index),
    /// else `(tokens_added, None)`. Reads prompt tokens straight from the slab (no copy).
    /// SAFETY: slot arrived via R2 → Core 2 owns it here (no R3 publish in this fn).
    fn prefill_chunk(
        &mut self,
        idx: usize,
        slab: &Slab,
        batch_base: i32,
        budget: usize,
    ) -> Result<(usize, Option<i32>), BackendError> {
        let (slot, seq_id, n_prompt, prefill_pos) = {
            let a = &self.active[idx];
            (a.slot, a.seq_id, a.n_prompt, a.prefill_pos)
        };
        let s = unsafe { slab.slot_mut(slot) };
        let chunk_end = (prefill_pos + budget).min(n_prompt);
        let mut logit_idx = None;
        for (offset, pos) in (prefill_pos..chunk_end).enumerate() {
            let is_last = pos == n_prompt - 1;
            self.batch
                .add(LlamaToken(s.tokens[pos]), pos as i32, &[seq_id], is_last)
                .map_err(|e| BackendError::Decode(e.to_string()))?;
            if is_last {
                logit_idx = Some(batch_base + offset as i32);
            }
        }
        let added = chunk_end - prefill_pos;
        // Advance the seq's prefill cursor + KV position by what we added.
        {
            let a = &mut self.active[idx];
            a.prefill_pos = chunk_end;
            a.pos = chunk_end as i32;
        }
        Ok((added, logit_idx))
    }

    /// One **unified batch** step (P7 chunked prefill): build a single `llama_batch` that
    /// mixes one decode token per **Decoding** seq (`prefill_pos == n_prompt`, logits on)
    /// with up to a per-step prefill budget of prompt tokens drawn from **Prefilling**
    /// seqs (logits only on a token that completes a prompt), `decode()` once (the GPU
    /// forward for every sequence at once — decode and prefill share the pass, so a long
    /// prompt never stalls the running set), then: emit + sample-next + retire the
    /// Decoding seqs, and transition each completed Prefilling seq to Decoding (sampling
    /// its first token, donating prefix cells if it was the establisher).
    pub fn step<P: Producer<RingEvent>>(&mut self, slab: &Slab, r3: &mut P) {
        self.batch.clear();
        let n_active = self.active.len();
        let mut build_err: Option<String> = None;
        // Pass 1a — decode tokens: one per Decoding seq. `batch_idx` tracks the token's
        // position in the batch, which is the index `get_logits_ith` reads back.
        let mut batch_idx = 0i32;
        self.decode_idx.clear(); // reused scratch — no per-step alloc; holds (active_idx, batch_idx)
        for i in 0..n_active {
            if self.active[i].prefill_pos == self.active[i].n_prompt {
                let (tok, pos, seq_id) = {
                    let a = &self.active[i];
                    (a.next_token, a.pos, a.seq_id)
                };
                if let Err(e) = self.batch.add(LlamaToken(tok), pos, &[seq_id], true) {
                    build_err = Some(e.to_string());
                    break;
                }
                self.decode_idx.push((i, batch_idx));
                batch_idx += 1;
            }
        }
        // Move the scratch into a local for the rest of `step` (avoids partial-borrow
        // friction with `self.active`/`self.ctx`); restored at the end so its capacity is
        // reused next iteration — no steady-state allocation.
        let decode_idx = std::mem::take(&mut self.decode_idx);
        let n_decode = decode_idx.len();
        // Pass 1b — prefill chunks: fill the rest of the batch from Prefilling seqs, up to
        // `step_prefill_budget` (>= 1 whenever any Prefilling seq exists, so a lone long
        // prompt always advances). `n_decode + budget <= n_batch`, so the batch never
        // overflows its token capacity.
        let mut prefill_complete: Vec<(usize, i32)> = Vec::new(); // (active_idx, logit batch idx)
        if build_err.is_none() {
            let mut budget = step_prefill_budget(n_decode, self.n_batch, self.max_batch_tokens);
            for i in 0..n_active {
                if budget == 0 {
                    break;
                }
                if self.active[i].prefill_pos < self.active[i].n_prompt {
                    match self.prefill_chunk(i, slab, batch_idx, budget) {
                        Ok((added, done_idx)) => {
                            batch_idx += added as i32;
                            budget -= added;
                            if let Some(li) = done_idx {
                                prefill_complete.push((i, li));
                            }
                        }
                        Err(e) => {
                            build_err = Some(e.to_string());
                            break;
                        }
                    }
                }
            }
        }
        if let Some(e) = build_err {
            return self.fail_all(r3, &e);
        }
        if batch_idx == 0 {
            return; // nothing to decode (defensive; Core 2 only steps a non-idle decoder)
        }
        if let Err(e) = self.ctx.decode(&mut self.batch) {
            return self.fail_all(r3, &e.to_string());
        }

        // Emit the fed token for every Decoding seq in ONE R3 burst (the design's
        // batch-publish throughput path). `n_decode <= max_batch_seqs <= ring_size`, so it
        // always fits. Prefilling seqs emit nothing this step.
        if n_decode > 0 {
            r3.batch_publish(n_decode, |iter| {
                for (e, &(ai, _)) in iter.zip(decode_idx.iter()) {
                    let seq = &self.active[ai];
                    *e = RingEvent { slot: seq.slot, kind: EventKind::Token(seq.next_token as u32) };
                }
            });
        }

        // `retire[ai] == true` marks a seq to drop after both passes; a single compaction
        // below frees its KV + returns its id (avoids swap_remove index churn mid-pass).
        // Reused scratch (restored at the end) — no per-step allocation.
        let mut retire = std::mem::take(&mut self.retire_mask);
        retire.clear();
        retire.resize(n_active, false);

        // Decoding seqs: advance, sample the next token (from the decode token's batch
        // index), retire on max_tokens/EOS.
        for &(ai, bidx) in &decode_idx {
            self.active[ai].n_generated += 1;
            self.active[ai].pos += 1;
            let finished = if self.active[ai].n_generated >= self.active[ai].max_tokens {
                Some(FinishReason::MaxTokens)
            } else {
                let next = argmax(self.ctx.get_logits_ith(bidx));
                if self.model.0.is_eog_token(LlamaToken(next)) {
                    Some(FinishReason::Eos)
                } else {
                    self.active[ai].next_token = next;
                    None
                }
            };
            if let Some(reason) = finished {
                let slot = self.active[ai].slot;
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(reason) });
                { debug_assert!(self.reserved >= self.active[ai].reserve, "reserved underflow"); self.reserved = self.reserved.saturating_sub(self.active[ai].reserve); }
                retire[ai] = true;
            }
        }

        // Completed Prefilling seqs: sample the first token from the prompt-final logit.
        // A non-EOG first token → the seq becomes Decoding (emits next step, mirroring the
        // admit-samples-first-token order); establishers donate their fork's cells to its
        // prefix seq now and mark it ready, so deferred sharers can proceed.
        for &(ai, li) in &prefill_complete {
            let first = argmax(self.ctx.get_logits_ith(li));
            let (slot, seq_id, establishes) = {
                let a = &self.active[ai];
                (a.slot, a.seq_id, a.establishes)
            };
            if self.model.0.is_eog_token(LlamaToken(first)) {
                // Immediate EOS at prefill completion: the seq won't decode. An establisher
                // never donated its cells → drop its (unready) fork so a deferred sharer
                // re-establishes rather than copy empty KV; release its full footprint.
                if let Some(node) = establishes {
                    if let Some((pseq, cells)) = self.trie.demote_node(node) {
                        self.prefix_seq_pool.push(pseq);
                        { debug_assert!(self.reserved >= cells, "reserved underflow"); self.reserved = self.reserved.saturating_sub(cells); } // the fork cells never became permanent
                    }
                }
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Eos) });
                { debug_assert!(self.reserved >= self.active[ai].reserve, "reserved underflow"); self.reserved = self.reserved.saturating_sub(self.active[ai].reserve); }
                retire[ai] = true;
            } else {
                self.active[ai].next_token = first; // Decoding now (prefill_pos == n_prompt)
                if let Some(node) = establishes {
                    // Donate this seq's `0..depth` cells to the fork's prefix seq once, so
                    // future matching requests copy them (a `seq_rm` on this seq later leaves
                    // the fork resident — shared cells). Confirm the crate's exact spelling on
                    // target (see module note).
                    let depth = self.trie.depth(node) as u32;
                    let dst = self.trie.seq_id(node).expect("materialized fork has a seq id");
                    match self.ctx.copy_kv_cache_seq(seq_id, dst, Some(0), Some(depth)) {
                        Ok(()) => self.trie.set_ready(node),
                        Err(e) => {
                            // Donation failed: drop the unready fork (this seq keeps its own
                            // cells + decodes normally; a later request re-establishes), and
                            // release the fork's cells that admit had kept reserved.
                            eprintln!("core2 step: prefix donation failed for seq {seq_id}: {e}");
                            if let Some((pseq, cells)) = self.trie.demote_node(node) {
                                self.prefix_seq_pool.push(pseq);
                                { debug_assert!(self.reserved >= cells, "reserved underflow"); self.reserved = self.reserved.saturating_sub(cells); }
                            }
                        }
                    }
                }
            }
        }

        // Compact: drop retired seqs, freeing KV + returning ids. Same write-index sweep
        // as P3; a KV-clear failure poisons the cache → fail survivors + wipe (below).
        let mut kv_clear_failed = false;
        let mut w = 0;
        // Indexed sweep: `r` reads `retire[r]` AND `self.active` (swap-compact), so an
        // iterator over one can't drive the other.
        #[allow(clippy::needless_range_loop)]
        for r in 0..n_active {
            if retire[r] {
                let seq_id = self.active[r].seq_id;
                match self.free_seq_kv(seq_id) {
                    Ok(()) => self.seq_pool.push(seq_id),
                    Err(e) => {
                        eprintln!("core2 step: KV clear failed for seq {seq_id}: {e}");
                        kv_clear_failed = true;
                    }
                }
            } else {
                if w != r {
                    self.active.swap(w, r);
                }
                w += 1;
            }
        }
        self.active.truncate(w);
        // Return the scratch buffers (with their grown capacity) for the next step.
        self.decode_idx = decode_idx;
        self.retire_mask = retire;
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
        // `clear_kv_cache_seq` returns `Ok(false)` only for a failed *partial* removal; a
        // full-range `(None, None)` seq_rm always succeeds, so the bool is discarded.
        self.ctx
            .clear_kv_cache_seq(Some(seq_id as u32), None, None)
            .map(|_| ())
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
