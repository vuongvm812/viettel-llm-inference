//! PyO3 bindings for the Rust KV-cache / scheduler core.
//!
//! Loaded inside the vLLM EngineCore process by `vtl/patches/rust_sched.py`. Nothing here
//! is authoritative unless an env gate says so; see that module for the flags.
//!
//! Boundary rules:
//!   * Only primitives, `bytes` and lists cross the boundary — never a vLLM object.
//!   * Block-ID output goes through persistent numpy buffers owned by Rust
//!     ([`KvManager::buffer`]); Python slices a view (`buf[:n]`) instead of building a
//!     list. `block_ids_lists()` exists only where vLLM's own contract demands
//!     `tuple[list[int], ...]` (`KVCacheBlocks.get_block_ids`).
//!
//! THREADING. The mutable core (`Manager` + `ScheduleCore` + the speculation state) lives
//! behind one `Arc<Mutex<Shared>>` so the `vtl-sched-spec` worker can run a whole
//! scheduling step without the GIL (`spec.rs`). Every method here is lock -> op:
//!
//!   * [`KvManager::w`] — the WRITE guard. Rolls back any in-flight speculation FIRST
//!     (invariant 2 in `spec.rs`), then hands over the state. Used by every method that
//!     mutates OR that reads state a speculative run would have already changed.
//!   * [`KvManager::r`] — the READ guard. Only for state speculation cannot touch
//!     (interning, block hashes, stop params) plus the two table probes, which are
//!     documented to read through a pending speculation on purpose.
//!
//! The lock is uncontended in steady state: the engine core is single-threaded and the
//! worker only holds it between a `kick` and the following `take_speculative`.
//! The numpy buffers stay OUTSIDE the mutex — they are GIL-protected, as before.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard};

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyTuple};

use crate::config::{Config, GroupConfig};
use crate::hash;
use crate::hash::{Digest32, Key, HASH_LEN};
use crate::manager::Manager;
use crate::sched::{Decisions, Params, SchedReq};
use crate::single_type::{cdiv, Kind};
use crate::spec::{lock_shared, Shared, SpecDriver};
use crate::update::{gather_sampled, StopParams};


fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

/// The locked half of a step commit: stop decisions, the resident-table delta, the R8 record
/// and (optionally) the inline shm publish.
///
/// Extracted verbatim from `update_step_pack_np`'s `allow_threads` body so the Rust model
/// runner drives THE SAME code rather than a parallel implementation, so one set of tests
/// covers both paths. Callers must not hold the GIL.
#[allow(clippy::too_many_arguments)]
fn step_pack_locked(
    shared: &Arc<Mutex<Shared>>,
    slots: &[u32],
    counts: &[u32],
    flat: &[i64],
    max_model_len: usize,
    engine_index: u32,
    timestamp: f64,
    finished: &[String],
    fold_cache: bool,
    publish: bool,
) -> Result<(Vec<(u32, u8, i64)>, Option<Vec<u8>>, bool), String> {
    let mut sh = lock_shared(shared);
    sh.invalidate();
    let (verdicts, packed) = sh.manager.update_step_pack_store(
        slots,
        counts,
        flat,
        max_model_len,
        engine_index,
        timestamp,
        finished,
        fold_cache,
    )?;
    let out: Vec<(u32, u8, i64)> = verdicts
        .iter()
        .map(|v| (v.num_accepted, v.status, v.stop_reason))
        .collect();
    if packed && publish {
        #[cfg(feature = "shm")]
        let delivered = match &sh.out {
            Some(ch) => ch.publish_record(sh.manager.raw.finish()?),
            None => false,
        };
        #[cfg(not(feature = "shm"))]
        let delivered = false;
        if delivered {
            return Ok((out, None, true));
        }
    }
    let rec = if packed {
        Some(sh.manager.raw.finish()?.to_vec())
    } else {
        None
    };
    Ok((out, rec, false))
}

fn get_usize(d: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing config key {key}")))?
        .extract()
}

fn get_bool(d: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing config key {key}")))?
        .extract()
}

fn write_buf(py: Python<'_>, buf: &Py<PyArray1<i64>>, cap: usize, src: &[u32]) -> PyResult<usize> {
    if src.len() > cap {
        return Err(PyRuntimeError::new_err(format!(
            "block-id arena overflow: {} > {cap} (max_model_len / block_size mismatch)",
            src.len()
        )));
    }
    let bound = buf.bind(py);
    // SAFETY: the buffer is owned by this KvManager and only ever handed to Python as a
    // read-only `[:n]` view; the GIL is held for the duration of the write.
    let dst = unsafe { bound.as_slice_mut()? };
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s as i64;
    }
    Ok(src.len())
}

#[pyclass(module = "vtl_sched")]
pub struct KvManager {
    /// The mutable core, shared with the `vtl-sched-spec` worker (`spec.rs`).
    pub(crate) shared: Arc<Mutex<Shared>>,
    /// Spawned on the first `kick`; `None` means speculation was never asked for.
    driver: Option<SpecDriver>,
    /// One persistent numpy buffer per KV cache group; block IDs are written here and
    /// Python takes a zero-copy `[:n]` view. Outside the mutex — GIL-protected.
    bufs: Vec<Py<PyArray1<i64>>>,
    buf_cap: usize,
    flat: Vec<u32>,
    n_groups: usize,
}

impl KvManager {
    /// READ guard. Only for state a speculative run cannot have changed — plus the two
    /// table probes, which deliberately read through a pending speculation.
    #[inline]
    fn r(&self) -> MutexGuard<'_, Shared> {
        lock_shared(&self.shared)
    }

    /// WRITE guard: rolls back any in-flight speculation before handing over the state.
    #[inline]
    fn w(&self) -> MutexGuard<'_, Shared> {
        let mut g = lock_shared(&self.shared);
        g.invalidate();
        g
    }
}

#[pymethods]
impl KvManager {
    /// `config` mirrors `KVCacheManager.__init__` + `KVCacheConfig`, flattened to plain
    /// data. See `rust_sched.py::build_config` for the producer.
    #[new]
    fn new(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let groups_obj = config
            .get_item("groups")?
            .ok_or_else(|| PyValueError::new_err("missing config key groups"))?;
        let groups_list: Vec<Bound<'_, PyDict>> = groups_obj.extract()?;
        let mut groups = Vec::with_capacity(groups_list.len());
        for g in &groups_list {
            let kind_s: String = g
                .get_item("kind")?
                .ok_or_else(|| PyValueError::new_err("group missing kind"))?
                .extract()?;
            let kind = match kind_s.as_str() {
                "full" => Kind::FullAttention,
                "mamba" => Kind::Mamba,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported kv cache spec kind {other:?}; this port covers full \
                         attention and mamba only"
                    )))
                }
            };
            groups.push(GroupConfig {
                kind,
                block_size: get_usize(g, "block_size")?,
                is_full_attention: get_bool(g, "is_full_attention")?,
                spec_signature: g
                    .get_item("spec_signature")?
                    .ok_or_else(|| PyValueError::new_err("group missing spec_signature"))?
                    .extract()?,
                mamba_align: get_bool(g, "mamba_align")?,
                num_speculative_blocks: get_usize(g, "num_speculative_blocks")?,
                use_eagle: get_bool(g, "use_eagle")?,
            });
        }
        let cfg = Config {
            num_blocks: get_usize(config, "num_blocks")?,
            enable_caching: get_bool(config, "enable_caching")?,
            max_model_len: get_usize(config, "max_model_len")?,
            scheduler_block_size: get_usize(config, "scheduler_block_size")?,
            hash_block_size: get_usize(config, "hash_block_size")?,
            log_stats: get_bool(config, "log_stats")?,
            watermark: config
                .get_item("watermark")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.0),
            radix: config
                .get_item("radix")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(false),
            groups,
        };
        cfg.validate().map_err(PyValueError::new_err)?;

        let min_block = cfg.groups.iter().map(|g| g.block_size).min().unwrap();
        let max_spec = cfg
            .groups
            .iter()
            .map(|g| g.num_speculative_blocks)
            .max()
            .unwrap_or(0);
        let buf_cap = cdiv(cfg.max_model_len, min_block) + max_spec + 2;
        let n_groups = cfg.groups.len();
        let inner = Manager::new(cfg).map_err(err)?;
        let bufs = (0..n_groups)
            .map(|_| PyArray1::<i64>::zeros_bound(py, buf_cap, false).unbind())
            .collect();
        Ok(KvManager {
            shared: Arc::new(Mutex::new(Shared::new(inner))),
            driver: None,
            bufs,
            buf_cap,
            flat: Vec::with_capacity(256),
            n_groups,
        })
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.n_groups
    }

    /// The persistent per-group block-ID buffer. Python holds these once and slices
    /// `buf[:n]` — no per-call allocation of the data.
    fn buffer(&self, py: Python<'_>, group: usize) -> PyObject {
        self.bufs[group].clone_ref(py).into_any()
    }

    // ---- request registry -------------------------------------------------

    fn intern(&self, req_id: &str) -> u32 {
        self.w().manager.intern(req_id)
    }

    fn lookup(&self, req_id: &str) -> Option<u32> {
        self.r().manager.lookup(req_id)
    }

    fn forget(&self, req_id: &str) {
        self.w().manager.forget(req_id)
    }

    /// Append the request's new block hashes (32 bytes each, concatenated).
    fn push_hashes(
        &self,
        slot: u32,
        packed: &Bound<'_, PyBytes>,
        num_prompt_tokens: usize,
    ) -> PyResult<()> {
        let b = packed.as_bytes();
        if b.len() % HASH_LEN != 0 {
            return Err(PyValueError::new_err(format!(
                "packed hashes must be a multiple of {HASH_LEN} bytes, got {}",
                b.len()
            )));
        }
        self.w().manager.push_hashes(slot, b, num_prompt_tokens);
        Ok(())
    }

    fn num_hashes(&self, slot: u32) -> usize {
        self.r().manager.num_hashes(slot)
    }

    // ---- R6a: batched stop decision (update_from_output) -------------------

    /// Intern a slot's immutable stop condition. Pushed once, when `rust_sched.py` first
    /// sees the request; `forget()` drops it along with the id.
    #[pyo3(signature = (slot, min_tokens, max_tokens, eos_token_id, stop_token_ids))]
    fn set_stop_params(
        &self,
        slot: u32,
        min_tokens: usize,
        max_tokens: usize,
        eos_token_id: Option<i64>,
        stop_token_ids: Vec<i64>,
    ) {
        self.w().manager.stops.set(
            slot,
            StopParams {
                min_tokens,
                max_tokens,
                eos_token_id,
                stop_token_ids: stop_token_ids.into_iter().collect(),
            },
        );
    }

    fn has_stop_params(&self, slot: u32) -> bool {
        self.r().manager.stops.has(slot)
    }

    /// One step's stop decisions, plus the resident table's update-time delta (R6b).
    ///
    /// Flat in, flat out: request `i` owns `token_ids[cu_lens[i]..cu_lens[i + 1]]`, and
    /// gets back `(num_accepted, status, stop_reason)`. `status == 255` means the slot has
    /// no interned params and Python must run `check_stop` itself for that request — the
    /// table is left untouched for those slots, so Python owns the fix-up via `table_set`.
    ///
    /// Runs with the GIL RELEASED: this is pure integer work over data already copied in.
    #[allow(clippy::too_many_arguments)]
    fn update_step(
        &self,
        py: Python<'_>,
        slots: Vec<u32>,
        cu_lens: Vec<u32>,
        token_ids: Vec<i64>,
        num_output_tokens: Vec<u32>,
        num_tokens: Vec<u32>,
        max_model_len: usize,
    ) -> PyResult<Vec<(u32, u8, i64)>> {
        let shared = self.shared.clone();
        py.allow_threads(move || {
            let mut sh = lock_shared(&shared);
            sh.invalidate();
            sh.manager
                .update_step(
                    &slots,
                    &cu_lens,
                    &token_ids,
                    &num_output_tokens,
                    &num_tokens,
                    max_model_len,
                )
                .map(|out| {
                    out.iter()
                        .map(|v| (v.num_accepted, v.status, v.stop_reason))
                        .collect()
                })
        })
        .map_err(err)
    }

    /// R8: [`KvManager::update_step`] that ALSO returns the finished shm output record.
    ///
    /// `(verdicts, record_or_none)`. `None` means "this step is not raw-packable" -- a
    /// refused slot, an un-interned name, a token id outside u32 -- and Python then builds
    /// `EngineCoreOutput` objects for the whole batch, exactly as it did before R8. The
    /// verdicts are applied either way, so the resident table stays correct on both arms.
    ///
    /// `finished` must arrive SORTED; the Python packer sorts a set and the bytes have to
    /// match. `engine_index` / `timestamp` are stamped by the output thread in stock vLLM,
    /// so the caller hands them in rather than leaving two holes to patch over a loan.
    ///
    /// GIL released for the pack, like `update_step` -- it is byte pushing over data
    /// already copied in. The record is copied once more on the way out (`PyBytes` cannot
    /// be built without the GIL); that is ~50 bytes for a decode step.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (slots, cu_lens, token_ids, num_output_tokens, num_tokens,
                        max_model_len, engine_index, timestamp, finished))]
    fn update_step_pack(
        &self,
        py: Python<'_>,
        slots: Vec<u32>,
        cu_lens: Vec<u32>,
        token_ids: Vec<i64>,
        num_output_tokens: Vec<u32>,
        num_tokens: Vec<u32>,
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: Vec<String>,
    ) -> PyResult<(Vec<(u32, u8, i64)>, Option<Py<PyBytes>>)> {
        let shared = self.shared.clone();
        let (verdicts, record) = py
            .allow_threads(move || {
                let mut sh = lock_shared(&shared);
                sh.invalidate();
                let (verdicts, packed) = sh.manager.update_step_pack(
                    &slots,
                    &cu_lens,
                    &token_ids,
                    &num_output_tokens,
                    &num_tokens,
                    max_model_len,
                    engine_index,
                    timestamp,
                    &finished,
                )?;
                let out: Vec<(u32, u8, i64)> = verdicts
                    .iter()
                    .map(|v| (v.num_accepted, v.status, v.stop_reason))
                    .collect();
                let rec = if packed {
                    Some(sh.manager.raw.finish()?.to_vec())
                } else {
                    None
                };
                Ok::<_, String>((out, rec))
            })
            .map_err(err)?;
        Ok((
            verdicts,
            record.map(|r| PyBytes::new_bound(py, &r).unbind()),
        ))
    }

    // ---- Port-2: the Rust-owned token store -------------------------------

    /// Hand over vLLM's live `NONE_HASH`. Until this runs, `store_init` refuses -- hashing
    /// against a guessed seed would fork the prefix-cache key space silently.
    fn store_arm(&self, none_hash: &Bound<'_, PyBytes>) -> PyResult<()> {
        let b = none_hash.as_bytes();
        if b.len() != HASH_LEN {
            return Err(PyValueError::new_err(format!(
                "none_hash must be {HASH_LEN} bytes, got {}",
                b.len()
            )));
        }
        let mut d: Digest32 = [0; HASH_LEN];
        d.copy_from_slice(b);
        self.w().manager.store_arm(d);
        Ok(())
    }

    /// Take over one slot's token bookkeeping. `pending` is the request's token tail past
    /// the last block hash Python already produced (so shorter than `hash_block_size`).
    fn store_init(
        &self,
        slot: u32,
        pending: Vec<i64>,
        num_tokens: usize,
        num_output_tokens: usize,
    ) -> PyResult<()> {
        self.w()
            .manager
            .store_init(slot, &pending, num_tokens, num_output_tokens)
            .map_err(err)
    }

    /// The slot's output tokens, as native Python ints (PyO3 converts `i64` -> `int`, so
    /// nothing numpy-shaped can reach a block-hash input or a `Request` token list).
    fn slot_tokens(&self, slot: u32) -> Option<Vec<i64>> {
        self.r().manager.slot_tokens(slot).map(|t| t.to_vec())
    }

    /// The slot's whole block-hash chain, 32 bytes per hash, for rebuilding
    /// `Request.block_hashes` when a request is handed back to the stock path.
    fn slot_hashes(&self, py: Python<'_>, slot: u32) -> Option<Py<PyBytes>> {
        self.r()
            .manager
            .slot_hashes(slot)
            .map(|v| PyBytes::new_bound(py, &v).unbind())
    }

    /// Release a slot from the store (materialization / permanent per-request fallback).
    fn store_forget(&self, slot: u32) {
        self.w().manager.tokens.forget(slot);
    }

    /// Port-2: [`KvManager::update_step_pack`] driven straight off the sampler's numpy
    /// output, with the per-request counters read from the token store.
    ///
    /// `sampled` is `AsyncOutput`'s `[max_num_reqs, num_sampled]` host array; `rows[i]` is
    /// the row entry `i` owns and `counts[i]` how many of that row's columns are valid, so
    /// the two `tolist()` calls in `AsyncOutput.get_output` and the per-request
    /// `toks.extend` loop in `decide()` both disappear -- the ids are copied once, here,
    /// straight out of the array's buffer.
    ///
    /// Same return contract as `update_step_pack`: `None` for the record means "not
    /// packable", and on that arm NOTHING is appended to the store, because Python is about
    /// to materialize the whole step back onto the stock path.
    ///
    /// `fold_cache` (R9, default `false`): when the step packs, also run each slot's
    /// `cache_blocks` INSIDE this call instead of `rust_sched.py` doing it one FFI
    /// crossing per request afterwards. See [`Manager::fold_cache_blocks`]; `false`
    /// reproduces every existing caller byte-for-byte (no store append either way is
    /// unaffected, since that is gated on `packed`, not on this flag).
    ///
    /// `publish` (Batch 3, default `false`): when the step packs AND the crate's output
    /// channel is open ([`KvManager::out_open`]), publish the record straight from the
    /// packer's buffer into the loaned shm sample INSIDE this same locked call -- no
    /// `Vec<u8>` copy, no `PyBytes`, no output-queue hop to the Python shm thread. On a
    /// delivered publish the returned record is `None` (nothing left to queue) and
    /// `published` is `true`; on anything else (not packed, no channel open, or the
    /// publish itself reported 0 receivers) behaviour is EXACTLY as `publish=false`:
    /// the caller gets the bytes back and must route them the way it always did.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (sampled, rows, counts, slots, max_model_len, engine_index,
                        timestamp, finished, fold_cache=false, publish=false))]
    fn update_step_pack_np(
        &self,
        py: Python<'_>,
        sampled: PyReadonlyArray2<'_, i64>,
        rows: Vec<u32>,
        counts: Vec<u32>,
        slots: Vec<u32>,
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: Vec<String>,
        fold_cache: bool,
        publish: bool,
    ) -> PyResult<(Vec<(u32, u8, i64)>, Option<Py<PyBytes>>, bool)> {
        let n = slots.len();
        if rows.len() != n || counts.len() != n {
            return Err(PyValueError::new_err(format!(
                "update_step_pack_np arity mismatch: {n} slots, {} rows, {} counts",
                rows.len(),
                counts.len()
            )));
        }
        let shape = sampled.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        // Contiguity is a property of the caller's array, not something to assume: a
        // non-contiguous one is refused so Python falls back rather than reading garbage.
        let arr = sampled.as_slice().map_err(|_| {
            PyValueError::new_err("update_step_pack_np needs a C-contiguous sampled array")
        })?;
        // Gather while the GIL is still held: `arr` borrows a Python-owned buffer.
        let flat = gather_sampled(arr, nrows, ncols, &rows, &counts).map_err(err)?;
        let shared = self.shared.clone();
        let (verdicts, record, published) = py
            .allow_threads(move || {
                step_pack_locked(
                    &shared,
                    &slots,
                    &counts,
                    &flat,
                    max_model_len,
                    engine_index,
                    timestamp,
                    &finished,
                    fold_cache,
                    publish,
                )
            })
            .map_err(err)?;
        Ok((
            verdicts,
            record.map(|r| PyBytes::new_bound(py, &r).unbind()),
            published,
        ))
    }

    // ---- Batch 3: crate-owned iceoryx2 output publisher -------------------

    /// Open `vtl/out/<seed>` for this boot's `input_address`. `false` on any failure
    /// (iceoryx2 refused the pairing, or the wheel was built without the `shm` cargo
    /// feature) -- the caller's fail-open ladder then builds its own publisher exactly as
    /// it did before this port. Idempotent: a channel already open for THIS process is left
    /// alone. `MAX_PUBLISHERS = 1` is baked into the service config on both halves (see
    /// `out.rs`'s module doc), so once this returns `true` Python must not also construct
    /// an `_OutputPublisher` -- ownership is exclusive, not shared.
    #[cfg(feature = "shm")]
    fn out_open(&self, input_address: &str) -> bool {
        let mut sh = self.w();
        if sh.out.is_some() {
            return true;
        }
        match crate::out::OutChannel::open(input_address) {
            Ok(ch) => {
                sh.out = Some(ch);
                true
            }
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "shm"))]
    fn out_open(&self, _input_address: &str) -> bool {
        false
    }

    /// `true` once `out_open` has succeeded. Read by `rust_sched.py`'s inline-publish
    /// ordering guard and by `shm_ipc.py` to decide whether a Python `_OutputPublisher` is
    /// needed at all.
    #[cfg(feature = "shm")]
    fn out_is_open(&self) -> bool {
        self.r().out.is_some()
    }

    #[cfg(not(feature = "shm"))]
    fn out_is_open(&self) -> bool {
        false
    }

    /// Publish `tag + payload` through the crate's channel (mirrors
    /// `_OutputPublisher.publish`): the fallback arms in `shm_ipc.py` (msgpack-encoded
    /// objects, the TAG_DEAD sentinel) call this instead of the iceoryx2 Python bindings,
    /// now that ownership of the channel has moved into the crate. `false` = no channel
    /// open, or undelivered (no subscriber / a full buffer) -- the caller's existing ZMQ
    /// fallback and permanent demotion latch take over exactly as before this port.
    #[cfg(feature = "shm")]
    fn out_publish(&self, tag: u8, payload: &Bound<'_, PyBytes>) -> bool {
        let sh = self.r();
        match &sh.out {
            Some(ch) => ch.publish(tag, payload.as_bytes()),
            None => false,
        }
    }

    #[cfg(not(feature = "shm"))]
    fn out_publish(&self, _tag: u8, _payload: &Bound<'_, PyBytes>) -> bool {
        false
    }

    /// Publish an ALREADY-FRAMED record (mirrors `_OutputPublisher.publish_record`): the
    /// tag byte is whatever the caller's packer wrote at offset 0, copied verbatim.
    #[cfg(feature = "shm")]
    fn out_publish_record(&self, record: &Bound<'_, PyBytes>) -> bool {
        let sh = self.r();
        match &sh.out {
            Some(ch) => ch.publish_record(record.as_bytes()),
            None => false,
        }
    }

    #[cfg(not(feature = "shm"))]
    fn out_publish_record(&self, _record: &Bound<'_, PyBytes>) -> bool {
        false
    }

    // ---- R6b: the resident request table ----------------------------------

    /// Overwrite one slot's entry. Same tuple layout as `pack_req` / `Scheduler.schedule`.
    /// The escape hatch for anything Python mutates out of band.
    fn table_set(&self, slot: u32, req: ReqTuple) {
        self.w().manager.table_set(slot, unpack(&req));
    }

    /// C1a: apply an N-step burst's extra `delta` tokens to `slots` in the resident
    /// table. Negative `delta` is the reconcile (the burst stopped short). One crossing
    /// per burst step, not per request.
    fn table_burst(&self, slots: Vec<u32>, delta: i64) {
        self.w().manager.table_burst(&slots, delta)
    }

    fn table_clear(&self, slot: u32) {
        self.w().manager.table_clear(slot);
    }

    /// FxHash over the entries at `slots`, in order.
    ///
    /// NON-INVALIDATING on purpose, so a probe cannot kill a live speculation. The flip
    /// side: between a `kick` and its `take_speculative` this reads the SPECULATIVE
    /// post-commit values. Call it outside that window.
    fn table_fingerprint(&self, slots: Vec<u32>) -> u64 {
        self.r().manager.table_fingerprint(&slots)
    }

    /// `(hits, misses, rollbacks, disabled)` for the speculation path.
    fn spec_stats(&self) -> (u64, u64, u64, bool) {
        let sh = self.r();
        (
            sh.spec.hits,
            sh.spec.misses,
            sh.spec.rollbacks,
            sh.spec.disabled,
        )
    }

    /// Order-sensitive hash of the entire KV state. O(pool size) — a diagnostic, not a
    /// hot-path call. Exposed because it is what proves a speculative rollback was exact.
    fn state_fingerprint(&self) -> u64 {
        self.w().manager.state_fingerprint()
    }

    // ---- KVCacheManager surface -------------------------------------------

    fn new_step_starts(&self) {
        self.w().manager.new_step_starts()
    }

    /// Returns `num_new_computed_tokens`; the hit blocks stay pending for the following
    /// `allocate_slots(use_pending_hit=True)`.
    fn get_computed_blocks(
        &self,
        slot: u32,
        num_tokens: usize,
        num_preemptions: u32,
        skip_reading_prefix_cache: bool,
    ) -> usize {
        self.w().manager.get_computed_blocks(
            slot,
            num_tokens,
            num_preemptions,
            skip_reading_prefix_cache,
        )
    }

    /// Read-only cache-hit walk that does not touch `prefix_cache_stats`
    /// (`vtl/patches/kv_cache_manager.py::plan_request`).
    fn peek_cache_hit(&self, slot: u32, num_tokens: usize) -> usize {
        self.w().manager.peek_cache_hit(slot, num_tokens)
    }

    /// Copy the pending prefix-cache hit for `group` into its buffer; returns the count.
    fn pending_hit_into_buffer(&self, py: Python<'_>, group: usize) -> PyResult<usize> {
        let sh = self.w();
        write_buf(py, &self.bufs[group], self.buf_cap, &sh.manager.pending_hit[group])
    }

    #[allow(clippy::too_many_arguments)]
    fn allocate_slots(
        &self,
        slot: u32,
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        use_pending_hit: bool,
        num_lookahead_tokens: usize,
        num_computed_tokens: usize,
        num_request_tokens: usize,
        status: u8,
        has_scheduled_reqs: bool,
    ) -> PyResult<bool> {
        self.w()
            .manager
            .allocate_slots(
                slot,
                num_new_tokens,
                num_new_computed_tokens,
                use_pending_hit,
                num_lookahead_tokens,
                num_computed_tokens,
                num_request_tokens,
                status,
                has_scheduled_reqs,
            )
            .map_err(err)
    }

    /// Blocks newly allocated by the last `allocate_slots`, per group.
    fn new_blocks_into_buffer(&self, py: Python<'_>, group: usize) -> PyResult<usize> {
        let sh = self.w();
        write_buf(py, &self.bufs[group], self.buf_cap, &sh.manager.new_blocks[group])
    }

    /// All blocks currently held by `slot` in `group` (`KVCacheManager.get_blocks`).
    fn blocks_into_buffer(&self, py: Python<'_>, slot: u32, group: usize) -> PyResult<usize> {
        let sh = self.w();
        write_buf(
            py,
            &self.bufs[group],
            self.buf_cap,
            sh.manager.group_blocks(slot, group),
        )
    }

    /// `KVCacheBlocks.get_block_ids` shape: `tuple[list[int], ...]`. vLLM's own
    /// consumers (`NewRequestData`, `_make_cached_request_data`) require real lists, so
    /// this is the one place the port materialises them.
    fn block_ids_lists<'py>(&self, py: Python<'py>, slot: u32) -> Bound<'py, PyTuple> {
        let sh = self.w();
        let per_group: Vec<Bound<'py, PyList>> = (0..self.n_groups)
            .map(|g| {
                PyList::new_bound(py, sh.manager.group_blocks(slot, g).iter().map(|&b| b as i64))
            })
            .collect();
        PyTuple::new_bound(py, per_group)
    }

    /// `get_block_ids_for_computed_tokens` (kv_cache_manager.py:600).
    fn block_ids_for_computed_tokens<'py>(
        &self,
        py: Python<'py>,
        slot: u32,
        num_computed_tokens: usize,
    ) -> Bound<'py, PyTuple> {
        let sh = self.w();
        let per_group: Vec<Bound<'py, PyList>> = (0..self.n_groups)
            .map(|g| {
                let n = sh
                    .manager
                    .num_blocks_for_computed_tokens(slot, g, num_computed_tokens);
                PyList::new_bound(
                    py,
                    sh.manager.group_blocks(slot, g)[..n].iter().map(|&b| b as i64),
                )
            })
            .collect();
        PyTuple::new_bound(py, per_group)
    }

    fn cache_blocks(&self, slot: u32, num_computed_tokens: usize) {
        self.w().manager.cache_blocks(slot, num_computed_tokens)
    }

    fn free(&self, slot: u32) {
        self.w().manager.free(slot)
    }

    fn pop_blocks_for_free<'py>(&mut self, py: Python<'py>, slot: u32) -> Bound<'py, PyList> {
        let mut out = std::mem::take(&mut self.flat);
        out.clear();
        {
            let mut sh = self.w();
            sh.manager.pop_blocks_for_free(slot, &mut out);
        }
        let list = PyList::new_bound(py, out.iter().map(|&b| b as i64));
        self.flat = out;
        list
    }

    fn remove_skipped_blocks(&self, slot: u32, total_computed_tokens: usize) {
        self.w()
            .manager
            .remove_skipped_blocks(slot, total_computed_tokens)
    }

    fn evict_blocks(&self, block_ids: Vec<u32>) {
        self.w().manager.evict_blocks(&block_ids)
    }

    fn reset_prefix_cache(&self) -> bool {
        self.w().manager.reset_prefix_cache()
    }

    #[getter]
    fn usage(&self) -> f64 {
        self.w().manager.usage()
    }

    #[getter]
    fn num_free_blocks(&self) -> usize {
        self.w().manager.num_free_blocks()
    }

    /// Marconi hint read by `_mamba_block_aligned_split` (scheduler.py:730).
    #[getter]
    fn num_uncached_common_prefix_tokens(&self) -> usize {
        self.w().manager.coord.num_uncached_common_prefix_tokens
    }

    /// Number of blocks currently held by `slot` in `group` — cheap parity probe.
    fn num_blocks(&self, slot: u32, group: usize) -> usize {
        self.w().manager.group_blocks(slot, group).len()
    }

    /// R9: count of `update_step_pack_np(fold_cache=true)` slots the fold could not apply
    /// `cache_blocks` to (no resident table entry, or a non-RUNNING one). Should stay 0
    /// for the lifetime of a correct boot; `rust_sched.py`'s R9 disable latch polls this
    /// rather than trusting the fold's silence.
    fn cache_fold_skips(&self) -> u64 {
        self.r().manager.cache_fold_skips
    }

    fn take_prefix_cache_stats<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(s) = self.w().manager.take_prefix_cache_stats() else {
            return Ok(None);
        };
        let d = PyDict::new_bound(py);
        d.set_item("requests", s.requests)?;
        d.set_item("queries", s.queries)?;
        d.set_item("hits", s.hits)?;
        d.set_item("preempted_requests", s.preempted_requests)?;
        d.set_item("preempted_queries", s.preempted_queries)?;
        d.set_item("preempted_hits", s.preempted_hits)?;
        d.set_item("reset", s.reset)?;
        Ok(Some(d))
    }

    fn take_new_block_ids<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyList> {
        let mut out = std::mem::take(&mut self.flat);
        out.clear();
        {
            let mut sh = self.w();
            sh.manager.take_new_block_ids(&mut out);
        }
        let list = PyList::new_bound(py, out.iter().map(|&b| b as i64));
        self.flat = out;
        list
    }

    fn num_common_prefix_blocks(&self, slot: u32) -> Vec<usize> {
        self.w().manager.get_num_common_prefix_blocks(slot).to_vec()
    }
}

/// Phase B: persistent i64 buffers the decisions are written into, replacing the per-step
/// PyDict of Python lists. One per decision vector, in the order the count tuple reports:
/// running pairs, admitted triples, flat block ids, block lens, preempted, waiting order.
///
/// Separate from [`KvManager::bufs`] on purpose — those are single-slot and are overwritten
/// mid-apply by `kv.get_blocks`.
///
/// ALIASING. Python consumes slot/count values inside the same `schedule()` call, so views
/// over these are safe; block-id slices end up inside a `RustBlocks` that `SchedulerOutput`
/// retains across steps (the async batch queue keeps two in flight), so the Python side
/// copies those out with `.tolist()`. `commit_burst` reads decisions in-step only.
const ARENA_BUFS: usize = 6;

#[cfg(feature = "python")]
struct Arena {
    bufs: Vec<Py<PyArray1<i64>>>,
}

#[pyclass(module = "vtl_sched")]
pub struct Scheduler {
    /// Engine constants, handed over once by `set_params` instead of rebuilt from a dict
    /// on every step. None until `set_params` runs.
    ///
    /// The scheduling STATE (`ScheduleCore`) lives in the `KvManager`'s `Shared` — the
    /// speculation worker needs the manager and the core under one lock, and this class
    /// stays a thin, backward-compatible handle.
    params: Option<Params>,
    /// Built on the first `*_arena` call. `None` on a dict-path boot, so a plugin that
    /// never asks for the arena pays nothing. Interior-mutable so the `*_arena` entry
    /// points can take `&self`: a `&mut self` pymethod is an EXCLUSIVE pyclass borrow
    /// held across the GIL-released scheduling window, which would turn any concurrent
    /// `Scheduler` call into `RuntimeError: Already borrowed`. The RefCell borrow here is
    /// scoped to the GIL-held marshalling only.
    arena: RefCell<Option<Arena>>,
}

impl Arena {
    fn new() -> Self {
        Arena {
            bufs: Vec::with_capacity(ARENA_BUFS),
        }
    }

    /// Write `n` values into buffer `idx`, growing it (to the next power of two) if it is
    /// too small. Returns true when the buffer was REPLACED — Python then has stale views
    /// and must re-read them from `arena_buffers()`.
    fn fill<I>(&mut self, py: Python<'_>, idx: usize, n: usize, vals: I) -> PyResult<bool>
    where
        I: Iterator<Item = i64>,
    {
        while self.bufs.len() <= idx {
            self.bufs
                .push(PyArray1::<i64>::zeros_bound(py, 0, false).unbind());
        }
        let mut grew = false;
        if self.bufs[idx].bind(py).len() < n {
            self.bufs[idx] =
                PyArray1::<i64>::zeros_bound(py, n.next_power_of_two(), false).unbind();
            grew = true;
        }
        let bound = self.bufs[idx].bind(py);
        // SAFETY: the buffer is owned by this Scheduler; Python only reads `[:n]` slices
        // of it (nothing on the Python side writes it), and the GIL is held for the
        // duration of this write.
        let dst = unsafe { bound.as_slice_mut()? };
        let mut written = 0usize;
        for (d, v) in dst[..n].iter_mut().zip(vals) {
            *d = v;
            written += 1;
        }
        // Python trusts `dst[..n]` blindly -- a short iterator would leave stale entries
        // it reads as real decisions.
        debug_assert_eq!(written, n, "arena buffer {idx}: iterator yielded {written} of {n}");
        Ok(grew)
    }
}

/// `(n_running, n_admitted, n_blocks, n_block_lens, n_preempted, n_waiting, grew, dict)`.
///
/// The trailing dict is `None` unless the caller passed `check=true`, in which case it is
/// the PyDict marshalling of the SAME decisions -- the only way to compare the two
/// marshalers without running `schedule()` twice (which would mutate state twice).
type ArenaCounts<'py> = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
    Option<Bound<'py, PyDict>>,
);

impl Scheduler {
    fn params(&self) -> PyResult<Params> {
        self.params
            .ok_or_else(|| PyRuntimeError::new_err("Scheduler.set_params was never called"))
    }

    /// Marshal `d` into the arena. Counts are ELEMENT counts of the flat buffers, not
    /// tuple counts, so Python's strided views need no arity constant.
    fn write_arena<'py>(
        &self,
        py: Python<'py>,
        d: &Decisions,
        check: bool,
    ) -> PyResult<ArenaCounts<'py>> {
        let mut slot = self.arena.borrow_mut();
        let arena = slot.get_or_insert_with(Arena::new);
        let n_run = d.scheduled_running.len() * 2;
        let n_adm = d.scheduled_admitted.len() * 3;
        let mut grew = arena.fill(
            py,
            0,
            n_run,
            d.scheduled_running
                .iter()
                .flat_map(|&(slot, num_new)| [i64::from(slot), num_new as i64]),
        )?;
        grew |= arena.fill(
            py,
            1,
            n_adm,
            d.scheduled_admitted.iter().flat_map(|&(slot, num_new, computed)| {
                [i64::from(slot), num_new as i64, computed as i64]
            }),
        )?;
        grew |= arena.fill(
            py,
            2,
            d.running_new_blocks.len(),
            d.running_new_blocks.iter().map(|&b| i64::from(b)),
        )?;
        grew |= arena.fill(
            py,
            3,
            d.running_new_lens.len(),
            d.running_new_lens.iter().map(|&n| i64::from(n)),
        )?;
        grew |= arena.fill(
            py,
            4,
            d.preempted.len(),
            d.preempted.iter().map(|&s| i64::from(s)),
        )?;
        grew |= arena.fill(
            py,
            5,
            d.waiting_order.len(),
            d.waiting_order.iter().map(|&s| i64::from(s)),
        )?;
        Ok((
            n_run,
            n_adm,
            d.running_new_blocks.len(),
            d.running_new_lens.len(),
            d.preempted.len(),
            d.waiting_order.len(),
            grew,
            if check {
                Some(decisions_dict(py, d)?)
            } else {
                None
            },
        ))
    }

    fn run_schedule(
        py: Python<'_>,
        kv: &KvManager,
        running: &[SchedReq],
        waiting: &[SchedReq],
        p: &Params,
    ) -> PyResult<Decisions> {
        let shared = kv.shared.clone();
        // The decisions are CLONED under the lock, not read back afterwards: a kick that
        // is still sitting in the worker's mailbox could be picked up in the gap between
        // dropping this guard and taking another, and would overwrite `core.decisions`
        // with a speculative run's answer.
        py.allow_threads(|| {
            let mut sh = lock_shared(&shared);
            sh.invalidate();
            let Shared { manager, core, .. } = &mut *sh;
            core.schedule(manager, running, waiting, p).cloned()
        })
        .map_err(err)
    }

    fn run_schedule_resident(
        py: Python<'_>,
        kv: &KvManager,
        running_slots: &[u32],
        waiting: &[SchedReq],
        p: &Params,
    ) -> PyResult<Decisions> {
        let shared = kv.shared.clone();
        py.allow_threads(|| {
            let mut sh = lock_shared(&shared);
            sh.invalidate();
            let Shared { manager, core, .. } = &mut *sh;
            core.schedule_resident(manager, running_slots, waiting, p)
                .cloned()
        })
        .map_err(err)
    }

    fn run_take_speculative(
        py: Python<'_>,
        kv: &KvManager,
        generation: u64,
        running_slots: &[u32],
        p: &Params,
    ) -> Option<Decisions> {
        let shared = kv.shared.clone();
        py.allow_threads(|| {
            let mut sh = lock_shared(&shared);
            sh.take_speculative(generation, running_slots, p)
        })
    }
}

/// A running/waiting request as a flat tuple. Order must match `rust_sched.py::pack_req`.
type ReqTuple = (u32, usize, usize, usize, usize, usize, usize, u8, u32, bool, bool);

fn unpack(t: &ReqTuple) -> SchedReq {
    SchedReq {
        slot: t.0,
        num_tokens: t.1,
        num_tokens_with_spec: t.2,
        num_computed_tokens: t.3,
        num_output_placeholders: t.4,
        num_prompt_tokens: t.5,
        max_tokens: t.6,
        status: t.7,
        num_preemptions: t.8,
        is_prefill_chunk: t.9,
        skip_reading_prefix_cache: t.10,
    }
}

fn pack(r: SchedReq) -> ReqTuple {
    (
        r.slot,
        r.num_tokens,
        r.num_tokens_with_spec,
        r.num_computed_tokens,
        r.num_output_placeholders,
        r.num_prompt_tokens,
        r.max_tokens,
        r.status,
        r.num_preemptions,
        r.is_prefill_chunk,
        r.skip_reading_prefix_cache,
    )
}

fn decisions_dict<'py>(py: Python<'py>, d: &Decisions) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("scheduled_running", d.scheduled_running.clone())?;
    out.set_item(
        "running_new_blocks",
        PyList::new_bound(py, d.running_new_blocks.iter().map(|&b| b as i64)),
    )?;
    out.set_item(
        "running_new_lens",
        PyList::new_bound(py, d.running_new_lens.iter().map(|&n| n as i64)),
    )?;
    out.set_item("scheduled_admitted", d.scheduled_admitted.clone())?;
    out.set_item("preempted", d.preempted.clone())?;
    out.set_item("waiting_order", d.waiting_order.clone())?;
    // Empty ONLY under `lean_decisions` -- both non-lean arms of the epilogue push one
    // entry per KV group, and there is always at least one group. Omitting the key beats
    // sending a zero list: Python then reuses a shared module-level constant.
    if !d.num_common_prefix_blocks.is_empty() {
        out.set_item("num_common_prefix_blocks", d.num_common_prefix_blocks.clone())?;
    }
    Ok(out)
}

#[pymethods]
impl Scheduler {
    #[new]
    fn new() -> Self {
        Scheduler {
            params: None,
            arena: RefCell::new(None),
        }
    }

    /// Install the engine constants. Called once at first schedule; every field is fixed
    /// for the lifetime of the engine, so re-parsing a dict per step is pure overhead.
    fn set_params(&mut self, params: &Bound<'_, PyDict>) -> PyResult<()> {
        self.params = Some(Params {
            max_num_scheduled_tokens: get_usize(params, "max_num_scheduled_tokens")?,
            max_num_running_reqs: get_usize(params, "max_num_running_reqs")?,
            max_model_len: get_usize(params, "max_model_len")?,
            num_sampled_tokens_per_step: get_usize(params, "num_sampled_tokens_per_step")?,
            long_prefill_token_threshold: get_usize(params, "long_prefill_token_threshold")?,
            enable_chunked_prefill: get_bool(params, "enable_chunked_prefill")?,
            need_mamba_block_aligned_split: get_bool(params, "need_mamba_block_aligned_split")?,
            cache_block_size: get_usize(params, "cache_block_size")?,
            num_lookahead_tokens: get_usize(params, "num_lookahead_tokens")?,
            sjf_reorder: get_bool(params, "sjf_reorder")?,
            sjf_usage_tight: params
                .get_item("sjf_usage_tight")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.90),
            // Optional key: a plugin older than this wheel does not send it, and false
            // is the shipped behaviour.
            lean_decisions: params
                .get_item("lean_decisions")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(false),
        });
        Ok(())
    }

    /// Run one scheduling step from marshalled snapshots. Returns the decisions;
    /// `SchedulerOutput` assembly and all request/queue mutation stay in Python.
    ///
    /// Also the resident table's FULL RESYNC: every `running` tuple is written into the
    /// table, so a step that bailed to stock vLLM is repaired by the next Rust step.
    ///
    /// The loop runs with the GIL released.
    fn schedule<'py>(
        &self,
        py: Python<'py>,
        kv: &KvManager,
        running: Vec<ReqTuple>,
        waiting: Vec<ReqTuple>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = self.params()?;
        let running: Vec<SchedReq> = running.iter().map(unpack).collect();
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        let d = Self::run_schedule(py, kv, &running, &waiting, &p)?;
        decisions_dict(py, &d)
    }

    /// The persistent decision buffers, in count-tuple order. Python reads them once and
    /// again whenever a `*_arena` call reports `grew`.
    fn arena_buffers(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let mut slot = self.arena.borrow_mut();
        let arena = slot.get_or_insert_with(Arena::new);
        while arena.bufs.len() < ARENA_BUFS {
            arena
                .bufs
                .push(PyArray1::<i64>::zeros_bound(py, 0, false).unbind());
        }
        Ok(arena
            .bufs
            .iter()
            .map(|b| b.clone_ref(py).into_any())
            .collect())
    }

    /// [`Self::schedule`] writing into the arena instead of building a PyDict.
    #[pyo3(signature = (kv, running, waiting, check=false))]
    fn schedule_arena<'py>(
        &self,
        py: Python<'py>,
        kv: &KvManager,
        running: Vec<ReqTuple>,
        waiting: Vec<ReqTuple>,
        check: bool,
    ) -> PyResult<ArenaCounts<'py>> {
        let p = self.params()?;
        let running: Vec<SchedReq> = running.iter().map(unpack).collect();
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        let d = Self::run_schedule(py, kv, &running, &waiting, &p)?;
        self.write_arena(py, &d, check)
    }

    /// [`Self::schedule_resident`] writing into the arena.
    #[pyo3(signature = (kv, running_slots, waiting, check=false))]
    fn schedule_resident_arena<'py>(
        &self,
        py: Python<'py>,
        kv: &KvManager,
        running_slots: Vec<u32>,
        waiting: Vec<ReqTuple>,
        check: bool,
    ) -> PyResult<ArenaCounts<'py>> {
        let p = self.params()?;
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        let d = Self::run_schedule_resident(py, kv, &running_slots, &waiting, &p)?;
        self.write_arena(py, &d, check)
    }

    /// [`Self::take_speculative`] writing into the arena. `None` still means "no usable
    /// speculation, call the resident path".
    #[pyo3(signature = (kv, generation, running_slots, check=false))]
    fn take_speculative_arena<'py>(
        &self,
        py: Python<'py>,
        kv: &KvManager,
        generation: u64,
        running_slots: Vec<u32>,
        check: bool,
    ) -> PyResult<Option<ArenaCounts<'py>>> {
        let p = self.params()?;
        match Self::run_take_speculative(py, kv, generation, &running_slots, &p) {
            Some(d) => Ok(Some(self.write_arena(py, &d, check)?)),
            None => Ok(None),
        }
    }

    /// R6b: the same step with the running set read from the Rust-resident table.
    /// `running_slots` carries Python's queue ORDER, which stays authoritative.
    fn schedule_resident<'py>(
        &self,
        py: Python<'py>,
        kv: &KvManager,
        running_slots: Vec<u32>,
        waiting: Vec<ReqTuple>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = self.params()?;
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        let d = Self::run_schedule_resident(py, kv, &running_slots, &waiting, &p)?;
        decisions_dict(py, &d)
    }

    /// Ask the background worker to precompute the next step. Returns immediately.
    ///
    /// ONLY valid when the waiting queue is empty — the worker speculates with an empty
    /// waiting slice, and consuming its answer for a step that has admissions would be
    /// wrong. `generation` is Python's counter of state mutations: `take_speculative`
    /// accepts the result only if it is unchanged.
    ///
    /// Returns False when speculation is unavailable (thread spawn failed, or it was
    /// permanently disabled by a worker panic).
    fn kick(&self, kv: &mut KvManager, generation: u64, running_slots: Vec<u32>) -> PyResult<bool> {
        let p = self.params()?;
        if kv.r().spec.disabled {
            return Ok(false);
        }
        if kv.driver.is_none() {
            let shared = kv.shared.clone();
            match SpecDriver::spawn(shared) {
                Ok(d) => kv.driver = Some(d),
                Err(e) => {
                    kv.r().spec.disabled = true;
                    return Err(PyRuntimeError::new_err(format!(
                        "could not spawn the vtl-sched-spec thread: {e}"
                    )));
                }
            }
        }
        let driver = kv.driver.as_ref().unwrap();
        if driver.is_disabled() {
            return Ok(false);
        }
        driver.kick(generation, running_slots, p);
        Ok(true)
    }

    /// Consume a speculative run. `Some(decisions)` only if `generation`, the slot order
    /// AND the params are identical to the kick — in which case the speculative mutations
    /// are COMMITTED (they were the real ones). Anything else rolls back and returns None,
    /// and the caller must fall back to `schedule_resident`.
    fn take_speculative<'py>(
        &self,
        py: Python<'py>,
        kv: &KvManager,
        generation: u64,
        running_slots: Vec<u32>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let p = self.params()?;
        match Self::run_take_speculative(py, kv, generation, &running_slots, &p) {
            Some(d) => Ok(Some(decisions_dict(py, &d)?)),
            None => Ok(None),
        }
    }
}

fn py_to_key(obj: &Bound<'_, PyAny>) -> PyResult<Key> {
    if obj.is_none() {
        return Ok(Key::None);
    }
    if obj.is_instance_of::<PyBytes>() {
        return Ok(Key::Bytes(obj.extract::<Vec<u8>>()?));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Key::Str(s));
    }
    // CPython pickles True/False as NEWTRUE/NEWFALSE, not as an int, so extracting a
    // bool into Key::Int would hash to different bytes than vLLM. vLLM's extra_keys are
    // only ever str / int / bytes / tuple / None, so refuse instead of guessing.
    if obj.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "bool extra keys are not supported (CPython pickles them as NEWTRUE/NEWFALSE)",
        ));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Key::Int(i));
    }
    if let Ok(t) = obj.downcast::<PyTuple>() {
        let mut items = Vec::with_capacity(t.len());
        for it in t.iter() {
            items.push(py_to_key(&it)?);
        }
        return Ok(Key::Tuple(items));
    }
    Err(PyValueError::new_err(
        "extra keys must be str / int / bytes / tuple / None",
    ))
}

fn extra_keys_from(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Key>>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            let t = o
                .downcast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("extra_keys must be a tuple or None"))?;
            let mut v = Vec::with_capacity(t.len());
            for it in t.iter() {
                v.push(py_to_key(&it)?);
            }
            Ok(Some(v))
        }
    }
}

fn digest_from(b: &[u8]) -> PyResult<Digest32> {
    if b.len() != HASH_LEN {
        return Err(PyValueError::new_err(format!(
            "expected a {HASH_LEN}-byte hash, got {}",
            b.len()
        )));
    }
    let mut d = [0u8; HASH_LEN];
    d.copy_from_slice(b);
    Ok(d)
}

/// `init_none_hash` with `PYTHONHASHSEED` set (kv_cache_utils.py:111).
#[pyfunction]
fn none_hash_from_seed<'py>(py: Python<'py>, seed: &str) -> Bound<'py, PyBytes> {
    PyBytes::new_bound(py, &hash::none_hash_from_seed(seed))
}

/// `hash_block_tokens` (kv_cache_utils.py:577) — one block.
#[pyfunction]
#[pyo3(signature = (none_hash, parent, token_ids, extra_keys=None))]
fn hash_block_tokens<'py>(
    py: Python<'py>,
    none_hash: &Bound<'py, PyBytes>,
    parent: Option<&Bound<'py, PyBytes>>,
    token_ids: Vec<i64>,
    extra_keys: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let nh = digest_from(none_hash.as_bytes())?;
    let keys = extra_keys_from(extra_keys)?;
    let h = hash::hash_block_tokens(
        &nh,
        parent.map(|p| p.as_bytes()),
        &token_ids,
        keys.as_deref(),
    );
    Ok(PyBytes::new_bound(py, &h))
}

/// `get_request_block_hasher`'s loop (kv_cache_utils.py:687) over a whole token sequence.
#[pyfunction]
#[pyo3(signature = (none_hash, hash_block_size, token_ids, extra_keys=None))]
fn block_hashes<'py>(
    py: Python<'py>,
    none_hash: &Bound<'py, PyBytes>,
    hash_block_size: usize,
    token_ids: Vec<i64>,
    extra_keys: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let nh = digest_from(none_hash.as_bytes())?;
    if hash_block_size == 0 {
        return Err(PyValueError::new_err("hash_block_size must be > 0"));
    }
    let keys = extra_keys_from(extra_keys)?;
    let hs = hash::hash_request_tokens(&nh, hash_block_size, &token_ids, keys.as_deref());
    Ok(PyList::new_bound(
        py,
        hs.iter().map(|h| PyBytes::new_bound(py, h)),
    ))
}

/// [`block_hashes`] fed by a numpy `uint32` array instead of a Python list.
///
/// Same hashes, one contiguous read instead of `n` `PyLong_AsLongLong` calls: with the raw
/// ADD record (`shm_ipc.py`) the prompt already exists as a `uint32` view over the shm
/// frame, and hashing a 4400-token prompt at admission is otherwise 4400 extractions.
///
/// ALIGNMENT is the caller's contract — the record's reserved `u16` exists to give the id
/// block a 4-byte boundary, and `rust_sched.py` re-checks `arr.flags.aligned` before
/// choosing this entry point. Non-contiguous is refused here (Python then falls back).
#[pyfunction]
#[pyo3(signature = (none_hash, hash_block_size, token_ids, extra_keys=None))]
fn block_hashes_u32<'py>(
    py: Python<'py>,
    none_hash: &Bound<'py, PyBytes>,
    hash_block_size: usize,
    token_ids: PyReadonlyArray1<'py, u32>,
    extra_keys: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let nh = digest_from(none_hash.as_bytes())?;
    if hash_block_size == 0 {
        return Err(PyValueError::new_err("hash_block_size must be > 0"));
    }
    let ids = token_ids.as_slice().map_err(|_| {
        PyValueError::new_err("block_hashes_u32 needs a C-contiguous uint32 array")
    })?;
    let keys = extra_keys_from(extra_keys)?;
    // vLLM's hashed byte stream is a pickle of a list of Python ints, which is i64-shaped;
    // the widening is one pass over a contiguous buffer, not a Python round trip.
    let ids: Vec<i64> = ids.iter().map(|&t| i64::from(t)).collect();
    let hs = hash::hash_request_tokens(&nh, hash_block_size, &ids, keys.as_deref());
    Ok(PyList::new_bound(
        py,
        hs.iter().map(|h| PyBytes::new_bound(py, h)),
    ))
}

/// The exact byte stream vLLM's `sha256()` hashes — exposed so the parity test can diff
/// it against `pickle.dumps(...)` instead of only comparing digests.
#[pyfunction]
#[pyo3(signature = (parent, token_ids, extra_keys=None))]
fn pickle_block_hash_input<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyBytes>,
    token_ids: Vec<i64>,
    extra_keys: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let keys = extra_keys_from(extra_keys)?;
    Ok(PyBytes::new_bound(
        py,
        &hash::pickle_block_hash_input(parent.as_bytes(), &token_ids, keys.as_deref()),
    ))
}

/// `make_block_hash_with_group_id` (kv_cache_utils.py:57).
#[pyfunction]
fn block_hash_with_group_id<'py>(
    py: Python<'py>,
    block_hash: &Bound<'py, PyBytes>,
    group_id: u32,
) -> Bound<'py, PyBytes> {
    let mut out = block_hash.as_bytes().to_vec();
    out.extend_from_slice(&group_id.to_be_bytes());
    PyBytes::new_bound(py, &out)
}

// ---------------------------------------------------------------------------------------
// The Rust steady-state model runner (round-2/RUST-RUNNER.md, phases 2-3).
// ---------------------------------------------------------------------------------------

/// Drives a captured whole-step CUDA graph and commits its output without re-entering Python.
///
/// One step is: launch the dispatched graph -> pre-planned D2H of the sampled tokens into a
/// pinned buffer this object owns -> `step_pack_locked`, the SAME core `update_step_pack_np`
/// runs -> stop verdicts + the R8 record + the inline shm publish. What that removes is the
/// Python between those points: `_burst_output`, `_emit`'s ModelRunnerOutput + AsyncOutput
/// construction and its three fresh-tensor D2H copies, `postprocess_sampled`, and the hop
/// back through the engine.
///
/// EVERY refusal path returns `None` rather than raising, and `rust_sched.py` runs the normal
/// Python step on `None`. That includes: no graph for this shape, a graph that is not FULL,
/// a null exec handle, the CUDA context having moved, or the runner never having been armed.
///
/// THREADING: `run_step` must be called from the thread that owns the CUDA context -- the
/// engine's step thread. It releases the GIL for the launch + wait + commit, but never moves
/// to another thread: `torch.inference_mode` and the current stream are thread-local, and
/// `Shared`'s module-state assumptions (RUST-RUNNER.md hazard 4) are single-step-loop
/// assumptions. The context recorded at `arm()` is re-checked every step so a violation is a
/// refusal, not corruption.
#[cfg(feature = "cuda")]
#[pyclass]
pub struct Runner {
    table: crate::gpu::GraphTable,
    shared: Arc<Mutex<Shared>>,
    /// `CUstream` the graphs were captured on, from `torch.cuda.current_stream().cuda_stream`.
    stream: u64,
    /// `cuCtxGetCurrent()` at arm time. A different context per step means someone moved us
    /// to another thread; refuse rather than launch into the wrong context.
    ctx: u64,
    /// Device address of the sampled-token accumulator (`BURST.accum`) and its `[rows, cols]`
    /// shape. Persistent for the boot -- that is what makes the D2H pre-plannable.
    out_ptr: u64,
    out_rows: usize,
    out_cols: usize,
    pinned: Option<crate::gpu::live::PinnedBuf>,
    event: u64,
    armed: bool,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl Runner {
    /// Borrows the manager's `Shared` so the commit half runs against the same state the
    /// scheduler mutates -- there is exactly one engine and one runner per process.
    #[new]
    fn new(kv: PyRef<'_, KvManager>) -> Self {
        Runner {
            table: crate::gpu::GraphTable::new(),
            shared: kv.shared.clone(),
            stream: 0,
            ctx: 0,
            out_ptr: 0,
            out_rows: 0,
            out_cols: 0,
            pinned: None,
            event: 0,
            armed: false,
        }
    }

    /// True only if the wheel can actually reach libcuda. Called before arming so a driver
    /// that will not dlopen is one logged line at boot, not a per-step surprise.
    #[staticmethod]
    fn driver_available() -> bool {
        crate::gpu::live::driver().is_ok()
    }

    #[staticmethod]
    fn driver_error() -> Option<String> {
        crate::gpu::live::driver().err()
    }

    /// Add one captured graph. `exec` is `torch.cuda.CUDAGraph.raw_cuda_graph_exec()` of the
    /// WHOLE-BURST graph, `cont` of the continuation graph for launches 2..k (0 = none, so
    /// this shape takes one launch per call).
    ///
    /// `cont` defaults so a plugin older than this wheel still registers the shapes it
    /// knows about instead of failing the arm on an arity mismatch.
    #[pyo3(signature = (num_tokens, num_reqs, uniform_token_count, num_active_loras, exec,
                        full, cont = 0))]
    fn register_graph(
        &mut self,
        num_tokens: u32,
        num_reqs: Option<u32>,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
        exec: u64,
        full: bool,
        cont: u64,
    ) -> u32 {
        self.table.push_entry(crate::gpu::GraphEntry {
            num_tokens,
            num_reqs,
            uniform_token_count,
            num_active_loras,
            exec,
            cont,
            full,
        })
    }

    /// One `mgr._candidates` entry, priority order preserved.
    fn set_candidates(&mut self, num_tokens: u32, loras: u32, idx: Vec<u32>) {
        self.table.set_candidates(num_tokens, loras, idx);
    }

    fn set_lora_dispatch(&mut self, map: std::collections::HashMap<u32, u32>, max_case: u32) {
        self.table
            .set_lora_dispatch(map.into_iter().collect(), max_case);
    }

    fn set_captured(&mut self, captured: bool) {
        self.table.set_captured(captured);
    }

    fn num_graphs(&self) -> usize {
        self.table.len()
    }

    /// The boot-time parity hook: what WOULD this dispatch to? Python compares the answer
    /// against `mgr.dispatch(...)` for every reachable tuple and refuses to arm on any
    /// mismatch, because a divergence here silently runs a different graph than vLLM would.
    ///
    /// Returns `(exec_handle, is_full)`, or `None` where vLLM would fall back to eager.
    #[pyo3(signature = (num_reqs, num_tokens, uniform_token_count, num_active_loras))]
    fn dispatch_probe(
        &self,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
    ) -> Option<(u64, bool)> {
        self.table
            .dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras)
            .map(|e| (e.exec, e.full))
    }

    /// Bind the stream, the context and the output buffer, and allocate the pinned
    /// destination + the blocking event. Idempotent-ish: re-arming reallocates.
    ///
    /// `out_ptr` is `BURST.accum.data_ptr()`; `[out_rows, out_cols]` its shape. The pinned
    /// buffer is sized for the whole accumulator once, so no step allocates.
    fn arm(
        &mut self,
        stream: u64,
        out_ptr: u64,
        out_rows: usize,
        out_cols: usize,
    ) -> PyResult<()> {
        let d = crate::gpu::live::driver().map_err(err)?;
        let mut ctx: *mut std::os::raw::c_void = std::ptr::null_mut();
        // SAFETY: valid out-param; the call only reads the calling thread's current context.
        crate::gpu::live::check("cuCtxGetCurrent", unsafe {
            (d.ctx_get_current)(&mut ctx as *mut _)
        })
        .map_err(err)?;
        if ctx.is_null() {
            return Err(err(
                "no current CUDA context on the arming thread; the runner would launch into \
                 nothing"
                    .to_string(),
            ));
        }
        let bytes = out_rows
            .checked_mul(out_cols)
            .and_then(|n| n.checked_mul(std::mem::size_of::<i64>()))
            .ok_or_else(|| err("accumulator size overflows".to_string()))?;
        let pinned = crate::gpu::live::PinnedBuf::new(bytes).map_err(err)?;
        let mut event: *mut std::os::raw::c_void = std::ptr::null_mut();
        // SAFETY: valid out-param. BLOCKING_SYNC so the wait parks instead of spinning --
        // the judge box has 3 vCPUs and a spin starves the shm ingest thread.
        crate::gpu::live::check("cuEventCreate", unsafe {
            (d.event_create)(
                &mut event as *mut _,
                crate::gpu::live::CU_EVENT_BLOCKING_SYNC,
            )
        })
        .map_err(err)?;
        self.stream = stream;
        self.ctx = ctx as u64;
        self.out_ptr = out_ptr;
        self.out_rows = out_rows;
        self.out_cols = out_cols;
        self.pinned = Some(pinned);
        self.event = event as u64;
        self.armed = true;
        Ok(())
    }

    fn is_armed(&self) -> bool {
        self.armed
    }

    /// Permanently stand down for this boot. Called by the Python side's degrade path so a
    /// later step cannot re-enter the Rust runner after a divergence.
    fn disarm(&mut self) {
        self.armed = false;
    }

    /// One steady-state decode step, entirely in Rust.
    ///
    /// `None` = declined; the caller must run the Python step. Anything else returns the same
    /// `(verdicts, record, published)` triple as `update_step_pack_np`, so `rust_sched.py`'s
    /// existing handling applies unchanged.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (num_reqs, num_tokens, uniform_token_count, num_active_loras, rows,
                        counts, slots, max_model_len, engine_index, timestamp, finished,
                        fold_cache, publish))]
    fn run_step(
        &self,
        py: Python<'_>,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
        rows: Vec<u32>,
        counts: Vec<u32>,
        slots: Vec<u32>,
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: Vec<String>,
        fold_cache: bool,
        publish: bool,
    ) -> PyResult<Option<(Vec<(u32, u8, i64)>, Option<Py<PyBytes>>, bool)>> {
        if !self.armed {
            return Ok(None);
        }
        if rows.len() != counts.len() || rows.len() != slots.len() {
            return Err(PyValueError::new_err(format!(
                "run_step arity mismatch: {} rows, {} counts, {} slots",
                rows.len(),
                counts.len(),
                slots.len()
            )));
        }
        // Decline BEFORE touching the driver if this shape has no FULL graph: vLLM would run
        // eager or PIECEWISE here, and substituting a different graph would be a wrong answer.
        let Some(exec) =
            self.table
                .launchable(num_reqs, num_tokens, uniform_token_count, num_active_loras)
        else {
            return Ok(None);
        };
        let d = crate::gpu::live::driver().map_err(err)?;
        let pinned = self
            .pinned
            .as_ref()
            .ok_or_else(|| err("runner armed without a pinned buffer".to_string()))?;

        let shared = self.shared.clone();
        // Everything crossing into `allow_threads` goes as an INTEGER: a raw pointer is not
        // `Send`, and making it so would be asserting something about the pointee we do not
        // want to assert. These are opaque driver handles and device addresses; they are cast
        // back inside, on the same thread, after the context check confirms they still mean
        // what they meant at arm().
        let stream_u = self.stream;
        let event_u = self.event;
        let pinned_ptr = pinned.as_ptr() as u64;
        let pinned_len = pinned.len();
        let out_ptr = self.out_ptr;
        let (out_rows, out_cols) = (self.out_rows, self.out_cols);
        let expect_ctx = self.ctx;

        let result = py.allow_threads(move || -> Result<_, String> {
            let stream = stream_u as *mut std::os::raw::c_void;
            let event = event_u as *mut std::os::raw::c_void;
            // Hazard 2: the context is thread-local. If we are not on the thread that armed,
            // every pointer below belongs to a different address space.
            let mut ctx: *mut std::os::raw::c_void = std::ptr::null_mut();
            // SAFETY: valid out-param, read-only call.
            crate::gpu::live::check("cuCtxGetCurrent", unsafe {
                (d.ctx_get_current)(&mut ctx as *mut _)
            })?;
            if ctx as u64 != expect_ctx {
                return Err(
                    "CUDA context changed since arm(); the runner is on the wrong thread"
                        .to_string(),
                );
            }
            // 1. Replay the whole-step graph. Inputs were written into the graph's own
            //    persistent buffers before this call, on this same stream, so ordering holds
            //    without an explicit wait.
            // SAFETY: `exec` is a cudaGraphExec_t torch instantiated and never re-instantiates
            // (verified at export); `stream` is the stream it was captured on.
            crate::gpu::live::check("cuGraphLaunch", unsafe {
                (d.graph_launch)(exec as *mut std::os::raw::c_void, stream)
            })?;
            // 2. Pre-planned D2H into our own pinned buffer -- this is what replaces
            //    AsyncOutput's freshly allocated CPU tensors. Same stream as the launch, so
            //    it cannot start before the graph finishes.
            let bytes = out_rows * out_cols * std::mem::size_of::<i64>();
            if bytes > pinned_len {
                return Err(format!(
                    "accumulator is {bytes} bytes but the pinned buffer is {pinned_len}"
                ));
            }
            // SAFETY: both pointers are live for the boot and the byte count is bounds-checked
            // against the pinned allocation immediately above.
            crate::gpu::live::check("cuMemcpyDtoHAsync", unsafe {
                (d.memcpy_dtoh_async)(pinned_ptr as *mut std::os::raw::c_void, out_ptr, bytes, stream)
            })?;
            // SAFETY: `event` was created in arm() and belongs to this context.
            crate::gpu::live::check("cuEventRecord", unsafe { (d.event_record)(event, stream) })?;
            // 3. Park in the driver until the copy lands (BLOCKING_SYNC, never a spin).
            // SAFETY: same event.
            crate::gpu::live::check("cuEventSynchronize", unsafe {
                (d.event_synchronize)(event)
            })?;
            // 4. Commit. SAFETY: cuMemHostAlloc returns memory aligned well past i64, the
            //    element count is the same one the bounds check above validated, and the
            //    synchronised event is precisely the "copy has completed" guarantee.
            let arr = unsafe {
                std::slice::from_raw_parts(pinned_ptr as *const i64, out_rows * out_cols)
            };
            let flat = crate::update::gather_sampled(arr, out_rows, out_cols, &rows, &counts)?;
            step_pack_locked(
                &shared,
                &slots,
                &counts,
                &flat,
                max_model_len,
                engine_index,
                timestamp,
                &finished,
                fold_cache,
                publish,
            )
        });

        let (verdicts, record, published) = result.map_err(err)?;
        Ok(Some((
            verdicts,
            record.map(|r| PyBytes::new_bound(py, &r).unbind()),
            published,
        )))
    }

    /// Phase 3: run up to `max_steps` decode steps back-to-back without returning to Python.
    ///
    /// Every iteration is the same sequence `run_step` performs. The loop exits as soon as
    /// anything happens that the scheduler must see -- a request finishing, or a shape with
    /// no FULL graph -- so the caller never has to unwind a step it did not expect.
    ///
    /// Returns `(records, exit_reason, steps_run)`. `records` holds one entry per step that
    /// produced an R8 record and did NOT publish it inline, in order; a step that published
    /// contributes nothing, exactly as in the single-step path. Verdicts are returned for the
    /// LAST step only -- the intermediate ones are, by the exit rule above, all "nothing
    /// stopped", and materialising them would allocate per step for data nobody reads.
    ///
    /// TWO EXEC CLASSES. Launch 1 replays the whole-burst graph, which begins with a
    /// prologue over the hidden states the stock FULL replay left behind; launches 2..k
    /// replay the CONTINUATION graph, which has no prologue and feeds off the last sampled
    /// token instead. Relaunching the first exec would re-read hidden states nothing
    /// refreshed in between and re-emit the token it already emitted, so a shape with no
    /// continuation handle clamps itself to a single launch -- the caller learns how many
    /// ran and reconciles the surplus it committed.
    ///
    /// Each launch fills the NEXT `width` accumulator columns (the continuation graph does
    /// not reset the column counter), so launch i is gathered at column offset `i * width`.
    ///
    /// WHY THE BATCH SHAPE IS FIXED FOR THE WHOLE CALL: the caller passes one
    /// `(num_reqs, num_tokens, uniform, loras)` and one slot/row/count layout. A step that
    /// would change any of them is precisely a step the scheduler has to re-plan, so it
    /// belongs to the next call. This is the same align-gate reasoning the burst already
    /// uses, one level up.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (max_steps, num_reqs, num_tokens, uniform_token_count,
                        num_active_loras, rows, counts, slots, max_model_len, engine_index,
                        timestamp, finished, fold_cache, publish))]
    fn run_steps(
        &self,
        py: Python<'_>,
        max_steps: u32,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
        rows: Vec<u32>,
        counts: Vec<u32>,
        slots: Vec<u32>,
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: Vec<String>,
        fold_cache: bool,
        publish: bool,
    ) -> PyResult<Option<(Vec<(u32, u8, i64)>, Vec<Py<PyBytes>>, String, u32)>> {
        if !self.armed || max_steps == 0 {
            return Ok(None);
        }
        if rows.len() != counts.len() || rows.len() != slots.len() {
            return Err(PyValueError::new_err(format!(
                "run_steps arity mismatch: {} rows, {} counts, {} slots",
                rows.len(),
                counts.len(),
                slots.len()
            )));
        }
        let Some((exec, cont)) =
            self.table
                .launchable_pair(num_reqs, num_tokens, uniform_token_count, num_active_loras)
        else {
            return Ok(None);
        };
        // No continuation graph for this shape: one launch, and the caller reconciles.
        let max_steps = if cont == 0 { 1 } else { max_steps };
        let width = crate::gpu::burst_width(&counts).map_err(err)? as usize;
        let d = crate::gpu::live::driver().map_err(err)?;
        let pinned = self
            .pinned
            .as_ref()
            .ok_or_else(|| err("runner armed without a pinned buffer".to_string()))?;

        let shared = self.shared.clone();
        let stream_u = self.stream;
        let event_u = self.event;
        let pinned_ptr = pinned.as_ptr() as u64;
        let pinned_len = pinned.len();
        let out_ptr = self.out_ptr;
        let (out_rows, out_cols) = (self.out_rows, self.out_cols);
        let expect_ctx = self.ctx;

        let outcome = py.allow_threads(move || -> Result<_, String> {
            let stream = stream_u as *mut std::os::raw::c_void;
            let event = event_u as *mut std::os::raw::c_void;
            let mut ctx: *mut std::os::raw::c_void = std::ptr::null_mut();
            // SAFETY: valid out-param, read-only call. Checked ONCE for the whole loop --
            // the context cannot change under us without the thread changing, and we never
            // yield the thread inside.
            crate::gpu::live::check("cuCtxGetCurrent", unsafe {
                (d.ctx_get_current)(&mut ctx as *mut _)
            })?;
            if ctx as u64 != expect_ctx {
                return Err(
                    "CUDA context changed since arm(); the runner is on the wrong thread"
                        .to_string(),
                );
            }
            let bytes = out_rows * out_cols * std::mem::size_of::<i64>();
            if bytes > pinned_len {
                return Err(format!(
                    "accumulator is {bytes} bytes but the pinned buffer is {pinned_len}"
                ));
            }
            // The whole call's columns have to exist before the first launch: the graphs
            // write `[i*width, (i+1)*width)` with no bound check of their own.
            if max_steps as usize * width > out_cols {
                return Err(format!(
                    "{max_steps} launches of {width} tokens overrun the {out_cols}-column \
                     accumulator"
                ));
            }
            // PRE-FLIGHT, before anything is launched: a step that cannot pack would still
            // have advanced the resident table and fed its tokens back into the input
            // buffers, so "decline" has to be decided while declining is still free.
            if !lock_shared(&shared).manager.runner_packable(&slots) {
                return Ok(None);
            }

            let mut records: Vec<Vec<u8>> = Vec::new();
            let mut last: Vec<(u32, u8, i64)> = Vec::new();
            let mut ran: u32 = 0;
            let mut exit = LoopExit::Budget;

            for i in 0..max_steps as usize {
                // SAFETY / ordering: identical to run_step's, once per iteration. The next
                // launch is enqueued on the same stream AFTER the previous event was
                // synchronised, so no step overlaps its predecessor's D2H.
                let this = if i == 0 { exec } else { cont };
                crate::gpu::live::check("cuGraphLaunch", unsafe {
                    (d.graph_launch)(this as *mut std::os::raw::c_void, stream)
                })?;
                crate::gpu::live::check("cuMemcpyDtoHAsync", unsafe {
                    (d.memcpy_dtoh_async)(
                        pinned_ptr as *mut std::os::raw::c_void,
                        out_ptr,
                        bytes,
                        stream,
                    )
                })?;
                crate::gpu::live::check("cuEventRecord", unsafe {
                    (d.event_record)(event, stream)
                })?;
                crate::gpu::live::check("cuEventSynchronize", unsafe {
                    (d.event_synchronize)(event)
                })?;
                // SAFETY: the synchronised event is the copy-completed guarantee.
                let arr = unsafe {
                    std::slice::from_raw_parts(pinned_ptr as *const i64, out_rows * out_cols)
                };
                // Launch i's tokens live `i * width` columns in. Every row shares the
                // offset (that is what `burst_width` enforces), so shifting the flat slice
                // moves every row's window by exactly one launch -- no second index, and
                // `gather_sampled`'s own bounds check tightens with the shift.
                let flat = crate::update::gather_sampled(
                    &arr[i * width..],
                    out_rows,
                    out_cols,
                    &rows,
                    &counts,
                )?;
                let (verdicts, rec, published) = step_pack_locked(
                    &shared,
                    &slots,
                    &counts,
                    &flat,
                    max_model_len,
                    engine_index,
                    timestamp,
                    &finished,
                    fold_cache,
                    publish,
                )?;
                ran += 1;
                let packed = published || rec.is_some();
                if let Some(r) = rec {
                    if !published {
                        records.push(r);
                    }
                }
                last = verdicts;
                if !packed {
                    // The pre-flight above says this cannot happen, so it is reported
                    // rather than absorbed: the launch DID run and advanced the resident
                    // table, but `store_apply` never appended its tokens. The caller drops
                    // this launch from the residue, resyncs the table and gives the tokens
                    // back through the burst reconcile.
                    exit = LoopExit::Unpacked;
                    break;
                }
                // A non-zero status is a finish. Stop BEFORE generating another token for a
                // slot the scheduler still thinks is running.
                if last.iter().any(|(_, status, _)| *status != 0) {
                    exit = LoopExit::Stopped;
                    break;
                }
            }
            Ok(Some((last, records, exit, ran)))
        });

        let Some((verdicts, records, exit, ran)) = outcome.map_err(err)? else {
            return Ok(None);
        };
        Ok(Some((
            verdicts,
            records
                .into_iter()
                .map(|r| PyBytes::new_bound(py, &r).unbind())
                .collect(),
            exit.as_str().to_string(),
            ran,
        )))
    }
}

/// Phase 3: the exits a Rust-owned multi-step loop can decide FOR ITSELF.
///
/// Everything here is derivable from state the runner already produced this step, which is
/// the whole point -- an exit condition that needed Python would defeat the loop. The
/// conditions that are NOT here (a request arriving, a batch-shape change) are the caller's:
/// it re-enters with fresh arguments, so they are naturally checked between calls.
///
/// "Dispatch declined" is deliberately NOT a variant: `run_steps` returns `None` before the
/// loop in that case, and giving it a second spelling here would let a caller handle one and
/// miss the other.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopExit {
    /// Ran the full `max_steps` budget without anything else intervening.
    Budget,
    /// A request finished (any non-zero verdict status). The scheduler must be told before
    /// another token is generated for a slot that is done.
    Stopped,
    /// The last launch ran but its step could not be PACKED, so its tokens were never
    /// appended to the store even though the graph produced them. The pre-flight makes this
    /// unreachable; it is a variant rather than an error because by the time it is known
    /// the launch has happened, and a caller that treated it as "nothing ran" would
    /// double-count the launches before it.
    Unpacked,
}

#[cfg(feature = "cuda")]
impl LoopExit {
    fn as_str(self) -> &'static str {
        match self {
            LoopExit::Budget => "budget",
            LoopExit::Stopped => "stopped",
            LoopExit::Unpacked => "unpacked",
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for Runner {
    fn drop(&mut self) {
        if self.event == 0 {
            return;
        }
        if let Ok(d) = crate::gpu::live::driver() {
            // SAFETY: created by cuEventCreate in arm(), destroyed exactly once here.
            unsafe { (d.event_destroy)(self.event as *mut std::os::raw::c_void) };
        }
    }
}

#[pymodule]
fn vtl_sched(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<KvManager>()?;
    m.add_class::<Scheduler>()?;
    // Absent unless the wheel was built with the `cuda` feature. `rust_sched.py` probes with
    // hasattr and leaves VTL_RUST_RUNNER inert if it is missing, so an old wheel degrades
    // instead of raising.
    #[cfg(feature = "cuda")]
    m.add_class::<Runner>()?;
    m.add("HAS_CUDA_RUNNER", cfg!(feature = "cuda"))?;
    m.add_function(wrap_pyfunction!(none_hash_from_seed, m)?)?;
    m.add_function(wrap_pyfunction!(hash_block_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(block_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(block_hashes_u32, m)?)?;
    m.add_function(wrap_pyfunction!(pickle_block_hash_input, m)?)?;
    m.add_function(wrap_pyfunction!(block_hash_with_group_id, m)?)?;
    Ok(())
}
