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

use std::sync::{Arc, Mutex, MutexGuard};

use numpy::{PyArray1, PyArrayMethods};
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
use crate::update::StopParams;


fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
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

    // ---- R6b: the resident request table ----------------------------------

    /// Overwrite one slot's entry. Same tuple layout as `pack_req` / `Scheduler.schedule`.
    /// The escape hatch for anything Python mutates out of band.
    fn table_set(&self, slot: u32, req: ReqTuple) {
        self.w().manager.table_set(slot, unpack(&req));
    }

    fn table_clear(&self, slot: u32) {
        self.w().manager.table_clear(slot);
    }

    /// One slot's entry in `pack_req` field order, for shadow-mode comparison.
    ///
    /// NON-INVALIDATING on purpose, so a shadow build can probe it without killing every
    /// speculation. The flip side: between a `kick` and its `take_speculative` this reads
    /// the SPECULATIVE post-commit values. Call it outside that window.
    fn table_entry(&self, slot: u32) -> Option<ReqTuple> {
        self.r().manager.table_get(slot).map(pack)
    }

    /// FxHash over the entries at `slots`, in order. Same non-invalidating caveat as
    /// [`KvManager::table_entry`].
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

    fn take_evicted(&self) -> Vec<u32> {
        self.w().manager.take_evicted()
    }
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
    out.set_item("num_common_prefix_blocks", d.num_common_prefix_blocks.clone())?;
    out.set_item("token_budget_left", d.token_budget_left)?;
    Ok(out)
}

#[pymethods]
impl Scheduler {
    #[new]
    fn new() -> Self {
        Scheduler { params: None }
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
        let p = self
            .params
            .ok_or_else(|| PyRuntimeError::new_err("Scheduler.set_params was never called"))?;
        let running: Vec<SchedReq> = running.iter().map(unpack).collect();
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        let shared = kv.shared.clone();
        // The decisions are CLONED under the lock, not read back afterwards: a kick that
        // is still sitting in the worker's mailbox could be picked up in the gap between
        // dropping this guard and taking another, and would overwrite `core.decisions`
        // with a speculative run's answer.
        let d = py
            .allow_threads(|| {
                let mut sh = lock_shared(&shared);
                sh.invalidate();
                let Shared { manager, core, .. } = &mut *sh;
                core.schedule(manager, &running, &waiting, &p).cloned()
            })
            .map_err(err)?;
        decisions_dict(py, &d)
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
        let p = self
            .params
            .ok_or_else(|| PyRuntimeError::new_err("Scheduler.set_params was never called"))?;
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        let shared = kv.shared.clone();
        let d = py
            .allow_threads(|| {
                let mut sh = lock_shared(&shared);
                sh.invalidate();
                let Shared { manager, core, .. } = &mut *sh;
                core.schedule_resident(manager, &running_slots, &waiting, &p)
                    .cloned()
            })
            .map_err(err)?;
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
        let p = self
            .params
            .ok_or_else(|| PyRuntimeError::new_err("Scheduler.set_params was never called"))?;
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
        let p = self
            .params
            .ok_or_else(|| PyRuntimeError::new_err("Scheduler.set_params was never called"))?;
        let shared = kv.shared.clone();
        let taken = py.allow_threads(|| {
            let mut sh = lock_shared(&shared);
            sh.take_speculative(generation, &running_slots, &p)
        });
        match taken {
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

#[pymodule]
fn vtl_sched(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<KvManager>()?;
    m.add_class::<Scheduler>()?;
    m.add_function(wrap_pyfunction!(none_hash_from_seed, m)?)?;
    m.add_function(wrap_pyfunction!(hash_block_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(block_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(pickle_block_hash_input, m)?)?;
    m.add_function(wrap_pyfunction!(block_hash_with_group_id, m)?)?;
    Ok(())
}
