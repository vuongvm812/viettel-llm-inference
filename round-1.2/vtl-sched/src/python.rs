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

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::config::{Config, GroupConfig};
use crate::hash;
use crate::hash::{Digest32, Key, HASH_LEN};
use crate::manager::Manager;
use crate::sched::{Params, SchedReq, ScheduleCore};
use crate::single_type::{cdiv, Kind};


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
    inner: Manager,
    /// One persistent numpy buffer per KV cache group; block IDs are written here and
    /// Python takes a zero-copy `[:n]` view.
    bufs: Vec<Py<PyArray1<i64>>>,
    buf_cap: usize,
    flat: Vec<u32>,
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
        let n = cfg.groups.len();
        let inner = Manager::new(cfg).map_err(err)?;
        let bufs = (0..n)
            .map(|_| PyArray1::<i64>::zeros_bound(py, buf_cap, false).unbind())
            .collect();
        Ok(KvManager {
            inner,
            bufs,
            buf_cap,
            flat: Vec::with_capacity(256),
        })
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.inner.coord.managers.len()
    }

    /// The persistent per-group block-ID buffer. Python holds these once and slices
    /// `buf[:n]` — no per-call allocation of the data.
    fn buffer(&self, py: Python<'_>, group: usize) -> PyObject {
        self.bufs[group].clone_ref(py).into_any()
    }

    // ---- request registry -------------------------------------------------

    fn intern(&mut self, req_id: &str) -> u32 {
        self.inner.intern(req_id)
    }

    fn lookup(&self, req_id: &str) -> Option<u32> {
        self.inner.lookup(req_id)
    }

    fn forget(&mut self, req_id: &str) {
        self.inner.forget(req_id)
    }

    /// Append the request's new block hashes (32 bytes each, concatenated).
    fn push_hashes(
        &mut self,
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
        self.inner.push_hashes(slot, b, num_prompt_tokens);
        Ok(())
    }

    fn num_hashes(&self, slot: u32) -> usize {
        self.inner.num_hashes(slot)
    }

    // ---- KVCacheManager surface -------------------------------------------

    fn new_step_starts(&mut self) {
        self.inner.new_step_starts()
    }

    /// Returns `num_new_computed_tokens`; the hit blocks stay pending for the following
    /// `allocate_slots(use_pending_hit=True)`.
    fn get_computed_blocks(
        &mut self,
        slot: u32,
        num_tokens: usize,
        num_preemptions: u32,
        skip_reading_prefix_cache: bool,
    ) -> usize {
        self.inner
            .get_computed_blocks(slot, num_tokens, num_preemptions, skip_reading_prefix_cache)
    }

    /// Read-only cache-hit walk that does not touch `prefix_cache_stats`
    /// (`vtl/patches/kv_cache_manager.py::plan_request`).
    fn peek_cache_hit(&mut self, slot: u32, num_tokens: usize) -> usize {
        self.inner.peek_cache_hit(slot, num_tokens)
    }

    /// Copy the pending prefix-cache hit for `group` into its buffer; returns the count.
    fn pending_hit_into_buffer(&mut self, py: Python<'_>, group: usize) -> PyResult<usize> {
        write_buf(
            py,
            &self.bufs[group],
            self.buf_cap,
            &self.inner.pending_hit[group],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn allocate_slots(
        &mut self,
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
        self.inner
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
    fn new_blocks_into_buffer(&mut self, py: Python<'_>, group: usize) -> PyResult<usize> {
        write_buf(
            py,
            &self.bufs[group],
            self.buf_cap,
            &self.inner.new_blocks[group],
        )
    }

    /// All blocks currently held by `slot` in `group` (`KVCacheManager.get_blocks`).
    fn blocks_into_buffer(&mut self, py: Python<'_>, slot: u32, group: usize) -> PyResult<usize> {
        let src = self.inner.group_blocks(slot, group);
        write_buf(py, &self.bufs[group], self.buf_cap, src)
    }

    /// `KVCacheBlocks.get_block_ids` shape: `tuple[list[int], ...]`. vLLM's own
    /// consumers (`NewRequestData`, `_make_cached_request_data`) require real lists, so
    /// this is the one place the port materialises them.
    fn block_ids_lists<'py>(&self, py: Python<'py>, slot: u32) -> Bound<'py, PyTuple> {
        let per_group: Vec<Bound<'py, PyList>> = (0..self.num_groups())
            .map(|g| {
                PyList::new_bound(
                    py,
                    self.inner.group_blocks(slot, g).iter().map(|&b| b as i64),
                )
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
        let per_group: Vec<Bound<'py, PyList>> = (0..self.num_groups())
            .map(|g| {
                let n = self
                    .inner
                    .num_blocks_for_computed_tokens(slot, g, num_computed_tokens);
                PyList::new_bound(
                    py,
                    self.inner.group_blocks(slot, g)[..n].iter().map(|&b| b as i64),
                )
            })
            .collect();
        PyTuple::new_bound(py, per_group)
    }

    fn cache_blocks(&mut self, slot: u32, num_computed_tokens: usize) {
        self.inner.cache_blocks(slot, num_computed_tokens)
    }

    fn free(&mut self, slot: u32) {
        self.inner.free(slot)
    }

    fn pop_blocks_for_free(&mut self, slot: u32) -> Vec<u32> {
        let mut out = std::mem::take(&mut self.flat);
        out.clear();
        self.inner.pop_blocks_for_free(slot, &mut out);
        let res = out.clone();
        self.flat = out;
        res
    }

    fn remove_skipped_blocks(&mut self, slot: u32, total_computed_tokens: usize) {
        self.inner.remove_skipped_blocks(slot, total_computed_tokens)
    }

    fn evict_blocks(&mut self, block_ids: Vec<u32>) {
        self.inner.evict_blocks(&block_ids)
    }

    fn reset_prefix_cache(&mut self) -> bool {
        self.inner.reset_prefix_cache()
    }

    #[getter]
    fn usage(&self) -> f64 {
        self.inner.usage()
    }

    #[getter]
    fn num_free_blocks(&self) -> usize {
        self.inner.num_free_blocks()
    }

    /// Marconi hint read by `_mamba_block_aligned_split` (scheduler.py:730).
    #[getter]
    fn num_uncached_common_prefix_tokens(&self) -> usize {
        self.inner.coord.num_uncached_common_prefix_tokens
    }

    /// Number of blocks currently held by `slot` in `group` — cheap parity probe.
    fn num_blocks(&self, slot: u32, group: usize) -> usize {
        self.inner.group_blocks(slot, group).len()
    }

    fn take_prefix_cache_stats<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(s) = self.inner.take_prefix_cache_stats() else {
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

    fn take_new_block_ids(&mut self) -> Vec<u32> {
        let mut out = std::mem::take(&mut self.flat);
        out.clear();
        self.inner.take_new_block_ids(&mut out);
        let res = out.clone();
        self.flat = out;
        res
    }

    fn num_common_prefix_blocks(&mut self, slot: u32) -> Vec<usize> {
        self.inner.get_num_common_prefix_blocks(slot).to_vec()
    }

    fn take_evicted(&mut self) -> Vec<u32> {
        self.inner.take_evicted()
    }
}

#[pyclass(module = "vtl_sched")]
pub struct Scheduler {
    core: ScheduleCore,
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

#[pymethods]
impl Scheduler {
    #[new]
    fn new() -> Self {
        Scheduler {
            core: ScheduleCore::new(),
        }
    }

    /// Run one scheduling step. Returns the decisions; `SchedulerOutput` assembly and
    /// all request/queue mutation stay in Python.
    fn schedule<'py>(
        &mut self,
        py: Python<'py>,
        kv: &mut KvManager,
        running: Vec<ReqTuple>,
        waiting: Vec<ReqTuple>,
        params: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = Params {
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
        };
        let running: Vec<SchedReq> = running.iter().map(unpack).collect();
        let waiting: Vec<SchedReq> = waiting.iter().map(unpack).collect();
        self.core
            .schedule(&mut kv.inner, &running, &waiting, &p)
            .map_err(err)?;
        let d = &self.core.decisions;
        let out = PyDict::new_bound(py);
        out.set_item("scheduled_running", d.scheduled_running.clone())?;
        out.set_item("scheduled_admitted", d.scheduled_admitted.clone())?;
        out.set_item("preempted", d.preempted.clone())?;
        out.set_item("waiting_order", d.waiting_order.clone())?;
        out.set_item("num_common_prefix_blocks", d.num_common_prefix_blocks.clone())?;
        out.set_item("token_budget_left", d.token_budget_left)?;
        Ok(out)
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
