//! The Rust steady-state runner's GPU half: a CUDA-graph dispatch table and the driver
//! calls that launch through it. See `round-2/RUST-RUNNER.md`.
//!
//! Split deliberately in two, the same way `out.rs` is:
//!
//!   * [`GraphTable`] and [`GraphTable::dispatch`] are PURE integer logic -- a port of
//!     `cudagraph_utils.py`'s `dispatch` / `_is_compatible` / `_resolve_effective_loras`.
//!     They compile and unit-test under DEFAULT features, which is what keeps the
//!     Dockerfile's `cargo test --release` gate (no libpython, no CUDA, no iceoryx2)
//!     meaningful for the part of this module that can actually be wrong in a way tests
//!     catch.
//!   * [`live`] holds the dlopen'd libcuda entry points and is behind the `cuda` feature.
//!
//! WHY THE TABLE IS EXPORTED, NOT DERIVED. `_init_candidates` (cudagraph_utils.py:181) builds
//! the priority-ordered candidate lists from cudagraph_mode, the capture-size list, the
//! decode query length and the LoRA capture cases. Re-deriving that in Rust would be a second
//! implementation of a policy that can change under us on a version bump, and a divergence
//! would silently pick a DIFFERENT graph than vLLM would -- a wrong-answer bug, not a crash.
//! So Python flattens `mgr._candidates` at boot and hands it over, and the boot-time export
//! then asserts `rust.dispatch(...) == mgr.dispatch(...)` across every reachable tuple before
//! the runner is allowed to arm.

use rustc_hash::FxHashMap;

/// One captured graph: a `BatchExecutionDescriptor` plus its instantiated `cudaGraphExec_t`.
///
/// `num_reqs` / `uniform_token_count` are `Option` for the same reason they are `| None` in
/// Python: a PIECEWISE descriptor imposes no request padding and accepts any uniform token
/// count. The runner only ever launches FULL entries, but the table carries whatever the
/// export gave it so `dispatch` can reproduce vLLM's answer exactly, including the cases
/// where the winning candidate is one the runner then declines to drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphEntry {
    pub num_tokens: u32,
    pub num_reqs: Option<u32>,
    pub uniform_token_count: Option<u32>,
    pub num_active_loras: u32,
    /// `cudaGraphExec_t` from `torch.cuda.CUDAGraph.raw_cuda_graph_exec()`. 0 = not a graph
    /// the runner may launch (exported for dispatch fidelity only).
    ///
    /// This is nstep's UNROLL graph, not stock's forward-only one: the runner's D2H reads
    /// the burst accumulator, which only the unroll graph fills (`rust_runner.py`).
    pub exec: u64,
    /// The CONTINUATION graph for launches 2..k of a multi-burst step: the same bodies with
    /// no prologue, so it feeds off the last sampled token instead of hidden states nothing
    /// refreshed. 0 = this shape can only take ONE launch per call; relaunching `exec`
    /// back-to-back would re-emit the previous burst's first token.
    pub cont: u64,
    /// True for `cg_mode == FULL`. Only these are launchable by the runner.
    pub full: bool,
}

/// The per-request token count a multi-launch call needs, or why it has none.
///
/// Launch i writes accumulator columns `[i*width, (i+1)*width)` for EVERY row, so one
/// shared width is what makes the column offset a single number instead of a per-row one.
/// A ragged batch would still gather plausible ids from the wrong columns, so it is refused
/// rather than approximated -- and being pure integer logic, it is testable without CUDA.
pub fn burst_width(counts: &[u32]) -> Result<u32, String> {
    let Some(&width) = counts.first() else {
        return Err("burst_width: no rows".to_string());
    };
    if width == 0 {
        return Err("burst_width: a row claims zero tokens".to_string());
    }
    if counts.iter().any(|&c| c != width) {
        return Err(format!("burst_width: ragged counts {counts:?}"));
    }
    Ok(width)
}

/// Priority-ordered candidates keyed by `(num_tokens, effective_loras)`, plus the LoRA
/// dispatch map, exactly as `CudaGraphManager` holds them.
#[derive(Default, Debug)]
pub struct GraphTable {
    entries: Vec<GraphEntry>,
    /// `(num_tokens, effective_loras) -> indices into `entries`, in vLLM's priority order.
    candidates: FxHashMap<(u32, u32), Vec<u32>>,
    /// `_build_lora_dispatch_map`'s precomputed actual -> captured-case mapping.
    lora_dispatch: FxHashMap<u32, u32>,
    max_lora_case: u32,
    /// `_graphs_captured`. False = dispatch always misses, same as vLLM before capture.
    captured: bool,
}

impl GraphTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one captured graph, returning its index.
    pub fn push_entry(&mut self, entry: GraphEntry) -> u32 {
        self.entries.push(entry);
        (self.entries.len() - 1) as u32
    }

    /// Register one `(num_tokens, effective_loras)` candidate list. Order is vLLM's priority
    /// order and is preserved verbatim -- `dispatch` returns the FIRST compatible entry, so a
    /// reordering here silently changes which graph runs.
    pub fn set_candidates(&mut self, num_tokens: u32, loras: u32, idx: Vec<u32>) {
        self.candidates.insert((num_tokens, loras), idx);
    }

    pub fn set_lora_dispatch(&mut self, map: FxHashMap<u32, u32>, max_case: u32) {
        self.lora_dispatch = map;
        self.max_lora_case = max_case;
    }

    pub fn set_captured(&mut self, captured: bool) {
        self.captured = captured;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entry(&self, idx: u32) -> Option<&GraphEntry> {
        self.entries.get(idx as usize)
    }

    /// `_resolve_effective_loras` (cudagraph_utils.py:174-179): map an actual active-LoRA
    /// count onto the captured case that can serve it, clamping above the largest.
    pub fn effective_loras(&self, num_active_loras: u32) -> u32 {
        if num_active_loras == 0 || self.lora_dispatch.is_empty() {
            return num_active_loras;
        }
        *self
            .lora_dispatch
            .get(&num_active_loras)
            .unwrap_or(&self.max_lora_case)
    }

    /// `_is_compatible` (cudagraph_utils.py:76-93), field for field.
    fn compatible(
        e: &GraphEntry,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        effective_loras: u32,
    ) -> bool {
        e.uniform_token_count
            .is_none_or(|u| Some(u) == uniform_token_count)
            && e.num_reqs.is_none_or(|n| n >= num_reqs)
            && e.num_tokens >= num_tokens
            && e.num_active_loras == effective_loras
    }

    /// `CudaGraphManager.dispatch` (cudagraph_utils.py:371-396).
    ///
    /// `None` means vLLM would fall back to `CUDAGraphMode.NONE` (eager) -- there is no graph
    /// for this shape, so the runner must hand the step back to Python rather than invent one.
    pub fn dispatch(
        &self,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
    ) -> Option<&GraphEntry> {
        let effective = self.effective_loras(num_active_loras);
        if !self.captured || num_tokens == 0 {
            return None;
        }
        let idx = self.candidates.get(&(num_tokens, effective))?;
        idx.iter()
            .filter_map(|i| self.entries.get(*i as usize))
            .find(|e| Self::compatible(e, num_reqs, num_tokens, uniform_token_count, effective))
    }

    /// What the runner actually needs: the launchable handle for this shape, or `None` to
    /// decline the step. A candidate that wins dispatch but is not FULL (a PIECEWISE graph)
    /// is a decline, NOT a fallthrough to the next candidate -- vLLM would have run that
    /// PIECEWISE graph, so trying the next one would diverge from it.
    pub fn launchable(
        &self,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
    ) -> Option<u64> {
        self.launchable_pair(num_reqs, num_tokens, uniform_token_count, num_active_loras)
            .map(|(exec, _)| exec)
    }

    /// [`GraphTable::launchable`] plus the continuation handle, which is `0` when this
    /// shape has none. A caller that gets `0` must run exactly ONE launch: the first exec
    /// begins with a prologue over hidden states nothing re-computes between launches, so
    /// relaunching it would re-emit the token it already emitted.
    pub fn launchable_pair(
        &self,
        num_reqs: u32,
        num_tokens: u32,
        uniform_token_count: Option<u32>,
        num_active_loras: u32,
    ) -> Option<(u64, u64)> {
        let e = self.dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras)?;
        (e.full && e.exec != 0).then_some((e.exec, e.cont))
    }
}

// ---------------------------------------------------------------------------------------
// The live driver half.
// ---------------------------------------------------------------------------------------

/// Errors from the driver half, as strings so the Python side can log one reason and
/// degrade -- every failure here means "run the step in Python", never "crash the engine".
#[cfg(feature = "cuda")]
pub type CudaResult<T> = Result<T, String>;

#[cfg(feature = "cuda")]
pub mod live {
    //! dlopen'd libcuda. No CUDA headers and no link-time CUDA, so this still compiles in a
    //! builder stage that has neither -- which is why `cargo test --features cuda --no-run`
    //! is a meaningful gate there.

    use super::CudaResult;
    use libloading::{Library, Symbol};
    use std::os::raw::{c_int, c_uint, c_void};
    use std::sync::OnceLock;

    pub type CUresult = c_int;
    pub type CUstream = *mut c_void;
    pub type CUevent = *mut c_void;
    pub type CUgraphExec = *mut c_void;
    pub type CUcontext = *mut c_void;

    const CUDA_SUCCESS: CUresult = 0;
    /// `CU_EVENT_BLOCKING_SYNC`: park the calling thread in the driver instead of spinning.
    /// Mandatory here -- the judge box has 3 vCPUs and a spin-wait starves the shm ingest
    /// thread (RUST-RUNNER.md hazard 5).
    pub const CU_EVENT_BLOCKING_SYNC: c_uint = 1;

    type FnGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
    type FnMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, u64, usize, CUstream) -> CUresult;
    type FnMemcpyHtoDAsync = unsafe extern "C" fn(u64, *const c_void, usize, CUstream) -> CUresult;
    type FnEventCreate = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
    type FnEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
    type FnEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
    type FnEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
    type FnMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult;
    type FnMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
    type FnCtxGetCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;

    /// The eight-ish driver entry points the runner needs, resolved once.
    pub struct Driver {
        _lib: Library,
        pub graph_launch: FnGraphLaunch,
        pub memcpy_dtoh_async: FnMemcpyDtoHAsync,
        pub memcpy_htod_async: FnMemcpyHtoDAsync,
        pub event_create: FnEventCreate,
        pub event_record: FnEventRecord,
        pub event_synchronize: FnEventSynchronize,
        pub event_destroy: FnEventDestroy,
        pub mem_host_alloc: FnMemHostAlloc,
        pub mem_free_host: FnMemFreeHost,
        pub ctx_get_current: FnCtxGetCurrent,
    }

    // SAFETY: every field is a plain function pointer into libcuda, which is thread-safe by
    // contract, and `_lib` is only held to keep the mapping alive. Nothing here is mutated
    // after construction.
    unsafe impl Send for Driver {}
    unsafe impl Sync for Driver {}

    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

    macro_rules! sym {
        ($lib:expr, $name:literal) => {{
            // SAFETY: the symbol names are CUDA driver API entry points and the signatures
            // above match the CUDA 12 headers. A missing symbol is a caught Err, not UB.
            let s: Symbol<'_, _> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
                .map_err(|e| format!("libcuda: no {}: {e}", $name))?;
            // SAFETY: lifting the symbol out of the Library borrow is sound because `_lib`
            // is moved into the same struct and outlives every use of the pointer.
            *s
        }};
    }

    fn load() -> Result<Driver, String> {
        // `libcuda.so.1` is the driver's stable soname; `libcuda.so` only exists with the
        // dev package installed, which the runtime image does not ship.
        // SAFETY: dlopen of a system library. No initialisers of ours run.
        let lib = unsafe { Library::new("libcuda.so.1") }
            .map_err(|e| format!("libcuda.so.1 could not be dlopened: {e}"))?;
        Ok(Driver {
            graph_launch: sym!(lib, "cuGraphLaunch"),
            memcpy_dtoh_async: sym!(lib, "cuMemcpyDtoHAsync_v2"),
            memcpy_htod_async: sym!(lib, "cuMemcpyHtoDAsync_v2"),
            event_create: sym!(lib, "cuEventCreate"),
            event_record: sym!(lib, "cuEventRecord"),
            event_synchronize: sym!(lib, "cuEventSynchronize"),
            event_destroy: sym!(lib, "cuEventDestroy"),
            mem_host_alloc: sym!(lib, "cuMemHostAlloc"),
            mem_free_host: sym!(lib, "cuMemFreeHost"),
            ctx_get_current: sym!(lib, "cuCtxGetCurrent"),
            _lib: lib,
        })
    }

    /// The process-wide driver handle, or the reason it is unavailable. Callers treat an
    /// `Err` as "the Rust runner cannot arm this boot" and fall back to Python.
    pub fn driver() -> Result<&'static Driver, String> {
        match DRIVER.get_or_init(load) {
            Ok(d) => Ok(d),
            Err(e) => Err(e.clone()),
        }
    }

    pub fn check(name: &str, rc: CUresult) -> CudaResult<()> {
        if rc == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{name} failed with CUresult {rc}"))
        }
    }

    /// Page-locked host memory the runner owns for the whole boot, so the per-step D2H has a
    /// pre-planned destination instead of `AsyncOutput`'s freshly allocated CPU tensors.
    pub struct PinnedBuf {
        ptr: *mut c_void,
        len: usize,
    }

    // SAFETY: a pinned host allocation is ordinary memory; the runner is the only owner and
    // access is serialised by the step loop.
    unsafe impl Send for PinnedBuf {}

    impl PinnedBuf {
        pub fn new(len: usize) -> CudaResult<PinnedBuf> {
            let d = driver()?;
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a valid out-param; `len` is the byte count we then record.
            check("cuMemHostAlloc", unsafe {
                (d.mem_host_alloc)(&mut ptr as *mut _, len, 0)
            })?;
            Ok(PinnedBuf { ptr, len })
        }

        pub fn as_ptr(&self) -> *mut c_void {
            self.ptr
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// The landed sampled-token ids. `count` elements, never more than the buffer holds.
        ///
        /// # Safety
        /// Only valid after the event recorded behind the D2H has been synchronised; reading
        /// earlier races the copy.
        pub unsafe fn as_i64(&self, count: usize) -> &[i64] {
            debug_assert!(count * std::mem::size_of::<i64>() <= self.len);
            // SAFETY: cuMemHostAlloc returns memory aligned well past i64, and the caller
            // guarantees the copy has completed (see the doc comment).
            unsafe { std::slice::from_raw_parts(self.ptr as *const i64, count) }
        }
    }

    impl Drop for PinnedBuf {
        fn drop(&mut self) {
            if self.ptr.is_null() {
                return;
            }
            if let Ok(d) = driver() {
                // SAFETY: `ptr` came from cuMemHostAlloc and is freed exactly once.
                unsafe { (d.mem_free_host)(self.ptr) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(num_tokens: u32, num_reqs: u32, exec: u64) -> GraphEntry {
        GraphEntry {
            num_tokens,
            num_reqs: Some(num_reqs),
            uniform_token_count: Some(1),
            num_active_loras: 0,
            exec,
            cont: 0,
            full: true,
        }
    }

    /// A decode-shaped table: FULL graphs at 1/2/4/8, one token per request.
    fn decode_table() -> GraphTable {
        let mut t = GraphTable::new();
        for (i, size) in [1u32, 2, 4, 8].iter().enumerate() {
            let idx = t.push_entry(full(*size, *size, 0x1000 + i as u64));
            t.set_candidates(*size, 0, vec![idx]);
        }
        t.set_captured(true);
        t
    }

    #[test]
    fn dispatch_picks_the_exact_size() {
        let t = decode_table();
        let e = t.dispatch(4, 4, Some(1), 0).expect("size 4 is captured");
        assert_eq!(e.num_tokens, 4);
        assert_eq!(t.launchable(4, 4, Some(1), 0), Some(0x1002));
    }

    #[test]
    fn dispatch_misses_before_capture() {
        let mut t = decode_table();
        t.set_captured(false);
        assert!(t.dispatch(4, 4, Some(1), 0).is_none());
    }

    #[test]
    fn zero_tokens_never_dispatch() {
        // vLLM guards `num_tokens > 0`; a zero-token step is not a graph shape.
        let t = decode_table();
        assert!(t.dispatch(0, 0, Some(1), 0).is_none());
    }

    #[test]
    fn uniform_token_count_must_match_when_the_graph_pins_it() {
        let t = decode_table();
        // A decode graph captured at uniform=1 must NOT serve a 2-tokens-per-request step.
        assert!(t.dispatch(4, 4, Some(2), 0).is_none());
        assert!(t.dispatch(4, 4, None, 0).is_none());
    }

    #[test]
    fn a_piecewise_style_entry_accepts_any_uniform_count() {
        let mut t = GraphTable::new();
        let idx = t.push_entry(GraphEntry {
            num_tokens: 16,
            num_reqs: None,
            uniform_token_count: None,
            num_active_loras: 0,
            exec: 0,
            cont: 0,
            full: false,
        });
        t.set_candidates(16, 0, vec![idx]);
        t.set_captured(true);
        assert!(t.dispatch(3, 16, Some(7), 0).is_some());
        assert!(t.dispatch(3, 16, None, 0).is_some());
        // ...but it is not launchable: vLLM would run the PIECEWISE graph, so the runner
        // declines the step rather than substituting a different graph.
        assert_eq!(t.launchable(3, 16, Some(7), 0), None);
    }

    #[test]
    fn num_reqs_is_an_upper_bound_but_num_tokens_must_fit_too() {
        let mut t = GraphTable::new();
        let idx = t.push_entry(full(8, 8, 0xAA));
        t.set_candidates(8, 0, vec![idx]);
        t.set_captured(true);
        // Padded batch: 5 real requests riding the size-8 graph.
        assert_eq!(t.launchable(5, 8, Some(1), 0), Some(0xAA));
        // More requests than the graph was captured for: refuse.
        assert!(t.dispatch(9, 8, Some(1), 0).is_none());
    }

    #[test]
    fn candidate_order_is_priority_order() {
        // Two entries serve the same key; dispatch must take the FIRST compatible one, not
        // the smallest or the last.
        let mut t = GraphTable::new();
        let big = t.push_entry(full(8, 8, 0xB16));
        let small = t.push_entry(full(8, 4, 0x5A11));
        t.set_candidates(8, 0, vec![small, big]);
        t.set_captured(true);
        assert_eq!(t.launchable(3, 8, Some(1), 0), Some(0x5A11));
        // 6 requests no longer fit the num_reqs=4 entry, so the next candidate wins.
        assert_eq!(t.launchable(6, 8, Some(1), 0), Some(0xB16));
    }

    #[test]
    fn lora_counts_clamp_to_the_largest_captured_case() {
        let mut t = GraphTable::new();
        let e = t.push_entry(GraphEntry {
            num_tokens: 4,
            num_reqs: Some(4),
            uniform_token_count: Some(1),
            num_active_loras: 2,
            exec: 0x10A,
            cont: 0,
            full: true,
        });
        t.set_candidates(4, 2, vec![e]);
        let mut map = FxHashMap::default();
        map.insert(1u32, 2u32);
        map.insert(2u32, 2u32);
        t.set_lora_dispatch(map, 2);
        t.set_captured(true);
        assert_eq!(t.effective_loras(0), 0);
        assert_eq!(t.effective_loras(1), 2);
        assert_eq!(t.effective_loras(2), 2);
        // Above the largest captured case: clamp, matching `_resolve_effective_loras`.
        assert_eq!(t.effective_loras(9), 2);
        assert!(t.dispatch(4, 4, Some(1), 1).is_some());
    }

    #[test]
    fn a_null_exec_is_never_launchable() {
        // Exported for dispatch fidelity but not drivable -- must not be handed to
        // cuGraphLaunch.
        let mut t = GraphTable::new();
        let idx = t.push_entry(full(4, 4, 0));
        t.set_candidates(4, 0, vec![idx]);
        t.set_captured(true);
        assert!(t.dispatch(4, 4, Some(1), 0).is_some());
        assert_eq!(t.launchable(4, 4, Some(1), 0), None);
    }

    #[test]
    fn an_unknown_shape_declines() {
        let t = decode_table();
        assert!(t.dispatch(3, 3, Some(1), 0).is_none(), "size 3 was never captured");
    }

    #[test]
    fn the_two_exec_classes_dispatch_together() {
        // A shape WITH a continuation graph: launch 1 takes `exec`, launches 2..k take
        // `cont`. Both come out of one dispatch -- looking the second one up separately
        // could pick a different row than the first.
        let mut t = GraphTable::new();
        let mut e = full(4, 4, 0xE1);
        e.cont = 0xC1;
        let idx = t.push_entry(e);
        t.set_candidates(4, 0, vec![idx]);
        t.set_captured(true);
        assert_eq!(t.launchable_pair(4, 4, Some(1), 0), Some((0xE1, 0xC1)));
        // `launchable` is the same answer without the continuation, so the single-launch
        // caller cannot accidentally see one.
        assert_eq!(t.launchable(4, 4, Some(1), 0), Some(0xE1));
    }

    #[test]
    fn a_missing_continuation_reports_zero_not_the_first_exec() {
        // The whole hazard: relaunching the FIRST exec back-to-back re-reads hidden states
        // nothing refreshed between launches and re-emits the token it already emitted. A
        // shape with no continuation graph must therefore come back with 0, never with
        // `exec` duplicated -- the caller clamps itself to one launch on 0.
        let t = decode_table();
        assert_eq!(t.launchable_pair(4, 4, Some(1), 0), Some((0x1002, 0)));
    }

    #[test]
    fn a_declined_shape_has_no_pair_at_all() {
        let t = decode_table();
        assert!(t.launchable_pair(3, 3, Some(1), 0).is_none());
    }

    #[test]
    fn burst_width_is_the_shared_row_count() {
        assert_eq!(burst_width(&[4, 4, 4]), Ok(4));
        assert_eq!(burst_width(&[1]), Ok(1));
        // Ragged, empty and zero-width batches all break the `i * width` column offset a
        // multi-launch call gathers with, so none of them may be guessed at.
        assert!(burst_width(&[4, 4, 2]).is_err());
        assert!(burst_width(&[]).is_err());
        assert!(burst_width(&[0, 0]).is_err());
    }
}
