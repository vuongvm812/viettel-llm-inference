//! Thread → core pinning. Real on Linux, silent no-op on macOS (the disruptor
//! crate pins its *managed* threads; our poller/loop threads are ours to pin).

/// Pin the current thread to `core`. On macOS `core_affinity::set_for_current`
/// is a no-op, matching the crate's own affinity behavior.
pub fn pin(core: usize) {
    if let Some(ids) = core_affinity::get_core_ids() {
        if let Some(id) = ids.get(core) {
            core_affinity::set_for_current(*id);
        }
    }
}
