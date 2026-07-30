//! Batch 3: the engine -> frontend output publisher, owned by the crate.
//!
//! Mirrors `vtl/patches/shm_ipc.py`'s `_OutputPublisher` byte-for-byte -- same service name,
//! same config constants, same framing -- because `MAX_PUBLISHERS = 1` is baked into the
//! service config on both halves (see that module's header comment): a crate publisher and
//! the Python publisher cannot coexist, so whichever one opens first for a boot owns the
//! channel for its lifetime.
//!
//! Behind the `shm` cargo feature: default `cargo test` must stay iceoryx2-free (no network
//! fetch, no libpython either -- see `python` feature). Only [`seed`] is unconditional, so its
//! golden test (shared with `shm_ipc.py::GOLDEN_SEED`) runs under default features too.

use sha2::{Digest, Sha256};

/// Per-boot service-name seed both processes derive independently from the same string
/// (Rust hashes the address it bound, Python hashes the one it was handed -- see
/// `shm_ipc.py::_seed`'s docstring). MUST match that function exactly: sha256 hex, lowercase,
/// first 12 characters.
pub fn seed(input_address: &str) -> String {
    let digest = Sha256::digest(input_address.as_bytes());
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex.truncate(12);
    hex
}

// Only the `shm` build actually opens a channel; a plain `cargo test` never calls this, so
// it would otherwise warn as dead code.
#[cfg(feature = "shm")]
fn output_service(input_address: &str) -> String {
    format!("vtl/out/{}", seed(input_address))
}

#[cfg(feature = "shm")]
mod live {
    use super::output_service;
    use iceoryx2::port::notifier::Notifier;
    use iceoryx2::port::publisher::Publisher;
    use iceoryx2::prelude::*;
    use iceoryx2::service::ipc_threadsafe::Service as ShmService;

    // --- service configuration; MUST stay identical to shm_ipc.py / the rust-frontend patch --
    const MAX_PUBLISHERS: usize = 1;
    const MAX_SUBSCRIBERS: usize = 1;
    const SUBSCRIBER_MAX_BUFFER_SIZE: usize = 512;
    const HISTORY_SIZE: usize = 0;
    // See shm_ipc.py:69-77 for why overflow stays off: with it off, a full subscriber buffer
    // makes `send()` report 0 receivers -- the same signal as "nobody subscribed", which
    // already routes callers to the ZMQ fallback -- instead of silently overwriting data or
    // (the iceoryx2 default) blocking this thread on a stalled frontend.
    const ENABLE_SAFE_OVERFLOW: bool = false;
    const OUTPUT_MAX_PAYLOAD: usize = 1 << 18;

    fn shm_error(e: impl core::fmt::Display) -> String {
        format!("iceoryx2: {e}")
    }

    /// Engine -> frontend output channel. Opens (or pairs with) the same
    /// `vtl/out/<seed>` + `<name>/ev` services `shm_ipc.py::_OutputPublisher` would, using
    /// `ipc_threadsafe` (matching the golden-seed-verified pairing the frontend subscriber
    /// dials into -- `shm_ipc.patch`'s `ShmService = ipc_threadsafe::Service`).
    pub struct OutChannel {
        publisher: Publisher<ShmService, [u8], ()>,
        notifier: Notifier<ShmService>,
        // Owns this process' iceoryx2 registration for these services; dropping it would
        // tear them down out from under the ports above.
        _node: Node<ShmService>,
    }

    impl OutChannel {
        pub fn open(input_address: &str) -> Result<OutChannel, String> {
            let node = NodeBuilder::new().create::<ShmService>().map_err(shm_error)?;
            let name = output_service(input_address);
            let data = node
                .service_builder(&name.as_str().try_into().map_err(shm_error)?)
                .publish_subscribe::<[u8]>()
                .max_publishers(MAX_PUBLISHERS)
                .max_subscribers(MAX_SUBSCRIBERS)
                .subscriber_max_buffer_size(SUBSCRIBER_MAX_BUFFER_SIZE)
                .history_size(HISTORY_SIZE)
                .enable_safe_overflow(ENABLE_SAFE_OVERFLOW)
                .open_or_create()
                .map_err(shm_error)?;
            let event = node
                .service_builder(&format!("{name}/ev").as_str().try_into().map_err(shm_error)?)
                .event()
                .max_notifiers(MAX_PUBLISHERS)
                .max_listeners(MAX_SUBSCRIBERS)
                .open_or_create()
                .map_err(shm_error)?;
            let publisher = data
                .publisher_builder()
                .initial_max_slice_len(OUTPUT_MAX_PAYLOAD)
                .allocation_strategy(AllocationStrategy::PowerOfTwo)
                // Mirrors shm_ipc.py: a full/receiverless buffer reports back rather than
                // blocking the engine's critical path (RetryUntilDelivered, the default,
                // would stall this call on a stalled frontend).
                .backpressure_strategy(BackpressureStrategy::DiscardData)
                .create()
                .map_err(shm_error)?;
            let notifier = event.notifier_builder().create().map_err(shm_error)?;
            Ok(OutChannel { publisher, notifier, _node: node })
        }

        /// Mirrors `_OutputPublisher.publish`: `tag + payload`, tag prepended by this call.
        /// `false` = undelivered (no subscriber, or a full buffer) -> caller falls back.
        pub fn publish(&self, tag: u8, payload: &[u8]) -> bool {
            let Ok(mut sample) = self.publisher.loan_slice_uninit(payload.len() + 1) else {
                return false;
            };
            {
                let buffer = sample.payload_mut();
                buffer[0].write(tag);
                // SAFETY: `MaybeUninit<u8>` is `#[repr(transparent)]` over `u8`, so an
                // initialized `&[u8]` is a valid source for a copy into `&mut
                // [MaybeUninit<u8>]` -- every byte is about to be overwritten either way.
                let tail = unsafe {
                    core::mem::transmute::<&[u8], &[core::mem::MaybeUninit<u8>]>(payload)
                };
                buffer[1..].copy_from_slice(tail);
            }
            // SAFETY: every byte of the loaned slice was written above.
            self.send(unsafe { sample.assume_init() })
        }

        /// Mirrors `_OutputPublisher.publish_record`: `record` is ALREADY framed (R8's
        /// packer writes its own tag byte at offset 0), so this copies it verbatim -- no
        /// tag prepended, unlike [`Self::publish`]. This is the delivered-publish path
        /// `python.rs::update_step_pack_np` takes: the record comes straight out of the
        /// crate's own `RawPacker` buffer, so this is the ONLY copy from pack to wire (no
        /// intermediate `Vec<u8>` / `PyBytes`).
        pub fn publish_record(&self, record: &[u8]) -> bool {
            let Ok(mut sample) = self.publisher.loan_slice_uninit(record.len()) else {
                return false;
            };
            {
                let buffer = sample.payload_mut();
                let src = unsafe {
                    core::mem::transmute::<&[u8], &[core::mem::MaybeUninit<u8>]>(record)
                };
                buffer.copy_from_slice(src);
            }
            self.send(unsafe { sample.assume_init() })
        }

        fn send(&self, sample: iceoryx2::sample_mut::SampleMut<ShmService, [u8], ()>) -> bool {
            match sample.send() {
                Ok(0) | Err(_) => false,
                Ok(_receivers) => {
                    let _ = self.notifier.notify();
                    true
                }
            }
        }
    }
}

#[cfg(feature = "shm")]
pub use live::OutChannel;

#[cfg(test)]
mod tests {
    use super::seed;

    // Shared with `shm_ipc.py::GOLDEN_SEED_ADDRESS` / `GOLDEN_SEED` -- if either side's hash
    // or truncation drifts, this fails instead of the two processes silently picking
    // different service names (a boot that pairs with nobody, forever on ZMQ).
    #[test]
    fn seed_matches_the_python_golden_vector() {
        assert_eq!(seed("ipc:///tmp/vllm-in-0"), "23c5030baa20");
    }

    #[test]
    fn seed_is_deterministic_and_address_sensitive() {
        assert_eq!(seed("a"), seed("a"));
        assert_ne!(seed("a"), seed("b"));
        assert_eq!(seed("a").len(), 12);
    }
}
