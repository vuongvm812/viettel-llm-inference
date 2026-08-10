//! Phase 3b: the frontend -> engine REQUEST subscriber, owned by the crate.
//!
//! Near-mirror of [`crate::out`] with the ports reversed: same `ipc_threadsafe` service
//! type, same seed, same service configuration -- and the configuration is not merely
//! "similar", it has to be IDENTICAL to `shm_ipc.py`'s constants, because `open_or_create`
//! refuses to pair with a peer whose config differs and the failure mode of a near-miss is a
//! boot where nothing ever pairs and every request quietly rides ZMQ instead.
//!
//! WHAT MOVES, AND WHY ONLY THIS. The Python `vtl-shm-in` thread structure survives whole:
//! the ready handshake ordering, the msgpack decoders, `preprocess_add_request`'s error
//! path, the ABORT dual-queue put and the one-bad-record-does-not-kill-the-channel rule.
//! Only the hot receive+split moves here, because that is the part that holds the GIL for a
//! `ctypes.string_at` of the whole record plus a `struct.unpack_from` per frame while the
//! decode loop is trying to run. A Rust-owned thread calling back into Python was rejected:
//! `blocking_wait_one` already parks without the GIL, so the callback shape would
//! re-implement every error semantic above for no additional GIL win.
//!
//! [`split`] and [`crate::out::seed`] are unconditional so their golden tests (shared with
//! `shm_ipc.py::GOLDEN_RAW_ADD` / `GOLDEN_SEED`) run under default `cargo test`; everything
//! that touches iceoryx2 is behind the `shm` cargo feature.

/// Request-channel tag for the raw ADD record. Mirrors `shm_ipc.py::TAG_RAW_ADD`: 0x10 sits
/// outside `EngineCoreRequestType`'s 0x00-0x03 so both tag spaces share the one ordered
/// channel.
pub const TAG_RAW_ADD: u8 = 0x10;
pub const RAW_ADD_VERSION: u8 = 1;
/// Offset of the u32 id block from the START of the record, tag byte included:
/// `1 (tag) + 1 (version) + 2 (reserved) + 4 (n_tokens)`. The reserved u16 exists exactly to
/// make this a multiple of 4 -- `shm_ipc.py` asserts the same thing, because the Python side
/// takes an ALIGNED `np.frombuffer` view over these bytes.
pub const RAW_ADD_IDS_AT: usize = 8;

const _: () = assert!(RAW_ADD_IDS_AT % 4 == 0);

/// One received record, split the way `shm_ipc.py::_run_input_thread` needs it.
///
/// `(tag, prompt_id_bytes, tail)`. `prompt_id_bytes` is `Some` only for [`TAG_RAW_ADD`],
/// where it is the little-endian u32 block Python views with `np.frombuffer`; `tail` is what
/// the msgpack decoder is handed -- the record past the ids for a raw ADD, and everything
/// past the tag byte for every other frame.
pub type Split = (u8, Option<Vec<u8>>, Vec<u8>);

/// Why a receive failed -- and, more usefully, WHOSE failure it is. The Python arm makes
/// exactly this distinction structurally (its framing `ValueError` is caught per record, its
/// listener/subscriber errors break the loop), so the crate has to hand it back rather than
/// collapse both into one string the caller would have to pattern-match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecvError {
    /// The channel itself failed. The caller drops the subscriber; the frontend sees 0
    /// receivers on its next publish and demotes itself to ZMQ for the boot.
    Channel(String),
    /// ONE record could not be framed. The caller drops the frame and keeps the channel --
    /// and says so loudly: the frontend committed that ADD on publish, so a dropped one is
    /// a request that hangs until the client times out.
    Record(String),
}

/// Split one framed request record. `Err` = drop the frame, exactly as the Python arm does.
///
/// The two owned `Vec`s are the SAME copy `ctypes.string_at` used to make on the Python
/// side, taken here instead -- with the GIL released. (The `PyBytes` the caller builds from
/// them is one extra memcpy of ~17 KB per arriving request; that buys the split, the park
/// and the copy all happening off the GIL, which is what the decode loop is waiting for.)
pub fn split(record: &[u8]) -> Result<Split, String> {
    let &tag = record
        .first()
        .ok_or_else(|| "empty request record".to_string())?;
    if tag != TAG_RAW_ADD {
        return Ok((tag, None, record[1..].to_vec()));
    }
    if record.len() < RAW_ADD_IDS_AT {
        return Err(format!(
            "raw ADD header is {} bytes, needs {RAW_ADD_IDS_AT}",
            record.len()
        ));
    }
    let version = record[1];
    if version != RAW_ADD_VERSION {
        return Err(format!(
            "raw ADD version {version}, this build speaks {RAW_ADD_VERSION}"
        ));
    }
    // record[2..4] is the reserved u16; deliberately not read.
    let n = u32::from_le_bytes([record[4], record[5], record[6], record[7]]) as usize;
    let end = RAW_ADD_IDS_AT
        .checked_add(n.checked_mul(4).ok_or_else(|| {
            format!("raw ADD claims {n} tokens, which does not fit a byte count")
        })?)
        .ok_or_else(|| format!("raw ADD claims {n} tokens, which overruns the record"))?;
    if end > record.len() {
        return Err(format!(
            "raw ADD claims {n} tokens, record is {} bytes",
            record.len()
        ));
    }
    Ok((
        tag,
        Some(record[RAW_ADD_IDS_AT..end].to_vec()),
        record[end..].to_vec(),
    ))
}

// Only the `shm` build opens a channel; a plain `cargo test` never calls this.
#[cfg(feature = "shm")]
fn request_service(input_address: &str, engine_index: u32) -> String {
    format!(
        "vtl/req/{}/{engine_index}",
        crate::out::seed(input_address)
    )
}

#[cfg(feature = "shm")]
mod live {
    use super::{request_service, split, RecvError, Split};
    use iceoryx2::port::listener::Listener;
    use iceoryx2::port::subscriber::Subscriber;
    use iceoryx2::prelude::*;
    use iceoryx2::service::ipc_threadsafe::Service as ShmService;

    // --- service configuration; MUST stay identical to shm_ipc.py's module constants ------
    // (`MAX_PUBLISHERS`/`MAX_SUBSCRIBERS`/`SUBSCRIBER_MAX_BUFFER_SIZE`/`HISTORY_SIZE`/
    // `ENABLE_SAFE_OVERFLOW`, shm_ipc.py:64-78, applied by its `_open_channel`.) A
    // mismatch does not fail loudly -- `open_or_create` simply refuses to pair, and the
    // frontend sees 0 subscribers and demotes itself to ZMQ for the boot.
    const MAX_PUBLISHERS: usize = 1;
    const MAX_SUBSCRIBERS: usize = 1;
    const SUBSCRIBER_MAX_BUFFER_SIZE: usize = 512;
    const HISTORY_SIZE: usize = 0;
    // See shm_ipc.py:70-77: with safe overflow OFF a full buffer makes the publisher's
    // `send()` report 0 receivers rather than silently overwriting an unread request.
    const ENABLE_SAFE_OVERFLOW: bool = false;

    fn shm_error(e: impl core::fmt::Display) -> String {
        format!("iceoryx2: {e}")
    }

    /// Frontend -> engine request channel. Pairs with the publisher the Rust frontend builds
    /// on `vtl/req/<seed>/<engine_index>` (+ `/ev`), i.e. the very services
    /// `shm_ipc.py::_run_input_thread` would otherwise open with the iceoryx2 Python
    /// bindings. `MAX_SUBSCRIBERS = 1` makes the two mutually exclusive: whichever opens
    /// first owns the channel for the boot, so the Python side builds one or the other and
    /// never both.
    pub struct InputChannel {
        subscriber: Subscriber<ShmService, [u8], ()>,
        listener: Listener<ShmService>,
        // Owns this process' iceoryx2 registration for these services; dropping it would
        // tear them down out from under the ports above.
        _node: Node<ShmService>,
    }

    impl InputChannel {
        pub fn open(input_address: &str, engine_index: u32) -> Result<InputChannel, String> {
            let node = NodeBuilder::new().create::<ShmService>().map_err(shm_error)?;
            let name = request_service(input_address, engine_index);
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
            let subscriber = data.subscriber_builder().create().map_err(shm_error)?;
            let listener = event.listener_builder().create().map_err(shm_error)?;
            Ok(InputChannel {
                subscriber,
                listener,
                _node: node,
            })
        }

        /// The next record, split. Blocks until one arrives.
        ///
        /// DRAIN-THEN-BLOCK: a queued sample is taken without touching the listener at all,
        /// so a burst of arrivals that produced one notification still yields every record.
        /// Only an empty queue parks -- in the driver's own primitive, never a spin (the
        /// judge box has 3 vCPUs, and this thread must not compete with the decode loop).
        /// A notification that lands between the failed `receive` and the park is not lost:
        /// the listener latches it and `blocking_wait_one` returns immediately.
        ///
        /// A spurious wakeup (`Ok(None)` from the listener) just re-enters the loop.
        pub fn recv_split(&self) -> Result<Split, RecvError> {
            loop {
                let pending = self
                    .subscriber
                    .receive()
                    .map_err(|e| RecvError::Channel(shm_error(e)))?;
                if let Some(sample) = pending {
                    return split(sample.payload()).map_err(RecvError::Record);
                }
                self.listener
                    .blocking_wait_one()
                    .map_err(|e| RecvError::Channel(shm_error(e)))?;
            }
        }
    }
}

#[cfg(feature = "shm")]
pub use live::InputChannel;

#[cfg(test)]
mod tests {
    use super::{split, RAW_ADD_IDS_AT, TAG_RAW_ADD};

    /// Copied VERBATIM from `shm_ipc.py::GOLDEN_RAW_ADD`, which its `_self_check` asserts
    /// `raw_add_split` against and which the frontend's `encode_raw_add` unit test builds.
    /// Three implementations, one vector: a layout drift fails a test instead of handing
    /// the engine a prompt it decodes into garbage.
    const GOLDEN_RAW_ADD: &str = concat!(
        "10", // TAG_RAW_ADD
        "01", "0000", "03000000", // version, reserved, n_tokens
        "01000000", "02000000", "2c010000", // the ids, little-endian u32
        "dc0014", // msgpack array16, 20 elements
        "ac", "76746c2d676f6c64656e2d30", // [0]  request_id
        "c0", // [1]  prompt_token_ids -> nil
        "c0", "c0", "c0", // [2]  mm/sampling/pooling params
        "cb3ff8000000000000", // [5]  arrival_time 1.5
        "c0", "c0", "c0", "c0", "c0", // [6]  lora..prompt_is_token_ids
        "00", "00", "00", // [11] client_index, current_wave, priority
        "c0", "c2", "c0", "c0", "c0", // [14] trace_headers..reasoning_parser_kwargs
        "c2", // [19] abort_immediately
    );

    fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn splits_the_python_golden_raw_add_vector() {
        let record = bytes(GOLDEN_RAW_ADD);
        let (tag, ids, tail) = split(&record).unwrap();
        assert_eq!(tag, TAG_RAW_ADD);
        let ids = ids.expect("a raw ADD carries its prompt ids");
        // 1, 2, 300 as little-endian u32 -- the same tuple as GOLDEN_RAW_ADD_TOKENS.
        assert_eq!(
            ids.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![1, 2, 300]
        );
        // Both halves must account for every byte, exactly as the Python check asserts.
        assert_eq!(RAW_ADD_IDS_AT + ids.len() + tail.len(), record.len());
        // The tail starts at the msgpack array16 header and carries the request id.
        assert_eq!(&tail[..3], &[0xdc, 0x00, 0x14]);
        assert!(tail
            .windows(12)
            .any(|w| w == b"vtl-golden-0"));
        // Element 1 (prompt_token_ids) is nil -- the ids ride in the header instead.
        assert_eq!(tail[3 + 1 + "vtl-golden-0".len()], 0xc0);
    }

    #[test]
    fn a_non_raw_add_frame_is_tag_plus_everything_after_it() {
        // `EngineCoreRequestType.ABORT` and friends: 0x00-0x03, msgpack body, no id block.
        let record = vec![0x01, 0x90, 0xc0];
        let (tag, ids, tail) = split(&record).unwrap();
        assert_eq!((tag, ids), (0x01, None));
        assert_eq!(tail, vec![0x90, 0xc0]);
    }

    #[test]
    fn a_version_this_build_does_not_speak_is_refused_not_guessed() {
        let mut record = bytes(GOLDEN_RAW_ADD);
        record[1] = 2;
        assert!(split(&record).is_err());
    }

    #[test]
    fn a_truncated_or_over_claiming_record_is_refused() {
        let record = bytes(GOLDEN_RAW_ADD);
        // Claims 0xffff tokens in a 60-byte record.
        let mut lying = record.clone();
        lying[4..8].copy_from_slice(&0xffffu32.to_le_bytes());
        assert!(split(&lying).is_err());
        // A count whose byte length overflows usize on a 32-bit target, and does not here --
        // either way it must be an Err, never a panic or a wrapped slice.
        let mut huge = record.clone();
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(split(&huge).is_err());
        // A header shorter than the fixed part.
        assert!(split(&record[..RAW_ADD_IDS_AT - 1]).is_err());
        assert!(split(&[]).is_err());
    }

    #[test]
    fn a_zero_token_raw_add_is_all_tail() {
        let mut record = bytes(GOLDEN_RAW_ADD);
        record.drain(RAW_ADD_IDS_AT..RAW_ADD_IDS_AT + 12);
        record[4..8].copy_from_slice(&0u32.to_le_bytes());
        let (_tag, ids, tail) = split(&record).unwrap();
        assert_eq!(ids.unwrap().len(), 0);
        assert_eq!(tail.len(), record.len() - RAW_ADD_IDS_AT);
    }
}
