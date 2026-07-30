"""iceoryx2 zero-copy shared-memory data plane for the Rust frontend <-> EngineCore hop.

Stock v0.25.0 moves every request and every output batch over ZMQ ``ipc://`` unix sockets
carrying msgpack. That is a syscall + copy per message on a box with 3 vCPUs. This patch
swaps the *transport* for iceoryx2 (lock-free shm pub/sub, wakeups via its event pattern so
reader threads park instead of spinning) and leaves the *serialization* alone.

Flags (all default off -- with them unset this module installs nothing):

* ``VTL_ENABLE_SHM_IPC=1`` -- registry gate, required.
* ``VTL_SHM_IPC=1``       -- master switch, read by BOTH processes. The Rust half
  (``vtl/vllm_patches/rust-frontend/shm_ipc.patch``) reads the same variable; if only one
  side has it the other degrades to ZMQ on its own (publishes report 0 receivers).
* ``VTL_SHM_IPC_RAW=1``   -- Phase B. Pack the common decode output as a flat little-endian
  struct instead of msgpack. Requires ``VTL_SHM_IPC=1``. Any batch the fixed layout cannot
  express falls back to a msgpack record transparently.

``docker-compose.yaml`` is the submission artifact and must stay literal, so these knobs are
NOT wired into it; ``docker-compose-optimized.yaml`` carries only the dev image tag. Set them
ad hoc for an A/B, e.g. ``docker compose ... -e VTL_ENABLE_SHM_IPC=1 -e VTL_SHM_IPC=1``.

What is patched, and why here rather than in ``__init__``:

1. ``EngineCoreProc.process_input_sockets`` is *wrapped*. The wrapper starts one extra input
   thread (iceoryx2 subscriber + listener on ``vtl/req/<seed>/<engine_index>``) and waits for
   it to report "subscribed" before delegating to the stock body. That ordering is
   load-bearing: the stock body's first act is to send ``ready_payload``, which is what
   releases the frontend's ``wait_for_input_registrations`` and therefore what lets the Rust
   side build its publisher. Subscribing first means the frontend's very first shm publish
   already has a receiver, so the transport never has to switch mid-stream (an ABORT
   overtaking its ADD). The stock ZMQ input thread is untouched -- handshake, registration,
   DP coordinator and fallback all keep working.
2. ``EngineCoreProc.process_output_sockets`` is *replaced* with a copy of the v0.25.0 body
   whose only transport change is the ``sockets[client_index].send_multipart(...)`` line, plus
   a DEAD record published alongside the existing ZMQ sentinel sends. See the inline
   ``upstream:`` comments for the exact lines mirrored.

Failure modes are deliberately boring: engine death still travels the stock ZMQ sentinel path
(`_send_engine_dead` -> `socket.send(ENGINE_CORE_DEAD)`, unchanged) and the frontend keeps its
ZMQ output loop running as the death detector; a frontend death is still caught by the stock
launcher sentinel; stale shm cannot outlive a boot because the service names are seeded from
the per-boot engine input address.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
from collections import deque
from contextlib import ExitStack

from vtl.registry import already_patched, mark_patched, register_patch

# Must be a child of "vllm.vtl": records logged to a bare "vtl" logger are dropped.
log = logging.getLogger("vllm.vtl.shm_ipc")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# --- service configuration; MUST stay identical to the Rust side --------------------------
MAX_PUBLISHERS = 1
MAX_SUBSCRIBERS = 1
SUBSCRIBER_MAX_BUFFER_SIZE = 512
HISTORY_SIZE = 0
REQUEST_MAX_PAYLOAD = 1 << 20
OUTPUT_MAX_PAYLOAD = 1 << 18
# iceoryx2 defaults to safe overflow: a publish into a FULL subscriber buffer silently
# overwrites the oldest unread sample. For a request/output stream that is silent data loss
# (a hung request, no log), so both sides turn it off -- the setting is part of the service
# config and `open_or_create` refuses to pair a service whose peer disagrees. With overflow
# off the publisher's backpressure strategy decides what a full buffer does; DiscardData
# makes `send()` report 0 receivers, which is the same signal as "nobody subscribed" and
# therefore already routes to the ZMQ fallback below. (RetryUntilDelivered, the iceoryx2
# default, would instead BLOCK the engine's output thread on a stalled frontend.)
ENABLE_SAFE_OVERFLOW = False
# Backlog math for the 512-sample buffer: the scored trace peaks at 8 concurrent generations
# and the engine emits ONE output batch per decode step, so the buffer holds ~512 decode
# steps ~= 2 s of stream at a 4 ms TPOT. A frontend that falls that far behind is not
# "momentarily busy", it is wedged -- reporting the failure and demoting to ZMQ is right.

# --- record framing; MUST stay identical to the Rust side ---------------------------------
TAG_MSGPACK = b"M"
TAG_RAW = b"R"
TAG_DEAD = b"D"
RAW_VERSION = 1

# Header: version u8, reserved u16, engine_index u32, timestamp f64, n_out u32, n_fin u32.
_RAW_HEADER = struct.Struct("<BHIdII")
# Per output: id_len u32, n_tok u32, n_nans u32, flags u8, finish u8, reserved u16, stop u32.
_RAW_OUTPUT_HEAD = struct.Struct("<IIIBBHI")
_U32 = struct.Struct("<I")
FLAG_FINISH_REASON = 0b0000_0001
FLAG_STOP_TOKEN = 0b0000_0010

assert _RAW_HEADER.size == 23, _RAW_HEADER.size
assert _RAW_OUTPUT_HEAD.size == 20, _RAW_OUTPUT_HEAD.size


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


# --- Batch 3: crate-owned output publisher --------------------------------------------
#
# ``VTL_SHM_IPC_RUST_PUB=1`` moves publisher OWNERSHIP for the engine -> frontend channel
# into the Rust crate: ``rust_sched.py``'s R9 residue path publishes the finished R8 record
# straight from ``update_step_pack_np`` on the engine thread, skipping the
# ``output_queue.put_nowait`` -> this module's output thread -> iceoryx2 hop entirely for
# the common case. ``MAX_PUBLISHERS = 1`` (below) means a crate publisher and a Python
# ``_OutputPublisher`` can never coexist, so ``_process_output_sockets`` builds one or the
# other, never both -- see its ``try``/``except`` at the top.


class _RustKvHandle:
    """One-slot registry: `rust_sched.py` calls `register_rust_kv` once its crate
    `KvManager` exists (it has no way to learn `input_address` itself -- that lives on
    `EngineCoreProc`, not the KV manager); `_process_output_sockets` -- which DOES know the
    address -- reads it back to open the channel. A list, not a bare module global, only
    because a late `register_rust_kv` call (patch install order is not guaranteed) must
    still be visible without needing a `global` statement at the call site.
    """

    __slots__ = ("kv",)

    def __init__(self) -> None:
        self.kv = None


_RUST_KV = _RustKvHandle()


def register_rust_kv(kv) -> None:
    """Cross-module handshake counterpart to `_RUST_KV` -- see its docstring."""
    _RUST_KV.kv = kv


class _PubCounters:
    """Inline-publish ordering guard: two SINGLE-WRITER counters, so neither needs a lock
    under the GIL (a lone `+= 1` on an int attribute is one atomic-enough read + write from
    the writer's own thread; only the OTHER thread's reads race it, and a stale read just
    delays the gate closing/opening by at most one step). ``enqueued`` is bumped by the
    engine thread (`rust_sched.py::update_from_output`) once per step whose output actually
    reaches `output_queue`; ``published`` is bumped here, by the output thread, once per
    queue item this thread finishes sending. `rust_sched.py`'s `decide()` only asks the
    crate to publish inline when the two agree -- otherwise an inline publish could
    overtake an item still in flight through the queue and reorder the wire stream.
    """

    __slots__ = ("enqueued", "published")

    def __init__(self) -> None:
        self.enqueued = 0
        self.published = 0


PUB = _PubCounters()


class _RustOutputAdapter:
    """Same method shape as `_OutputPublisher`, backed by the crate's `out_publish` /
    `out_publish_record` FFIs instead of the iceoryx2 Python bindings, so
    `_process_output_sockets`'s body -- the demotion latch, the TAG_DEAD send, the R8 bytes
    arm -- is textually unchanged whichever one is live. Constructed only after
    `kv.out_open()` has succeeded (``MAX_PUBLISHERS = 1`` makes the two mutually exclusive
    by construction -- see this module's header).
    """

    __slots__ = ("_kv",)

    def __init__(self, kv) -> None:
        self._kv = kv

    def publish(self, tag: bytes, payload) -> bool:
        if not isinstance(payload, (bytes, bytearray)):
            payload = bytes(payload)
        return self._kv.out_publish(tag[0], bytes(payload))

    def publish_record(self, record) -> bool:
        if not isinstance(record, bytes):
            record = bytes(record)
        return self._kv.out_publish_record(record)

    def publish_raw(self, outputs) -> bool:
        """No crate FFI packs `outputs` straight into a loaned shm sample the way Python's
        `_OutputPublisher.publish_raw` does -- this is the object-fallback arm (R8 did NOT
        apply this step), so one extra `bytearray` copy here is not on the fast path this
        port exists to shorten. `raw_pack_into` writes ITS OWN tag byte at offset 0 (same
        framing `publish_record` expects, NOT `publish`'s "tag + payload" -- a second tag
        byte here would corrupt the record with a duplicate leading byte)."""
        buf = bytearray(raw_size(outputs))
        written = raw_pack_into(buf, outputs)
        assert written == len(buf), (written, len(buf))
        return self._kv.out_publish_record(bytes(buf))


def _seed(input_address: str) -> str:
    """Per-boot service-name seed both processes derive independently.

    Rust hashes the address it bound (``ConnectedTransport::input_address``); Python hashes
    the one it was handed in the INIT message (``addresses.inputs[0]``) -- the same string.
    """
    import hashlib

    return hashlib.sha256(input_address.encode()).hexdigest()[:12]


def request_service(input_address: str, engine_index: int) -> str:
    return f"vtl/req/{_seed(input_address)}/{engine_index}"


def output_service(input_address: str) -> str:
    return f"vtl/out/{_seed(input_address)}"


# ------------------------------------------------------------------------------------------
# Phase B: fixed-layout records. Layout spec (little-endian, unaligned, tag byte at offset 0):
#
#   +0   u8   tag (TAG_RAW)
#   +1   u8   version (RAW_VERSION)
#   +2   u16  reserved, 0
#   +4   u32  engine_index
#   +8   f64  timestamp
#   +16  u32  num_outputs
#   +20  u32  num_finished
#   then num_outputs records:
#       +0   u32  request_id_len
#       +4   u32  num_new_token_ids
#       +8   u32  num_nans_in_logits
#       +12  u8   flags (bit0 finish_reason present, bit1 stop_reason token id present)
#       +13  u8   finish_reason
#       +14  u16  reserved, 0
#       +16  u32  stop_reason token id
#       +20  request_id utf-8
#       +..  num_new_token_ids * u32
#   then num_finished entries: u32 length + utf-8 bytes.
#
# ``raw_pack`` returns None whenever the batch carries anything the layout cannot express, and
# the caller then emits a msgpack record instead. Keeping the reject list explicit (rather
# than "encode what we can") is what makes the Rust decoder total.
# ------------------------------------------------------------------------------------------


# msgspec struct fields as of the ported v0.25.0 (``vllm/v1/engine/__init__.py``). The raw
# layout is a *reject list* (``raw_packable``), so a field ADDED by a vLLM bump would be
# silently dropped rather than falling back to msgpack. ``apply()`` diffs these against the
# live structs and forces raw off on any drift -- fail closed. Set by ``apply()``.
_RAW_ENABLED = True
# Set by apply() once the output thread has actually been replaced AND the raw layout is
# live. `rust_sched.py`'s R8 path reads it and refuses to emit bytes onto the output queue
# if it is False -- stock's `process_output_sockets` would do `outputs.engine_index = ...`
# on a `bytes` and take EngineCore down. Fail closed: the env gates alone are not proof
# that this module installed (VTL_ENABLE_SHM_IPC=0 disables the registry entry while
# VTL_SHM_IPC=1 is still set).
RAW_OUTPUT_PATH = False
KNOWN_OUTPUT_FIELDS = frozenset(
    {
        "request_id",
        "new_token_ids",
        "new_logprobs",
        "new_prompt_logprobs_tensors",
        "pooling_output",
        "finish_reason",
        "stop_reason",
        "events",
        "kv_transfer_params",
        "trace_headers",
        "prefill_stats",
        "routed_experts",
        "num_nans_in_logits",
    }
)
KNOWN_OUTPUTS_FIELDS = frozenset(
    {
        "engine_index",
        "outputs",
        "scheduler_stats",
        "timestamp",
        "utility_output",
        "finished_requests",
        "wave_complete",
        "start_wave",
    }
)


def raw_schema_ok(output_cls, outputs_cls) -> bool:
    """True iff both msgspec structs still carry exactly the fields the raw layout knows."""
    ok = True
    for cls, known in ((output_cls, KNOWN_OUTPUT_FIELDS), (outputs_cls, KNOWN_OUTPUTS_FIELDS)):
        drift = frozenset(cls.__struct_fields__) ^ known
        if drift:
            log.warning(
                "vtl shm_ipc: %s fields drifted from the ported layout (%s), forcing raw off",
                cls.__name__,
                sorted(drift),
            )
            ok = False
    return ok


def raw_packable(outputs) -> bool:
    """True iff every field of ``outputs`` survives the fixed layout."""
    if (
        outputs.scheduler_stats is not None
        or outputs.utility_output is not None
        or outputs.wave_complete is not None
        or outputs.start_wave is not None
    ):
        return False
    for out in outputs.outputs:
        if (
            out.new_logprobs is not None
            or out.new_prompt_logprobs_tensors is not None
            or out.pooling_output is not None
            or out.events is not None
            or out.kv_transfer_params is not None
            or out.trace_headers is not None
            or out.prefill_stats is not None
            or out.routed_experts is not None
        ):
            return False
        # A *string* stop_reason has no fixed-width slot; only a token id does.
        if out.stop_reason is not None and not isinstance(out.stop_reason, int):
            return False
    return True


def raw_plan(outputs):
    """``(byte length, encoded request ids, encoded finished ids)``.

    Every request id is utf-8 encoded exactly ONCE here and handed to ``raw_pack_into``;
    sizing and packing used to encode each id separately.
    """
    ids = [out.request_id.encode() for out in outputs.outputs]
    finished = [
        req_id.encode() for req_id in sorted(outputs.finished_requests or ())
    ]
    total = 1 + _RAW_HEADER.size
    for out, req_id in zip(outputs.outputs, ids):
        total += _RAW_OUTPUT_HEAD.size + len(req_id) + 4 * len(out.new_token_ids)
    for req_id in finished:
        total += 4 + len(req_id)
    return total, ids, finished


def raw_size(outputs) -> int:
    """Exact byte length of the record ``raw_pack_into`` would write."""
    return raw_plan(outputs)[0]


def raw_pack_into(buffer, outputs, plan=None) -> int:
    """Pack ``outputs`` into ``buffer`` (any writable buffer). Returns bytes written."""
    _, ids, finished = plan if plan is not None else raw_plan(outputs)
    buffer[0:1] = TAG_RAW
    _RAW_HEADER.pack_into(
        buffer,
        1,
        RAW_VERSION,
        0,
        outputs.engine_index,
        outputs.timestamp,
        len(outputs.outputs),
        len(finished),
    )
    at = 1 + _RAW_HEADER.size
    for out, req_id in zip(outputs.outputs, ids):
        flags = 0
        finish = 0
        if out.finish_reason is not None:
            flags |= FLAG_FINISH_REASON
            finish = int(out.finish_reason)
        stop = 0
        if out.stop_reason is not None:
            flags |= FLAG_STOP_TOKEN
            stop = int(out.stop_reason)
        _RAW_OUTPUT_HEAD.pack_into(
            buffer,
            at,
            len(req_id),
            len(out.new_token_ids),
            out.num_nans_in_logits,
            flags,
            finish,
            0,
            stop,
        )
        at += _RAW_OUTPUT_HEAD.size
        buffer[at : at + len(req_id)] = req_id
        at += len(req_id)
        for token in out.new_token_ids:
            _U32.pack_into(buffer, at, token)
            at += 4
    for req_id in finished:
        _U32.pack_into(buffer, at, len(req_id))
        at += 4
        buffer[at : at + len(req_id)] = req_id
        at += len(req_id)
    return at


def raw_unpack(record):
    """Inverse of ``raw_pack_into``: rebuild ``EngineCoreOutputs`` from a TAG_RAW record.

    R8 (``VTL_RUST_SCHED_R8``) has Rust emit these bytes directly, so the EngineCore no
    longer holds the objects at all. The shm publish takes the bytes as they are -- but the
    ZMQ arm still needs real objects, and the shm->ZMQ demotion is a PERMANENT one-way latch
    whose *tripping* batch must still be delivered. Decoding here removes that race
    entirely: whichever transport wins, the same record is what gets sent.

    Cold path by construction (it runs at most once per boot, on the demotion), so clarity
    beats speed. Raises on any malformed record -- the caller treats that as "drop shm".
    """
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

    mv = memoryview(record)
    if bytes(mv[0:1]) != TAG_RAW:
        raise ValueError(f"not a raw record: tag {bytes(mv[0:1])!r}")
    version, _reserved, engine_index, timestamp, n_out, n_fin = _RAW_HEADER.unpack_from(mv, 1)
    if version != RAW_VERSION:
        raise ValueError(f"raw record version {version} != {RAW_VERSION}")
    at = 1 + _RAW_HEADER.size
    outputs = []
    for _ in range(n_out):
        id_len, n_tok, n_nans, flags, finish, _r, stop = _RAW_OUTPUT_HEAD.unpack_from(mv, at)
        at += _RAW_OUTPUT_HEAD.size
        request_id = bytes(mv[at : at + id_len]).decode()
        at += id_len
        token_ids = list(struct.unpack_from(f"<{n_tok}I", mv, at)) if n_tok else []
        at += 4 * n_tok
        outputs.append(
            EngineCoreOutput(
                request_id=request_id,
                new_token_ids=token_ids,
                finish_reason=(
                    FinishReason(finish) if flags & FLAG_FINISH_REASON else None
                ),
                stop_reason=stop if flags & FLAG_STOP_TOKEN else None,
                num_nans_in_logits=n_nans,
            )
        )
    finished = set()
    for _ in range(n_fin):
        (id_len,) = _U32.unpack_from(mv, at)
        at += 4
        finished.add(bytes(mv[at : at + id_len]).decode())
        at += id_len
    if at != len(record):
        raise ValueError(f"raw record has {len(record) - at} trailing bytes")
    return EngineCoreOutputs(
        engine_index=engine_index,
        outputs=outputs,
        timestamp=timestamp,
        finished_requests=finished or None,
    )


# ------------------------------------------------------------------------------------------
# iceoryx2 plumbing
# ------------------------------------------------------------------------------------------


# One ctypes array type for every shm view. ``ctypes.c_ubyte * n`` mints -- and permanently
# caches -- a distinct type per distinct ``n``, so building one per record size leaks types
# for the process' lifetime. Nothing is read or written past the slice taken in ``_shm_view``,
# so a nominal length larger than the loan is harmless (``from_address`` touches no memory).
_VIEW_MAX = 1 << 24
_ShmBytes = ctypes.c_ubyte * _VIEW_MAX


def _shm_view(address: int, size: int) -> memoryview:
    """Writable ``memoryview`` of exactly ``size`` bytes at ``address``."""
    assert 0 < size <= _VIEW_MAX, size
    return memoryview(_ShmBytes.from_address(address)).cast("B")[:size]


def _address(buffer) -> int:
    """Address of a writable buffer. ``ctypes.memmove`` takes ints or ``bytes`` but not a
    ``bytearray``/``memoryview``, and ``c_char`` is a fixed type (no cache growth)."""
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def _open_channel(node, name: str):
    """Open (or create) the byte-slice pub/sub service plus its companion event service."""
    import iceoryx2 as ix

    data = (
        node.service_builder(ix.ServiceName.new(name))
        .publish_subscribe(ix.Slice[ctypes.c_uint8])
        .max_publishers(MAX_PUBLISHERS)
        .max_subscribers(MAX_SUBSCRIBERS)
        .subscriber_max_buffer_size(SUBSCRIBER_MAX_BUFFER_SIZE)
        .history_size(HISTORY_SIZE)
        .enable_safe_overflow(ENABLE_SAFE_OVERFLOW)
        .open_or_create()
    )
    event = (
        node.service_builder(ix.ServiceName.new(f"{name}/ev"))
        .event()
        .max_notifiers(MAX_PUBLISHERS)
        .max_listeners(MAX_SUBSCRIBERS)
        .open_or_create()
    )
    return data, event


class _OutputPublisher:
    """Publisher for the engine -> frontend channel."""

    def __init__(self, input_address: str):
        import iceoryx2 as ix

        # The node owns this process' iceoryx2 registration; dropping it would tear the
        # services down under the ports, so it is held for the publisher's lifetime.
        self._node = ix.NodeBuilder.new().create(ix.ServiceType.Ipc)
        name = output_service(input_address)
        data, event = _open_channel(self._node, name)
        self._publisher = (
            data.publisher_builder()
            .initial_max_slice_len(OUTPUT_MAX_PAYLOAD)
            .allocation_strategy(ix.AllocationStrategy.PowerOfTwo)
            # With safe overflow off, this is what a full subscriber buffer does: report
            # (send() -> 0 receivers) instead of blocking the engine's output thread.
            .backpressure_strategy(ix.BackpressureStrategy.DiscardData)
            .create()
        )
        self._notifier = event.notifier_builder().create()
        log.info("vtl shm_ipc: output channel ready (%s)", name)

    def publish(self, tag: bytes, payload) -> bool:
        """Publish ``tag + payload``. False means undelivered (no subscriber, or a full
        buffer) -> use ZMQ. ``payload`` is any buffer, so the caller can reuse one."""
        size = len(payload)
        sample = self._publisher.loan_slice_uninit(size + 1)
        ctypes.memmove(sample.payload_ptr, tag, 1)
        if size:
            src = payload if isinstance(payload, bytes) else _address(payload)
            ctypes.memmove(sample.payload_ptr + 1, src, size)
        return self._send(sample)

    def publish_record(self, record) -> bool:
        """Publish an ALREADY-FRAMED record (R8: Rust packed it, tag byte included)."""
        size = len(record)
        sample = self._publisher.loan_slice_uninit(size)
        ctypes.memmove(sample.payload_ptr, record, size)
        return self._send(sample)

    def publish_raw(self, outputs) -> bool:
        """Pack ``outputs`` straight into the loaned shm buffer -- no msgpack, no bytes copy."""
        plan = raw_plan(outputs)
        size = plan[0]
        sample = self._publisher.loan_slice_uninit(size)
        written = raw_pack_into(_shm_view(sample.payload_ptr, size), outputs, plan)
        assert written == size, (written, size)
        return self._send(sample)

    def _send(self, sample) -> bool:
        if sample.assume_init().send() == 0:
            return False
        self._notifier.notify()
        return True


def _run_input_thread(engine, input_address: str, engine_index: int, ready: threading.Event):
    """Mirror of ``core.py`` ``process_input_sockets``' recv loop over the shm channel.

    Decoders are built exactly as upstream (core.py:1494-1497) and every decoded request goes
    through the *same* methods, so ADD preprocessing errors and the ABORT dual-queue put keep
    their upstream semantics.
    """
    import iceoryx2 as ix

    from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
    from vllm.v1.serial_utils import MsgpackDecoder

    # upstream: core.py:1494-1497
    add_request_decoder = MsgpackDecoder(
        EngineCoreRequest, oob_tensor_provider=engine.tensor_ipc_receiver
    )
    generic_decoder = MsgpackDecoder(oob_tensor_provider=engine.tensor_ipc_receiver)

    node = ix.NodeBuilder.new().create(ix.ServiceType.Ipc)
    name = request_service(input_address, engine_index)
    data, event = _open_channel(node, name)
    subscriber = data.subscriber_builder().create()
    listener = event.listener_builder().create()
    log.info("vtl shm_ipc: request channel ready (%s)", name)
    ready.set()

    # A fatal error here drops the subscriber, which the frontend notices on its very next
    # publish (0 receivers) and answers by demoting itself to ZMQ for good -- so the engine
    # keeps taking requests. Log it loudly all the same; a silent stall would be the one
    # failure mode nothing else catches.
    try:
        while True:
            # Parks the thread on the iceoryx2 event primitive (releases the GIL) -- no
            # polling, no spin. Critical on the 3-vCPU box.
            listener.blocking_wait_one()
            while (sample := subscriber.receive()) is not None:
                payload = sample.payload()
                # ONE copy out of shm. The msgpack decoder is share_mem=True, so a decoded
                # ndarray/tensor would BORROW whatever buffer it is handed -- handing it a
                # view of the shm sample (which is recycled at the end of this iteration)
                # would be a use-after-free. `record` is Python-owned, and the memoryview
                # slice keeps it alive for any such borrow while avoiding the second
                # whole-payload copy that `record[1:]` used to make.
                record = ctypes.string_at(payload.as_ptr(), payload.len())
                try:
                    # upstream: core.py:1558-1587, frame split replaced by our framing.
                    request_type = EngineCoreRequestType(record[:1])
                    data_frames = [memoryview(record)[1:]]
                    if request_type == EngineCoreRequestType.ADD:
                        req = add_request_decoder.decode(data_frames)
                        try:
                            request = engine.preprocess_add_request(req)
                        except Exception:
                            engine._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)
                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts go to *both* queues, exactly as upstream: eager
                            # processing plus input-queue ordering so no request leaks.
                            # Idempotent in the scheduler.
                            engine.aborts_queue.put_nowait(request)
                    engine.input_queue.put_nowait((request_type, request))
                except Exception:
                    # One bad record must not take the channel down with it -- upstream's
                    # process_input_sockets survives a malformed frame the same way, and
                    # killing the thread here would strand every record already confirmed
                    # delivered. Channel-level failures still break out below.
                    log.exception("vtl shm_ipc: dropping malformed request record")
    except Exception:
        log.exception("vtl shm_ipc: request thread died, frontend will fall back to ZMQ")


def _process_output_sockets(self, output_paths, coord_output_path, engine_index):
    """Copy of v0.25.0 ``EngineCoreProc.process_output_sockets`` (core.py:1589-1654).

    The ONLY transport change is the ``sockets[client_index].send_multipart(...)`` line, which
    becomes an iceoryx2 publish; everything else (buffer reuse, the msgpack tracker bookkeeping,
    the coordinator socket, the linger settings, the DEAD sentinel) is upstream verbatim so a
    fallback is always byte-identical to stock.
    """
    import zmq

    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.serial_utils import MsgpackEncoder

    encoder = MsgpackEncoder()
    reuse_buffers: list[bytearray] = []
    pending = deque()

    shm = None
    shm_delivered = False  # shm has carried at least one batch
    shm_buffer = bytearray()  # reused across publishes, like upstream's reuse_buffers
    use_raw = False
    try:
        # Batch 3: MAX_PUBLISHERS=1 means the crate publisher and `_OutputPublisher` are
        # mutually exclusive -- try the crate first (it may already be registered from a
        # prior boot's stale reference, hence `out_open`'s own idempotence), and build the
        # Python one only if that is not live. `_RUST_KV.kv` is `None` whenever
        # `rust_sched.py` never registered (gate off, shadow-only mode, or authority mode
        # never installed) -- same fail-open shape as everything else in this module.
        rust_kv = _RUST_KV.kv if _env_flag("VTL_SHM_IPC_RUST_PUB") else None
        if rust_kv is not None and rust_kv.out_open(self.addresses.inputs[0]):
            shm = _RustOutputAdapter(rust_kv)
            log.info("vtl shm_ipc: output channel owned by the rust crate (VTL_SHM_IPC_RUST_PUB)")
        else:
            shm = _OutputPublisher(self.addresses.inputs[0])
        use_raw = _env_flag("VTL_SHM_IPC_RAW") and _RAW_ENABLED
    except Exception:
        log.exception("vtl shm_ipc: output channel setup failed, staying on ZMQ")

    with ExitStack() as stack, zmq.Context() as ctx:
        sockets = [
            stack.enter_context(make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
            for output_path in output_paths
        ]
        coord_socket = (
            stack.enter_context(
                make_zmq_socket(ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000)
            )
            if coord_output_path is not None
            else None
        )
        max_reuse_bufs = len(sockets) + 1

        while True:
            output = self.output_queue.get()
            if output == EngineCoreProc.ENGINE_CORE_DEAD:
                # The ZMQ sentinel is untouched -- it is what the frontend's still-running
                # ZMQ output loop watches. The shm DEAD record is additive so a frontend
                # parked on the shm listener does not wait for the socket to notice.
                if shm is not None:
                    try:
                        shm.publish(TAG_DEAD, b"")
                    except Exception:
                        log.exception("vtl shm_ipc: DEAD record publish failed")
                for socket in sockets:
                    socket.send(output)
                break
            assert not isinstance(output, bytes)
            client_index, outputs = output
            if type(outputs) is bytes:
                # R8: the scheduler handed us a finished TAG_RAW record instead of objects
                # (rust_sched.py's `_install_r8`). Publish the bytes as they are; the
                # engine_index and timestamp are already baked in by the packer.
                if shm is not None and client_index == 0 and len(sockets) == 1:
                    try:
                        if shm.publish_record(outputs):
                            shm_delivered = True
                            PUB.published += 1
                            continue
                    except Exception:
                        log.exception("vtl shm_ipc: R8 publish failed, ZMQ from here on")
                        shm = None
                    if shm is not None and shm_delivered:
                        log.warning(
                            "vtl shm_ipc: output publish undelivered after a live stream, "
                            "falling back to ZMQ for good"
                        )
                        shm = None
                # ZMQ arm: msgpack needs real objects back. Decoding costs one cold call
                # per boot (the demotion latch below makes it the last shm attempt), and
                # it is what lets the TRIPPING batch still be delivered.
                outputs = raw_unpack(outputs)
            outputs.engine_index = engine_index

            if client_index == -1:
                assert coord_socket is not None
                coord_socket.send_multipart(encoder.encode(outputs))
                continue

            # Reclaim buffers that zmq is finished with.
            while pending and pending[-1][0].done:
                reuse_buffers.append(pending.pop()[2])

            if shm is not None and client_index == 0 and len(sockets) == 1:
                sent = False
                try:
                    if use_raw and raw_packable(outputs):
                        sent = shm.publish_raw(outputs)
                    else:
                        # encode_into reuses one buffer, exactly like the ZMQ path below;
                        # encoder.encode() allocated a fresh one per decode step.
                        buffers = encoder.encode_into(outputs, shm_buffer)
                        # ponytail: multi-frame batches cannot go over shm. That is the zmq
                        # tensor zero-copy path (out-of-band logprob/pooling tensors); the
                        # served config -- greedy, no logprobs, text-only decode -- never
                        # produces one. UPGRADE PATH: publish the frames as one shm record
                        # with a [u32 count][u32 len...] table and hand the slices to Rust's
                        # decode_engine_core_outputs, which already takes a frame list.
                        sent = len(buffers) == 1 and shm.publish(TAG_MSGPACK, buffers[0])
                    if sent:
                        shm_delivered = True
                        PUB.published += 1
                        continue
                except Exception:
                    log.exception("vtl shm_ipc: publish failed, ZMQ from here on")
                    shm = None
                if shm is not None and shm_delivered:
                    # Latch, mirroring the Rust input side's `live: AtomicBool`: once shm has
                    # carried a batch, the first undelivered one demotes the transport FOR
                    # GOOD. Retrying per message would interleave ZMQ and shm, and a record
                    # published-but-unread (e.g. the Rust output thread died) would be lost
                    # behind later ZMQ traffic -- reordering plus silent loss.
                    log.warning(
                        "vtl shm_ipc: output publish undelivered after a live stream, "
                        "falling back to ZMQ for good"
                    )
                    shm = None

            buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
            buffers = encoder.encode_into(outputs, buffer)
            tracker = sockets[client_index].send_multipart(buffers, copy=False, track=True)
            if not tracker.done:
                ref = outputs if len(buffers) > 1 else None
                pending.appendleft((tracker, ref, buffer))
            elif len(reuse_buffers) < max_reuse_bufs:
                reuse_buffers.append(buffer)
            # Batch 3 ordering guard: this is the ZMQ fallback for the same client_index==0
            # stream `PUB.enqueued` counts (the coord (-1) branch above returns before this
            # point and is deliberately not counted -- `rust_sched.py` never enqueues those).
            PUB.published += 1


@register_patch("shm_ipc", default=False)
def apply() -> None:
    if not _env_flag("VTL_SHM_IPC"):
        log.info("vtl: shm_ipc selected but VTL_SHM_IPC is unset, leaving ZMQ in place")
        return

    import iceoryx2  # noqa: F401  -- fail the patch (not the server) if the wheel is absent

    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.engine.core import EngineCoreProc

    global _RAW_ENABLED
    _RAW_ENABLED = raw_schema_ok(EngineCoreOutput, EngineCoreOutputs)

    if already_patched(EngineCoreProc, "process_input_sockets"):
        return

    original_input = EngineCoreProc.process_input_sockets

    def process_input_sockets(self, input_addresses, coord_input_address, identity, ready_event):
        # Subscribe BEFORE delegating: the stock body's first act is to send ready_payload,
        # which releases the frontend's registration wait and lets it build its publisher.
        subscribed = threading.Event()
        thread = threading.Thread(
            target=_run_input_thread,
            args=(self, input_addresses[0], self.engine_index, subscribed),
            name="vtl-shm-in",
            daemon=True,
        )
        thread.start()
        if not subscribed.wait(timeout=30):
            log.error("vtl shm_ipc: request channel did not come up, ZMQ input only")
        return original_input(self, input_addresses, coord_input_address, identity, ready_event)

    EngineCoreProc.process_input_sockets = mark_patched(process_input_sockets, original_input)
    EngineCoreProc.process_output_sockets = mark_patched(
        _process_output_sockets, EngineCoreProc.process_output_sockets
    )
    global RAW_OUTPUT_PATH
    RAW_OUTPUT_PATH = bool(_env_flag("VTL_SHM_IPC_RAW") and _RAW_ENABLED)
    log.info(
        "vtl: shm_ipc installed (iceoryx2 data plane, raw=%s)", RAW_OUTPUT_PATH
    )


# ------------------------------------------------------------------------------------------
# self-checks
# ------------------------------------------------------------------------------------------


class _Out:
    """Stand-in for ``vllm.v1.engine.EngineCoreOutput`` -- the packer only reads attributes."""

    def __init__(self, request_id, new_token_ids, **kw):
        self.request_id = request_id
        self.new_token_ids = new_token_ids
        self.new_logprobs = None
        self.new_prompt_logprobs_tensors = None
        self.pooling_output = None
        self.finish_reason = None
        self.stop_reason = None
        self.events = None
        self.kv_transfer_params = None
        self.trace_headers = None
        self.prefill_stats = None
        self.routed_experts = None
        self.num_nans_in_logits = 0
        self.__dict__.update(kw)


class _Outs:
    """Stand-in for ``vllm.v1.engine.EngineCoreOutputs``."""

    def __init__(self, **kw):
        self.engine_index = 0
        self.outputs = []
        self.scheduler_stats = None
        self.timestamp = 0.0
        self.utility_output = None
        self.finished_requests = None
        self.wave_complete = None
        self.start_wave = None
        self.__dict__.update(kw)


def _pack(outputs) -> bytes:
    buffer = bytearray(raw_size(outputs))
    written = raw_pack_into(buffer, outputs)
    assert written == len(buffer), (written, len(buffer))
    return bytes(buffer)


# Golden cross-language vectors. The identical hex strings are asserted by the Rust unit test
# in vtl/vllm_patches/rust-frontend/shm_ipc.patch (`mod tests` in shm_ipc.rs); if either side
# changes the layout, both tests fail. Note the Rust decoder is handed the record WITHOUT the
# leading tag byte, so its vectors are these minus the leading "52".
GOLDEN_PLAIN_DECODE = (
    "52"
    "01" "0000" "07000000" "000000000000f03f" "01000000" "00000000"
    "06000000" "01000000" "00000000" "00" "00" "0000" "00000000"
    "7265712d3161" "05000000"
)
# The service-name seed is implemented twice (here and in Rust `seed()`); this one literal is
# asserted by both, so a divergence in the hash or the truncation fails a test instead of
# quietly giving the two processes different service names (= a silent ZMQ-only boot).
GOLDEN_SEED_ADDRESS = "ipc:///tmp/vllm-in-0"
GOLDEN_SEED = "23c5030baa20"
GOLDEN_FINISHED = (
    "52"
    "01" "0000" "00000000" "0000000000000000" "02000000" "02000000"
    "05000000" "02000000" "00000000" "03" "01" "0000" "02000000"
    "7265712d61" "03000000" "04000000"
    "05000000" "00000000" "00000000" "01" "02" "0000" "00000000"
    "7265712d62"
    "05000000" "7265712d61"
    "05000000" "7265712d62"
)


def _self_check() -> None:
    # --- pure-struct parts: no iceoryx2, no vLLM, no GPU ----------------------------------
    plain = _Outs(engine_index=7, timestamp=1.0, outputs=[_Out("req-1a", [5])])
    assert raw_packable(plain)
    packed = _pack(plain)
    assert packed[:1] == TAG_RAW
    assert packed.hex() == GOLDEN_PLAIN_DECODE, packed.hex()

    finished = _Outs(
        outputs=[
            _Out("req-a", [3, 4], finish_reason=1, stop_reason=2),
            _Out("req-b", [], finish_reason=2),
        ],
        finished_requests={"req-b", "req-a"},
    )
    assert raw_packable(finished)
    packed = _pack(finished)
    assert packed.hex() == GOLDEN_FINISHED, packed.hex()

    # finished_requests is a set; the packer must sort so the bytes are deterministic.
    reordered = _Outs(
        outputs=finished.outputs, finished_requests={"req-a", "req-b"}
    )
    assert _pack(reordered) == packed

    # size accounting is exact for every shape (that is what makes the shm loan safe)
    for case in (plain, finished, _Outs(), _Outs(outputs=[_Out("x", list(range(64)))])):
        assert len(_pack(case)) == raw_size(case)

    # every non-fixed-layout field forces the msgpack fallback
    assert not raw_packable(_Outs(scheduler_stats=object()))
    assert not raw_packable(_Outs(utility_output=object()))
    assert not raw_packable(_Outs(wave_complete=1))
    assert not raw_packable(_Outs(start_wave=1))
    for field in (
        "new_logprobs",
        "new_prompt_logprobs_tensors",
        "pooling_output",
        "events",
        "kv_transfer_params",
        "trace_headers",
        "prefill_stats",
        "routed_experts",
    ):
        assert not raw_packable(_Outs(outputs=[_Out("r", [1], **{field: object()})])), field
    assert not raw_packable(_Outs(outputs=[_Out("r", [1], stop_reason="</s>")]))
    assert raw_packable(_Outs(outputs=[_Out("r", [1], stop_reason=7)]))

    # utf-8 request ids are length-prefixed in BYTES, not code points
    unicode_case = _Outs(outputs=[_Out("req-é", [1])])
    assert len(_pack(unicode_case)) == raw_size(unicode_case)
    assert "req-é".encode() in _pack(unicode_case)

    # --- R8 round trip (pack -> unpack) ----------------------------------------------------
    # raw_unpack needs the real msgspec structs; off-box `make check` has no vLLM, so the
    # round trip runs only where the import lands (the image, and `make test-kernel`).
    try:
        from vllm.v1.engine import EngineCoreOutputs  # noqa: F401
    except Exception:
        print("shm_ipc: raw_unpack round trip skipped (no vLLM in this interpreter)")
    else:
        for case in (plain, finished, unicode_case):
            back = raw_unpack(_pack(case))
            assert back.engine_index == case.engine_index
            assert back.timestamp == case.timestamp
            assert len(back.outputs) == len(case.outputs)
            for got, want in zip(back.outputs, case.outputs):
                assert got.request_id == want.request_id
                assert got.new_token_ids == list(want.new_token_ids)
                assert (got.finish_reason is None) == (want.finish_reason is None)
                assert got.stop_reason == want.stop_reason
                assert got.num_nans_in_logits == want.num_nans_in_logits
            assert (back.finished_requests or set()) == set(case.finished_requests or ())
        # A truncated / mistagged record must raise, not decode into garbage.
        good = _pack(plain)
        for bad in (good[:-1], good + b"\x00", b"M" + good[1:]):
            try:
                raw_unpack(bad)
            except (ValueError, struct.error):
                pass
            else:
                raise AssertionError(f"raw_unpack accepted a malformed record: {bad.hex()}")

    # --- Batch 3: crate-owned publisher plumbing (fakes only, no live iceoryx2) -----------
    class _FakeKv:
        """Duck-types the three crate FFIs `_RustOutputAdapter` calls, with a `deliver`
        knob so both the delivered and undelivered arms are exercised."""

        def __init__(self, deliver=True):
            self.deliver = deliver
            self.opened = None
            self.publishes: list = []

        def out_open(self, input_address):
            self.opened = input_address
            return True

        def out_publish(self, tag, payload):
            self.publishes.append((tag, bytes(payload)))
            return self.deliver

        def out_publish_record(self, record):
            self.publishes.append((None, bytes(record)))
            return self.deliver

    adapter = _RustOutputAdapter(_FakeKv())
    assert adapter.publish(TAG_MSGPACK, b"\x90")
    assert adapter._kv.publishes == [(TAG_MSGPACK[0], b"\x90")]
    assert adapter.publish_record(b"Rhello")
    assert adapter._kv.publishes[-1] == (None, b"Rhello")

    undelivered = _RustOutputAdapter(_FakeKv(deliver=False))
    assert not undelivered.publish_record(b"Rgone")

    outs = _Outs(engine_index=0, timestamp=0.0, outputs=[_Out("req-1a", [5])])
    raw_adapter = _RustOutputAdapter(_FakeKv())
    assert raw_adapter.publish_raw(outs)
    # `publish_record` framing (no separate tag arg): the packed bytes already start with
    # TAG_RAW at offset 0, so a second tag byte must NOT be prepended -- see the docstring.
    assert raw_adapter._kv.publishes[-1] == (None, _pack(outs))

    # `register_rust_kv` / `_RUST_KV`: last write wins, visible without a `global` at the
    # reader (mirrors the cross-module handshake `_process_output_sockets` relies on).
    assert _RUST_KV.kv is None
    sentinel = object()
    register_rust_kv(sentinel)
    assert _RUST_KV.kv is sentinel
    register_rust_kv(None)  # leave the module state clean for any test run after this one

    # `_PubCounters`: the ordering guard's whole contract is this equality check.
    pc = _PubCounters()
    assert pc.enqueued == pc.published == 0
    pc.enqueued += 1
    assert pc.enqueued != pc.published
    pc.published += 1
    assert pc.enqueued == pc.published

    # --- schema drift gate ----------------------------------------------------------------
    class _StructOut:
        __struct_fields__ = tuple(KNOWN_OUTPUT_FIELDS)

    class _StructOuts:
        __struct_fields__ = tuple(KNOWN_OUTPUTS_FIELDS)

    assert raw_schema_ok(_StructOut, _StructOuts)

    # A field ADDED upstream is exactly the case the reject-list packer cannot see.
    class _DriftedOut:
        __struct_fields__ = (*KNOWN_OUTPUT_FIELDS, "brand_new_field")

    class _DriftedOuts:
        __struct_fields__ = tuple(KNOWN_OUTPUTS_FIELDS - {"start_wave"})

    log.disabled = True  # both cases warn by design; keep the self-check output clean
    assert not raw_schema_ok(_DriftedOut, _StructOuts)
    assert not raw_schema_ok(_StructOut, _DriftedOuts)
    log.disabled = False

    # --- service naming -------------------------------------------------------------------
    addr = GOLDEN_SEED_ADDRESS
    assert _seed(addr) == GOLDEN_SEED, _seed(addr)  # shared with the Rust test
    assert request_service(addr, 0) == request_service(addr, 0)
    assert request_service(addr, 0) != request_service(addr, 1)
    assert request_service(addr, 0) != request_service("ipc:///tmp/other", 0)
    assert request_service(addr, 0).startswith("vtl/req/")
    assert output_service(addr).startswith("vtl/out/")
    assert len(_seed(addr)) == 12

    # --- env gating -----------------------------------------------------------------------
    saved = os.environ.pop("VTL_SHM_IPC", None)
    try:
        assert not _env_flag("VTL_SHM_IPC")
        os.environ["VTL_SHM_IPC"] = "on"
        assert _env_flag("VTL_SHM_IPC")
        os.environ["VTL_SHM_IPC"] = "0"
        assert not _env_flag("VTL_SHM_IPC")
    finally:
        os.environ.pop("VTL_SHM_IPC", None)
        if saved is not None:
            os.environ["VTL_SHM_IPC"] = saved

    # --- live shm parts: skipped wherever iceoryx2 is not installed (e.g. the dev Mac) -----
    try:
        import iceoryx2 as ix
    except ImportError:
        print("shm_ipc self-check ok (iceoryx2 absent: live-shm parts skipped)")
        return

    node = ix.NodeBuilder.new().create(ix.ServiceType.Ipc)
    name = f"vtl/selfcheck/{os.getpid()}"
    data, event = _open_channel(node, name)
    publisher = (
        data.publisher_builder()
        .initial_max_slice_len(OUTPUT_MAX_PAYLOAD)
        .allocation_strategy(ix.AllocationStrategy.PowerOfTwo)
        .backpressure_strategy(ix.BackpressureStrategy.DiscardData)
        .create()
    )
    subscriber = data.subscriber_builder().create()
    notifier = event.notifier_builder().create()
    listener = event.listener_builder().create()

    record = _pack(finished)
    sample = publisher.loan_slice_uninit(len(record))
    ctypes.memmove(sample.payload_ptr, record, len(record))
    assert sample.assume_init().send() == 1
    notifier.notify()
    assert listener.timed_wait_all(ix.Duration.from_secs(2))

    got = subscriber.receive()
    assert got is not None
    view = got.payload()
    assert ctypes.string_at(view.as_ptr(), view.len()) == record

    # A full buffer must be REPORTED (0 receivers -> ZMQ fallback), never silently overwrite
    # the oldest unread sample. This is the whole point of ENABLE_SAFE_OVERFLOW = False.
    del got
    for _ in range(SUBSCRIBER_MAX_BUFFER_SIZE):
        publisher.loan_slice_uninit(1).assume_init().send()
    assert publisher.loan_slice_uninit(1).assume_init().send() == 0, "buffer overflowed"
    print("shm_ipc self-check ok (live shm round-trip + no-overflow policy verified)")


def _integration() -> None:
    """In-container smoke test of the real transport. Needs the built image (iceoryx2 wheel).

        docker run --rm --entrypoint python3 <image> -m vtl.patches.shm_ipc --integration

    Verifies, in one process across two threads: publish -> notify -> park -> wake -> receive,
    for both a msgpack-tagged record and a Phase-B raw record, plus the DEAD record. The
    cross-*process* / cross-*language* path is exercised by booting the server with
    VTL_SHM_IPC=1 and watching for "vtl shm_ipc: request channel ready" from BOTH sides in the
    container log. Kill-engine behaviour is unchanged from stock: the ZMQ ENGINE_CORE_DEAD
    sentinel path in ``_process_output_sockets`` is upstream verbatim and the frontend's ZMQ
    output loop keeps running as the detector; the shm DEAD record is purely additive.
    """
    import iceoryx2 as ix

    # Drive the REAL publisher class, so `publish_raw`'s pack-straight-into-the-loaned-buffer
    # path (which the pure-Python self-check cannot reach) is covered too.
    address = f"ipc:///tmp/vtl-integration-{os.getpid()}"
    publisher = _OutputPublisher(address)

    node = ix.NodeBuilder.new().create(ix.ServiceType.Ipc)
    data, event = _open_channel(node, output_service(address))
    subscriber = data.subscriber_builder().create()
    listener = event.listener_builder().create()

    seen: list[bytes] = []
    done = threading.Event()

    def reader() -> None:
        while not done.is_set():
            # Same park-on-event shape as the frontend's Rust reader and the engine's shm
            # input thread: no polling loop, no spin.
            listener.blocking_wait_one()
            while (sample := subscriber.receive()) is not None:
                view = sample.payload()
                record = ctypes.string_at(view.as_ptr(), view.len())
                seen.append(record)
                if record[:1] == TAG_DEAD:
                    done.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    outputs = _Outs(engine_index=1, timestamp=2.5, outputs=[_Out("req-int", [11, 12])])
    assert publisher.publish(TAG_MSGPACK, b"\x90"), "reader is not subscribed"
    assert publisher.publish_raw(outputs), "reader is not subscribed"
    assert publisher.publish(TAG_DEAD, b""), "reader is not subscribed"

    assert done.wait(timeout=10), f"reader never saw the DEAD record (got {seen})"
    assert seen[0] == TAG_MSGPACK + b"\x90"
    assert seen[1] == _pack(outputs), (seen[1].hex(), _pack(outputs).hex())
    assert seen[2] == TAG_DEAD
    thread.join(timeout=5)  # the bench below reads the subscriber itself
    print(f"shm_ipc integration ok ({len(seen)} records round-tripped through shm)")

    # --- isolated transport microbench ----------------------------------------------------
    # Raw-pack + publish + wakeup + receive for one typical decode-step batch, with the reader
    # in this same thread so the number is the transport cost, not a scheduling artefact.
    # (Decoding the record back is the Rust side's job -- see the shm_ipc.rs unit tests.)
    import time

    reps = 10_000
    start = time.perf_counter()
    for _ in range(reps):
        assert publisher.publish_raw(outputs)
        listener.try_wait_all()  # drain the wakeup, as the real reader does
        sample = subscriber.receive()
        assert sample is not None
        view = sample.payload()
        assert view.len() == raw_size(outputs)
        del sample
    micros = (time.perf_counter() - start) * 1e6 / reps
    print(f"shm_ipc microbench: {micros:.2f} us/record (pack+publish+receive, n={reps})")


if __name__ == "__main__":
    import sys

    if "--integration" in sys.argv:
        _integration()
    else:
        _self_check()
