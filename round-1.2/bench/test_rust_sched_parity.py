"""WS4 parity harness: Rust `vtl_sched` vs vLLM v0.25.0 Python.

Three tiers, each degrading to a clear skip rather than a false pass:

  1. HASH PARITY (always runs, stdlib only). vLLM's block hash is
     ``sha256(pickle.dumps((parent, tuple(tokens), extra_keys), protocol=5)).digest()``
     chained over the prefix (`vllm/utils/hashing.py:26`, `v1/core/kv_cache_utils.py:577`
     and `:687`). The reference here is that two-line composition using stdlib `pickle`
     — a fixture transcribed from the source, not a re-implementation — plus the REAL
     vLLM functions whenever `vllm` imports. A silent hash mismatch zeroes the 82.8%
     prefix-hit rate, so this tier compares the pickle BYTE STREAM as well as the digest.

  2. MANAGER PARITY (needs `vllm` importable). Drives a real `KVCacheManager` and the
     Rust `KvManager` through the same synthetic multi-turn workload and asserts
     identical block IDs per group, computed-token counts, free-block counts and usage.

  3. ORACLE PARITY (needs only the crate). Independent brute-force models, all runnable
     off-box where vLLM cannot be imported: (a) what the longest cached prefix must be,
     recomputed from the request history; (b) `FreeQueueOracle`, a direct transcription
     of vLLM's free queue, compared block ID by block ID under a pool small enough to
     recycle ~40x; (c) a replay of the RECORDED trace (`data/input/trace-round2.jsonl`)
     rather than only the synthetic fixture.

  4. WRAPPER PARITY (needs `vllm` importable). Tiers 2-3 drive `vtl_sched.KvManager`
     directly; this one drives the actual `VtlShadowKVCacheManager` /
     `VtlRustKVCacheManager` subclasses, which is the only place RustMirror, the
     BaseException guards, RustBlocks and `_refuse_unported` run.

Also `--bench`: ops/step microbench of the Rust manager (no Python side needed).

Off-box note: on a dev machine without torch, tier 2 SKIPS. It runs for real inside the
serving image, where `make test-kernel` picks this file up via the `/bench/test_*.py` glob.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import sys
import time

import pytest

# In the serving image vLLM is installed, so nothing extra is needed. On a dev box, point
# VTL_VLLM_SRC at a vLLM v0.25.0 source tree to make tier 2 attempt the real import
# (it still needs torch and the rest of vLLM's runtime deps).
VLLM_SRC = os.environ.get("VTL_VLLM_SRC", "")

try:
    import vtl_sched
except Exception as exc:  # pragma: no cover
    vtl_sched = None
    _CRATE_ERR = exc

requires_crate = pytest.mark.skipif(
    vtl_sched is None, reason="vtl_sched extension not built (maturin develop)"
)


# ---------------------------------------------------------------------------
# tier 1 -- hash parity
# ---------------------------------------------------------------------------


def ref_sha256(obj) -> bytes:
    """vllm/utils/hashing.py:26 `sha256`, verbatim."""
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).digest()


def ref_hash_block_tokens(none_hash, parent, tokens, extra_keys=None) -> bytes:
    """v1/core/kv_cache_utils.py:577 `hash_block_tokens`, verbatim."""
    if not parent:
        parent = none_hash
    return ref_sha256((parent, tuple(tokens), extra_keys))


def ref_block_hashes(none_hash, hash_block_size, token_ids, extra_keys=None):
    """v1/core/kv_cache_utils.py:687 `request_block_hasher` for the no-MM case."""
    out = []
    prev = None
    start = 0
    while start + hash_block_size <= len(token_ids):
        h = ref_hash_block_tokens(
            none_hash, prev, token_ids[start : start + hash_block_size], extra_keys
        )
        out.append(h)
        prev = h
        start += hash_block_size
    return out


@requires_crate
@pytest.mark.parametrize("hash_block_size", [16, 32])
def test_hash_chain_matches_reference(hash_block_size):
    none_hash = vtl_sched.none_hash_from_seed("0")
    assert none_hash == ref_sha256("0"), "NONE_HASH derivation from PYTHONHASHSEED"

    rng = random.Random(1234)
    for length in (0, 1, 15, 16, 17, 64, 257):
        tokens = [rng.randrange(0, 65536) for _ in range(length)]
        assert vtl_sched.block_hashes(none_hash, hash_block_size, tokens) == ref_block_hashes(
            none_hash, hash_block_size, tokens
        ), f"length={length}"


@requires_crate
def test_pickle_byte_stream_matches_cpython():
    """Compare the exact bytes fed to SHA-256, not just the digest."""
    rng = random.Random(7)
    cases = [
        (b"\x00" * 32, [1, 2, 3], None),
        (b"\xaa" * 32, list(range(20)), None),
        (b"", [1], None),
        (b"\xbb" * 32, [300, 70000, 5_000_000_000, -3], None),
        (b"\xcc" * 32, [rng.randrange(0, 200000) for _ in range(64)], None),
        (b"\xdd" * 32, [1, 2], ("lora-name",)),
        (b"\xee" * 32, [1, 2], ("lora-name", ("mm-hash-abc", 0))),
        (b"\xff" * 32, [1, 2], (b"\x01\x02\x03",)),
        (b"\x11" * 32, [1, 2], ()),
    ]
    for parent, tokens, extra in cases:
        want = pickle.dumps((parent, tuple(tokens), extra), protocol=pickle.HIGHEST_PROTOCOL)
        got = vtl_sched.pickle_block_hash_input(parent, tokens, extra)
        assert got == want, f"pickle mismatch for {extra!r}"
        # hash_block_tokens substitutes NONE_HASH for a falsy parent (`if not
        # parent_block_hash`), so an EMPTY parent is not hashed literally.
        none_hash = b"\x00" * 32
        assert vtl_sched.hash_block_tokens(
            none_hash, parent, tokens, extra
        ) == ref_hash_block_tokens(none_hash, parent, tokens, extra)


@requires_crate
def test_token_ids_at_int_encoding_boundaries():
    """The BININT1/BININT2/BININT/LONG1 ladder is where a hand-written pickler breaks."""
    none_hash = b"\x5a" * 32
    boundaries = [0, 1, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**63 - 1]
    assert vtl_sched.block_hashes(none_hash, len(boundaries), boundaries) == ref_block_hashes(
        none_hash, len(boundaries), boundaries
    )


@requires_crate
def test_group_id_packing():
    h = b"\x77" * 32
    assert vtl_sched.block_hash_with_group_id(h, 3) == h + (3).to_bytes(4, "big")


@requires_crate
def test_hash_parity_against_real_vllm():
    """Same as tier 1 but against the real functions, when vLLM imports."""
    vllm = _try_import_vllm()
    if vllm is None:
        pytest.skip("vllm not importable here (needs torch); covered in-container")
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import hash_block_tokens

    none_hash = vtl_sched.none_hash_from_seed("0")
    assert none_hash == sha256("0")
    rng = random.Random(99)
    tokens = [rng.randrange(0, 65536) for _ in range(16)]
    want = hash_block_tokens(sha256, none_hash, tokens, None)
    assert vtl_sched.hash_block_tokens(none_hash, none_hash, tokens, None) == want


# ---------------------------------------------------------------------------
# synthetic workload -- mimics data/input/trace-round2.jsonl
# ---------------------------------------------------------------------------


def synth_conversations(num_convs=70, turns=6, seed=0):
    """70 conversations x 6 turns with a shared system preamble.

    Matches the recorded trace's shape (data/input/trace-round2.jsonl: 420 requests,
    multi-turn, heavy prefix reuse): every conversation shares one long system block, and
    turn N is a strict prefix extension of turn N-1. Measured reuse through the Rust
    manager is ~75%; the recorded trace measures 82.8% (its system preamble is longer
    relative to the turns). Close enough to exercise the same code paths — this is a
    parity fixture, not a performance model.
    """
    rng = random.Random(seed)
    system = [rng.randrange(1000, 60000) for _ in range(320)]
    convs = []
    for c in range(num_convs):
        tokens = list(system)
        conv = []
        for _ in range(turns):
            tokens = tokens + [rng.randrange(1000, 60000) for _ in range(rng.randrange(40, 160))]
            conv.append(list(tokens))
            tokens = tokens + [rng.randrange(1000, 60000) for _ in range(rng.randrange(24, 96))]
        convs.append(conv)
    # Interleave conversations so turns from different conversations compete, but keep
    # each conversation's turns in order — that is what the recorded trace does, and it
    # is what produces the measured ~82% prefix reuse.
    order = []
    for t in range(turns):
        round_ = [(c, t) for c in range(num_convs)]
        rng.shuffle(round_)
        order.extend(round_)
    return convs, order


def rust_config(num_blocks=4096, block_size=16, max_model_len=32768, radix=False):
    """The served LFM2.5-1.2B shape: one full-attention group + one mamba/align group."""
    return {
        "num_blocks": num_blocks,
        "enable_caching": True,
        "max_model_len": max_model_len,
        "scheduler_block_size": block_size,
        "hash_block_size": block_size,
        "log_stats": True,
        "watermark": 0.0,
        "radix": radix,
        "groups": [
            {
                "kind": "full",
                "block_size": block_size,
                "is_full_attention": True,
                "spec_signature": "FullAttentionSpec(16)",
                "mamba_align": False,
                "num_speculative_blocks": 0,
                "use_eagle": False,
            },
            {
                "kind": "mamba",
                "block_size": block_size,
                "is_full_attention": False,
                "spec_signature": "MambaSpec(16,align)",
                "mamba_align": True,
                "num_speculative_blocks": 0,
                "use_eagle": False,
            },
        ],
    }


def aligned_target(num_tokens, block_size):
    """What `_mamba_block_aligned_split` (scheduler.py:338) makes the prefill land on.

    A mamba state block only becomes a prefix-cache key when it sits on a block boundary,
    so the scheduler floors each prefill chunk to a block multiple. Replaying the raw
    prompt length instead would silently never populate the mamba group — the harness has
    to model this or it tests nothing.
    """
    return num_tokens // block_size * block_size


def drive_rust(kv, none_hash, block_size, convs, order, limit=None):
    """Replay the workload through the Rust manager; return per-request observations."""
    obs = []
    for n, (c, t) in enumerate(order):
        if limit is not None and n >= limit:
            break
        tokens = convs[c][t]
        rid = f"c{c}-t{t}"
        slot = kv.intern(rid)
        packed = b"".join(vtl_sched.block_hashes(none_hash, block_size, tokens))
        kv.push_hashes(slot, packed, len(tokens))
        kv.new_step_starts()
        hit = kv.get_computed_blocks(slot, len(tokens), 0, False)
        target = aligned_target(len(tokens), block_size)
        if target <= hit:
            obs.append((rid, hit, None, None, kv.num_free_blocks))
            kv.forget(rid)
            continue
        ok = kv.allocate_slots(slot, target - hit, hit, True, 0, 0, len(tokens), 0, False)
        ids = None
        if ok:
            ids = tuple(
                kv.buffer(g)[: kv.blocks_into_buffer(slot, g)].tolist()
                for g in range(kv.num_groups)
            )
        obs.append((rid, hit, ok, ids, kv.num_free_blocks))
        if ok:
            kv.free(slot)
        kv.forget(rid)
    return obs


# ---------------------------------------------------------------------------
# tier 3 -- independent oracle (no vLLM needed)
# ---------------------------------------------------------------------------


@requires_crate
def test_prefix_hit_matches_independent_oracle():
    """Recompute the expected hit from first principles and compare.

    The oracle knows nothing about vLLM's data structures: it tracks, per group, which
    prefix lengths have been cached, and derives what a hybrid full-attention + mamba
    lookup must return. Its job is to catch a Rust logic error off-box, where the real
    vLLM cannot be imported.
    """
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    kv = vtl_sched.KvManager(rust_config(num_blocks=1 << 15))
    convs, order = synth_conversations(num_convs=12, turns=4, seed=5)

    # Cached prefix-chain hashes per group (what cache_full_blocks would have inserted).
    cached_full: set[bytes] = set()
    cached_mamba: set[bytes] = set()
    live = []

    for c, t in order:
        tokens = convs[c][t]
        rid = f"c{c}-t{t}"
        slot = kv.intern(rid)
        hashes = vtl_sched.block_hashes(none_hash, block_size, tokens)
        kv.push_hashes(slot, b"".join(hashes), len(tokens))
        kv.new_step_starts()

        # Oracle: full attention matches the longest chain prefix present; mamba needs
        # exactly one cached state at or below that boundary; the hybrid fixed point is
        # the mamba answer (full attention gets truncated to it), capped at
        # num_tokens - 1.
        max_blocks = (len(tokens) - 1) // block_size
        full_hit = 0
        while full_hit < max_blocks and hashes[full_hit] in cached_full:
            full_hit += 1
        expect = 0
        for n in range(full_hit, 0, -1):
            if hashes[n - 1] in cached_mamba:
                expect = n
                break

        got = kv.get_computed_blocks(slot, len(tokens), 0, False)
        assert got == expect * block_size, (
            f"{rid}: rust hit {got} tokens, oracle expected {expect * block_size} "
            f"(full-attn chain {full_hit} blocks)"
        )

        target = aligned_target(len(tokens), block_size)
        if target <= got:
            kv.forget(rid)
            continue
        assert kv.allocate_slots(slot, target - got, got, True, 0, 0, len(tokens), 0, False)
        live.append((rid, slot))
        # cache_blocks ran inside allocate_slots: every full block is now cached for full
        # attention, and the LAST full block is the cached mamba state (align mode keeps
        # exactly one live state block, at the end of the block list).
        num_full = target // block_size
        for h in hashes[:num_full]:
            cached_full.add(h)
        if num_full:
            cached_mamba.add(hashes[num_full - 1])

    # Nothing leaks: freeing everything returns the pool to its initial state.
    for rid, slot in live:
        kv.free(slot)
        kv.forget(rid)
    assert kv.num_free_blocks == (1 << 15) - 1, "null block is the only one still held"


@requires_crate
def test_radix_index_gives_identical_decisions():
    """VTL_RUST_SCHED_RADIX must be a pure data-structure swap."""
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    convs, order = synth_conversations(num_convs=10, turns=4, seed=11)
    flat = drive_rust(
        vtl_sched.KvManager(rust_config(num_blocks=2048)), none_hash, block_size, convs, order
    )
    radix = drive_rust(
        vtl_sched.KvManager(rust_config(num_blocks=2048, radix=True)),
        none_hash,
        block_size,
        convs,
        order,
    )
    assert flat == radix


@requires_crate
def test_pool_pressure_forces_eviction_without_corruption():
    """Small pool: blocks recycle, hashes must be evicted with them."""
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    kv = vtl_sched.KvManager(rust_config(num_blocks=256, max_model_len=4096))
    convs, order = synth_conversations(num_convs=30, turns=3, seed=3)
    obs = drive_rust(kv, none_hash, block_size, convs, order)
    assert all(ok for _, _, ok, _, _ in obs), "a request failed to allocate in a fresh pool"
    # Everything is freed each iteration, so the pool must come back to exactly one
    # held block (the null block) no matter how much churn happened.
    assert kv.num_free_blocks == 255


# ---------------------------------------------------------------------------
# tier 3b -- eviction-order oracle under pool pressure (no vLLM needed)
# ---------------------------------------------------------------------------


def full_attention_config(num_blocks, block_size=16, max_model_len=8192):
    """One full-attention group. Eviction order lives in the shared BlockPool, so the
    simplest layout that reaches it is the right one to model in python."""
    cfg = rust_config(num_blocks, block_size, max_model_len)
    cfg["groups"] = cfg["groups"][:1]
    return cfg


class FreeQueueOracle:
    """`FreeKVCacheBlockQueue` + the cached-hash map, straight from the source.

    `popleft` hands out the least-recently-freed block and drops whatever it had cached
    (block_pool.py:574); `free_blocks` appends hashed blocks in EVICTION order, i.e. the
    caller's list already reversed (`:614`); `touch` pulls a freed-but-cached block back
    out of the queue (`:597`). Block 0 is the null block and never circulates.

    Only the all-blocks-hashed case is modelled: the driver below always allocates a whole
    number of blocks, so `free_blocks`' unhashed-first split never triggers.
    """

    def __init__(self, num_blocks):
        from collections import deque

        self.free = deque(range(1, num_blocks))
        self.by_hash = {}
        self.by_block = {}

    def touch(self, blocks):
        for b in blocks:
            self.free.remove(b)

    def alloc(self, n):
        out = []
        for _ in range(n):
            b = self.free.popleft()
            h = self.by_block.pop(b, None)
            if h is not None:
                del self.by_hash[h]
            out.append(b)
        return out

    def cache(self, blocks, hashes, start):
        for i in range(start, len(hashes)):
            self.by_hash[hashes[i]] = blocks[i]
            self.by_block[blocks[i]] = hashes[i]

    def release(self, blocks):
        self.free.extend(reversed(blocks))


@requires_crate
def test_eviction_order_matches_the_python_oracle_under_pressure():
    """Small pool, long workload: every handed-out block ID must match the oracle.

    The pool recycles ~40x over the run, so a single wrong eviction victim diverges the
    ID stream immediately and never recovers.
    """
    block_size, num_blocks = 16, 512
    none_hash = vtl_sched.none_hash_from_seed("0")
    kv = vtl_sched.KvManager(full_attention_config(num_blocks, block_size))
    oracle = FreeQueueOracle(num_blocks)
    convs, order = synth_conversations(num_convs=40, turns=4, seed=17)

    recycled = 0
    for c, t in order:
        tokens = convs[c][t]
        rid = f"c{c}-t{t}"
        slot = kv.intern(rid)
        hashes = vtl_sched.block_hashes(none_hash, block_size, tokens)
        kv.push_hashes(slot, b"".join(hashes), len(tokens))
        kv.new_step_starts()

        max_blocks = (len(tokens) - 1) // block_size
        n_hit = 0
        while n_hit < max_blocks and hashes[n_hit] in oracle.by_hash:
            n_hit += 1
        hit_blocks = [oracle.by_hash[h] for h in hashes[:n_hit]]

        got_hit = kv.get_computed_blocks(slot, len(tokens), 0, False)
        assert got_hit == n_hit * block_size, f"{rid}: hit tokens"
        n = kv.pending_hit_into_buffer(0)
        assert kv.buffer(0)[:n].tolist() == hit_blocks, f"{rid}: hit block ids"

        target = aligned_target(len(tokens), block_size)
        if target <= got_hit:
            kv.forget(rid)
            continue
        oracle.touch(hit_blocks)
        new = oracle.alloc(target // block_size - n_hit)
        recycled += sum(1 for b in new if b in oracle.by_block)
        blocks = hit_blocks + new

        assert kv.allocate_slots(
            slot, target - got_hit, got_hit, True, 0, 0, len(tokens), 0, False
        ), rid
        n = kv.new_blocks_into_buffer(0)
        assert kv.buffer(0)[:n].tolist() == new, f"{rid}: new block ids"
        n = kv.blocks_into_buffer(slot, 0)
        assert kv.buffer(0)[:n].tolist() == blocks, f"{rid}: full block table"

        oracle.cache(blocks, hashes[: target // block_size], n_hit)
        kv.free(slot)
        kv.forget(rid)
        oracle.release(blocks)

    assert kv.num_free_blocks == num_blocks - 1
    assert len(oracle.free) == num_blocks - 1


# ---------------------------------------------------------------------------
# tier 3c -- replay of the recorded trace (no vLLM needed)
# ---------------------------------------------------------------------------

TRACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "input",
    "trace-round2.jsonl",
)


def trace_conversations(path=TRACE):
    """The recorded round-2 trace as ``(conversations, arrival_order)``.

    The trace stores chat MESSAGES, not token ids, and no tokenizer is available off-box,
    so each whitespace-delimited word maps to a stable id. That keeps the only property
    the prefix cache reacts to — which requests share which leading token runs, and how
    long they are — while the tier stays runnable without torch or the model. 420
    requests, 70 conversations x 6 turns, 1482-3104 words each.
    """
    import json
    import re

    vocab = {}
    convs, index, order = [], {}, []
    with open(path) as fh:
        for line in fh:
            messages = json.loads(line)["body"]["messages"]
            tokens = [
                vocab.setdefault(w, len(vocab) + 1000)
                for m in messages
                for w in re.findall(r"\S+", m["content"])
            ]
            key = messages[0]["content"], messages[1]["content"]
            c = index.get(key)
            if c is None:
                c = index[key] = len(convs)
                convs.append([])
            order.append((c, len(convs[c])))
            convs[c].append(tokens)
    return convs, order


@requires_crate
def test_recorded_trace_replays_with_high_reuse():
    """Drive the real trace through the Rust manager, not just the synthetic fixture."""
    if not os.path.exists(TRACE):
        pytest.skip(f"{TRACE} not present")
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    convs, order = trace_conversations()
    assert len(order) == 420 and len(convs) == 70, (len(order), len(convs))

    kv = vtl_sched.KvManager(rust_config(num_blocks=1 << 14, max_model_len=8192))
    obs = drive_rust(kv, none_hash, block_size, convs, order)
    assert len(obs) == len(order)
    assert all(ok is not False for _, _, ok, _, _ in obs), "a trace request failed to fit"
    reuse = sum(h for _, h, _, _, _ in obs) / sum(len(convs[c][t]) for c, t in order)
    # The trace measures 82.8% block-hash reuse with the real tokenizer; word-level ids
    # give the same shape. Anything near zero means the replay lost the prefix structure.
    assert reuse > 0.6, f"trace replay is not reusing prefixes (reuse={reuse:.2f})"
    assert kv.num_free_blocks == (1 << 14) - 1, "everything freed"


@requires_crate
def test_schedule_returns_per_step_deltas_for_running_requests():
    """The `running_new_blocks` / `running_new_lens` contract `rust_sched.py` slices.

    scheduler.py:588 stores only the step's new blocks for a running request. If this
    ever handed back the whole table, the V2 runner's `append_block_ids(overwrite=False)`
    would duplicate the block table on every decode step.
    """
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    kv = vtl_sched.KvManager(rust_config(num_blocks=512, max_model_len=4096))
    core = vtl_sched.Scheduler()
    core.set_params(
        {
            "max_num_scheduled_tokens": 8192,
            "max_num_running_reqs": 64,
            "max_model_len": 4096,
            "num_sampled_tokens_per_step": 1,
            "long_prefill_token_threshold": 0,
            "enable_chunked_prefill": True,
            "need_mamba_block_aligned_split": True,
            "cache_block_size": block_size,
            "num_lookahead_tokens": 0,
            "sjf_reorder": False,
        }
    )
    tokens = [7000 + i for i in range(80)]
    slot = kv.intern("r0")
    kv.push_hashes(slot, b"".join(vtl_sched.block_hashes(none_hash, block_size, tokens)), 80)

    def req(num_tokens, computed, status):
        return (slot, num_tokens, num_tokens, computed, 0, 64, 128, status, 0, False, False)

    admitted = core.schedule(kv, [], [req(64, 0, 0)])
    assert len(admitted["scheduled_admitted"]) == 1
    before = [kv.blocks_into_buffer(slot, g) for g in range(kv.num_groups)]

    d = core.schedule(kv, [req(65, 64, 1)], [])
    assert len(d["scheduled_running"]) == 1

    flat, lens = d["running_new_blocks"], d["running_new_lens"]
    assert len(lens) == kv.num_groups
    off = 0
    for g in range(kv.num_groups):
        n = lens[g]
        span = flat[off : off + n]
        off += n
        after = kv.buffer(g)[: kv.blocks_into_buffer(slot, g)].tolist()
        assert n == len(after) - before[g], f"group {g}: span length == table growth"
        if n:
            assert span == after[before[g] :], f"group {g}: span == the new tail"
    assert off == len(flat)
    assert sum(before) > len(flat), "a delta, not the whole table"


@requires_crate
def test_workload_actually_exercises_the_prefix_cache():
    """Guard against a vacuously-passing suite: the fixture must produce real reuse."""
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    convs, order = synth_conversations(seed=1)
    kv = vtl_sched.KvManager(rust_config(num_blocks=1 << 16))
    obs = drive_rust(kv, none_hash, block_size, convs, order)
    reuse = sum(h for _, h, _, _, _ in obs) / sum(len(convs[c][t]) for c, t in order)
    assert reuse > 0.5, f"workload is not exercising the prefix cache (reuse={reuse:.2f})"


# ---------------------------------------------------------------------------
# tier 2 -- real vLLM KVCacheManager parity
# ---------------------------------------------------------------------------


def _try_import_vllm():
    if VLLM_SRC and os.path.isdir(VLLM_SRC) and VLLM_SRC not in sys.path:
        sys.path.append(VLLM_SRC)
    try:
        # Import a real submodule, not just the package. Running pytest from the repo root
        # puts the source-only `vllm/` checkout on sys.path as a NAMESPACE package, so a
        # bare `import vllm` succeeds and every later import blows up mid-test instead of
        # skipping.
        from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: F401
        import vllm

        return vllm
    except Exception:
        return None


def _computed_blocks(py, request):
    """``KVCacheManager.get_computed_blocks`` -> ``(blocks, num_computed_tokens)``.

    v0.25.0 returns a 2-tuple (kv_cache_manager.py:206); read it by position so a tree
    that grows a third element still drives this harness instead of raising ValueError.
    """
    out = py.get_computed_blocks(request)
    assert len(out) >= 2, f"unexpected get_computed_blocks arity: {len(out)}"
    return out[0], out[1]


def _build_python_manager(num_blocks, block_size, max_model_len):
    """Real vLLM KVCacheManager with the served hybrid layout."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    full = FullAttentionSpec(block_size=block_size, num_kv_heads=8, head_size=64, dtype="auto")
    mamba = MambaSpec(
        shapes=((4, 2048),),
        dtypes=("auto",),
        block_size=block_size,
        page_size_padded=None,
        mamba_type="mamba2",
        num_speculative_blocks=0,
    )
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["a"], full), KVCacheGroupSpec(["b"], mamba)],
    )
    return KVCacheManager(
        kv_cache_config=cfg,
        max_model_len=max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
        log_stats=True,
    )


@requires_crate
def test_manager_parity_against_vllm():
    if _try_import_vllm() is None:
        pytest.skip(
            "vllm not importable here (needs torch and the full dep set); this tier runs "
            "in-container via `make test-kernel`"
        )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    init_none_hash(sha256)
    import vllm.v1.core.kv_cache_utils as ku

    block_size, num_blocks, max_model_len = 16, 4096, 32768
    py = _build_python_manager(num_blocks, block_size, max_model_len)
    rs = vtl_sched.KvManager(rust_config(num_blocks, block_size, max_model_len))
    hasher = get_request_block_hasher(block_size, sha256)

    convs, order = synth_conversations(num_convs=20, turns=5, seed=42)
    from vllm.v1.request import Request

    live = {}
    for c, t in order:
        tokens = convs[c][t]
        rid = f"c{c}-t{t}"
        request = Request(
            request_id=rid,
            prompt_token_ids=tokens,
            sampling_params=None,
            pooling_params=None,
            eos_token_id=None,
            block_hasher=hasher,
        )
        slot = rs.intern(rid)
        rs.push_hashes(slot, b"".join(request.block_hashes), len(tokens))

        py.new_step_starts()
        rs.new_step_starts()

        py_blocks, py_hit = _computed_blocks(py, request)
        rs_hit = rs.get_computed_blocks(slot, len(tokens), 0, False)
        assert py_hit == rs_hit, f"{rid}: computed tokens {py_hit} vs {rs_hit}"
        py_ids = py_blocks.get_block_ids()
        for g in range(rs.num_groups):
            n = rs.pending_hit_into_buffer(g)
            assert list(py_ids[g]) == rs.buffer(g)[:n].tolist(), f"{rid}: hit blocks g{g}"

        py_new = py.allocate_slots(request, len(tokens) - py_hit, py_hit, py_blocks)
        rs_ok = rs.allocate_slots(
            slot, len(tokens) - rs_hit, rs_hit, True, 0, 0, len(tokens), 0, bool(live)
        )
        assert (py_new is not None) == rs_ok, f"{rid}: admission decision"
        if py_new is None:
            continue
        py_new_ids = py_new.get_block_ids()
        for g in range(rs.num_groups):
            n = rs.new_blocks_into_buffer(g)
            assert list(py_new_ids[g]) == rs.buffer(g)[:n].tolist(), f"{rid}: new blocks g{g}"
        assert py.block_pool.get_num_free_blocks() == rs.num_free_blocks, rid
        assert abs(py.usage - rs.usage) < 1e-12, rid

        live[rid] = (request, slot)
        # Retire the oldest live request to keep the pool churning (and to exercise the
        # free path's eviction ordering).
        if len(live) > 8:
            old_rid = next(iter(live))
            old_request, old_slot = live.pop(old_rid)
            py.free(old_request)
            rs.free(old_slot)
            rs.forget(old_rid)
            assert py.block_pool.get_num_free_blocks() == rs.num_free_blocks, old_rid

    py_stats = py.make_prefix_cache_stats()
    rs_stats = rs.take_prefix_cache_stats()
    assert py_stats.queries == rs_stats["queries"]
    assert py_stats.hits == rs_stats["hits"]
    assert py_stats.requests == rs_stats["requests"]
    assert ku.NONE_HASH  # keep the import meaningful


# ---------------------------------------------------------------------------
# tier 4 -- the rust_sched.py wrapper classes themselves
# ---------------------------------------------------------------------------


def _install_wrappers():
    """`_install_shadow` / `_install_authority` over the stock v0.25.0 KVCacheManager."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from vtl.patches.rust_sched import _install_authority, _install_shadow

    modes = {"radix": False, "strict": True}
    return (
        _install_shadow(KVCacheManager, modes),
        _install_authority(KVCacheManager, modes),
    )


def _wrapper_manager(cls, num_blocks, block_size, max_model_len):
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    full = FullAttentionSpec(block_size=block_size, num_kv_heads=8, head_size=64, dtype="auto")
    mamba = MambaSpec(
        shapes=((4, 2048),),
        dtypes=("auto",),
        block_size=block_size,
        page_size_padded=None,
        mamba_type="mamba2",
        num_speculative_blocks=0,
    )
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["a"], full), KVCacheGroupSpec(["b"], mamba)],
    )
    return cls(
        kv_cache_config=cfg,
        max_model_len=max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
        log_stats=True,
    )


@requires_crate
def test_shadow_and_authority_wrappers_agree_with_python():
    """Tier 2 drives `vtl_sched.KvManager` directly; this drives the real subclasses.

    Everything between the engine and the crate — RustMirror slot interning, the
    BaseException guards, RustBlocks, `_refuse_unported` — only runs here.
    """
    if _try_import_vllm() is None:
        pytest.skip("vllm not importable here; this tier runs in-container")
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    shadow_cls, authority_cls = _install_wrappers()
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)

    block_size, num_blocks, max_model_len = 16, 4096, 32768
    shadow = _wrapper_manager(shadow_cls, num_blocks, block_size, max_model_len)
    authority = _wrapper_manager(authority_cls, num_blocks, block_size, max_model_len)
    assert shadow._rust is not None, "shadow mirror refused to start"

    convs, order = synth_conversations(num_convs=15, turns=4, seed=77)
    live = []
    for c, t in order:
        tokens = convs[c][t]
        rid = f"c{c}-t{t}"
        reqs = [
            Request(
                request_id=rid,
                prompt_token_ids=tokens,
                sampling_params=None,
                pooling_params=None,
                eos_token_id=None,
                block_hasher=hasher,
            )
            for _ in range(2)
        ]
        for mgr, request in zip((shadow, authority), reqs):
            mgr.new_step_starts()
            blocks, hit = _computed_blocks(mgr, request)
            mgr.allocate_slots(request, len(tokens) - hit, hit, blocks)
        assert shadow.get_block_ids(rid) == authority.get_block_ids(rid), rid
        live.append(reqs)
        if len(live) > 6:
            for mgr, request in zip((shadow, authority), live.pop(0)):
                mgr.free(request)

    assert shadow.rust_shadow_mismatches == 0
    assert shadow._rust is not None, "a guard fired and disabled the mirror"
    assert authority.usage == pytest.approx(shadow.usage)

    # Deferred freeing would drain the stale python pool -- authority must refuse.
    with pytest.raises(RuntimeError):
        authority.pop_blocks_for_free(live[0][1])


# ---------------------------------------------------------------------------
# microbench
# ---------------------------------------------------------------------------


def bench(iters=3):
    if vtl_sched is None:  # pragma: no cover
        print(f"vtl_sched not importable: {_CRATE_ERR}")
        return
    block_size = 16
    none_hash = vtl_sched.none_hash_from_seed("0")
    convs, order = synth_conversations(seed=1)
    print(f"{len(order)} requests, {sum(len(c[-1]) for c in convs)} tail tokens")
    for _ in range(iters):
        kv = vtl_sched.KvManager(rust_config(num_blocks=1 << 16))
        t0 = time.perf_counter()
        drive_rust(kv, none_hash, block_size, convs, order)
        dt = time.perf_counter() - t0
        print(f"rust manager: {dt * 1e3:8.2f} ms total  {dt / len(order) * 1e6:7.2f} us/request")

    py_available = _try_import_vllm() is not None
    if not py_available:
        print("python side: skipped (vllm not importable here)")
        return
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    for _ in range(iters):
        py = _build_python_manager(1 << 16, block_size, 32768)
        t0 = time.perf_counter()
        for c, t in order:
            tokens = convs[c][t]
            request = Request(
                request_id=f"c{c}-t{t}",
                prompt_token_ids=tokens,
                sampling_params=None,
                pooling_params=None,
                eos_token_id=None,
                block_hasher=hasher,
            )
            py.new_step_starts()
            blocks, hit = _computed_blocks(py, request)
            if py.allocate_slots(request, len(tokens) - hit, hit, blocks) is not None:
                py.free(request)
        dt = time.perf_counter() - t0
        print(f"python manager: {dt * 1e3:8.2f} ms total  {dt / len(order) * 1e6:7.2f} us/request")


# ---------------------------------------------------------------------------
# tier 4 -- R6a update core: the batched stop decision vs vLLM's check_stop
# ---------------------------------------------------------------------------
#
# The oracle is `check_stop` (v1/core/sched/utils.py:94) transcribed with the counters
# inlined, driven over a full stop matrix. A false "not stopped" leaks a request past its
# max_tokens; a false "stopped" truncates an answer. Both are silent, so this tier is
# exhaustive over the branch cross-product rather than sampled.

NOT_STOPPED, FIN_STOPPED, FIN_LENGTH = 0, 1, 2


def ref_check_stop(params, tokens, num_output, num_tokens, max_model_len):
    """utils.py:94 `check_stop` inside scheduler.py:1904's append loop, verbatim."""
    min_tokens, max_tokens, eos, stop_ids = params
    for i, tok in enumerate(tokens, 1):
        num_output += 1
        num_tokens += 1
        if num_output < min_tokens:
            continue
        if tok == eos:
            return i, FIN_STOPPED, -1
        if tok in stop_ids:
            return i, FIN_STOPPED, tok
        if num_tokens >= max_model_len or num_output >= max_tokens:
            return i, FIN_LENGTH, -1
    return len(tokens), NOT_STOPPED, -1


@requires_crate
def test_update_step_matches_check_stop():
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    max_model_len = 64

    # (min_tokens, max_tokens, eos, stop_token_ids) x (num_output, num_tokens) x tokens.
    param_sets = [
        (0, 4, 2, []),
        (0, 4, 2, [13, 99]),
        (3, 4, 2, [2]),          # eos also listed as a stop id: eos must win
        (0, 1, None, [7]),       # no eos configured
        (2, 100, 5, []),         # min_tokens gate, far from the length cap
    ]
    token_sets = [[7], [2], [99], [13], [5, 6, 2, 7], [7, 7, 7, 7, 7], []]
    counters = [(0, 10), (3, 10), (7, 10), (0, max_model_len - 1), (0, max_model_len)]

    cases = []
    for pi, p in enumerate(param_sets):
        slot = kv.intern(f"r{pi}")
        kv.set_stop_params(slot, p[0], p[1], p[2], p[3])
        assert kv.has_stop_params(slot)
        for toks in token_sets:
            for n_out, n_tok in counters:
                cases.append((slot, p, toks, n_out, n_tok))

    slots, cu, flat, n_out, n_tok = [], [0], [], [], []
    for slot, _p, toks, o, t in cases:
        slots.append(slot)
        flat.extend(toks)
        cu.append(len(flat))
        n_out.append(o)
        n_tok.append(t)

    got = kv.update_step(slots, cu, flat, n_out, n_tok, max_model_len)
    assert len(got) == len(cases)
    for (slot, p, toks, o, t), rust in zip(cases, got):
        want = ref_check_stop(p, toks, o, t, max_model_len)
        assert tuple(rust) == want, (p, toks, o, t, rust, want)


@requires_crate
def test_update_step_refuses_unregistered_and_malformed_batches():
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    known = kv.intern("known")
    kv.set_stop_params(known, 0, 8, 2, [])
    unknown = kv.intern("unknown")
    assert not kv.has_stop_params(unknown)

    got = kv.update_step([known, unknown], [0, 1, 2], [7, 7], [0, 0], [1, 1], 64)
    assert got[0][1] == NOT_STOPPED
    assert got[1][1] == 255, "an unregistered slot must be refused, not answered"

    # Arity errors are exceptions, not silent wrong answers.
    for bad in (
        ([known], [0], [7], [0], [1]),            # cu_lens too short
        ([known], [1, 2], [7], [0], [1]),         # cu_lens does not start at 0
        ([known], [0, 5], [7], [0], [1]),         # cu_lens overruns token_ids
        ([known], [0, 1], [7], [0, 0], [1]),      # counter arity mismatch
    ):
        with pytest.raises(Exception):
            kv.update_step(*bad, 64)

    # forget() drops the stop params with the id, so a recycled slot cannot inherit them.
    kv.forget("known")
    reused = kv.intern("fresh")
    assert not kv.has_stop_params(reused)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
