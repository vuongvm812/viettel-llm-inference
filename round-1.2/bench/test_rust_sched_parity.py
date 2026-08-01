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


# ------------------------------------------------------------------------------------
# R8 tier -- Rust's raw output record vs the Python packer, byte for byte.
#
# The Python packer (`vtl/patches/shm_ipc.py`) is the reference because it is the one the
# frontend's decoder was written against and the one the golden vectors pin. If these two
# ever disagree, the frontend decodes a record built by the other side into garbage --
# silently, because the layout has no checksum.
# ------------------------------------------------------------------------------------


def _packer():
    try:
        from vtl.patches.shm_ipc import _Out, _Outs, raw_pack_into, raw_size
    except Exception:  # pragma: no cover - PYTHONPATH without round-1.2
        pytest.skip("vtl.patches.shm_ipc not importable")

    def pack(outputs):
        buf = bytearray(raw_size(outputs))
        n = raw_pack_into(buf, outputs)
        assert n == len(buf), (n, len(buf))
        return bytes(buf)

    return _Out, _Outs, pack


# `FinishReason` (vllm/v1/engine/__init__.py) for the statuses check_stop can produce.
FINISH_FROM_STATUS = {FIN_STOPPED: 0, FIN_LENGTH: 1}


@requires_crate
def test_update_step_pack_bytes_match_the_python_packer():
    _Out, _Outs, pack = _packer()
    max_model_len = 64
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))

    # (name, params, offered tokens, num_output, num_tokens) covering: plain decode,
    # multi-token bursts, a mid-burst EOS trim, a stop-token-id trim (the only case that
    # carries a stop_reason), and the length cap.
    cases = [
        ("req-1a", (0, 8, 2, []), [5], 0, 10),
        ("req-2", (0, 8, 2, []), [5, 6, 7, 8], 0, 10),
        ("req-3", (0, 8, 2, []), [5, 2, 7, 8], 0, 10),       # EOS at index 1 -> keep 2
        ("req-4", (0, 8, 2, [99]), [5, 99, 7], 0, 10),       # stop id -> stop_reason=99
        ("req-5", (0, 3, 2, []), [5, 6, 7, 8], 0, 10),       # max_tokens caps at 3
        ("req-\u00e9", (0, 8, 2, []), [5, 6], 0, 10),           # utf-8 id, byte-length prefix
    ]

    slots, cu, flat, n_out, n_tok = [], [0], [], [], []
    for name, params, toks, o, t in cases:
        slot = kv.intern(name)
        kv.set_stop_params(slot, *params)
        slots.append(slot)
        flat.extend(toks)
        cu.append(len(flat))
        n_out.append(o)
        n_tok.append(t)

    verdicts, record = kv.update_step_pack(
        slots, cu, flat, n_out, n_tok, max_model_len, 7, 1.5, []
    )
    assert record is not None, "every case here is packable"

    # Same verdicts as the non-packing entry point, and the same as the Python oracle.
    for (_name, params, toks, o, t), rust in zip(cases, verdicts):
        assert tuple(rust) == ref_check_stop(params, toks, o, t, max_model_len)

    outputs = []
    for (name, _p, toks, _o, _t), (num_accepted, status, stop_reason) in zip(cases, verdicts):
        outputs.append(
            _Out(
                name,
                list(toks[:num_accepted]),
                finish_reason=FINISH_FROM_STATUS.get(status),
                stop_reason=stop_reason if stop_reason >= 0 else None,
            )
        )
    want = pack(_Outs(engine_index=7, timestamp=1.5, outputs=outputs))
    assert record == want, f"\n  rust  ={record.hex()}\n  python={want.hex()}"


@requires_crate
def test_update_step_pack_carries_sorted_finished_ids():
    _Out, _Outs, pack = _packer()
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    slot = kv.intern("req-a")
    kv.set_stop_params(slot, 0, 8, 2, [])
    finished = ["req-a", "req-b"]
    _v, record = kv.update_step_pack([slot], [0, 1], [2], [0], [10], 64, 0, 0.0, finished)
    want = pack(
        _Outs(
            outputs=[_Out("req-a", [2], finish_reason=0)],
            finished_requests=set(finished),
        )
    )
    assert record == want


@requires_crate
def test_update_step_pack_refuses_rather_than_emitting_a_partial_record():
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    known = kv.intern("known")
    kv.set_stop_params(known, 0, 8, 2, [])
    unknown = kv.intern("unknown")
    verdicts, record = kv.update_step_pack(
        [known, unknown], [0, 1, 2], [7, 7], [0, 0], [1, 1], 64, 0, 0.0, []
    )
    assert verdicts[1][1] == 255
    assert record is None, "one refused slot must drop the WHOLE record, not half of it"
    # ...and the verdicts were still applied, so the resident table is not left behind.
    assert verdicts[0][1] == NOT_STOPPED


# ------------------------------------------------------------------------------------
# C1a tier -- the N-step burst's num_computed_tokens trajectory in the resident table.
#
# This is the one that can poison the prefix cache: a burst that stops after j < N tokens
# has N tokens' worth of num_computed committed, and the down-correction is the only thing
# that stops `cache_blocks` from fingerprinting KV that was never computed.
# ------------------------------------------------------------------------------------


def _entry(kv, slot):
    e = kv.table_entry(slot)
    assert e is not None
    # pack_req order: slot, num_tokens, num_tokens_with_spec, num_computed_tokens,
    # num_output_placeholders, num_prompt_tokens, max_tokens, status, num_preemptions,
    # is_prefill_chunk, skip_reading_prefix_cache
    return {"num_tokens": e[1], "num_computed": e[3], "placeholders": e[4],
            "is_prefill_chunk": e[9]}


def _decode_entry(slot, num_tokens=100):
    """A steady-state decode entry as the post-schedule commit leaves it."""
    return (slot, num_tokens, num_tokens, num_tokens + 1, 2, 10, 4096, 1, 0, False, False)


@requires_crate
@pytest.mark.parametrize("n", [2, 4, 8])
def test_table_burst_commits_and_reconciles_every_stop_point(n):
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    slot = kv.intern("burst")
    kv.set_stop_params(slot, 0, 1000, 2, [])

    for kept in range(1, n + 1):
        kv.table_set(slot, _decode_entry(slot))
        base = _entry(kv, slot)
        kv.table_burst([slot], n - 1)
        after = _entry(kv, slot)
        assert after["num_computed"] == base["num_computed"] + n - 1
        assert after["placeholders"] == base["placeholders"] + n - 1

        # The step comes back with `kept` tokens: EOS at index kept-1 trims the rest.
        toks = [7] * (kept - 1) + ([2] if kept < n else [7]) + [7] * (n - kept)
        verdicts = kv.update_step(
            [slot], [0, n], toks[:n], [0], [base["num_tokens"]], 4096
        )
        assert verdicts[0][0] == kept, (kept, verdicts)
        short = n - kept
        if short:
            kv.table_burst([slot], -short)
        final = _entry(kv, slot)
        # num_tokens is synced by update_step to Python's pre-append count + accepted,
        # so the freed request never carries tokens kept+1..N anywhere.
        assert final["num_tokens"] == base["num_tokens"] + kept

        # THE INVARIANT. After any step -- burst or not, stopped or not -- the entry must
        # satisfy `num_computed == num_tokens`: every token that exists has been through
        # the model, and no position that was not computed is counted. A missing
        # down-correction shows up here as a POSITIVE drift, which is exactly the shape
        # that would make `cache_blocks` fingerprint KV that was never written.
        assert final["num_computed"] == final["num_tokens"], (
            f"N={n} kept={kept}: num_computed drifted by "
            f"{final['num_computed'] - final['num_tokens']}"
        )


@requires_crate
def test_table_burst_is_a_no_op_on_an_unknown_slot():
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    kv.table_burst([4242], 3)  # must not raise
    assert kv.table_entry(4242) is None


@requires_crate
def test_update_step_pack_multi_token_wire_shape():
    """The wire format is length-prefixed per output, so N tokens need no wire work --
    assert that directly rather than trusting the layout comment."""
    _Out, _Outs, pack = _packer()
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    slot = kv.intern("r")
    kv.set_stop_params(slot, 0, 1000, 2, [])
    for n in (1, 2, 4, 8):
        toks = list(range(100, 100 + n))
        _v, record = kv.update_step_pack(
            [slot], [0, n], toks, [0], [10], 4096, 0, 0.0, []
        )
        assert record == pack(_Outs(outputs=[_Out("r", toks)])), n
        # header(24) + row head(20) + id(1) + n * u32
        assert len(record) == 24 + 20 + 1 + 4 * n


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


# ------------------------------------------------------------------------------------
# R9 tier -- update_step_pack_np driven straight from Python, and fold_cache.
# ------------------------------------------------------------------------------------

try:
    import numpy as np
except Exception:  # pragma: no cover - vtl_sched's own `python` feature needs numpy too
    np = None

requires_numpy = pytest.mark.skipif(np is None, reason="numpy not importable")


@requires_crate
@requires_numpy
def test_update_step_pack_np_matches_update_step_pack():
    """The numpy crossing must be a faithful stand-in for the flat-list one: same
    verdicts, same record, for the same tokens taken off a 2D array instead of a
    flattened + cu_lens-indexed buffer."""
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    kv_np = vtl_sched.KvManager(rust_config(num_blocks=256))
    kv.store_arm(vtl_sched.none_hash_from_seed("0"))
    kv_np.store_arm(vtl_sched.none_hash_from_seed("0"))

    names = ["r0", "r1", "r2"]
    toks = [[5], [6, 7, 2], [8, 9]]  # r1's EOS(2) trims at index 2
    n_toks = [10, 10, 10]
    slots, cu, flat = [], [0], []
    for name, t, n in zip(names, toks, n_toks):
        for kv_ in (kv, kv_np):
            s = kv_.intern(name)
            kv_.set_stop_params(s, 0, 1000, 2, [])
            kv_.store_init(s, [], n, 0)
        slots.append(kv_np.lookup(name))
        flat.extend(t)
        cu.append(len(flat))
    counts = [len(t) for t in toks]

    v_pack, rec_pack = kv.update_step_pack(
        [kv.lookup(n) for n in names], cu, flat, [0] * len(names), n_toks, 4096, 0, 1.5, []
    )
    rows = list(range(len(names)))
    arr = np.zeros((len(names), max(counts)), dtype=np.int64)
    for row, t in zip(rows, toks):
        arr[row, : len(t)] = t
    v_np, rec_np, published = kv_np.update_step_pack_np(arr, rows, counts, slots, 4096, 0, 1.5, ())
    assert [tuple(v) for v in v_np] == [tuple(v) for v in v_pack]
    assert rec_np == rec_pack
    assert published is False, "publish defaults off -- nothing should have been delivered"


@requires_crate
@requires_numpy
def test_update_step_pack_np_refuses_a_noncontiguous_array():
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    kv.store_arm(vtl_sched.none_hash_from_seed("0"))
    slot = kv.intern("r")
    kv.set_stop_params(slot, 0, 1000, 2, [])
    kv.store_init(slot, [], 10, 0)
    arr = np.zeros((4, 8), dtype=np.int64)
    arr[0, :2] = [5, 6]
    noncontig = arr[:, ::2]  # a view, not C-contiguous
    with pytest.raises(Exception):
        kv.update_step_pack_np(noncontig, [0], [2], [slot], 4096, 0, 0.0, ())


@requires_crate
@requires_numpy
def test_update_step_pack_np_fold_cache_defaults_off():
    """`fold_cache` must default to `False`: every caller written before R9 existed
    (this file's other tests included) must keep behaving exactly as it did without
    passing the new keyword."""
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    kv.store_arm(vtl_sched.none_hash_from_seed("0"))
    slot = kv.intern("r")
    kv.set_stop_params(slot, 0, 1000, 2, [])
    kv.store_init(slot, [], 10, 0)
    arr = np.array([[5, 6, 7, 8]], dtype=np.int64)
    verdicts, record, published = kv.update_step_pack_np(arr, [0], [4], [slot], 4096, 0, 0.0, ())
    assert record is not None
    assert published is False, "publish defaults off -- nothing should have been delivered"
    assert kv.cache_fold_skips() == 0, "fold_cache defaulted off -- the fold never ran"


@requires_crate
@requires_numpy
def test_update_step_pack_np_fold_cache_is_idempotent_with_an_explicit_call():
    """A folded step, then an explicit `cache_blocks` call with the SAME argument the
    fold would have used, must change nothing further: the fold already did exactly
    that much work. (The rigorous cache-content proof, with real block allocation, is
    the crate's own `cache_fold_matches_explicit_cache_blocks` golden test; this is the
    FFI-surface smoke test -- the keyword plumbs through and the pool is left stable.)
    """
    kv = vtl_sched.KvManager(rust_config(num_blocks=256))
    kv.store_arm(vtl_sched.none_hash_from_seed("0"))
    slot = kv.intern("r")
    kv.set_stop_params(slot, 0, 1000, 2, [])
    kv.store_init(slot, [], 16, 0)
    kv.table_set(slot, (slot, 16, 16, 16, 0, 16, 1000, 1, 0, False, False))
    arr = np.array([[100, 101, 102, 103]], dtype=np.int64)
    verdicts, record, published = kv.update_step_pack_np(
        arr, [0], [4], [slot], 4096, 0, 0.0, (), fold_cache=True
    )
    assert record is not None
    assert published is False, "publish defaults off -- nothing should have been delivered"
    assert kv.cache_fold_skips() == 0
    after_fold = kv.num_free_blocks
    e = kv.table_entry(slot)
    kv.cache_blocks(slot, e[3] - e[4])  # the same math the fold just ran internally
    assert kv.num_free_blocks == after_fold, "an explicit call after the fold is a no-op"


# ---------------------------------------------------------------------------
# tier 5 -- R6b/R6c: the resident request table and the speculative precompute
# ---------------------------------------------------------------------------
#
# Both are decision-preserving optimisations, so the oracle is the SAME crate driven the
# SAME way through the marshalled `Scheduler.schedule`. Three arms run in lockstep over one
# scripted workload -- marshalled / resident / resident+speculation -- and every step's
# decisions dict must be identical, block IDs included. A wrong block ID is not a slower
# schedule, it is corrupted KV state.
#
# THE HARNESS MODELS ASYNC SCHEDULING: `schedule(N)` runs BEFORE `update_from_output(N-1)`
# (v0.25.0 AsyncScheduler). That is the whole reason `num_output_placeholders` exists, and
# it is transient WITHIN a step -- a driver that updated before scheduling would leave the
# placeholder arithmetic in `sched.rs::advance` / `manager.rs::update_step` untested.

ST_WAITING, ST_RUNNING, ST_PREEMPTED = 0, 1, 2
SPEC_TOKEN = 4242  # never an eos and never a stop id, so only max_tokens ends a request
# The worker is parked on a condvar and a decode-step schedule is microseconds of work;
# 20ms is four orders of magnitude of slack for a loaded CI box. Only the HIT count
# depends on it -- every correctness assertion below holds whether or not it lands.
SPEC_SETTLE = 0.02


class Req:
    """The `pack_req` fields plus the two counters `update_step` needs. Deliberately a
    plain object: the point is to mirror what `rust_sched.py` reads off a vLLM Request."""

    __slots__ = (
        "rid", "tokens", "slot", "num_tokens", "num_tokens_with_spec",
        "num_computed_tokens", "num_output_placeholders", "num_prompt_tokens",
        "max_tokens", "status", "num_preemptions", "is_prefill_chunk",
        "skip_reading_prefix_cache", "num_output_tokens", "finished", "pushed",
    )

    def __init__(self, rid, tokens, max_tokens):
        self.rid, self.tokens, self.max_tokens = rid, tokens, max_tokens
        self.slot = -1
        self.num_tokens = self.num_tokens_with_spec = self.num_prompt_tokens = len(tokens)
        self.num_computed_tokens = self.num_output_placeholders = 0
        self.num_output_tokens = self.num_preemptions = 0
        self.status = ST_WAITING
        self.is_prefill_chunk = False
        self.skip_reading_prefix_cache = False
        self.finished = False
        self.pushed = 0


def pack(r):
    """`rust_sched.py::pack_req`, field for field."""
    return (
        r.slot, r.num_tokens, r.num_tokens_with_spec, r.num_computed_tokens,
        r.num_output_placeholders, r.num_prompt_tokens, r.max_tokens, r.status,
        r.num_preemptions, r.is_prefill_chunk, r.skip_reading_prefix_cache,
    )


def advance(r, num_new, sampled=1):
    """`sched.rs::advance` == `Scheduler._update_after_schedule` + the Async override."""
    r.num_computed_tokens += num_new
    r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens + r.num_output_placeholders
    if not r.is_prefill_chunk:
        r.num_output_placeholders += sampled


def sched_params(block_size=16, max_model_len=2048, max_tokens_per_step=96):
    return {
        "max_num_scheduled_tokens": max_tokens_per_step,
        "max_num_running_reqs": 8,
        "max_model_len": max_model_len,
        "num_sampled_tokens_per_step": 1,
        "long_prefill_token_threshold": 0,
        "enable_chunked_prefill": True,
        "need_mamba_block_aligned_split": True,
        "cache_block_size": block_size,
        "num_lookahead_tokens": 0,
        "sjf_reorder": False,
    }


def _prompt(rid, length, shared):
    """Shared prefix + a request-specific tail, so the prefix cache actually participates
    in the block IDs the arms are compared on."""
    rng = random.Random(rid)
    return list(shared) + [rng.randrange(2000, 60000) for _ in range(length - len(shared))]


class Engine:
    """One arm. `mode` is 'marshalled' | 'resident' | 'spec'.

    Everything Python-side (queues, counters, admission bookkeeping) is identical across
    arms; only the Rust entry point differs. `unregistered` names requests whose stop
    params are never interned, which is how the status-255 fallback is provoked.
    """

    def __init__(self, mode, script, num_blocks=48, unregistered=(), inject=None,
                 block_size=16, max_model_len=2048):
        self.mode = mode
        self.script = script
        self.unregistered = set(unregistered)
        self.inject = list(inject or ())
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.none_hash = vtl_sched.none_hash_from_seed("0")
        self.kv = vtl_sched.KvManager(
            rust_config(num_blocks=num_blocks, block_size=block_size,
                        max_model_len=max_model_len)
        )
        self.core = vtl_sched.Scheduler()
        self.core.set_params(sched_params(block_size, max_model_len))
        self.shared = [1000 + i for i in range(64)]
        self.running, self.waiting, self.by_slot = [], [], {}
        self.n = 0
        self.prev_batch = []
        # rust_sched.py::TableState, in miniature.
        self.dirty, self.gen, self.armed, self.clean = True, 0, False, True
        self.hits, self.resyncs, self.resident_steps = 0, 0, 0
        self.probes = {}  # injection kind -> how many times it actually refused one

    # -- driver ----------------------------------------------------------

    def step(self):
        self.n += 1
        for rid, plen, max_tokens in self.script.get(self.n, ()):
            self._arrive(rid, plen, max_tokens)
        prev = self.prev_batch
        d = self._schedule()
        self.prev_batch = self._apply(d)
        self._update(prev)
        return d

    def _arrive(self, rid, plen, max_tokens):
        r = Req(rid, _prompt(rid, plen, self.shared), max_tokens)
        r.slot = self.kv.intern(rid)
        self._push(r)
        if rid not in self.unregistered:
            self.kv.set_stop_params(r.slot, 0, max_tokens, None, [])
        self.by_slot[r.slot] = r
        self.waiting.append(r)
        self.gen += 1  # rust_sched.py bump site 1 (Scheduler.add_request)
        self.armed = False

    def _push(self, r):
        """`RustMirror.slot`: hand Rust the block hashes Python's hasher produced, for the
        prompt at arrival and for every full block the generated tokens complete after.

        Deliberately NOT a generation bump: `push_hashes` goes through the crate's
        invalidating `w()` guard, so a pending speculation is rolled back by Rust itself
        and the following `take_speculative` simply misses -- which is what serving does.
        """
        hashes = vtl_sched.block_hashes(self.none_hash, self.block_size, r.tokens)
        if len(hashes) > r.pushed:
            self.kv.push_hashes(
                r.slot, b"".join(hashes[r.pushed:]), r.num_prompt_tokens
            )
            r.pushed = len(hashes)

    def _schedule(self):
        # Hashes are pushed BEFORE scheduling on every path, exactly as rust_sched.py's
        # marshalling loop does.
        for r in self.running:
            self._push(r)
        for r in self.waiting:
            self._push(r)
        slots = [r.slot for r in self.running]
        waiting = [pack(r) for r in self.waiting]
        if self.mode == "marshalled" or self.dirty:
            if self.dirty and self.mode != "marshalled":
                self.resyncs += 1
            d = self.core.schedule(self.kv, [pack(r) for r in self.running], waiting)
            self.dirty = self.armed = False
            return d
        # Clean resident step. The table must already equal what pack_req would send;
        # assert it directly, but only outside a kick/take window (`table_entry` reads
        # through a pending speculation by design -- python.rs).
        self.resident_steps += 1
        if self.mode == "resident":
            for r in self.running:
                assert self.kv.table_entry(r.slot) == pack(r), f"step {self.n}: {r.rid}"
        d = None
        if self.mode == "spec" and self.armed and not waiting:
            d = self._take(slots)
        self.armed = False
        if d is None:
            d = self.core.schedule_resident(self.kv, slots, waiting)
        return d

    def _take(self, slots):
        """Consume the speculation, or prove that an invalidation refused it.

        Injections are consumed by ORDINAL armed take, not by step number: which steps end
        up with a speculation armed depends on the pool pressure, so a fixed step list
        silently degrades into no coverage at all.
        """
        probe = self.inject[0] if self.inject else None
        if probe == "order" and len(slots) < 2:
            probe = None  # needs two running requests to be a different order at all
        if probe is not None:
            self.inject.pop(0)
            self.probes[probe] = self.probes.get(probe, 0) + 1
        if probe == "gen":
            assert self.core.take_speculative(self.kv, self.gen + 1, slots) is None, (
                "a stale generation must not be consumed"
            )
            return None
        if probe == "order":
            assert len(slots) > 1, "the slot-order probe needs at least two running reqs"
            assert self.core.take_speculative(self.kv, self.gen, slots[::-1]) is None, (
                "a different running order must not be consumed"
            )
            return None
        if probe == "table_set":
            # Simulates an out-of-band mutation (abort, KV-load rollback): writing the
            # SAME values back still invalidates, so the decisions cannot change.
            self.kv.table_set(slots[0], pack(self.by_slot[slots[0]]))
            assert self.core.take_speculative(self.kv, self.gen, slots) is None, (
                "table_set between kick and take must invalidate"
            )
            return None
        d = self.core.take_speculative(self.kv, self.gen, slots)
        if d is not None:
            self.hits += 1
        return d

    def _apply(self, d):
        """`rust_sched.py`'s phase C, python-object half. Returns the requests that will
        carry a sampled token into the next `update_step`."""
        sampled = []
        for slot, num_new in d["scheduled_running"]:
            r = self.by_slot[slot]
            advance(r, num_new)
            if not r.is_prefill_chunk:
                sampled.append(r)
        self.waiting = [self.by_slot[s] for s in d["waiting_order"]]
        for slot in d["preempted"]:
            r = self.by_slot[slot]
            self.running.remove(r)
            r.status = ST_PREEMPTED
            r.num_computed_tokens = 0
            r.num_preemptions += 1
            self.waiting.insert(0, r)  # one-at-a-time prepend, as `_preempt_request` does
        for slot, num_new, num_computed in d["scheduled_admitted"]:
            r = self.by_slot[slot]
            self.running.append(r)
            r.status = ST_RUNNING
            r.num_computed_tokens = num_computed
            advance(r, num_new)
            if not r.is_prefill_chunk:
                sampled.append(r)
        return sampled

    def _update(self, batch):
        """`update_from_output` for the PREVIOUS step, then the kick."""
        self.clean = True
        batch = [r for r in batch if not r.finished]
        if batch:
            slots, cu, toks, n_out, n_tok = [], [0], [], [], []
            for r in batch:
                slots.append(r.slot)
                toks.append(SPEC_TOKEN)
                cu.append(len(toks))
                n_out.append(r.num_output_tokens)
                n_tok.append(r.num_tokens)
            out = self.kv.update_step(slots, cu, toks, n_out, n_tok, self.max_model_len)
            for r, (accepted, status, _reason) in zip(batch, out):
                if status == 255:
                    # No interned params: Rust refused and left the table untouched, so
                    # python owns both the verdict and the resync.
                    accepted, status, _ = ref_check_stop(
                        (0, r.max_tokens, None, ()), [SPEC_TOKEN],
                        r.num_output_tokens, r.num_tokens, self.max_model_len,
                    )
                    self.dirty = True
                    self.clean = False
                    self.gen += 1
                    self.armed = False
                r.tokens.extend([SPEC_TOKEN] * accepted)
                r.num_tokens += accepted
                r.num_tokens_with_spec += accepted
                r.num_output_tokens += accepted
                r.num_output_placeholders = max(0, r.num_output_placeholders - accepted)
                if status != NOT_STOPPED:
                    self._finish(r)
        if self.mode == "spec":
            self._kick()

    def _finish(self, r):
        r.finished = True
        if r in self.running:
            self.running.remove(r)
        if r in self.waiting:
            self.waiting.remove(r)
        self.kv.free(r.slot)
        self.kv.forget(r.rid)
        self.gen += 1  # rust_sched.py bump site 2 (finish_requests / _free_request)
        self.armed = False

    def _kick(self):
        if self.dirty or not self.clean or self.waiting or not self.running:
            return
        # Push the hashes the tokens just appended completed, BEFORE kicking: the worker
        # walks the resident table, which already counts those tokens. rust_sched.py's
        # kick site does the same, via `mirror.slot`.
        for r in self.running:
            self._push(r)
        slots = [r.slot for r in self.running]
        if self.core.kick(self.kv, self.gen, slots):
            self.armed = True
            time.sleep(SPEC_SETTLE)


def _workload():
    """One script that reaches every state transition the table has to survive.

    * step 1-6 arrivals -> admissions, and 260-token prompts against a 96-token step
      budget -> chunked prefill over three steps each
    * 48 blocks x 16 tokens = 768 tokens of pool against ~1300 tokens of running set
      -> preemption, and the preempted requests resume from the waiting queue
    * max_tokens 4-7 -> finishes through `update_step`, mid-run, at different steps
    * `c3` never registers its stop params -> a status-255 step and the resync after it
    """
    return {
        1: [("c0", 260, 16)],
        2: [("c1", 260, 12)],
        3: [("c2", 300, 20)],
        5: [("c3", 220, 9)],
        6: [("c4", 300, 14)],
        11: [("c5", 180, 18)],
    }


def _run(mode, steps=70, **kw):
    eng = Engine(mode, _workload(), **kw)
    return eng, [eng.step() for _ in range(steps)]


@requires_crate
def test_resident_table_matches_the_marshalled_schedule():
    """R6b: `schedule_resident` must decide exactly what `schedule` decides, every step."""
    ref, want = _run("marshalled", unregistered=("c3",))
    got_eng, got = _run("resident", unregistered=("c3",))

    for i, (a, b) in enumerate(zip(want, got), 1):
        assert a == b, f"step {i}: marshalled {a} != resident {b}"
    assert got_eng.resyncs, "the status-255 fallback never forced a resync"
    assert got_eng.resident_steps > got_eng.resyncs * 2, (
        f"only {got_eng.resident_steps} steps actually took the resident path"
    )
    assert ref.kv.num_free_blocks == got_eng.kv.num_free_blocks

    # The fixture has to actually reach the transitions it claims to cover, or this tier
    # passes vacuously.
    kinds = {
        "admitted": sum(len(d["scheduled_admitted"]) for d in want),
        "preempted": sum(len(d["preempted"]) for d in want),
        "running": sum(len(d["scheduled_running"]) for d in want),
    }
    assert kinds["admitted"] >= 6, kinds
    assert kinds["preempted"] >= 1, kinds
    assert kinds["running"] >= 20, kinds
    assert any(r.finished for r in ref.by_slot.values()), "no request finished"


@requires_crate
def test_speculation_returns_the_same_decisions():
    """R6c: a consumed speculation is the real schedule, and it must be bit-identical."""
    _ref, want = _run("marshalled")
    eng, got = _run("spec")
    for i, (a, b) in enumerate(zip(want, got), 1):
        assert a == b, f"step {i}: marshalled {a} != speculative {b}"
    assert eng.hits > 0, "no speculation was ever consumed -- the tier proved nothing"
    hits, misses, rollbacks, disabled = eng.kv.spec_stats()
    assert not disabled, "the worker disabled itself (panic or poisoned lock)"
    assert hits == eng.hits, (hits, eng.hits)
    assert misses + rollbacks >= 0


@requires_crate
def test_speculation_refuses_every_invalidation():
    """Each injection must return None from `take_speculative` AND leave the fallback
    schedule identical to the no-speculation oracle -- i.e. the rollback was exact."""
    _ref, want = _run("marshalled")
    inject = ["gen", "order", "table_set", "order", "gen", "table_set"]
    eng, got = _run("spec", inject=inject)
    for i, (a, b) in enumerate(zip(want, got), 1):
        assert a == b, f"step {i}: marshalled {a} != post-invalidation {b}"
    # An injection only proves something on a step that actually had a speculation armed;
    # all three KINDS must have refused one, or the tier is vacuous.
    assert set(eng.probes) == {"gen", "order", "table_set"}, eng.probes
    hits, misses, rollbacks, disabled = eng.kv.spec_stats()
    assert not disabled
    # Each refusal is a miss, and a refusal with a pending speculation is also a rollback.
    assert misses >= sum(eng.probes.values()), (misses, eng.probes)
    assert rollbacks >= len(eng.probes), rollbacks


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
