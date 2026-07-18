"""
Phase-0 spike (Crux 1): LFM2 short-conv TREE-commit correctness.

The fork claim under test: after a speculative *tree* verify, the LFM2 short-conv
state can be committed by gathering the last K-1 conv-inputs (Bx) along the
ACCEPTED PATH -- and vLLM's existing scalar `num_accepted_tokens` contiguous-prefix
commit is WRONG for a tree (it grabs whichever nodes sit at those flat-buffer slots,
which include rejected siblings).

Why it's correct: the short conv is a depthwise causal FIR of width K=conv_L_cache.
A position's output depends only on its last K conv-inputs, and the post-position
state is the last K-1 of them -- zero dependence on anything older or on sibling
branches. So the accepted leaf's state == the last K-1 Bx along its root->leaf path.

LFM2.5 (hf-model/config.json): conv_L_cache=3 -> K=3, window = K-1 = 2, conv_dim=2048.
Pure Python (no torch/numpy) so it runs on a laptop and on the H200 box unchanged.
"""

K = 3                 # conv_L_cache for LFM2.5
STATE_LEN = K - 1     # committed conv-state window = 2
C = 4                 # channels (conv_dim=2048 in the real model; small here for the check)

# Deterministic depthwise causal FIR: per-channel taps w[c][k] + bias.
w = [[(c + 1) * 0.1 + k * 0.03 for k in range(K)] for c in range(C)]
bias = [0.5 - 0.07 * c for c in range(C)]


def vec(seed):
    """Deterministic value vector, distinct per seed (stands in for a Bx input)."""
    return [((seed * 7 + c * 13 + 3) % 11) * 0.1 - 0.5 for c in range(C)]


def conv_out(window):
    """Depthwise causal conv output for the current position; window = K inputs, oldest..newest."""
    return [bias[c] + sum(w[c][k] * window[k][c] for k in range(K)) for c in range(C)]


# Conv state already cached before the speculative tree (last K-1 real-token inputs).
init_state = [vec(101), vec(102)]  # length STATE_LEN

# Draft tree via parent pointers. Node 0 = root (extends init_state).
#   0 -> {1, 2}; 1 -> {3, 4}; 3 -> {5}
parent = {0: None, 1: 0, 2: 0, 3: 1, 4: 1, 5: 3}
bx = {n: vec(200 + n) for n in parent}        # each node's staged conv-input Bx
bfs_flat = [0, 1, 2, 3, 4, 5]                  # a linearization (like vLLM's flat draft buffer)


def path_inputs(accepted_path):
    """Full conv-input timeline: cached window + the accepted path's Bx, in order."""
    return list(init_state) + [bx[n] for n in accepted_path]


def reference_chain(accepted_path):
    """Ground truth: run the conv as a plain linear sequence over the accepted path."""
    seq, outs = list(init_state), []
    for n in accepted_path:
        seq.append(bx[n])
        outs.append(conv_out(seq[-K:]))
    return outs, seq[-STATE_LEN:]              # per-step outputs, final state window


def path_gather_commit(accepted_path):
    """Fork's tree-aware commit: last K-1 conv-inputs along the accepted path."""
    return path_inputs(accepted_path)[-STATE_LEN:]


def scalar_prefix_commit(accepted_path):
    """vLLM's current mechanism: last K-1 of the flat buffer's first `num_accepted` slots."""
    num_accepted = len(accepted_path)
    prefix_nodes = bfs_flat[:num_accepted]
    return [bx[n] for n in prefix_nodes][-STATE_LEN:]


def approx(a, b, eps=1e-12):
    fa = [x for v in a for x in v]
    fb = [x for v in b for x in v]
    return len(fa) == len(fb) and all(abs(x - y) <= eps for x, y in zip(fa, fb))


def check(accepted_path, label):
    ref_outs, ref_state = reference_chain(accepted_path)
    # Per-node ancestor-window output equals the chain output (FIR equivalence anchor).
    for i, n in enumerate(accepted_path):
        seq = path_inputs(accepted_path[: i + 1])
        assert approx([conv_out(seq[-K:])], [ref_outs[i]]), f"{label}: node {n} output mismatch"
    # The load-bearing claim: path-gather commit == reference final state.
    assert approx(path_gather_commit(accepted_path), ref_state), f"{label}: gather commit wrong"
    return ref_state


# 1) Full accept root->1->3->5: gather commit matches the chain reference.
check([0, 1, 3, 5], "full-accept")

# 2) Partial accept root->1 (drafts 3,5 rejected): commit must follow the accepted leaf.
check([0, 1], "partial-accept")

# 3) Sibling branches must not collide: root->1 and root->2 give different states.
s1 = check([0, 1], "branch-1")
s2 = check([0, 2], "branch-2")
assert not approx(s1, s2), "sibling branches produced identical committed state"

# 4) The decisive one: vLLM's scalar-prefix commit is WRONG for a tree; the gather is right.
accepted = [0, 1, 3, 5]
_, ref_state = reference_chain(accepted)
assert approx(path_gather_commit(accepted), ref_state), "gather commit should equal reference"
assert not approx(scalar_prefix_commit(accepted), ref_state), (
    "scalar-prefix commit unexpectedly matched -- pick a tree where it must differ"
)
# Concretely: for num_accepted=4 the prefix is nodes [0,1,2,3] -> window [bx[2], bx[3]],
# but node 2 is a REJECTED sibling of node 1. The correct window is [bx[3], bx[5]].
assert bx[2] in scalar_prefix_commit(accepted) and bx[2] not in path_gather_commit(accepted), (
    "expected the rejected sibling (node 2) to leak into the scalar-prefix commit only"
)

print("PASS: LFM2 short-conv tree-commit spike (K=3, window=2)")
print("  - path-gather commit (last K-1 accepted Bx) == linear-chain reference, all cases")
print("  - full/partial accept + sibling branches all correct and isolated")
print("  - vLLM's scalar-prefix commit provably wrong: rejected sibling node 2 leaks in")
print("  => Crux 1 mechanism validated; the fork must thread accepted-path node ids, not a scalar.")
