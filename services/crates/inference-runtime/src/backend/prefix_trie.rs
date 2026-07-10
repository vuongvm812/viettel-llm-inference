//! Pure token radix trie for the P7 shared-prefix cache — generalizes P4's single
//! fixed-K prefix to **arbitrary, variable-length, nested** shared prefixes with
//! longest-prefix match.
//!
//! This module is backend-independent and **fully unit-tested in the default build**:
//! it holds no KV cache and does no GPU work. It tracks, for each *materialized* prefix,
//! a llama sequence id whose KV holds that prefix's `0..depth` cells (the real backend
//! attaches the actual `copy_kv_cache_seq`; the mock only exercises the accounting). Both
//! backends drive the same trie, so the mock stays an exact oracle for the real path.
//!
//! Structure: an edge-compressed (radix) trie over token ids. A node's `depth` is its
//! absolute prefix length (root = 0). A node is **materialized** when it owns a prefix
//! `seq_id` (its `0..depth` cells are resident) and becomes **ready** once the
//! establishing request finishes prefilling it (P7 chunked prefill — until then a matching
//! request must defer rather than copy empty KV). [`longest_match`](PrefixTrie::longest_match)
//! returns the deepest ready materialized node whose path is a prefix of the query.
//!
//! Physical model / ceiling: llama.cpp KV is **copy-based**, not block-ref-counted, so each
//! materialized node owns an independent `0..depth` cell run — a nested pair (5K ⊂ 8K) that
//! are both materialized cost 5K+8K cells, not 8K. Nested *matching* still works (an 8K
//! sharer reuses 8K, a 5K-only sharer reuses 5K), and bounded capacity + LRU eviction keep
//! the seq count in check. // ponytail: copy-based physical sharing, no cross-node KV dedup;
//! upgrade path is block-ref-counted KV (SGLang-style) if the crate ever exposes it.

#![allow(dead_code)] // internal utility: some methods are used by the mock, some by the real
                     // backend, some only by the unit tests — the full API stays for both builds.

/// Arena index of a trie node.
pub type NodeId = usize;

/// Result of [`PrefixTrie::longest_match`]: how many leading tokens of the query are
/// already resident in some ready materialized node, and which node holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Length of the longest ready materialized prefix of the query (`0` = no match).
    pub reused: usize,
    /// The node holding those `reused` cells (`None` when `reused == 0`).
    pub node: Option<NodeId>,
}

/// The full reuse/establish decision from [`PrefixTrie::plan_reuse`] — the shared cache
/// policy both backends apply (see that method).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixReuse {
    /// Cells to copy from a resident ready prefix (0 = full prefill). Capped to `n_prompt-1`.
    pub reused: usize,
    /// The node to copy those `reused` cells from (`None` when `reused == 0`).
    pub match_node: Option<NodeId>,
    /// The fork node to materialize when establishing (`None` when not establishing).
    pub fork: Option<NodeId>,
    /// Incremental fork cells to keep reserved when establishing (`None` = don't establish).
    pub establish_new_cells: Option<usize>,
    /// The request matches a materialized-but-not-yet-ready prefix (its establisher is still
    /// prefilling) → the caller must defer, not copy from empty KV.
    pub defer_unready: bool,
}

/// Everything one [`PrefixTrie::walk`] of a query yields (all `None`/`false` if only the root
/// matched): the deepest ready-materialized node, the deepest materialized node, and the
/// deepest unmaterialized fork.
struct WalkResult {
    deepest_ready: Option<NodeId>,
    deepest_mat: Option<NodeId>,
    deepest_fork: Option<NodeId>,
}

struct Node {
    /// Token run on the edge from `parent` into this node (radix / edge-compressed).
    edge: Vec<i32>,
    /// Absolute prefix depth = `parent.depth + edge.len()` (root depth `0`).
    depth: usize,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    /// llama seq id holding this prefix's `0..depth` KV cells, or `None` for a purely
    /// structural node (an internal split point that isn't itself cached).
    seq_id: Option<i32>,
    /// `false` while the establishing request is still prefilling these cells.
    ready: bool,
    /// Monotonic use tick for LRU eviction (bumped on establish + every reuse).
    last_used: u64,
}

/// Bounded token radix trie of materialized shared prefixes.
pub struct PrefixTrie {
    nodes: Vec<Node>,
    /// Max materialized (seq-owning) nodes — bounds the reserved prefix seq ids.
    cap: usize,
    /// Minimum shared depth worth materializing (the `shared_prefix_tokens` knob): a
    /// prefix shorter than this falls back to a full prefill rather than caching a tiny run.
    min_depth: usize,
    /// LRU clock; `0` is never a valid stamp so it can flag "never used".
    tick: u64,
}

impl PrefixTrie {
    /// `cap` materialized nodes max; `min_depth` = `shared_prefix_tokens` (0 disables — the
    /// trie then never materializes, i.e. P3 behavior).
    pub fn new(cap: usize, min_depth: usize) -> Self {
        let root = Node {
            edge: Vec::new(),
            depth: 0,
            parent: None,
            children: Vec::new(),
            seq_id: None,
            ready: true, // the empty prefix is trivially resident
            last_used: 0,
        };
        PrefixTrie { nodes: vec![root], cap, min_depth, tick: 0 }
    }

    /// Sharing is disabled (P3 behavior) when `min_depth == 0`.
    pub fn enabled(&self) -> bool {
        self.min_depth > 0
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Count of materialized (seq-owning) nodes — bounded by `cap`.
    pub fn materialized_len(&self) -> usize {
        self.nodes.iter().filter(|n| n.seq_id.is_some()).count()
    }

    /// Total resident prefix cells = Σ each materialized node's *incremental* depth over its
    /// nearest materialized ancestor. A lone chain (root → A@5000 → B@8000) with both
    /// materialized counts 8000 (A=5000, B=+3000) — the copy-based backend physically stores
    /// B's 0..8000, but a request only ever reserves the *incremental* suffix it prefills, so
    /// the KV budget (which admission checks) charges each shared run once. Independent
    /// prefixes (distinct roots' children) sum their full depths.
    pub fn reserved_cells(&self) -> usize {
        let mut total = 0;
        for (id, n) in self.nodes.iter().enumerate() {
            if n.seq_id.is_some() {
                let anc = self.nearest_materialized_ancestor(id);
                let anc_depth = anc.map(|a| self.nodes[a].depth).unwrap_or(0);
                total += n.depth - anc_depth;
            }
        }
        total
    }

    fn nearest_materialized_ancestor(&self, id: NodeId) -> Option<NodeId> {
        let mut cur = self.nodes[id].parent;
        while let Some(p) = cur {
            if self.nodes[p].seq_id.is_some() {
                return Some(p);
            }
            cur = self.nodes[p].parent;
        }
        None
    }

    /// Longest ready materialized prefix of `tokens`. Walks edges greedily; along the way
    /// records the deepest node that is both materialized and ready.
    pub fn longest_match(&self, tokens: &[i32]) -> Match {
        let w = self.walk(tokens);
        match w.deepest_ready {
            Some(id) => Match { reused: self.nodes[id].depth, node: Some(id) },
            None => Match { reused: 0, node: None },
        }
    }

    /// The complete shared-prefix reuse/establish decision for `tokens` — the SINGLE source
    /// of truth both backends consume (they then apply only their own KV side effects). Pure,
    /// so it stays unit-tested in the default build and the mock can't drift from the real
    /// backend's cache policy.
    ///
    /// `reused` is the deepest ready materialized prefix (capped at `n_prompt - 1` so ≥ 1
    /// token still decodes for the first-sample logits). `establish_new_cells` is `Some(new)`
    /// when this request should cache a newly-discovered fork, where `new` is the fork's
    /// *incremental* depth over what's already reused.
    ///
    /// Establish gate — a fork is only cached when it (a) is deeper than `reused`, (b) fits
    /// the cache (`can_materialize`), and (c) has **no materialized descendant**. Rule (c) is
    /// the accounting invariant: `new = depth - reused` charges the run once only if the
    /// fork's nearest materialized ancestor is exactly the `reused` node. A fork with a
    /// materialized descendant would be a shallower ancestor established *after* a deeper one
    /// — its shared run is already charged by the descendant, so caching it here would
    /// double-count `reserved` (a permanent leak that inflates admission pressure). Skipping
    /// it costs a little sharing on an out-of-order lineage; correctness wins.
    pub fn plan_reuse(&self, tokens: &[i32], n_prompt: usize) -> PrefixReuse {
        // One trie walk yields everything: the deepest ready match, the deepest materialized
        // node (for the readiness gate), and the deepest unmaterialized fork — so the fast
        // loop traverses a long prompt once, not three times.
        let w = self.walk(tokens);
        let reused = w.deepest_ready.map_or(0, |id| self.nodes[id].depth).min(n_prompt.saturating_sub(1));
        let defer_unready = w.deepest_mat.is_some_and(|id| !self.nodes[id].ready);
        let (fork, establish_new_cells) = match w.deepest_fork {
            Some(f)
                if self.nodes[f].depth > reused
                    && self.can_materialize()
                    && !self.has_materialized_descendant(f) =>
            {
                (Some(f), Some(self.nodes[f].depth - reused))
            }
            _ => (None, None),
        };
        PrefixReuse { reused, match_node: w.deepest_ready, fork, establish_new_cells, defer_unready }
    }

    /// Does some node matching `tokens` exist that is materialized but **not yet ready**
    /// (its establisher is still prefilling)? The caller defers such a request so it can't
    /// copy from empty KV. Only the deepest matched materialized node matters — a ready
    /// ancestor of an unready node is still safe to share, so we report unready only when
    /// the *deepest reachable* materialized node on the path is unready.
    pub fn matches_unready(&self, tokens: &[i32]) -> bool {
        self.walk(tokens).deepest_mat.is_some_and(|id| !self.nodes[id].ready)
    }

    /// Walk `tokens` down the trie as far as the path matches, recording in one pass the
    /// deepest ready-materialized node, the deepest materialized node, and the deepest
    /// unmaterialized fork (≥ 2 children, depth ≥ `min_depth`) — all `None` if only the root
    /// matched. A partial edge match stops the walk.
    fn walk(&self, tokens: &[i32]) -> WalkResult {
        let mut cur = 0usize; // root
        let mut w = WalkResult { deepest_ready: None, deepest_mat: None, deepest_fork: None };
        self.visit(cur, &mut w);
        'walk: loop {
            let base = self.nodes[cur].depth;
            if base >= tokens.len() {
                break;
            }
            // Find the child whose edge's first token matches tokens[base].
            let mut advanced = false;
            for &child in &self.nodes[cur].children {
                let edge = &self.nodes[child].edge;
                if edge.first() == Some(&tokens[base]) {
                    // Match the whole edge run against tokens.
                    let end = base + edge.len();
                    if end <= tokens.len() && tokens[base..end] == edge[..] {
                        cur = child;
                        self.visit(cur, &mut w);
                        advanced = true;
                        break;
                    } else {
                        // Partial edge match — the query diverges mid-edge; stop here.
                        break 'walk;
                    }
                }
            }
            if !advanced {
                break;
            }
        }
        w
    }

    /// Fold a node visited on the walk into the running `WalkResult`.
    fn visit(&self, id: NodeId, w: &mut WalkResult) {
        let n = &self.nodes[id];
        if n.seq_id.is_some() {
            w.deepest_mat = Some(id);
            if n.ready {
                w.deepest_ready = Some(id);
            }
        } else if self.enabled() && n.depth >= self.min_depth && n.children.len() >= 2 {
            // Unmaterialized fork (genuine shared prefix of ≥ 2 requests) worth caching.
            // Gated on `enabled()` so sharing-off (min_depth == 0) never establishes.
            w.deepest_fork = Some(id);
        }
    }

    /// Insert a materialized prefix of `tokens` at `depth`, bound to `seq_id`, `ready =
    /// false`. Splits an edge when the path diverges mid-run (radix insert). Returns the new
    /// node's id, or `None` if `depth < min_depth`, `depth == 0`, `depth > tokens.len()`, or
    /// the trie is at `cap` with nothing idle-evictable (caller then falls back to a plain
    /// full prefill). The caller is responsible for donating the actual KV cells to `seq_id`.
    pub fn insert(&mut self, tokens: &[i32], depth: usize, seq_id: i32) -> Option<NodeId> {
        if !self.enabled() || depth == 0 || depth < self.min_depth || depth > tokens.len() {
            return None;
        }
        // Descend to the node whose depth is the largest <= `depth` on the matching path,
        // splitting an edge if `depth` lands inside it.
        let node = self.descend_to_depth(tokens, depth)?;
        if self.nodes[node].depth != depth {
            return None; // couldn't reach exactly `depth` (path diverged before it)
        }
        if self.nodes[node].seq_id.is_some() {
            // Already materialized at this exact depth — reuse it (refresh readiness).
            self.nodes[node].seq_id = Some(seq_id);
            self.nodes[node].ready = false;
            let t = self.next_tick();
            self.nodes[node].last_used = t;
            return Some(node);
        }
        if self.materialized_len() >= self.cap {
            return None; // full — caller evicts via `evict_lru_leaf` (keeps its budget in sync) then retries
        }
        self.nodes[node].seq_id = Some(seq_id);
        self.nodes[node].ready = false;
        let t = self.next_tick();
        self.nodes[node].last_used = t;
        Some(node)
    }

    /// Descend/extend the trie along `tokens` until reaching a node at exactly `depth`,
    /// splitting the edge that straddles `depth` and creating any missing tail. Returns the
    /// node at `depth`, or `None` if the existing path diverges from `tokens` before `depth`.
    fn descend_to_depth(&mut self, tokens: &[i32], depth: usize) -> Option<NodeId> {
        let mut cur = 0usize;
        loop {
            let base = self.nodes[cur].depth;
            if base == depth {
                return Some(cur);
            }
            debug_assert!(base < depth);
            // Find a child whose edge starts with tokens[base].
            let mut matched_child = None;
            for &child in &self.nodes[cur].children {
                if self.nodes[child].edge.first() == Some(&tokens[base]) {
                    matched_child = Some(child);
                    break;
                }
            }
            match matched_child {
                Some(child) => {
                    let child_depth = self.nodes[child].depth;
                    let edge_len = self.nodes[child].edge.len();
                    // How far does the query match this edge? Capped at `depth - base` so a
                    // `depth` landing strictly inside a longer fully-matching edge splits at
                    // `depth` instead of overshooting past it (would trip the `base < depth`
                    // assertion next iteration). `depth > base` holds (asserted above).
                    let limit = edge_len.min(depth - base);
                    let mut m = 0;
                    while m < limit && base + m < tokens.len() && tokens[base + m] == self.nodes[child].edge[m] {
                        m += 1;
                    }
                    if base + m == depth {
                        // `depth` lands inside (or at the end of) this edge.
                        if m == edge_len {
                            cur = child; // exactly at the child node
                        } else {
                            return Some(self.split_edge(child, m)); // split mid-edge at `depth`
                        }
                    } else if m == edge_len {
                        // Consumed the whole edge, still short of `depth` → descend.
                        cur = child;
                    } else {
                        // Diverged mid-edge before reaching `depth`: split, then extend a new
                        // branch for the query's tail (a genuine radix fork).
                        let split = self.split_edge(child, m);
                        return Some(self.extend(split, &tokens[base + m..depth]));
                    }
                    let _ = child_depth;
                }
                None => {
                    // No child starts here → append a fresh edge for the remaining tail.
                    return Some(self.extend(cur, &tokens[base..depth]));
                }
            }
        }
    }

    /// Split `child`'s edge after `m` tokens, inserting a new structural node between
    /// `child`'s parent and `child`. Returns the new intermediate node.
    fn split_edge(&mut self, child: NodeId, m: usize) -> NodeId {
        let parent = self.nodes[child].parent.expect("root has no edge to split");
        let base = self.nodes[parent].depth;
        let head: Vec<i32> = self.nodes[child].edge[..m].to_vec();
        let tail: Vec<i32> = self.nodes[child].edge[m..].to_vec();
        let mid = self.nodes.len();
        self.nodes.push(Node {
            edge: head,
            depth: base + m,
            parent: Some(parent),
            children: vec![child],
            seq_id: None,
            ready: true, // structural node has no cells to wait on
            last_used: 0,
        });
        // Re-parent child under mid with the shortened tail edge.
        self.nodes[child].edge = tail;
        self.nodes[child].parent = Some(mid);
        // Replace child with mid in parent's child list.
        let slot = self.nodes[parent].children.iter().position(|&c| c == child).expect("child linked to parent");
        self.nodes[parent].children[slot] = mid;
        mid
    }

    /// Append a fresh structural node under `parent` carrying edge `tail`.
    fn extend(&mut self, parent: NodeId, tail: &[i32]) -> NodeId {
        let base = self.nodes[parent].depth;
        let id = self.nodes.len();
        self.nodes.push(Node {
            edge: tail.to_vec(),
            depth: base + tail.len(),
            parent: Some(parent),
            children: Vec::new(),
            seq_id: None,
            ready: true,
            last_used: 0,
        });
        self.nodes[parent].children.push(id);
        id
    }

    /// Record a request's full prompt `tokens` as a **structural** path (no KV, no cap),
    /// splitting edges where it diverges from prior requests. This is how the trie *discovers*
    /// shared prefixes: when a second request diverges from a first, the split creates a fork
    /// node at the exact shared depth, which [`deepest_unmaterialized_fork`] can then cache.
    /// Bounded by `max_nodes` to keep the arena from growing without limit under a stream of
    /// distinct prompts; over the cap it no-ops (new sharing simply isn't discovered — the
    /// request falls back to a full prefill). // ponytail: arena cap, no structural-node LRU;
    /// upgrade to pruning cold structural leaves if distinct-prompt churn dominates.
    pub fn insert_structural(&mut self, tokens: &[i32], max_nodes: usize) {
        if !self.enabled() || tokens.is_empty() || self.nodes.len() >= max_nodes {
            return;
        }
        // Descending to the full length creates every fork along the way as a side effect.
        let _ = self.descend_to_depth(tokens, tokens.len());
    }

    /// The deepest **fork** node on `tokens`' path (a node with ≥ 2 children — a genuine
    /// shared prefix of two or more distinct requests) at depth `>= min_depth` that is not
    /// yet materialized. This is the best new prefix to cache: materializing it lets every
    /// request through that fork reuse its cells. `None` when no such fork exists (e.g. the
    /// first request of a lineage, before any divergence). Shares the single [`walk`] pass.
    pub fn deepest_unmaterialized_fork(&self, tokens: &[i32]) -> Option<NodeId> {
        self.walk(tokens).deepest_fork
    }

    /// Materialize an existing (structural, unmaterialized) node — bind it to `seq_id`,
    /// `ready = false`. Fails (returns `false`) if the trie is at `cap`. Used to cache a fork
    /// discovered by [`plan_reuse`], which only ever surfaces unmaterialized forks — so
    /// overwriting a live `seq_id` (which would leak the old prefix seq id) must not happen.
    pub fn materialize(&mut self, node: NodeId, seq_id: i32) -> bool {
        debug_assert!(
            self.nodes[node].seq_id.is_none(),
            "materialize on an already-materialized node would leak its prefix seq id"
        );
        if self.materialized_len() >= self.cap {
            return false; // full — caller evicts first (see `evict_lru_leaf`)
        }
        self.nodes[node].seq_id = Some(seq_id);
        self.nodes[node].ready = false;
        let t = self.next_tick();
        self.nodes[node].last_used = t;
        true
    }

    /// Room to materialize another prefix without eviction.
    pub fn can_materialize(&self) -> bool {
        self.materialized_len() < self.cap
    }

    /// Mark `node`'s cells resident (its establisher finished prefilling); bumps LRU.
    pub fn set_ready(&mut self, node: NodeId) {
        let t = self.next_tick();
        self.nodes[node].ready = true;
        self.nodes[node].last_used = t;
    }

    /// Record a reuse of `node` (bumps its LRU stamp so hot prefixes aren't evicted).
    pub fn touch(&mut self, node: NodeId) {
        let t = self.next_tick();
        self.nodes[node].last_used = t;
    }

    /// The prefix seq id backing `node` (for the real backend's `copy_kv_cache_seq`).
    pub fn seq_id(&self, node: NodeId) -> Option<i32> {
        self.nodes[node].seq_id
    }

    pub fn depth(&self, node: NodeId) -> usize {
        self.nodes[node].depth
    }

    /// Evict the least-recently-used **ready leaf** materialized node (a materialized node
    /// with no materialized descendants), freeing its seq id. Returns `(seq_id,
    /// incremental_cells)` — `incremental_cells` is the node's depth over its nearest
    /// materialized ancestor (what the KV budget charged for it, matching [`reserved_cells`]),
    /// for the caller to subtract from its running reservation; `seq_id` frees the real KV.
    /// Returns `None` if nothing is evictable. Only ready leaves are evicted — an unready node
    /// has an in-flight establisher, and an ancestor of a live deeper prefix must stay resident.
    pub fn evict_lru_leaf(&mut self) -> Option<(i32, usize)> {
        let victim = self.lru_evictable()?;
        self.demote_node(victim)
    }

    /// De-materialize a specific node (free its seq id, drop its cells). Returns `(seq_id,
    /// incremental_cells)` — same accounting as [`evict_lru_leaf`] — or `None` if it wasn't
    /// materialized. Used to drop an establisher's fork when its prefill ends in immediate
    /// EOS (the cells were never donated) or when the donation fails.
    pub fn demote_node(&mut self, node: NodeId) -> Option<(i32, usize)> {
        let anc = self.nearest_materialized_ancestor(node);
        let anc_depth = anc.map(|a| self.nodes[a].depth).unwrap_or(0);
        let incremental = self.nodes[node].depth - anc_depth;
        let seq_id = self.nodes[node].seq_id.take()?;
        self.nodes[node].ready = false;
        self.prune_if_structural(node);
        Some((seq_id, incremental))
    }

    fn lru_evictable(&self) -> Option<NodeId> {
        let mut best: Option<(u64, NodeId)> = None;
        for (id, n) in self.nodes.iter().enumerate() {
            if n.seq_id.is_none() || !n.ready {
                continue;
            }
            // A materialized leaf = no materialized descendant.
            if self.has_materialized_descendant(id) {
                continue;
            }
            match best {
                Some((t, _)) if n.last_used >= t => {}
                _ => best = Some((n.last_used, id)),
            }
        }
        best.map(|(_, id)| id)
    }

    fn has_materialized_descendant(&self, id: NodeId) -> bool {
        self.nodes[id].children.iter().any(|&c| self.nodes[c].seq_id.is_some() || self.has_materialized_descendant(c))
    }

    /// After de-materializing `id`, drop it if it's now a childless structural leaf (keeps
    /// the arena from accreting dead split nodes). Structural interior nodes are kept — they
    /// still route the radix walk.
    fn prune_if_structural(&mut self, id: NodeId) {
        if self.nodes[id].seq_id.is_none() && self.nodes[id].children.is_empty() {
            if let Some(parent) = self.nodes[id].parent {
                self.nodes[parent].children.retain(|&c| c != id);
            }
            // Leave the slot in place (arena index stability); it's unreachable now.
            self.nodes[id].parent = None;
        }
    }

    /// Drop everything (a full KV wipe): reset to just the root.
    pub fn clear(&mut self) {
        self.nodes.truncate(1);
        self.nodes[0].children.clear();
        self.tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixTrie;

    const CAP: usize = 8;
    const K: usize = 3; // min cache depth

    fn ready_insert(t: &mut PrefixTrie, tokens: &[i32], depth: usize, seq: i32) -> usize {
        let n = t.insert(tokens, depth, seq).expect("insert");
        t.set_ready(n);
        n
    }

    #[test]
    fn longest_match_returns_deepest_ready_node() {
        let mut t = PrefixTrie::new(CAP, K);
        ready_insert(&mut t, &[1, 2, 3], 3, 10);
        ready_insert(&mut t, &[1, 2, 3, 4, 5], 5, 11);
        // A query extending the deep node matches depth 5.
        assert_eq!(t.longest_match(&[1, 2, 3, 4, 5, 9]).reused, 5);
        // A query that follows the shallow prefix but diverges before 5 → depth 3.
        assert_eq!(t.longest_match(&[1, 2, 3, 9]).reused, 3);
        // No shared leading token → 0.
        assert_eq!(t.longest_match(&[9, 9, 9]).reused, 0);
    }

    #[test]
    fn insert_splits_edge_on_divergence() {
        let mut t = PrefixTrie::new(CAP, K);
        ready_insert(&mut t, &[1, 2, 3, 4], 4, 10);
        // A divergent prefix sharing [1,2] forces a split at depth 2 and a new branch.
        ready_insert(&mut t, &[1, 2, 9, 9], 4, 11);
        // Both deep prefixes are independently reusable...
        assert_eq!(t.longest_match(&[1, 2, 3, 4, 0]).reused, 4);
        assert_eq!(t.longest_match(&[1, 2, 9, 9, 0]).reused, 4);
        // ...and the shared [1,2] run is a single structural fork (a query diverging right
        // after [1,2] reuses neither deep leaf → 0, since [1,2] itself is below min_depth K=3
        // and never materialized).
        assert_eq!(t.longest_match(&[1, 2, 0]).reused, 0);
    }

    #[test]
    fn nested_shared_prefix_charges_incremental_cells() {
        let mut t = PrefixTrie::new(CAP, K);
        // A 3-deep prefix and a nested 5-deep prefix under it, both materialized.
        ready_insert(&mut t, &[1, 2, 3], 3, 10);
        ready_insert(&mut t, &[1, 2, 3, 4, 5], 5, 11);
        // Reserved counts the shared run once: 3 (ancestor) + 2 (incremental child) = 5.
        assert_eq!(t.reserved_cells(), 5, "nested prefixes charge the shared run once");
    }

    #[test]
    fn distinct_prefixes_sum_their_full_depths() {
        let mut t = PrefixTrie::new(CAP, K);
        ready_insert(&mut t, &[1, 2, 3], 3, 10);
        ready_insert(&mut t, &[7, 8, 9], 3, 11);
        assert_eq!(t.reserved_cells(), 6, "independent prefixes each charge their full depth");
    }

    #[test]
    fn matches_unready_gates_double_establish() {
        let mut t = PrefixTrie::new(CAP, K);
        let n = t.insert(&[1, 2, 3], 3, 10).expect("insert"); // not ready yet
        assert!(t.matches_unready(&[1, 2, 3, 4]), "matching an unready prefix must defer");
        t.set_ready(n);
        assert!(!t.matches_unready(&[1, 2, 3, 4]), "ready prefix no longer defers");
    }

    #[test]
    fn evict_lru_leaf_frees_oldest_when_full() {
        let mut t = PrefixTrie::new(2, K); // cap 2 materialized nodes
        ready_insert(&mut t, &[1, 1, 1], 3, 10);
        ready_insert(&mut t, &[2, 2, 2], 3, 11);
        // Touch the second so the first is the LRU victim.
        let n2 = t.longest_match(&[2, 2, 2]).node.unwrap();
        t.touch(n2);
        let (seq, depth) = t.evict_lru_leaf().expect("something evictable");
        assert_eq!((seq, depth), (10, 3), "evicts the least-recently-used leaf");
        assert_eq!(t.longest_match(&[1, 1, 1]).reused, 0, "evicted prefix no longer matches");
        assert_eq!(t.longest_match(&[2, 2, 2]).reused, 3, "the touched prefix survives");
    }

    #[test]
    fn fork_discovery_caches_the_shared_prefix_of_two_requests() {
        // Two requests share a 4-token prefix [1,2,3,4] then diverge. Structural inserts of
        // both create a fork at depth 4; the second request can then cache exactly that
        // shared prefix and later requests reuse it (variable-length, discovered — not a
        // fixed K window). Uses cap large, min_depth K=3.
        let mut t = PrefixTrie::new(CAP, K);
        let r1 = [1, 2, 3, 4, 5, 5];
        let r2 = [1, 2, 3, 4, 8, 8];
        t.insert_structural(&r1, 1024);
        // Before r2 there is no fork (single path) → nothing to cache.
        assert!(t.deepest_unmaterialized_fork(&r1).is_none(), "no fork on a lone path");
        t.insert_structural(&r2, 1024);
        let fork = t.deepest_unmaterialized_fork(&r2).expect("fork at the shared prefix");
        assert_eq!(t.depth(fork), 4, "fork is the exact 4-token shared prefix");
        // Cache it (r2 donates its 0..4 cells), mark ready.
        assert!(t.materialize(fork, 42));
        t.set_ready(fork);
        // A third request sharing the prefix reuses all 4 shared tokens.
        assert_eq!(t.longest_match(&[1, 2, 3, 4, 9]).reused, 4, "later requests reuse the discovered prefix");
        // A request sharing a *deeper* nested run establishes a deeper fork on top.
        let r3 = [1, 2, 3, 4, 8, 8, 7]; // extends r2's branch
        t.insert_structural(&r3, 1024);
        let r4 = [1, 2, 3, 4, 8, 8, 6];
        t.insert_structural(&r4, 1024);
        let deep = t.deepest_unmaterialized_fork(&r4).expect("deeper fork");
        assert_eq!(t.depth(deep), 6, "nested fork at the deeper shared run [1,2,3,4,8,8]");
    }

    #[test]
    fn insert_rejects_below_min_depth_and_when_disabled() {
        let mut t = PrefixTrie::new(CAP, K);
        assert!(t.insert(&[1, 2], 2, 10).is_none(), "depth < min_depth K rejected");
        assert!(t.insert(&[1, 2, 3], 0, 10).is_none(), "zero depth rejected");
        assert!(t.insert(&[1, 2, 3], 9, 10).is_none(), "depth > tokens.len() rejected");
        let mut off = PrefixTrie::new(CAP, 0);
        assert!(!off.enabled());
        assert!(off.insert(&[1, 2, 3], 3, 10).is_none(), "min_depth 0 never materializes");
    }

    #[test]
    fn plan_reuse_matches_and_establishes_a_fresh_fork() {
        let mut t = PrefixTrie::new(CAP, K);
        // Two requests share [1,2,3,4] → a fork the second can cache.
        t.insert_structural(&[1, 2, 3, 4, 5], 1024);
        t.insert_structural(&[1, 2, 3, 4, 8], 1024);
        // No ready prefix yet → reused 0, but a fork at depth 4 is establishable (fresh, fits).
        let p = t.plan_reuse(&[1, 2, 3, 4, 8], 5);
        assert_eq!(p.reused, 0);
        assert_eq!(p.establish_new_cells, Some(4), "cache the whole fresh 4-token fork");
        assert_eq!(p.fork.map(|f| t.depth(f)), Some(4));
        // Materialize it; a later matching request now reuses 4 and establishes nothing.
        let f = p.fork.unwrap();
        t.materialize(f, 50);
        t.set_ready(f);
        let p2 = t.plan_reuse(&[1, 2, 3, 4, 9], 5);
        assert_eq!(p2.reused, 4, "reuses the resident prefix");
        assert_eq!(p2.establish_new_cells, None, "nothing new to cache");
    }

    #[test]
    fn plan_reuse_skips_ancestor_fork_of_a_materialized_node() {
        // Regression for the deep-before-shallow `reserved` drift: with a deep fork F@4
        // already materialized, a request that would fork at the SHALLOWER ancestor H@2 must
        // NOT establish it — F already charges the [1,2] run, so caching H would double-count.
        let mut t = PrefixTrie::new(CAP, 2); // min_depth 2 so a depth-2 fork is otherwise eligible
        t.insert_structural(&[1, 2, 3, 4, 5, 6], 1024);
        t.insert_structural(&[1, 2, 3, 4, 7, 7], 1024);
        let f = t.deepest_unmaterialized_fork(&[1, 2, 3, 4, 5, 6]).expect("fork F@4");
        assert_eq!(t.depth(f), 4);
        t.materialize(f, 100);
        t.set_ready(f);
        // A third request diverges at depth 2 → structural fork H@2, an ancestor of F.
        t.insert_structural(&[1, 2, 9, 9], 1024);
        let p = t.plan_reuse(&[1, 2, 9, 9], 4);
        assert_eq!(p.reused, 0, "F@4 doesn't match [1,2,9,9]");
        assert_eq!(p.establish_new_cells, None, "ancestor of a materialized fork is not cached");
        assert!(p.fork.is_none());
    }

    #[test]
    fn demote_node_frees_seq_and_returns_incremental_cells() {
        let mut t = PrefixTrie::new(CAP, K);
        ready_insert(&mut t, &[1, 2, 3], 3, 10); // ancestor@3
        let child = t.insert(&[1, 2, 3, 4, 5], 5, 11).expect("child@5");
        t.set_ready(child);
        // Demote the child → frees seq 11, returns its incremental depth over the ancestor (5-3).
        assert_eq!(t.demote_node(child), Some((11, 2)));
        assert_eq!(t.longest_match(&[1, 2, 3, 9]).reused, 3, "ancestor still resident");
        assert_eq!(t.longest_match(&[1, 2, 3, 4, 5]).reused, 3, "child no longer materialized");
        assert!(t.demote_node(child).is_none(), "already demoted → None");
    }

    #[test]
    fn evict_never_picks_a_live_nested_ancestor() {
        let mut t = PrefixTrie::new(CAP, K);
        ready_insert(&mut t, &[1, 2, 3], 3, 10); // ancestor@3 (older)
        let child = t.insert(&[1, 2, 3, 4, 5], 5, 11).expect("child@5");
        t.set_ready(child); // child@5 (newer) — a leaf under the ancestor
        // The ancestor is older but has a materialized descendant → not a leaf → never evicted.
        assert_eq!(t.evict_lru_leaf().map(|(s, _)| s), Some(11), "evicts the child leaf, not the live ancestor");
        // Now the ancestor is itself a leaf → evictable.
        assert_eq!(t.evict_lru_leaf().map(|(s, _)| s), Some(10));
        assert_eq!(t.evict_lru_leaf(), None, "nothing left to evict");
    }

    #[test]
    fn clear_resets_to_empty_root() {
        let mut t = PrefixTrie::new(CAP, K);
        ready_insert(&mut t, &[1, 2, 3], 3, 10);
        assert_eq!(t.materialized_len(), 1);
        t.clear();
        assert_eq!(t.materialized_len(), 0);
        assert_eq!(t.longest_match(&[1, 2, 3, 4]).reused, 0, "no prefix survives a wipe");
    }
}
