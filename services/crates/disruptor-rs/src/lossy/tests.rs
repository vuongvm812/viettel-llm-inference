#![allow(clippy::cast_possible_truncation)]

use super::{TryRecvError, channel};

// ── Single-threaded behaviour (also the Miri-checked path) ─────────────────────
//
// Excluded under `--cfg loom`: these drive the ring outside `loom::model`, where
// loom's instrumented atomics panic. Loom mode runs only the `loom_*` models.

#[cfg(not(loom))]
mod sync {
    use super::{TryRecvError, channel};
    use std::mem::MaybeUninit;
    use std::thread;

    /// A two-word payload whose invariant (`b == !a`) is broken by any torn read.
    /// Used to prove that a concurrent overwrite is never *delivered*, only dropped.
    #[derive(Clone, Copy, bytemuck::NoUninit)]
    #[repr(C)]
    struct Pair {
        a: u64,
        b: u64,
    }

    #[test]
    fn single_consumer_fifo() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [rx] = rxs;

        tx.publish(1);
        tx.publish(2);

        assert_eq!(rx.try_recv(), Ok(1));
        assert_eq!(rx.try_recv(), Ok(2));
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn two_consumers_each_see_every_item() {
        let (tx, rxs) = channel::<u64, 8, 2>();
        let [rx0, rx1] = rxs;

        for v in 0..6u64 {
            tx.publish(v);
        }

        for rx in [&rx0, &rx1] {
            for expected in 0..6u64 {
                assert_eq!(rx.try_recv(), Ok(expected));
            }
            assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        }
    }

    #[test]
    fn multiword_payload_round_trips() {
        let (tx, rxs) = channel::<Pair, 4, 1>();
        let [rx] = rxs;

        tx.publish(Pair { a: 7, b: !7 });
        let got = rx.try_recv().unwrap();
        assert_eq!(got.a, 7);
        assert_eq!(got.b, !7);
    }

    #[test]
    fn lapped_consumer_resyncs_to_oldest_live_and_counts_drops() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [rx] = rxs;

        // Phase 2 tail-versioning: effective capacity is `N - 1 = 3`, not `N`.
        // Ring depth 4, publish 6 → tail = 6, live window = [tail - (N - 1),
        // tail) = [3, 6) = {3, 4, 5}; positions 0,1,2 are unreadable (the slot
        // at `idx(tail) == idx(2)` shares the producer's next write target).
        // The lapped cursor resyncs to position 3, dropping the three skipped
        // positions 0..3 into `lagged`.
        for v in 0..6u64 {
            tx.publish(v);
        }

        assert_eq!(rx.try_recv(), Err(TryRecvError::Lagged(3)));
        assert_eq!(rx.try_recv(), Ok(3));
        assert_eq!(rx.try_recv(), Ok(4));
        assert_eq!(rx.try_recv(), Ok(5));
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        assert_eq!(rx.lagged(), 3);
    }

    #[test]
    fn filter_skips_unwanted_without_counting_as_lag() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;

        tx.publish_with_header(10, 0);
        tx.publish_with_header(11, 1);
        tx.publish_with_header(12, 0);

        let keep_even = |h: u64| h.is_multiple_of(2);
        assert_eq!(rx.try_recv_filter(keep_even), Ok(10));
        assert_eq!(rx.try_recv_filter(keep_even), Ok(12)); // 11 skipped by filter
        assert_eq!(rx.try_recv_filter(keep_even), Err(TryRecvError::Empty));
        assert_eq!(rx.lagged(), 0); // a filtered skip is not a dropped item
    }

    #[test]
    fn disconnected_after_producer_drops_and_drains() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [rx] = rxs;

        tx.publish(1);
        tx.publish(2);
        drop(tx);

        assert_eq!(rx.try_recv(), Ok(1));
        assert_eq!(rx.try_recv(), Ok(2));
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn producer_observes_all_consumers_dropping() {
        let (tx, rxs) = channel::<u64, 4, 2>();
        assert!(tx.is_any_consumer_alive());

        let [rx0, rx1] = rxs;
        drop(rx0);
        assert!(tx.is_any_consumer_alive());
        drop(rx1);
        assert!(!tx.is_any_consumer_alive());
    }

    #[test]
    fn pop_batch_drains_in_order() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        for v in 0..5u64 {
            tx.publish(v);
        }
        let mut out = [MaybeUninit::uninit(); 8];
        let n = rx.pop_batch(&mut out);
        assert_eq!(n, 5);
        for (i, slot) in out.iter().take(n).enumerate() {
            assert_eq!(unsafe { slot.assume_init() }, i as u64);
        }
    }

    /// `available()` reports the number of slots drainable now, clamped to
    /// `N - 1` (Phase 2 safe horizon; was `N` under the seqlock).
    /// Before any publish: 0. After publishing 3 to a depth-8 ring: 3. After
    /// over-publishing (6 to depth 4): clamped to `N - 1 = 3` — the consumer
    /// is lapped-to-full and the next read will resync.
    #[test]
    fn available_reports_drainable_count_clamped_to_capacity() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        assert_eq!(rx.available(), 0);

        for v in 0..3u64 {
            tx.publish(v);
        }
        assert_eq!(rx.available(), 3);

        // Drain one → available shrinks.
        let _ = rx.try_recv().unwrap();
        assert_eq!(rx.available(), 2);

        // Lapped-to-full case: depth 4, publish 6.
        let (tx2, rxs2) = channel::<u64, 4, 1>();
        let [rx2] = rxs2;
        for v in 0..6u64 {
            tx2.publish(v);
        }
        assert_eq!(rx2.available(), 3, "clamp to N - 1 when lapped");
    }

    /// `drain_available` delivers in FIFO order and tags `end_of_batch = true`
    /// **only** on the last delivery of the call. With a single batch
    /// containing every published item, exactly one event carries EOB.
    #[test]
    fn drain_available_tags_only_last_delivery_as_end_of_batch() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [mut rx] = rxs;
        for v in 0..5u64 {
            tx.publish(v);
        }

        let mut seen: Vec<(u64, bool)> = Vec::new();
        let n = rx.drain_available(8, |v, eob| seen.push((*v, eob)));
        assert_eq!(n, 5);
        assert_eq!(
            seen,
            vec![(0, false), (1, false), (2, false), (3, false), (4, true)]
        );
        // Cursor advanced past everything → no more available.
        assert_eq!(rx.available(), 0);
    }

    /// `max` caps the batch size, EOB flags the last *cap-limited* delivery,
    /// and the cursor is left positioned so the next call resumes cleanly.
    #[test]
    fn drain_available_respects_max_and_resumes() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [mut rx] = rxs;
        for v in 0..6u64 {
            tx.publish(v);
        }

        let mut batch1: Vec<(u64, bool)> = Vec::new();
        let n1 = rx.drain_available(3, |v, eob| batch1.push((*v, eob)));
        assert_eq!(n1, 3);
        assert_eq!(batch1, vec![(0, false), (1, false), (2, true)]);

        let mut batch2: Vec<(u64, bool)> = Vec::new();
        let n2 = rx.drain_available(8, |v, eob| batch2.push((*v, eob)));
        assert_eq!(n2, 3);
        assert_eq!(batch2, vec![(3, false), (4, false), (5, true)]);
    }

    /// `drain_available(_, _)` on an empty ring returns 0 and never invokes `f`.
    /// `drain_available(0, _)` is a no-op even when items are pending.
    #[test]
    fn drain_available_empty_and_zero_max_are_noops() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [mut rx] = rxs;

        let mut invoked = false;
        assert_eq!(rx.drain_available(8, |_, _| invoked = true), 0);
        assert!(!invoked, "no items, closure must not run");

        tx.publish(42);
        assert_eq!(rx.drain_available(0, |_, _| invoked = true), 0);
        assert!(
            !invoked,
            "max=0 must skip the closure even with pending items"
        );
        assert_eq!(rx.try_recv(), Ok(42), "cursor must not advance");
    }

    /// Mid-batch lap: the consumer is already trailing when `drain_available`
    /// starts. The closure runs zero times; the cursor resyncs to the oldest
    /// still-live position; `lagged()` records the gap. A follow-up call
    /// delivers the recovered tail with the final item tagged EOB.
    #[test]
    fn drain_available_handles_lap_before_first_slot() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [mut rx] = rxs;
        for v in 0..6u64 {
            tx.publish(v);
        }

        let mut seen: Vec<(u64, bool)> = Vec::new();
        let n = rx.drain_available(8, |v, eob| seen.push((*v, eob)));
        assert_eq!(n, 0, "first call detects the lap and resyncs");
        assert!(seen.is_empty());
        // Phase 2 tail-versioning: oldest live = tail - (N - 1) = 6 - 3 = 3,
        // so positions 0,1,2 are skipped (3 dropped, not 2 as under the
        // seqlock with its `tail - N` horizon).
        assert_eq!(rx.lagged(), 3);

        let n2 = rx.drain_available(8, |v, eob| seen.push((*v, eob)));
        assert_eq!(n2, 3);
        assert_eq!(
            seen,
            vec![(3, false), (4, false), (5, true)],
            "recovered tail [3..6) with EOB on the last"
        );
    }

    /// Panic-safety contract: when `f` panics on the k-th delivery, the cursor
    /// is already advanced past that item (it is "consumed") AND every prior
    /// item delivered in this batch is also consumed — none of [0..=k-1] nor k
    /// is re-delivered on the next call. The next call resumes cleanly at k+1.
    #[test]
    fn drain_available_panic_in_f_does_not_replay_delivered_items() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let (tx, rxs) = channel::<u64, 8, 1>();
        let [mut rx] = rxs;
        for v in 0..5u64 {
            tx.publish(v);
        }

        // First call: deliver 0, 1 successfully, then panic on 2.
        let mut delivered_before_panic: Vec<u64> = Vec::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            rx.drain_available(8, |v, _eob| {
                if *v == 2 {
                    panic!("synthetic failure on item 2");
                }
                delivered_before_panic.push(*v);
            })
        }));
        assert!(result.is_err(), "f must have panicked");
        assert_eq!(
            delivered_before_panic,
            vec![0, 1],
            "items 0,1 reached the callback before the panic"
        );

        // Second call: must resume at 3, NOT replay 0/1, NOT replay the
        // panicking item 2 (it is consumed-and-lost per the documented
        // semantic).
        let mut seen_after: Vec<(u64, bool)> = Vec::new();
        let n = rx.drain_available(8, |v, eob| seen_after.push((*v, eob)));
        assert_eq!(n, 2);
        assert_eq!(seen_after, vec![(3, false), (4, true)]);
        assert_eq!(rx.lagged(), 0, "panic is not a lag");
    }

    /// Stress: a lapping producer races a consumer that pulls everything through
    /// `drain_available`. Per-position accounting (`delivered + lagged == COUNT`)
    /// must hold, deliveries strictly increasing, and each batch must tag
    /// exactly one EOB.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "threaded stress loop; Miri runs the single-threaded paths"
    )]
    fn drain_available_lapping_accounts_for_every_item() {
        const COUNT: u64 = 200_000;
        let (tx, rxs) = channel::<u64, 1024, 1>();
        let [rx] = rxs;

        let producer = thread::spawn(move || {
            for v in 0..COUNT {
                tx.publish(v);
            }
            drop(tx);
        });

        let consumer = thread::spawn(move || {
            let mut rx = rx;
            let mut delivered = 0u64;
            let mut last: Option<u64> = None;
            loop {
                let mut batch_eobs = 0u32;
                let mut batch_n = 0u32;
                let n = rx.drain_available(64, |v, eob| {
                    if let Some(prev) = last {
                        assert!(*v > prev, "non-monotonic delivery {prev} -> {v}");
                    }
                    assert!(*v < COUNT, "fabricated value {v}");
                    last = Some(*v);
                    batch_n += 1;
                    if eob {
                        batch_eobs += 1;
                    }
                });
                if n > 0 {
                    assert_eq!(batch_n, n as u32);
                    assert_eq!(batch_eobs, 1, "exactly one EOB per non-empty batch");
                    delivered += n as u64;
                } else if !rx.is_producer_alive() && rx.available() == 0 {
                    break;
                } else {
                    std::hint::spin_loop();
                }
            }
            (delivered, rx.lagged())
        });

        producer.join().unwrap();
        let (delivered, lagged) = consumer.join().unwrap();
        assert_eq!(delivered + lagged, COUNT, "per-position accounting");
    }

    /// A fast producer laps a small ring while two consumers read. The invariant
    /// `b == !a` must hold for *every delivered* item — a torn read breaks it.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "threaded stress loop; Miri runs the single-threaded paths"
    )]
    fn multiword_no_torn_delivery_under_lapping() {
        const ITERS: u64 = 200_000;
        let (tx, rxs) = channel::<Pair, 16, 2>();
        let [rx0, rx1] = rxs;

        let producer = thread::spawn(move || {
            for x in 0..ITERS {
                tx.publish(Pair { a: x, b: !x });
            }
        });

        let spawn_drain = |rx: crate::lossy::Consumer<Pair, 16, 2>| {
            thread::spawn(move || {
                let mut delivered = 0u64;
                loop {
                    match rx.try_recv() {
                        Ok(p) => {
                            assert_eq!(p.b, !p.a, "torn read delivered: a={} b={}", p.a, p.b);
                            delivered += 1;
                        }
                        Err(TryRecvError::Empty) => {
                            if !rx.is_producer_alive() {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                        Err(TryRecvError::Lagged(_)) => {}
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
                while let Ok(p) = rx.try_recv() {
                    assert_eq!(p.b, !p.a, "torn read delivered at drain");
                    delivered += 1;
                }
                delivered
            })
        };

        let h0 = spawn_drain(rx0);
        let h1 = spawn_drain(rx1);
        producer.join().unwrap();
        let d0 = h0.join().unwrap();
        let d1 = h1.join().unwrap();
        assert!(d0 > 0 && d1 > 0, "consumers delivered d0={d0} d1={d1}");
    }

    // ── Peek (register-style) — does not advance the cursor ─────────────────

    /// On a fresh ring `peek_latest` returns `None` (no items published yet),
    /// and the cursor stays at 0 so the first `try_recv` still sees the first
    /// publish.
    #[test]
    fn peek_latest_empty_returns_none() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [rx] = rxs;
        assert_eq!(rx.peek_latest(), None);

        tx.publish(7);
        assert_eq!(rx.peek_latest(), Some(7));
        // Cursor untouched by peek.
        assert_eq!(rx.try_recv(), Ok(7));
    }

    /// Repeated peeks return the **same** freshest item — peek does not
    /// consume. After K publishes, `peek_latest` returns the K-th, regardless
    /// of how many times it has been called.
    #[test]
    fn peek_latest_is_idempotent_and_does_not_advance_cursor() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        for v in 0..5u64 {
            tx.publish(v);
        }
        for _ in 0..16 {
            assert_eq!(rx.peek_latest(), Some(4));
        }
        // Stream delivery is intact: every item still readable in order.
        for expected in 0..5u64 {
            assert_eq!(rx.try_recv(), Ok(expected));
        }
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        // Pure peek without subsequent advance does not count as lag.
        assert_eq!(rx.lagged(), 0);
    }

    /// `peek(n)` walks the live history: `peek(0)` is the latest,
    /// `peek(N - 2)` is the oldest still-peekable. `peek(N - 1)` is rejected
    /// even with a fully published ring, because the slot at `idx(tail)`
    /// shares its index with the producer's next write target.
    ///
    /// Publish exactly `N - 1` (not `N`) so the stream cursor at position 0
    /// stays inside the live window — letting the test also assert that peek
    /// did not advance the cursor (`try_recv` still delivers position 0).
    #[test]
    fn peek_n_covers_horizon_and_rejects_top_offset() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        // tail = 7 → live window [tail - (N - 1), tail) = [0, 7).
        for v in 0..7u64 {
            tx.publish(v);
        }
        for n in 0..7usize {
            // n=0 ⇒ latest=6; n=1 ⇒ 5; …; n=6 ⇒ 0.
            assert_eq!(rx.peek(n), Some(6 - n as u64), "peek({n})");
        }
        assert_eq!(rx.peek(7), None, "peek(N - 1) rejected by safety bound");
        assert_eq!(rx.peek(99), None, "peek(out-of-range) rejected");
        // peek doesn't move the cursor.
        assert_eq!(rx.try_recv(), Ok(0));
    }

    /// `peek(n)` returns `None` when there are fewer than `n + 1` items
    /// published (the ring has no slot at the requested offset).
    #[test]
    fn peek_returns_none_when_ring_too_shallow() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        tx.publish(100);
        tx.publish(101);
        assert_eq!(rx.peek(0), Some(101));
        assert_eq!(rx.peek(1), Some(100));
        assert_eq!(rx.peek(2), None, "only 2 items published");
        assert_eq!(rx.peek(5), None);
    }

    /// `peek_range` provides an atomic snapshot in publication order
    /// (oldest → latest). With `0..K` and at least K items published, the
    /// snapshot is the K most recent items.
    ///
    /// Publish `N - 1` so the stream cursor at position 0 stays inside the
    /// live window (under Phase 2 tail-versioning, publishing `N` would lap
    /// cursor 0 because position 0 lives at `idx(tail) == idx(N)`).
    #[test]
    fn peek_range_atomic_snapshot_in_publication_order() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        // tail = 7 → live window [0, 7); positions 0..=6 are safely peekable.
        for v in 0..7u64 {
            tx.publish(v);
        }
        let mut buf: Vec<u64> = Vec::new();
        rx.peek_range(0..4, &mut buf)
            .expect("snapshot must succeed when ring is quiescent");
        // Latest is position 6, oldest in `0..4` is position 3.
        assert_eq!(buf, vec![3, 4, 5, 6], "oldest-first order");

        // Repeating with the same buffer truncates back via the caller — we
        // assert the API does not clear it on success.
        let entry_len = buf.len();
        rx.peek_range(0..2, &mut buf)
            .expect("second snapshot succeeds");
        assert_eq!(&buf[entry_len..], &[5, 6]);

        // Unbounded `..` covers the maximum window (offsets 0..N - 1).
        let mut full: Vec<u64> = Vec::new();
        rx.peek_range(.., &mut full).expect("full window");
        assert_eq!(full, vec![0, 1, 2, 3, 4, 5, 6], "N - 1 = 7 items");

        // Cursor is still at 0.
        assert_eq!(rx.try_recv(), Ok(0));
    }

    /// Invalid ranges (empty or over-wide) return `None` and leave `out`
    /// untouched. An over-wide range that includes the writing-locked offset
    /// (`>= N - 1`) is rejected before touching the buffer.
    #[test]
    #[allow(clippy::reversed_empty_ranges)] // intentional — verifies rejection of bad ranges
    fn peek_range_rejects_invalid_bounds() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [rx] = rxs;
        for v in 0..3u64 {
            tx.publish(v);
        }

        let mut buf: Vec<u64> = vec![99];
        assert_eq!(rx.peek_range(3..1, &mut buf), None, "empty / inverted");
        assert_eq!(buf, vec![99], "buf untouched on rejection");

        assert_eq!(rx.peek_range(0..4, &mut buf), None, "exceeds N - 1 horizon");
        assert_eq!(buf, vec![99], "buf untouched on rejection");

        assert_eq!(rx.peek_range(2..2, &mut buf), None, "zero-width range");
        assert_eq!(buf, vec![99]);
    }

    /// `peek_range` rolls back to entry length when the ring lacks enough
    /// published items — the snapshot precondition (enough history) failed
    /// before any push.
    #[test]
    fn peek_range_rejects_when_ring_too_shallow() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        tx.publish(1);
        tx.publish(2);
        let mut buf: Vec<u64> = vec![55];
        assert_eq!(rx.peek_range(0..4, &mut buf), None);
        assert_eq!(buf, vec![55], "buf untouched, only 2 items live");
    }

    /// After a lap, the slots holding lapped positions no longer satisfy the
    /// post-read `tail` distance check — `peek` against them returns `None`
    /// (the tail-version rejects rather than delivering a different
    /// position's value). The consumer's `lagged()` counter is **unchanged**
    /// by a failed peek: peek is a snapshot API, not a stream-skip.
    #[test]
    fn peek_at_lapped_position_returns_none_without_counting_lag() {
        let (tx, rxs) = channel::<u64, 4, 1>();
        let [rx] = rxs;
        // Phase 2 tail-versioning: publish 6 → tail = 6, live window =
        // [tail - (N - 1), tail) = [3, 6) = {3, 4, 5}. Positions 0,1,2 are
        // unreadable (the slot at `idx(tail) == idx(2)` shares the producer's
        // next write target).
        for v in 0..6u64 {
            tx.publish(v);
        }
        assert_eq!(rx.peek_latest(), Some(5));
        assert_eq!(rx.peek(0), Some(5));
        assert_eq!(rx.peek(1), Some(4));
        assert_eq!(rx.peek(2), Some(3));
        // Offset 3 would target position 2 — outside the safe horizon.
        // Rejected by the entry bound `n < N - 1 = 3`.
        assert_eq!(rx.peek(3), None, "blocked by n < N - 1 safe horizon");
        assert_eq!(rx.lagged(), 0, "peek failure is NOT a stream-skip");
    }

    // ── Miri-checked peek paths ────────────────────────────────────────────
    //
    // Existing single-threaded paths run under Miri. These add two more:
    // peek after publish, and peek_range snapshot. Both exercise the racy
    // memcpy + tail-version validation under Miri's tree-borrows.

    /// Miri-checked: a single peek after publish exercises `load_value` and
    /// the tail-version validation on a quiescent ring.
    #[test]
    fn miri_peek_latest_after_publish() {
        let (tx, rxs) = channel::<Pair, 4, 1>();
        let [rx] = rxs;
        tx.publish(Pair { a: 11, b: !11 });
        let got = rx.peek_latest().expect("peek delivers fresh item");
        assert_eq!(got.a, 11);
        assert_eq!(got.b, !11);
    }

    /// Miri-checked: snapshot via `peek_range` on a quiescent ring writes
    /// every position in order.
    #[test]
    fn miri_peek_range_after_publish() {
        let (tx, rxs) = channel::<u64, 8, 1>();
        let [rx] = rxs;
        for v in 10..14u64 {
            tx.publish(v);
        }
        let mut buf: Vec<u64> = Vec::new();
        rx.peek_range(0..4, &mut buf).expect("snapshot");
        assert_eq!(buf, vec![10, 11, 12, 13]);
    }

    /// Under a free-running (lapping) producer the ring is lossy, but its
    /// accounting must be exact: for each consumer, every one of `COUNT`
    /// positions is **either** delivered exactly once **or** counted in a lag gap.
    /// Delivered values are strictly increasing (no dup/reorder/garbage).
    #[test]
    #[cfg_attr(
        miri,
        ignore = "threaded stress loop; Miri runs the single-threaded paths"
    )]
    fn lapping_accounts_for_every_item() {
        const COUNT: u64 = 200_000;
        let (tx, rxs) = channel::<u64, 1024, 2>();
        let [rx0, rx1] = rxs;

        let producer = thread::spawn(move || {
            for v in 0..COUNT {
                tx.publish(v);
            }
            drop(tx);
        });

        let spawn_drain = |rx: crate::lossy::Consumer<u64, 1024, 2>| {
            thread::spawn(move || {
                let mut delivered = 0u64;
                let mut last: Option<u64> = None;
                loop {
                    match rx.try_recv() {
                        Ok(v) => {
                            assert!(v < COUNT, "fabricated value {v}");
                            if let Some(prev) = last {
                                assert!(v > prev, "non-monotonic delivery {prev} -> {v}");
                            }
                            last = Some(v);
                            delivered += 1;
                        }
                        Err(TryRecvError::Empty) => std::hint::spin_loop(),
                        Err(TryRecvError::Lagged(_)) => {}
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
                (delivered, rx.lagged())
            })
        };

        let h0 = spawn_drain(rx0);
        let h1 = spawn_drain(rx1);
        producer.join().unwrap();
        let (delivered0, lagged0) = h0.join().unwrap();
        let (delivered1, lagged1) = h1.join().unwrap();
        assert_eq!(delivered0 + lagged0, COUNT, "consumer 0 accounting");
        assert_eq!(delivered1 + lagged1, COUNT, "consumer 1 accounting");
    }
}

// ── loom model checks ──────────────────────────────────────────────────────────
//
// Kept to two threads (producer + one consumer): loom verifies the
// tail-version Acquire/Release handshake and the lapped-mid-read resync.
// Multi-consumer independence is structural (each consumer owns its cursor)
// and covered by the deterministic `sync::two_consumers_each_see_every_item`
// test.
//
// Run with: `RUSTFLAGS="--cfg loom" cargo test -p spmc --lib loom`

/// Producer publishes 1,2 then drops; consumer drains. No lapping (depth ≥ items).
/// Every delivered value must be one that was published, in order.
#[cfg(loom)]
#[test]
fn loom_handoff_no_garbage() {
    loom::model(|| {
        let (tx, rxs) = channel::<usize, 4, 1>();
        let [rx] = rxs;

        let producer = loom::thread::spawn(move || {
            tx.publish(1);
            tx.publish(2);
        });

        let consumer = loom::thread::spawn(move || {
            let mut seen = Vec::new();
            loop {
                match rx.try_recv() {
                    Ok(v) => seen.push(v),
                    Err(TryRecvError::Empty) => {
                        if !rx.is_producer_alive() {
                            break;
                        }
                        loom::thread::yield_now();
                    }
                    Err(TryRecvError::Lagged(_)) => {}
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            while let Ok(v) = rx.try_recv() {
                seen.push(v);
            }
            seen
        });

        producer.join().unwrap();
        let seen = consumer.join().unwrap();
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "out of order: {seen:?}"
        );
        assert!(seen.iter().all(|&v| v == 1 || v == 2), "garbage: {seen:?}");
    });
}

/// Same lapping scenario as `loom_lapping_resync_no_garbage`, but drives the
/// **batched** receive path: a successful slot validated mid-batch must not be
/// retroactively invalidated by a later lap in the same batch (the deferred
/// flush only emits items whose tail-version re-check passed), and a lap-on-first-slot
/// must not pre-deliver garbage through the EOB flush.
#[cfg(loom)]
#[test]
fn loom_drain_available_lapping_no_garbage() {
    loom::model(|| {
        let (tx, rxs) = channel::<usize, 2, 1>();
        let [rx] = rxs;

        let producer = loom::thread::spawn(move || {
            tx.publish(1);
            tx.publish(2);
            tx.publish(3);
        });

        let consumer = loom::thread::spawn(move || {
            let mut rx = rx;
            let mut seen: Vec<(usize, bool)> = Vec::new();
            loop {
                let n = rx.drain_available(3, |v, eob| seen.push((*v, eob)));
                if n == 0 {
                    if !rx.is_producer_alive() {
                        break;
                    }
                    loom::thread::yield_now();
                }
            }
            // Producer is gone; drain any tail items the loop above missed.
            loop {
                let n = rx.drain_available(3, |v, eob| seen.push((*v, eob)));
                if n == 0 {
                    break;
                }
            }
            seen
        });

        producer.join().unwrap();
        let seen = consumer.join().unwrap();

        // Every delivered value is one the producer wrote; no torn garbage.
        assert!(
            seen.iter().all(|(v, _)| (1..=3).contains(v)),
            "garbage: {seen:?}"
        );
        // Deliveries are strictly increasing (no dup, no reorder).
        assert!(
            seen.windows(2).all(|w| w[0].0 < w[1].0),
            "out of order: {seen:?}"
        );
        // Every non-empty `drain_available` call tags exactly one EOB, so EOBs
        // partition the delivery stream into batches. Each batch's last item is
        // the EOB; counting EOBs counts batches.
        if !seen.is_empty() {
            assert!(seen.last().unwrap().1, "last delivery must carry EOB");
        }
    });
}

/// `peek_latest` is a register-style accessor: it must never advance the
/// consumer cursor, and it must never deliver a value the producer did not
/// publish. After arbitrary interleaving of peeks against a 2-publish
/// producer, a subsequent drain via `try_recv` must still see {1, 2} as a
/// contiguous prefix.
#[cfg(loom)]
#[test]
fn loom_peek_does_not_advance_cursor() {
    loom::model(|| {
        let (tx, rxs) = channel::<usize, 4, 1>();
        let [rx] = rxs;

        let producer = loom::thread::spawn(move || {
            tx.publish(1);
            tx.publish(2);
        });

        let consumer = loom::thread::spawn(move || {
            // Race peeks against the publisher; any returned value MUST be one
            // the producer wrote.
            let mut peeked = Vec::new();
            for _ in 0..4 {
                if let Some(v) = rx.peek_latest() {
                    peeked.push(v);
                }
            }

            // Drain via the cursor-stream API. peek must not have advanced the
            // cursor, so the drained sequence is a contiguous prefix of
            // {1, 2}.
            let mut seen = Vec::new();
            loop {
                match rx.try_recv() {
                    Ok(v) => seen.push(v),
                    Err(TryRecvError::Empty) => {
                        if !rx.is_producer_alive() {
                            break;
                        }
                        loom::thread::yield_now();
                    }
                    Err(TryRecvError::Lagged(_)) => {}
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            while let Ok(v) = rx.try_recv() {
                seen.push(v);
            }
            (peeked, seen)
        });

        producer.join().unwrap();
        let (peeked, seen) = consumer.join().unwrap();

        // Every peeked value was actually published.
        assert!(
            peeked.iter().all(|&v| v == 1 || v == 2),
            "peek garbage: {peeked:?}"
        );
        // Drain is a contiguous prefix of [1, 2] — no item is missed because
        // peek consumed it.
        assert!(
            seen.iter().all(|&v| v == 1 || v == 2),
            "drain garbage: {seen:?}"
        );
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "drain out of order: {seen:?}"
        );
    });
}

/// Phase 2 accounting invariant: every published position is **either**
/// delivered exactly once **or** counted in the lag gap. Crucially, this
/// also catches a backward cursor drag — `resync` setting `self.cursor` to a
/// value at or behind its current position would cause some positions to be
/// counted twice (once as delivered, once as lagged), breaking the equality.
///
/// This is the explicit Phase 2 model called out in `PLAN.md §2.3`: tighter
/// than "no garbage" because it requires the exact `tail - (N - 1)` resync
/// target to leave no slot double-counted.
#[cfg(loom)]
#[test]
fn loom_delivered_plus_lagged_equals_total() {
    loom::model(|| {
        const TOTAL: usize = 3;
        let (tx, rxs) = channel::<usize, 2, 1>();
        let [rx] = rxs;

        let producer = loom::thread::spawn(move || {
            for v in 1..=TOTAL {
                tx.publish(v);
            }
        });

        let consumer = loom::thread::spawn(move || {
            let mut delivered = 0u64;
            let mut last_value: Option<usize> = None;
            let mut process_ok = |v: usize| {
                assert!((1..=TOTAL).contains(&v), "delivered garbage value {v}");
                if let Some(prev) = last_value {
                    assert!(v > prev, "non-monotonic delivery {prev} -> {v}");
                }
                last_value = Some(v);
                delivered += 1;
            };
            loop {
                match rx.try_recv() {
                    Ok(v) => process_ok(v),
                    Err(TryRecvError::Empty) => {
                        if !rx.is_producer_alive() {
                            break;
                        }
                        loom::thread::yield_now();
                    }
                    Err(TryRecvError::Lagged(_)) => {}
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            // Final drain after the producer is gone. Must also handle
            // `Lagged` here — the producer could have lapped past the
            // consumer's cursor before the consumer first saw any tail
            // advance, in which case the first post-drop `try_recv` returns
            // `Lagged(_)` and the *next* one delivers the resync target.
            loop {
                match rx.try_recv() {
                    Ok(v) => process_ok(v),
                    Err(TryRecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }
            (delivered, rx.lagged())
        });

        producer.join().unwrap();
        let (delivered, lagged) = consumer.join().unwrap();
        // The Phase 2 invariant: each of the TOTAL published positions is
        // either delivered or counted in the lag gap, never both, never
        // neither. A backward `cursor` drag (resync target <= current cursor)
        // would inflate `delivered + lagged > TOTAL`; a forward gap that
        // skips a position without counting it would deflate it.
        assert_eq!(
            delivered + lagged,
            TOTAL as u64,
            "delivered({delivered}) + lagged({lagged}) != published({TOTAL}) \
             — Phase 2 tail - (N - 1) resync target is wrong"
        );
    });
}

/// Ring depth 2, three publishes → position 2 laps position 0. Verifies the
/// lapped-mid-read resync never delivers a stale/garbage value.
#[cfg(loom)]
#[test]
fn loom_lapping_resync_no_garbage() {
    loom::model(|| {
        let (tx, rxs) = channel::<usize, 2, 1>();
        let [rx] = rxs;

        let producer = loom::thread::spawn(move || {
            tx.publish(1);
            tx.publish(2);
            tx.publish(3);
        });

        let consumer = loom::thread::spawn(move || {
            let mut seen = Vec::new();
            loop {
                match rx.try_recv() {
                    Ok(v) => seen.push(v),
                    Err(TryRecvError::Empty) => {
                        if !rx.is_producer_alive() {
                            break;
                        }
                        loom::thread::yield_now();
                    }
                    Err(TryRecvError::Lagged(_)) => {}
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            while let Ok(v) = rx.try_recv() {
                seen.push(v);
            }
            seen
        });

        producer.join().unwrap();
        let seen = consumer.join().unwrap();
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "out of order: {seen:?}"
        );
        assert!(
            seen.iter().all(|&v| (1..=3).contains(&v)),
            "garbage: {seen:?}"
        );
    });
}
