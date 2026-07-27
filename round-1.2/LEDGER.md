# LEDGER — round-1.2 scored-latency hypotheses

One line per hypothesis. A row exists only after a wall-clock A/B; reasoning alone never
produces one. Measurement rule: multiple BOOTS per arm (`make ab`), not reps — the noise floor
is boot-to-boot (~0.5 ms TPOT, ~0.05 ms rep-to-rep). Rank by ERS, not by the TTFT/TPOT columns.

Box: RTX 3060 12 GB dev box (`andromeda`), NOT the judge's H200/MIG slice. It cannot resolve
host-side wins below ~1 ms — a null here means *unknown*, not negative. Scheduler-shape changes
(how many steps a prefill spans) it can resolve.

| # | Date | Hypothesis | Predicted | Measured | Kept? | Why |
|---|------|-----------|-----------|----------|-------|-----|
| 1 | 2026-07-27 | `--max-num-scheduled-tokens`: shipped 2048 chunks the 4,281-tok turn-1 prefill into 3 scheduler steps; 8192 (== `max_num_batched_tokens`, i.e. flag-absent) runs it whole | 8192 wins ~10 ms TTFT, TPOT within noise → small ERS win for 8192 | **NULL, and backwards.** 3 boots/arm × 150 req. ERS 0.4326 ±0.0050 (2048) vs 0.4249 ±0.0118 (8192); delta 0.0077 < the 0.0118 within-arm floor. TTFT p50 77 ms vs 80 ms — 8192 is 3 ms *worse*, not 10 ms better. TPOT mean 6.00 vs 6.05 ms | **No change** (2048 already ships) | The chunk cap does not shape TTFT on this trace: p50 is set by the many short prompts, and the one 4,281-tok prompt is a single request out of 150. Removing the cap only widens the batch, and the boot-to-boot spread with it (±0.0118 vs ±0.0050 ERS; TPOT p99 ±0.37 vs ±0.06 ms) — 8192 is the *less reproducible* arm. Do not re-test this bracket (1024/4096 included) without a trace whose prefills dominate p50 |

## Harness notes (learned the hard way, 2026-07-27)

- **`make ab` did not exist on this branch.** It lives only on `autoresearch`, where it sweeps
  *env vars* through the profile overlay's host-slack delay probe. The version added here sweeps
  a *serve flag* through `bench/arm_compose.py` instead, so it needs neither overlay.
- **`arm_compose.py` refused the control arm.** Its no-op guard (`overrides … changed nothing`)
  is a typo catcher, but an A/B control restates the shipped value by construction. Hence
  `--allow-unchanged`. Booting the control *without* the overlay instead would have measured the
  overlay, not the flag.
- **The dev box has no `aiohttp`, no pip, no venv, and a host python of 3.14.** `replay.py`
  cannot run on the host at all — `~/vtl-ab.sh` on that box had always run it inside the image.
  `REPLAY_IN_IMAGE=1` does that from the Makefile. Any future runbook step that says
  "`python3 bench/…`" fails there unless it is a pure-stdlib script (`compare.py`, `metrics.py`).
- Smoke-test a new sweep target with `AB_ROUNDS=1 AB_LIMIT=10 AB_WARM=5` (~4 min) before
  spending 30 min of boots on it.
- **`compare.py` cannot read an `ab` run.** It prints one column per *file*, and every boot's
  header is the same `localhost:8000`, so six columns are indistinguishable and its "spread"
  mixes within-arm boot noise into the between-arm delta — the exact comparison the boots were
  added to avoid. `bench/ab_summary.py` groups by arm (from the `bench-ab-<arm>-r<n>-<i>` name,
  reps inside a boot averaged) and tests the arm delta against the **within-arm boot spread the
  run itself observed**. Use it for `make ab`; `compare.py` is still right for two single runs.
- **The 0.5 ms boot-to-boot TPOT floor is pessimistic for an interleaved same-image run.** Row
  1's six boots held TPOT mean to ±0.04 ms within the 2048 arm and ±0.13 ms within 8192 — 4–13×
  tighter than the folklore number `compare.py` hardcodes, which would have called a real win
  null. Measure the floor per run (ab_summary does) rather than assuming it. Note the floor is
  *arm-dependent*: the wider-batch arm was 2–6× noisier, so it is a property of the config under
  test, not of the box.
- A boot+warm+measure cycle at `--limit 150` is ~2.5 min, not the ~5 min first estimated: a
  3-boot-per-arm two-arm A/B costs ~16 min. Six boots per arm is affordable when a delta lands
  near the floor.
