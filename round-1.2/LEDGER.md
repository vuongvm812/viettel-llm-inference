# LEDGER — round-1.2 scored-latency hypotheses

One line per hypothesis. A row exists only after a wall-clock A/B; reasoning alone never
produces one. Measurement rule: multiple BOOTS per arm (`make ab`), not reps — the noise floor
is boot-to-boot (~0.5 ms TPOT, ~0.05 ms rep-to-rep). Rank by ERS, not by the TTFT/TPOT columns.

Box: RTX 3060 12 GB dev box (`andromeda`), NOT the judge's H200/MIG slice. It cannot resolve
host-side wins below ~1 ms — a null here means *unknown*, not negative. Scheduler-shape changes
(how many steps a prefill spans) it can resolve.

| # | Date | Hypothesis | Predicted | Measured | Kept? | Why |
|---|------|-----------|-----------|----------|-------|-----|
| 1 | 2026-07-27 | `--max-num-scheduled-tokens`: shipped 2048 chunks the 4,281-tok turn-1 prefill into 3 scheduler steps; 8192 (== `max_num_batched_tokens`, i.e. flag-absent) runs it whole | 8192 wins ~10 ms TTFT, TPOT within noise → small ERS win for 8192 | _in flight_ | _pending_ | 6-boot run launched on `andromeda` 2026-07-27 12:20 UTC (3 boots/arm, interleaved, `--limit 150`); ~30 min. Read `/tmp/ab-run.log` then `python3 bench/compare.py bench-ab-*.json` |

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
