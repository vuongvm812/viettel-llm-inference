# Before travelling

Two items, both scoped to `round-2` — the earlier rounds are frozen and CI does not target them.
One means round-2 is currently running against something other than what we think. One would tell
us *now*, on hardware we already have, whether the model-agnostic claim actually holds.

---

## B1 — round-2 is booting against stock vLLM

`round-2/round.mk` is unpinned on purpose, so the five rust-frontend patches — sonic-rs decode, shm
IPC, per-token streaming, and the rest — **are not in play today**. Any measurement taken now is of
a different server than the one we intend to submit.

Build and pin the fork before travelling. This is not just tidiness: `Dockerfile.vllm-fork` is the
one image in the repo that cannot build offline at all, so it must not be on the critical path at
the venue.

## B2 — Run the acceptance test on the RTX 3060 now

HANDOFF §5 already defines it: boot a plain dense model unlike the previous round's — it names
`Qwen/Qwen3-0.6B` — and require both a green `make verify` and the Rust scheduler reporting
`AUTHORITY mode active`. That is the entire model-agnostic claim under test, and 0.6B fits
comfortably on a 12 GB card, so it needs neither the H200 nor the venue.

**This is the highest-value pre-contest work.** If the scheduler does not engage on an unseen dense
model, adding a `Kind` variant becomes day-1 critical path — a third of the contest spent on
plumbing instead of optimizing. Finding that out now costs an afternoon. Finding it out on day 1
costs the round.

Make it a permanent CI job so it stays true as the stack changes.
