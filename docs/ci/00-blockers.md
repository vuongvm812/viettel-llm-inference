# Before travelling

Three items, all scoped to `round-2`. One means round-2 is currently running against something
other than what we think. One would tell us *now*, on hardware we already have, whether the
model-agnostic claim actually holds. One makes the venue workflow possible at all.

---

## B1 — round-2 is booting against stock vLLM

`round-2/round.mk` is unpinned on purpose, so the five rust-frontend patches — sonic-rs decode, shm
IPC, per-token streaming, and the rest — **are not in play today**. Any measurement taken now is of
a different server than the one we intend to submit.

Build and pin the fork before travelling. This is not just tidiness: `Dockerfile.vllm-fork` is the
one image in the repo that cannot build offline at all, and the contest VM has no direct internet,
so it can never be built there.

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

## B3 — Add the Docker Hub secrets, and dispatch Cloud Build once

`cloud-build.yml` is the **only** way to build an image for the contest: the VM has no direct
internet, and building there silently degrades (see [`README.md`](README.md)). It needs two repo
secrets — `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` for the `traitimbanggia` account — set under
Settings → Secrets and variables → Actions.

Then dispatch it once before travelling. That single run proves three things at once: the secrets
work, the hosted runner has enough disk for the ~10 GB base plus build layers, and the digest lands
in the step summary where the re-pin step expects it. Discovering any of those at the venue costs a
day of the contest.

## Settled — how round 2 is actually submitted

Not an artifact. Per `../../round-2/submission-CLI-README.md`, the LLM task is scored by
registering our own running server's address:

```bash
airace endpoint --task llm --url http://<internal-VM-IP>:8000
```

Two facts to internalise before day 1, because both cost real points:

- **Each registration consumes one of a small number of attempts** — including a registration that
  points at a dead or malformed URL. `curl http://<ip>:8000/v1/models` first, every time.
- **The server must stay up for the whole grading run**; dying mid-scoring is a zero for that
  attempt. Run it in `tmux`, and treat a grading run like a bench — nobody else touches the GPU.

Nothing in the repo needs to change for this: the submission compose already binds
`--host=0.0.0.0` and publishes `8000:8000`. The digest pin remains useful for reproducibility on
our side, not for the judge.
