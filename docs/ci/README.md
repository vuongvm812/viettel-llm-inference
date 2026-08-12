# Round-2 CI + collaboration — the plan

## The situation

Three days on-site. Three people, one of whom is remote with GitHub access only. One H200 VPS whose
shape (MIG 1g.18gb slice or full card) and served model are both unknown until the day. The on-site
machine is sealed each evening and unsealed the next morning. Venue network is unknown.

Two variables drive everything, and only one of them is under our control:

- **Where the CI runner can live** — on the VPS, or on the on-site laptop acting as a jump host.
  This is the thing we cannot decide until day 0, so the design must not care.
- **What the model and GPU turn out to be** — decides roughly half the tuning surface, and is
  discoverable in the first hour with the right tooling in place.

## What we build

One idea carries the whole plan: **CI logic lives in scripts and Make targets, not in workflow
YAML, and every GPU command goes through a single transport shim.** The consequence is that the two
cases below differ by two environment variables, and a move to GitLab is a change of *where CI runs*
rather than a rewrite.

Concretely, four pieces:

1. **A transport shim** (`scripts/ci/exec.sh`) — runs a command locally, or rsyncs to the VPS and
   runs it there. This is what absorbs the runner-location uncertainty.
2. **Probes** — hardware (`hw-profile`) and model (`model-probe`), so day 0 answers "what box is
   this, what model is this, what will silently not work" in minutes rather than by discovery.
3. **A chain, not a pipeline** — one manual dispatch runs build → test → bench → profile end to end.
   Triggers stay manual: there is one shared GPU, and auto-triggering would produce benchmark
   numbers taken while another job was running.
4. **A committed results ledger** — every bench appends a row on a `results` branch. This is the
   only way the remote teammate sees whether anything helped.

**Scope: `round-2` only.** `round-1.1` and `round-1.2` are frozen and CI does not target them.
`ROUND=` stays parameterized so an older round can be run by hand, but nothing in CI fans out over
rounds — a gate that is red on an abandoned round is a gate nobody reads.

## The two cases we plan for

| | [Case 1 — runner on the VPS](case-1-vps-runner.md) | [Case 2 — laptop as jump host](case-2-laptop-jumphost.md) |
|---|---|---|
| Precondition | VPS reaches github.com | Only the laptop does |
| Runner sits | Outside the seal | **Inside** the seal |
| Overnight work | Possible | Not possible |
| Remote teammate | Fully first-class | First-class, but dispatches queue overnight |
| Setup cost | ~20 min | ~40 min |

Decide on day 0 with one command: `ssh <vps> curl -sS -o /dev/null -w '%{http_code}' https://api.github.com`.
Then follow [`bootstrap.md`](bootstrap.md) — the command-level runbook for standing up either case
and dispatching the first chain.

Everything else — GitHub blocked, no internet at all, the GPU disappearing — is covered briefly in
[contingencies](contingencies.md). Those are real but unlikely, and none of them changes what we
build; they only change where it runs.

## The three-day shape

**Day 0 (first hour)** — probe hardware, probe model, register the runner, set the four serve
literals, get a scored baseline into the ledger. Everything after is a delta against that baseline;
without it the first day's work is unfalsifiable.

**Days 1–2** — optimize against the ledger. On-site owns anything needing the GPU; the remote
teammate owns the CPU-bound surfaces — the cheap gate, the ledger and reporting, the GitLab
standby, and `vtl-sched` (its tests are stdlib-only, so the whole crate is workable from a laptop).

**Day 3** — freeze early. `make warm`, re-pin the digest, `compose-lint`, final bench. The
submission is a compose file pinned by digest; a build that fails scores zero, so the last day is
for validation, not for landing changes.

**Every evening** — the machine is sealed. Push everything, tag, and write a `STATE.md` for the
remote teammate; across a sealed night it is the only handoff they get.

**Every morning** — re-probe the hardware *before* anything else. A reboot can reset MIG geometry,
and a changed SM count silently invalidates every tile schedule pinned the day before.

## Before travelling

[`00-blockers.md`](00-blockers.md) — two items. One means round-2 is currently running against stock
vLLM; one is a test that would tell us now, on hardware we already have, whether the model-agnostic
claim actually holds.

## Source of truth

`round-2/HANDOFF.md` is authoritative for the stack. Its §3 is already a written "adding a model"
procedure and its §5 already names what is unverified. This plan automates and asserts §3 — it does
not restate it.
