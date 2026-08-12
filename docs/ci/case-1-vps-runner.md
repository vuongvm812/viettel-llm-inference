# Case 1 — CI runner on the VPS

**Precondition:** the H200 VPS reaches `github.com` outbound on 443. Nothing inbound is needed —
self-hosted runners are outbound long-poll only, which is also the sentence that gets a firewall
exception approved.

## The picture

The VPS runs the runner, the GPU, and the work. The laptop is just a terminal. GitHub is the only
coordination point, and it is one every team member already has.

```
remote teammate ──┐
                  ├──► GitHub (dispatch + artifacts + ledger) ◄──► runner on VPS ──► H200
on-site laptops ──┘
```

## What this means for the team

Everyone is symmetric. The remote teammate dispatches the same workflows as the people in the room,
reads the same ledger, and never needs VPS access — which is the entire reason this case is worth
fighting for on day 0.

The runner living **outside the seal** is the second reason. Long sweeps, profile captures and
overnight A/B arms are possible here and nowhere else. Given a 3-day contest, the two sealed nights
are roughly 40% of the wall clock; being able to use them is a material advantage, not a convenience.

Setup commands: [`bootstrap.md`](bootstrap.md).

## Where this case lives in the repo

VPS CI is additive, never a modification of the dev-box CI: `.github/workflows/vps-gpu.yml` (the
build → test → bench chain, dispatched via `make ci-gpu`), `.github/runner/vps/` (the runner
compose, labels `h200,sm90,gpu`), and `scripts/ci/` (the transport shim and bench preflight). The
legacy `build-push` / `bench` / `bootstrap` workflows stay pinned to the rtx3060 and untouched.

## Shape of the work

**Day 0** — register the runner on the VPS with labels derived from the hardware probe, so a MIG
slice and a full card produce different labels without anything being hardcoded. Baseline into the
ledger. Announce to the remote teammate that they are unblocked.

**Days 1–2** — GPU work is serialized by a concurrency group, not by asking in chat. Anyone
dispatches; jobs queue. Overnight, queue the sweeps that are too slow for working hours.

**Day 3** — freeze, warm, re-pin, validate.

## Risks and what they cost

| Risk | Cost | Response |
|---|---|---|
| Two benches overlap on the one GPU | Both numbers are garbage, and you may not notice | Concurrency group in the GPU workflow; verify on day 0 by dispatching twice |
| VPS rebooted overnight, MIG geometry reset | A morning spent chasing tile schedules that no longer fit | Re-probe hardware before anything else each morning |
| `make warm` / NVRTC cache baked on the wrong box | Cache is arch-keyed, so it is inert — harmless but wasted | Warm on the target box only |
| VPS loses GitHub mid-contest | CI stops; work continues manually | Fall through to [Case 2](case-2-laptop-jumphost.md), then [contingencies](contingencies.md) |

## When to abandon this case

If the VPS cannot reach GitHub within ~20 minutes of trying, stop and switch to
[Case 2](case-2-laptop-jumphost.md). Do not spend day-0 hours negotiating a firewall exception —
Case 2 works with a strictly smaller network ask, and the hour is worth more than the overnight
capability.
