# Case 2 — laptop as jump host to the VPS

**Precondition:** the VPS is reachable only from the venue LAN; the on-site laptop reaches both the
VPS (SSH) and `github.com`. This is the more likely case if the organizers hand out VPS access
scoped to the venue network.

## The picture

The laptop runs the runner and coordinates; the VPS runs every GPU command. The transport shim
rsyncs the round directory out, runs the command on the VPS, and rsyncs artifacts back — so the
workflows are byte-identical to Case 1 and only two environment variables differ.

```
remote teammate ──► GitHub ──► runner on laptop ──ssh──► VPS ──► H200
                                    (sealed nightly)
```

## What this means for the team

**The runner is inside the seal.** This is the defining cost of the case and it is organisational,
not technical:

- **No overnight work.** Everything that needs a GPU has to fit inside working hours. Sweeps get
  smaller, or they get cut. Plan the day-1 arm list assuming roughly two-thirds of the GPU time of
  Case 1.
- **The remote teammate's dispatches queue.** They stay first-class — same workflows, same ledger —
  but feedback can be overnight rather than minutes. Tell them the queue times explicitly; silence
  gets read as "my push broke the build."
- **One laptop becomes load-bearing.** If it fails or is confiscated, CI stops. Register a second
  laptop's runner as a standby on day 0 while it is cheap to do.

## Shape of the work

**Day 0** — same probes and baseline as Case 1, plus: SSH key on the runner (no passphrase prompt),
the two transport variables, and a smoke test that the rsync round-trip actually returns artifacts.
Budget ~20 minutes more than Case 1.

**Days 1–2** — GPU work is bounded by the working day. Front-load the long arms; keep the last hour
for the seal ritual rather than for one more bench that will not finish.

**Every unseal** — confirm the runner rejoined before dispatching. It restarts on its own, but
"rejoined" and "restarted" are not the same thing, and a job dispatched at a runner that is not
listening just sits.

## Risks and what they cost

| Risk | Cost | Response |
|---|---|---|
| Bench target pointed at the VPS from the laptop | Measures the venue LAN, not the server; inflates every TTFT and looks like a real regression | The replay must run **on the VPS**, where `localhost` is correct. Never "fix" this with a port-forward |
| Remote daemon path confusion | Tests silently run against the wrong mount and pass | Send whole commands over SSH, not a remote Docker socket — bind mounts resolve on the wrong filesystem |
| Sealed overnight, queue drains at 9am | An hour of the morning gone to jobs nobody is waiting for anymore | Cancel stale queued jobs at unseal before dispatching fresh ones |
| Laptop dies | CI stops entirely | Standby runner registered on day 0 |
| SSH flakiness over the venue LAN | Jobs fail mid-chain, artifacts lost | Make the chain resumable per stage; keep artifacts on the VPS until pulled |

## When to escalate

If GitHub becomes unreachable from the laptop too, go to [contingencies](contingencies.md) — the
GitLab standby is the next step and it reuses everything already built here.
