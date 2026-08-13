# Case 3 — console-only VPS

**Precondition:** no SSH to the VPS. Access is a browser console (Guacamole/noVNC-style) whose
clipboard is one-way: paste **to** the VPS works, copy **from** it does not. Organizers use this
setup precisely to stop model weights leaving the box — which means it is plausible here.

Cases 1 and 2 both die (no runner registration, no ssh hop). What replaces them depends entirely
on **egress**, so the first act on day 0 is to probe the ladder below — each rung is one pasted
command, and the answer decides the whole working mode.

## The egress ladder

| Rung | Reachable from the VPS | Working mode |
|---|---|---|
| 0 | nothing (UI only) | `paste-pack.sh` in; transcribe results off the screen; `Dockerfile.console` rebuilds; `cloud-build` mints the submission |
| 1 | + `github.com` | **`git-runner.sh`** — two-way code/results via git; near-CI. Images stay frozen at the pre-pulls |
| 2 | + Docker Hub | VPS pulls images again; one pasted `docker login` restores local build+push |
| 3 | + Actions endpoints | The real runner connects — this is just [Case 1](case-1-vps-runner.md) |

Probe (paste in order; stop at the first failure):

```bash
git ls-remote https://github.com/vuongvm812/viettel-llm-inference HEAD   # rung 1
docker pull hello-world                                                  # rung 2
curl -sS -o /dev/null -w '%{http_code}\n' https://pipelines.actions.githubusercontent.com  # rung 3-ish
```

## Rung 1 — the mode to aim for

One pasted fine-grained PAT (contents:RW, contest-scoped) turns git into a **two-way channel**,
and the copy-out restriction stops mattering: results leave the box as commits.

```bash
# paste once into the console:
git clone https://<PAT>@github.com/vuongvm812/viettel-llm-inference /opt/vtl && cd /opt/vtl
ln -s /data/models/<model> hf-model
scripts/ci/git-runner.sh cuongvd/build &
```

From then on nobody touches the UI. Anyone — laptop or remote teammate — pushes a `ci/RUN` file
on the inbox branch (`nonce:` first line, then make commands); the runner executes, commits
`output.log` + exit code + every `bench-*.json` path-preserved to the `vps-results` branch, and
pushes. Same-nonce re-push never re-runs (a new run needs a new nonce); state lives on the
remote, so runner restarts are safe.

## Rung 0 — survival mode

- **In:** `scripts/ci/paste-pack.sh <file> <target>` chunks anything into paste-sized,
  self-verifying snippets (assemble step says PASTE OK or PASTE CORRUPT — never a silent
  half-file). Measure the console's real limit **first** with `paste-pack.sh --probe`.
- **Out:** transcribe the one-line `::VTL_BENCH::` JSON that `bench/_ci_report.py` already
  prints; screenshots for anything bulkier. Every transcribed result gets committed from the
  laptop, so the ledger discipline survives.
- **What survives:** env/flag tuning (compose edits are text), the NVRTC kernel loop (paste a
  `.cu`, restart, ~15s), on-box trace regeneration (`bench/build_trace_round2.py` is seeded —
  the 6MB trace never crosses the paste channel, only the 4KB spec), `make verify` greps.
- **What dies:** CI, image pulls, the fork rebuild (network-fatal by design), artifact upload.
  The remote teammate is results-read-only until rung 1.

## The image problem

The VPS can build **only for itself**: with the pinned base pre-pulled,
`round-2/Dockerfile.console` reinstalls pasted/pulled plugin source with zero network
(`--network=none` proves it) — full-fat except `vtl._C_w4a8`, whose CUTLASS headers are gone.
But no egress means no push, and a submission is a registry digest the judge pulls. So:

```
paste/pull source → test on the H200 via Dockerfile.console
                  → dispatch cloud-build.yml on the SAME commit (GitHub-hosted, no GPU needed)
                  → re-pin the printed digest from the laptop → judge pulls from Docker Hub
```

Honest gap: the VPS-tested build and the pushed cloud build are same-source, not bit-identical.

## Day-0 checklist

1. Probe the egress ladder — before anything else; it selects the mode.
2. `docker images --digests` — verify the pre-pulls: `traitimbanggia/yasuoadc@sha256:42cfc…`
   and `vllm/vllm-openai:v0.25.0`. **At rungs 0–1 these are the only way our code reaches the
   box** — this is the pre-contest organizer ask this case elevates to critical.
3. Rung 0: `paste-pack.sh --probe` to measure the paste limit; then paste the bench subset.
   Rung 1: the three-line bootstrap above.
4. Regenerate the trace on-box; baseline bench; get the first result out (transcribed or pushed).

## When to escalate

Rungs only go up: re-probe after any organizer network change. If GitHub itself is unreachable
from the *laptop* too, that is [contingencies](contingencies.md) territory, not this case.
