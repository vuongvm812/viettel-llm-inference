# Round-2 workflow — build in the cloud, run on the VM

The organizers' access rules (`../../round-2/H200-server-README.md`) leave exactly one workable
shape. This is it. There is no menu of cases: earlier drafts planned four topologies, and the
published rules ruled out three.

## What the rules force

| Rule (§ of the access doc) | Consequence |
|---|---|
| §3 SSH is **interactive terminal only** — no `scp`/`rsync`, no `ssh host "cmd"`, no `-L/-D/-R` | No remote-exec automation into the VM. Long-running work lives in `tmux`, started by hand. |
| §5/§7 no direct internet; egress is the proxy `10.10.1.126:3128` (apt, pip, npm, conda, **docker pull**) + a submission API | A self-hosted Actions runner on the VM cannot reach the Actions endpoints. **CI does not run on the VM.** |
| §4 web IDE (VS Code + JupyterLab) over `/srv/contest-workspace` | The sanctioned two-way file channel: drag-drop up, editor download. |

One more constraint comes from our own Dockerfile: its fetches from `sh.rustup.rs`, `crates.io`,
the CUTLASS tarball and `archive.ubuntu.com` are all **deliberately non-fatal**. Behind a
whitelist proxy a VM build therefore does not fail — it silently produces a gutted image (no Rust
scheduler wheel, no `vtl._C_w4a8`, possibly glibc malloc) that benches worse for reasons nobody
would trace back to the network. **So the VM never builds. It pulls.**

## The workflow

```
laptop / remote teammate            GitHub-hosted runner            contest VM (H200)
  push code ──────────────────────► cloud-build.yml                  ssh teamNN  (interactive)
                                    make push (full internet)        tmux session
                                    digest ──► step summary           │
  re-pin digest, commit ◄──────────────────────────────────────────── docker pull   (proxy)
                                                                      make ci-up / make verify
                                                                      preflight.sh / make bench
  results committed ◄──── web IDE download (or git push if allowed) ◄─ bench-*.json
                                                                      │
                                                    scoring ◄───────── airace endpoint --task llm
```

- **Build:** `cloud-build.yml` on GitHub-hosted runners — full internet, so the image is
  full-fat. Compiling needs nvcc, not a GPU. Anyone with repo write access can dispatch it.
- **Ship:** `docker pull` the pinned digest through the proxy — explicitly supported by §5.
- **Run:** `make ci-up` → `make verify` → `scripts/ci/preflight.sh` → `make bench`, in `tmux`.
- **Iterate fast** on plugin-only edits without a cloud round-trip: `round-2/Dockerfile.console`
  rebuilds `FROM` the pulled image with **no network at all**. Full-fat except `vtl._C_w4a8`,
  whose CUTLASS headers were deleted after the original build.
- **Results out:** download `bench-*.json` via the web IDE (or `git push` if github.com turns out
  to be whitelisted), then commit them from the laptop.

Setup commands: [`bootstrap.md`](bootstrap.md). Pre-contest work: [`00-blockers.md`](00-blockers.md).

## Who does what

**On-site** holds the only VM access: SSH key + web IDE password. They run the loop, and they are
the only ones who can see a number the moment it exists.

**The remote teammate has no VM access at all.** That is a deliberate constraint, not an
oversight, and it shapes two habits:

- Every bench result gets **committed** — a number that only ever appeared on someone's terminal
  does not exist for half the team.
- Their work is the GitHub side, which needs no VM: dispatching `cloud-build.yml`, reviewing PRs,
  the `vtl-sched` Rust crate (its tests are stdlib-only), the bench tooling, and analysis of
  committed results.

## Day 0, in order

1. **Probe the proxy whitelist** — it decides how code reaches the VM:
   `git ls-remote https://github.com/vuongvm812/viettel-llm-inference HEAD` (git in → use git;
   fails → use the web IDE) and `docker pull hello-world` (must work per §5).
2. Confirm the pinned image pulls: `docker pull traitimbanggia/yasuoadc@sha256:42cfc1ae…`.
3. Stand up the loop and get a **baseline ERS committed** in the first hour — everything after is
   a delta against it.

## Submission is an endpoint, not an image

Round 2 does not submit an artifact (`../../round-2/submission-CLI-README.md`). We run the server
on our own VM and register its address with the `airace` CLI, which is pre-authenticated on the
box:

```bash
airace endpoint --task llm --url http://<internal-VM-IP>:8000
```

The judge then calls that endpoint **from another machine**. Three consequences that change how we
work, not just how we submit:

- **The server must survive the grading run.** It dies mid-scoring, that attempt is a zero. So the
  server lives in `tmux`, and nobody touches the GPU while grading is in flight — the same
  discipline `preflight.sh` enforces for benches, now with a score attached.
- **Attempts are a finite resource.** Every registration consumes one, *including* one that points
  at a dead or malformed URL. Verify with `curl` before registering, every time.
- **The URL format is unforgiving**: `http://` required, internal VM IP (never `localhost` — the
  judge is on a different host), no trailing slash, no path. Our compose already binds
  `--host=0.0.0.0` and publishes `8000:8000`, which is what makes this reachable.

The image digest still matters, just not to the judge: it is how *we* get a reproducible,
full-fat server onto the VM. `airace list` shows scores without leaving the box.
