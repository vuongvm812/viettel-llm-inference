# Contingencies

Real but unlikely. None of these changes *what* we build — only where it runs. Kept short on
purpose; if one happens, the work is mostly already done.

> These cover the laptop/GitHub side. The VPS-side degraded modes — no SSH, console-only access,
> partial egress — are [Case 3](case-3-console-only.md), which has its own ladder and tooling.

## GitHub unreachable

**First, five minutes of cheap moves.** Most "GitHub is blocked" turns out to be narrower than it
looks: SSH over port 443 (`ssh.github.com`) survives most port filters, and if git works but pushes
fail it is the *registry* that is blocked, not GitHub — switching to GHCR keeps everything else.

**If it is really blocked:** move to a private GitLab. Either `gitlab.com` or a self-hosted instance
on the VPS; the self-hosted variant only needs the VPS reachable, so it survives cases where nothing
external does.

The switch should be a mechanical ~20 minutes — mirror the repo, register the runner, set the CI
variables — because `.gitlab-ci.yml` is committed in advance and its jobs call the same scripts as
the GitHub workflows. **Dry-run it once before the contest.** A standby that has never been executed
is not a standby, and the remote teammate needs a GitLab account created while email still works.

## No outbound internet at all

A bare repo or Gitea on the VPS becomes the git host for everyone on-site. CI degrades to running
the chain by hand; the ledger still gets written, which is what matters.

The remote teammate is the real casualty. One person, at fixed times twice a day, carries a git
bundle to a phone hotspot and back. Rules that make this survivable: fixed times announced in
advance, cumulative bundles rather than incremental ones, the ledger always included, and merges
happening on-site so two people who cannot talk are never resolving the same conflict.

**This case is won or lost before travelling**, not during it: images pre-pulled by digest, the
CUTLASS tarball and cargo vendor dir carried, and the vLLM fork already built — that last one
because its Dockerfile is the single thing in the repo that cannot build offline at all.

## The GPU disappears

Fall back to the RTX 3060 runner. What still works: the cheap gate, image builds, non-sm90 kernel
parity, accuracy evaluation, and the whole CPU-side surface.

What cannot be validated there, and must be flagged rather than assumed: the sm_90a W4A8 extension
and its tile schedules, megakernel co-residency, L2 persistence, and every MIG-shaped tuning
decision. A submission tuned on a 3060 and never re-validated on the H200 is a guess.
