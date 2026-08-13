# Bootstrap runbook

Command-level setup for both CI cases. Decide the case first (decision tree in
[`README.md`](README.md)): can the **VPS** reach github.com → Case 1; only the **laptop** can →
Case 2. Everything downstream — workflow, dispatch, labels — is identical between them; only the
runner you start differs.

## VPS host prerequisites (both cases)

Organizer-provided per the server spec: Docker + Compose v2, NVIDIA Container Toolkit, driver.
Ours to check on day 0:

- `python3` + `aiohttp` on the host python — `make bench` runs `bench/replay.py` on the VPS host
  (Case 2 over ssh; Case 1's caveat below). Ubuntu 24.04 is PEP 668 externally-managed, so
  `python3 -m venv` or `pip --break-system-packages` for
  `pip install -r round-2/bench/requirements.txt`.
- Model weights staged (organizers: `/data/models/<model>`, read-only).
- `docker login -u traitimbanggia` — round-2 images live under the on-site team's account
  (`traitimbanggia/yasuoadc`, fork `traitimbanggia/slowleveling`), NOT the remote teammate's
  `unseenablefuture`, precisely so pushes work at the venue. Login on **whichever host runs
  `make push` / `make vllm-fork PUSH=1`** — the VPS in both cases (in Case 2 the push runs there
  over ssh; login state is per-host, so logging in on the laptop does nothing). Pulls need no
  login: both repos must stay **public**, because the judge pulls the submission pin anonymously.

## Case 1 — runner on the VPS

```bash
# on the VPS
git clone git@github.com:vuongvm812/viettel-llm-inference.git && cd viettel-llm-inference
ln -s /data/models/<model> hf-model      # feeds the ../hf-model:/model:ro mount in the overlays
cd .github/runner/vps
GH_PAT=<PAT-with-repo-scope> docker compose -f docker-compose.runner.yaml up -d
```

Confirm the runner shows **Idle** under repo Settings → Actions → Runners (labels
`h200,sm90,gpu`).

## Case 2 — laptop jump host

```bash
# once: deploy key + VPS prep
ssh-keygen -t ed25519 -N '' -f .github/runner/jump/key    # gitignored; keep chmod 600
ssh-copy-id -i .github/runner/jump/key.pub user@vps
ssh user@vps 'sudo mkdir -p /opt/vtl-ci && sudo chown $USER /opt/vtl-ci && \
              ln -s /data/models/<model> /opt/hf-model'
# /opt/hf-model, not inside /opt/vtl-ci: exec.sh rsyncs the repo to /opt/vtl-ci, and the
# overlays' ../hf-model mount resolves to its SIBLING. rsync --delete never touches it.

# on the laptop
cd .github/runner/jump
VTL_GPU_HOST=user@vps VTL_SSH_KEY=./key GH_PAT=<PAT> \
  docker compose -f docker-compose.runner.yaml up -d

# prove the hop from INSIDE the container before any dispatch:
docker compose -f docker-compose.runner.yaml exec runner \
  bash -lc 'command -v rsync ssh python3 && ssh -o BatchMode=yes -i /ssh/key user@vps true' \
  && echo hop-ok
```

The laptop clone needs nothing else — the stock runner image already ships `rsync`/`ssh`/`python3`
(the smoke test verifies, since the tag is rolling), and every GPU command runs on the VPS.

Builds never require a GPU box: `cloud-build.yml` runs `make push` (or the fork build) on
GitHub-hosted runners with the Docker Hub secrets — the digest lands in the run summary. It is
the *only* submission-build path in [Case 3](case-3-console-only.md), and a free fallback here.

## Dispatch (identical in both cases)

```bash
make ci-gpu CUDA_ARCHS='9.0+PTX'                           # full build -> test -> bench chain
make ci-gpu CUDA_ARCHS='9.0+PTX' STAGES=bench              # bench only
make ci-gpu CUDA_ARCHS='8.6;9.0+PTX' RUNNER_LABEL=rtx3060  # rehearse on the dev box, fat binary
```

Or from the Actions tab: workflow **VPS GPU**, fill `cuda_archs` (it is required on purpose — a
wrong arch does not degrade, it dies at first kernel launch). Both runners answer the same `h200`
label, so whichever is online takes the job and the `gpu-h200` concurrency group keeps it to one
job per GPU. If both cases' runners are somehow up at once, that is fine — either produces a
correct run.

## Known gap — Case 1 bench stage

In Case 1 the bench runs *inside* the runner container, which sits on its own bridge network:
`localhost:8000` does not reach the model server published on the VPS host, and the stock runner
image lacks `aiohttp`. (Pre-existing — the legacy `bench.yml` has the same problem on the
rtx3060.) Fix when it bites: `network_mode: host` plus a `dockerfile_inline`
`pip install aiohttp` in `.github/runner/vps/docker-compose.runner.yaml`. Case 2 is immune — its
bench runs on the VPS host itself, where `localhost` is correct.
