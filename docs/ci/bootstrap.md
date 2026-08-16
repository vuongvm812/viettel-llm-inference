# Bootstrap runbook

Command-level setup for the one workflow ([`README.md`](README.md)). Commands come from the
organizers' access doc (`../../round-2/H200-server-README.md`); replace `teamNN` throughout.

## 1. Laptop — SSH access

Put the BTC-issued private key in place and add the ProxyJump stanza (§2 of the access doc):

```bash
cp teamNN ~/.ssh/teamNN && chmod 600 ~/.ssh/teamNN
```

```
Host contest-gw-teamNN
    HostName 171.226.125.255
    User teamNN
    IdentityFile ~/.ssh/teamNN
    IdentitiesOnly yes

Host teamNN
    HostName <internal VM IP from BTC>
    User teamNN
    IdentityFile ~/.ssh/teamNN
    IdentitiesOnly yes
    ProxyJump contest-gw-teamNN
    RequestTTY yes
```

Then `ssh teamNN`. That interactive terminal is the only shell channel — `scp`, `rsync`,
`ssh teamNN "cmd"` and port forwarding are all rejected by design.

## 2. VM — probe the proxy whitelist first

This decides how code reaches the box, so do it before anything else:

```bash
docker pull hello-world                                                   # must work (§5)
git ls-remote https://github.com/vuongvm812/viettel-llm-inference HEAD     # git in, or web IDE
```

Most tools already read the proxy. For any that miss it:

```bash
export HTTP_PROXY=http://10.10.1.126:3128 HTTPS_PROXY=http://10.10.1.126:3128
export NO_PROXY=localhost,127.0.0.1
```

## 3. VM — workspace and code

Work in `/srv/contest-workspace`: it is the tree the web IDE edits, so the terminal and the
browser see the same files.

```bash
cd /srv/contest-workspace
git clone https://github.com/vuongvm812/viettel-llm-inference .   # or: drag-drop via the web IDE
ln -s /data/models/<model> hf-model    # feeds the ../hf-model:/model:ro mount in the overlays
pip install -r round-2/bench/requirements.txt   # aiohttp for bench/replay.py (proxy handles it)
```

Ubuntu 24.04 is PEP 668 externally-managed — use `python3 -m venv` or
`pip --break-system-packages`.

## 4. VM — everything long-running goes in tmux

The SSH session will drop, and anything started bare dies with it.

```bash
tmux new -s vtl          # detach: Ctrl-b d      reattach: tmux attach -t vtl
```

## 5. VM — the loop

```bash
docker pull traitimbanggia/yasuoadc@sha256:<digest>          # image comes from cloud-build.yml
make ci-up   ROUND=round-2 IMAGE_DIGEST=sha256:<digest>      # pinned image, no build
make verify  ROUND=round-2                                   # plugin loaded? quant registered?
scripts/ci/preflight.sh round-2                              # GPU idle, no stray stack, flags on
make bench   ROUND=round-2 TARGET=http://localhost:8000
python3 round-2/bench/_ci_report.py                          # ERS table + ::VTL_BENCH:: line
make ci-down ROUND=round-2
```

`preflight.sh` is not ceremony: two people on one GPU means a bench can run while someone else
holds the card, and that number is wrong in a way nothing else catches.

**Fast inner loop** for plugin-only edits — no network, no cloud round-trip:

```bash
cd round-2 && docker build --network=none -f Dockerfile.console -t yasuoadc:console .
make ci-up ROUND=round-2 IMAGE_DIGEST= TAG=console
```

## 6. Getting results out

Open the web IDE (`https://code.teamNN.171.226.125.255.nip.io/`, Basic Auth `teamNN` + the BTC
password), download `round-2/bench-*.json`, and commit them from the laptop. If the day-0 probe
showed git working, `git push` from the VM instead.

Either way the results get committed: the remote teammate has no VM access, so an uncommitted
number is invisible to half the team.

## 7. Submitting — register the endpoint

Round 2 scores a **running server**, not an artifact (`../../round-2/submission-CLI-README.md`).
`airace` is pre-authenticated on the VM; it refuses to run anywhere else (`IP_MISMATCH`).

```bash
airace status                       # must print OUR team name before anything else
ip -4 addr show | grep 10.10        # the internal IP the judge will call
curl -s http://10.10.1.107:8000/v1/models   # MUST return before you register
airace endpoint --task llm --url http://10.10.1.107:8000
airace list                         # status + score, without leaving the box
```

The `curl` is not optional. **Every registration consumes one of a small number of attempts, even
if the URL is dead or malformed**, and the format is strict:

| | |
|---|---|
| ✅ `http://10.10.1.107:8000` | internal IP, no trailing slash, no path |
| ❌ `http://10.10.1.107:8000/` | trailing slash |
| ❌ `http://127.0.0.1:8000` | the judge calls from another host |
| ❌ `10.10.1.107:8000` | missing scheme |

The server must stay up for the entire grading run — dying mid-scoring scores that attempt zero.
Keep it in `tmux`, and treat grading like a bench: nobody else touches the GPU while it is in
flight.

## 8. Building a new image

Never on the VM — its build would silently degrade (see [`README.md`](README.md)). Dispatch
**Cloud Build** from the Actions tab (or `gh workflow run cloud-build.yml`) with
`cuda_archs=9.0+PTX`; the digest lands in the run summary. Re-pin it in
`round-2/docker-compose.yaml`, commit, then `docker pull` it on the VM.
