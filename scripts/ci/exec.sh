#!/usr/bin/env bash
# Transport shim. Runs a command where the GPU is.
#
# WHY THIS EXISTS. Round-2 is contested on-site and we do not know until day 0 whether the CI
# runner can live on the H200 VPS (Case 1) or has to sit on the laptop and reach the VPS over SSH
# (Case 2). That is the one variable we cannot decide in advance, so nothing above this file is
# allowed to care. Every GPU command in the workflows goes through here; the two cases differ by
# two environment variables and nothing else.
#
#   VTL_EXEC=local                      # default -- runner and GPU are the same box (Case 1)
#   VTL_EXEC=ssh VTL_GPU_HOST=u@vps     # runner is elsewhere (Case 2)
#
# NOT `DOCKER_HOST=ssh://...`, which looks cheaper and is a trap: `make test-kernel` bind-mounts
# $(PWD)/$(ROUND)/bench and docker-compose.localtest.yaml mounts ../hf-model. With a remote daemon
# both resolve against the REMOTE filesystem, so the mount silently succeeds against the wrong path
# and the tests run against nothing. Sending whole commands over SSH has no such ambiguity.
#
#   scripts/ci/exec.sh make build ROUND=round-2 CUDA_ARCHS='9.0+PTX'
set -euo pipefail

VTL_EXEC="${VTL_EXEC:-local}"

case "$VTL_EXEC" in
  local)
    exec "$@"
    ;;

  ssh)
    : "${VTL_GPU_HOST:?VTL_EXEC=ssh needs VTL_GPU_HOST (user@host)}"
    REMOTE_DIR="${VTL_REMOTE_DIR:-/opt/vtl-ci}"
    # rsync --include patterns for the pull-back, matched against the whole tree. NOT bare globs:
    # `make bench` runs under `cd $(ROUND) &&`, so its JSON lands in round-2/, and a flat
    # `rsync host:.../bench-*.json ./` would both miss it and flatten the path if it hit.
    ARTIFACTS="${VTL_ARTIFACTS:-bench-*.json bench-profile-summary.json}"

    # --delete so a file removed locally does not linger remotely and get picked up by a glob.
    # hf-model is excluded: it is the model mount and is staged on the GPU box already, not
    # something to push across a venue LAN.
    echo "exec: rsync -> ${VTL_GPU_HOST}:${REMOTE_DIR}"
    rsync -az --delete \
      --exclude '.git' \
      --exclude 'hf-model' \
      --exclude 'bench-*.json' \
      ./ "${VTL_GPU_HOST}:${REMOTE_DIR}/"

    echo "exec: run on ${VTL_GPU_HOST}"
    # Quote each argument so `make bench TARGET=http://...` and friends survive the extra shell.
    # printf %q is bash-only, which is why this script declares bash.
    remote_cmd="cd $(printf '%q' "$REMOTE_DIR")"
    for arg in "$@"; do remote_cmd+=" $(printf '%q' "$arg")"; done
    # rc is captured rather than propagated by `set -e` so the artifact pull still happens on a
    # failed bench -- a failing run's JSON is exactly what someone needs to read.
    rc=0
    ssh -o BatchMode=yes "$VTL_GPU_HOST" "bash -lc $(printf '%q' "$remote_cmd")" || rc=$?

    echo "exec: rsync <- artifacts"
    # --include='*/' + --exclude='*' walks every directory but transfers only the named patterns,
    # preserving the path -- so round-2/bench-open.json comes back as round-2/bench-open.json.
    # --prune-empty-dirs keeps the traversal from creating the whole empty tree locally.
    includes=()
    for pat in $ARTIFACTS; do includes+=(--include="$pat"); done
    rsync -az --prune-empty-dirs \
      --include='*/' "${includes[@]}" --exclude='*' \
      "${VTL_GPU_HOST}:${REMOTE_DIR}/" ./ || true

    exit "$rc"
    ;;

  *)
    echo "exec: unknown VTL_EXEC='$VTL_EXEC' (want: local | ssh)" >&2
    exit 2
    ;;
esac
