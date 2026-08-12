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
#   VTL_SSH_OPTS='-i /ssh/key -p 22'    # optional; reaches BOTH ssh and rsync. The jump runner
#                                       # is a container whose key is a bind mount at a fixed
#                                       # path, so default-key lookup cannot be assumed.
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
    # BatchMode so a missing/rejected key fails fast instead of hanging a CI job on a password
    # prompt nobody will ever see. VTL_SSH_OPTS is word-split on purpose (it is a list of flags).
    # shellcheck disable=SC2206
    SSH_CMD=(ssh -o BatchMode=yes ${VTL_SSH_OPTS:-})
    # rsync --include patterns for the pull-back, matched against the whole tree. NOT bare globs:
    # `make bench` runs under `cd $(ROUND) &&`, so its JSON lands in round-2/, and a flat
    # `rsync host:.../bench-*.json ./` would both miss it and flatten the path if it hit.
    ARTIFACTS="${VTL_ARTIFACTS:-bench-*.json bench-profile-summary.json}"

    # --delete so a file removed locally does not linger remotely and get picked up by a glob.
    # hf-model is excluded: it is the model mount and is staged on the GPU box already, not
    # something to push across a venue LAN.
    echo "exec: rsync -> ${VTL_GPU_HOST}:${REMOTE_DIR}"
    rsync -az --delete -e "${SSH_CMD[*]}" \
      --exclude '.git' \
      --exclude 'hf-model' \
      --exclude 'bench-*.json' \
      ./ "${VTL_GPU_HOST}:${REMOTE_DIR}/"

    echo "exec: run on ${VTL_GPU_HOST}"
    [ $# -ge 1 ] || { echo "exec: no command given" >&2; exit 2; }
    # Quote each argument so `make bench TARGET=http://...` and friends survive the extra shell.
    # printf %q is bash-only, which is why this script declares bash.
    # The `&&` after cd is LOAD-BEARING: without it the command becomes an extra argument to cd
    # itself -- which modern bash rejects ("too many arguments", every remote command fails) and
    # bash 3.2 silently IGNORES, i.e. cd succeeds, the real command never runs, and rc is 0.
    # Caught live: an exec.sh test run on macOS reported success for a command that never ran.
    remote_cmd="cd $(printf '%q' "$REMOTE_DIR") &&"
    for arg in "$@"; do remote_cmd+=" $(printf '%q' "$arg")"; done
    # rc is captured rather than propagated by `set -e` so the artifact pull still happens on a
    # failed bench -- a failing run's JSON is exactly what someone needs to read.
    rc=0
    "${SSH_CMD[@]}" "$VTL_GPU_HOST" "bash -lc $(printf '%q' "$remote_cmd")" || rc=$?

    echo "exec: rsync <- artifacts"
    # --include='*/' + --exclude='*' walks every directory but transfers only the named patterns,
    # preserving the path -- so round-2/bench-open.json comes back as round-2/bench-open.json.
    # --prune-empty-dirs keeps the traversal from creating the whole empty tree locally.
    includes=()
    for pat in $ARTIFACTS; do includes+=(--include="$pat"); done
    rsync -az --prune-empty-dirs -e "${SSH_CMD[*]}" \
      --include='*/' "${includes[@]}" --exclude='*' \
      "${VTL_GPU_HOST}:${REMOTE_DIR}/" ./ || true

    exit "$rc"
    ;;

  *)
    echo "exec: unknown VTL_EXEC='$VTL_EXEC' (want: local | ssh)" >&2
    exit 2
    ;;
esac
