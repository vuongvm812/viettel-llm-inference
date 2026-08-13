#!/usr/bin/env bash
# Poor-man's CI runner for the rung-1 scenario (docs/ci/case-3-console-only.md): the VPS can
# reach github.com and NOTHING else -- no Actions endpoints, so no real runner; no copy-out, so
# no artifacts. But git is a two-way channel, and that is enough:
#
#   laptop / remote teammate                 VPS (this script, started once via the console)
#   ------------------------                 --------------------------------------------
#   write ci/RUN on the inbox branch  --->   poll, execute the commands
#   git push                                 commit output + bench-*.json to the results
#   read runs/<nonce>/ on results     <---   branch, git push
#
# ci/RUN format (on the inbox branch):
#     nonce: bench-day1-03            <- first line; the dedupe key. NEW RUN = NEW NONCE.
#     make bench ROUND=round-2 TARGET=http://localhost:8000
#     # comments and blank lines are skipped; commands run in order, stop at first failure
#
# Dedupe is REMOTE state: a nonce is done iff runs/<nonce>/ exists on origin/<results>. That
# makes restarts safe (state survives the process) and gives at-least-once semantics: a run
# that finished but failed to push will re-run after a restart -- the acceptable direction for
# a bench harness, where a silently dropped run is worse than a repeated one.
#
# Deliberately dumb: one loop, no daemon, no queue, no sandbox. The repo is private and ci/RUN
# is authored by the same three people who could type the command into the console themselves.
#
# Run from a DEDICATED clone (it does `git checkout -f` on the inbox commit). Auth: the clone's
# origin URL must embed a fine-grained PAT (contents:RW) -- pasted once at bootstrap.
#
#   scripts/ci/git-runner.sh <inbox-branch> [results-branch]      # results default: vps-results
#   VTL_POLL_SECS=15  VTL_RESULTS_WT=/tmp/vtl-results-wt
set -uo pipefail

INBOX="${1:?usage: git-runner.sh <inbox-branch> [results-branch]}"
RESULTS="${2:-vps-results}"
POLL="${VTL_POLL_SECS:-15}"
WT="${VTL_RESULTS_WT:-/tmp/vtl-results-wt}"
MAIN="$(pwd)"

say() { printf '[git-runner %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
trap 'say "stopped"; exit 0' INT TERM

# Results worktree, created once. If the branch does not exist yet, root it at an empty tree --
# results history stays tiny and never entangles with the code history.
ensure_results_wt() {
  [ -d "$WT/.git" ] || [ -f "$WT/.git" ] && return 0
  git worktree prune 2>/dev/null || true   # a deleted-but-registered worktree blocks re-add
  git fetch origin "$RESULTS" 2>/dev/null || true
  if ! git rev-parse -q --verify "refs/remotes/origin/$RESULTS" >/dev/null; then
    root=$(git commit-tree "$(git hash-object -t tree /dev/null)" -m "results root") || return 1
    git branch -f "$RESULTS" "$root"
  else
    git branch -f "$RESULTS" "origin/$RESULTS"
  fi
  git worktree add -f "$WT" "$RESULTS" || return 1
}

nonce_done() {  # remote state is the truth; refresh then check
  git fetch origin "$RESULTS" 2>/dev/null || true
  git rev-parse -q --verify "refs/remotes/origin/$RESULTS" >/dev/null || return 1
  git ls-tree --name-only "origin/$RESULTS" runs/ 2>/dev/null | grep -qx "runs/$1"
}

publish() {  # $1 nonce, $2 inbox sha, $3 exit code, $4 log file
  cd "$WT" || return 1
  git fetch origin "$RESULTS" 2>/dev/null || true
  git rev-parse -q --verify "refs/remotes/origin/$RESULTS" >/dev/null \
    && git reset --hard "origin/$RESULTS" >/dev/null
  mkdir -p "runs/$1"
  cp "$4" "runs/$1/output.log"
  echo "$3" > "runs/$1/exit"
  git -C "$MAIN" show "$2:ci/RUN" > "runs/$1/RUN" 2>/dev/null || true
  # everything the bench wrote, path-preserved -- same globs exec.sh pulls in Case 2
  ( cd "$MAIN" && find . -path ./.git -prune -o \( -name 'bench-*.json' -o -name 'bench-profile-summary.json' \) -print ) \
    | while read -r f; do mkdir -p "runs/$1/$(dirname "$f")"; cp "$MAIN/$f" "runs/$1/$f"; done
  git add -A "runs/$1"
  git commit -q -m "run $1 (inbox ${2:0:7}, exit $3)"
  for try in 1 2 3; do
    git push origin "$RESULTS" 2>/dev/null && { cd "$MAIN"; return 0; }
    say "push rejected (attempt $try), rebasing"
    git pull --rebase origin "$RESULTS" 2>/dev/null || true
  done
  cd "$MAIN"; say "WARN: could not push results for $1 -- will re-run after restart"; return 1
}

ensure_results_wt || { say "FATAL: cannot set up results worktree at $WT"; exit 1; }
say "polling origin/$INBOX every ${POLL}s; results -> origin/$RESULTS; worktree $WT"

while :; do
  git fetch origin "$INBOX" 2>/dev/null || { say "fetch failed (network?); retrying"; sleep "$POLL"; continue; }
  sha=$(git rev-parse -q --verify "refs/remotes/origin/$INBOX") || { sleep "$POLL"; continue; }

  run_file=$(git show "$sha:ci/RUN" 2>/dev/null) || { sleep "$POLL"; continue; }
  nonce=$(printf '%s\n' "$run_file" | sed -n '1s/^nonce:[[:space:]]*//p')
  [ -n "$nonce" ] || { say "ci/RUN on $INBOX has no 'nonce:' first line; ignoring"; sleep "$POLL"; continue; }
  # nonce becomes a directory name -- refuse anything that could escape runs/
  case "$nonce" in */*|..*|*' '*) say "bad nonce '$nonce'; ignoring"; sleep "$POLL"; continue;; esac

  if nonce_done "$nonce"; then sleep "$POLL"; continue; fi

  say "run $nonce @ ${sha:0:7}"
  git checkout -qf "$sha"
  log=$(mktemp); rc=0
  while IFS= read -r cmd; do
    case "$cmd" in ''|'#'*|nonce:*) continue;; esac
    say "  \$ $cmd"
    printf '$ %s\n' "$cmd" >> "$log"
    bash -c "$cmd" >> "$log" 2>&1 || { rc=$?; printf '(exit %s)\n' "$rc" >> "$log"; break; }
  done <<< "$run_file"

  say "run $nonce done (exit $rc); publishing"
  publish "$nonce" "$sha" "$rc" "$log"
  rm -f "$log"
  sleep "$POLL"
done
