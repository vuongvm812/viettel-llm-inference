#!/usr/bin/env bash
# airace submission driver for round-2. Runs ON the contest VM (airace is TTY-only there,
# not remotely drivable). Install: crontab -e -> `0 * * * * /bin/bash <repo>/scripts/vm/submit-cron.sh`
#   submit-cron.sh                   one guarded submission (what cron calls hourly)
#   submit-cron.sh --submissions 3   3 submissions, 3 min apart. Submissions queue on the
#                                    judge, so this does NOT wait for grading between them.
set -u
URL="http://10.10.1.117:8000"          # this VM's internal IP (ip -4 addr show | grep 10.10)
DAILY_QUOTA=30
INTERVAL=180                           # 3 min between queued submissions
N=1
[ "${1:-}" = "--submissions" ] && N="${2:?usage: submit-cron.sh [--submissions N]}"
DIR="$HOME/submit-log"; mkdir -p "$DIR"
LOG="$DIR/submit-$(date +%F).log"
REM_FILE="$DIR/remaining-$(date +%F)"  # freshest "Lượt còn lại" seen today
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) invocation, submissions=$N"

remaining() {   # autodetect attempts left today
  # 1) freshest: "Lượt còn lại: NN" cached from the last registration output today
  if [ -s "$REM_FILE" ]; then cat "$REM_FILE"; return; fi
  # 2) derive: quota minus today's registrations counted from airace list ("[17/08 07:14] ...")
  local used; used=$(airace list 2>/dev/null | grep -cF "[$(date +%d/%m) ") || used=0
  echo $((DAILY_QUOTA - used))
}

for i in $(seq 1 "$N"); do
  rem=$(remaining)
  [ "$rem" -le 0 ] && { echo "STOP: no attempts left today"; exit 0; }
  # a dead endpoint still burns an attempt (submission-CLI-README) -- one cheap curl
  # protects the budget
  curl -fsm 10 "$URL/v1/models" >/dev/null \
    || { echo "STOP: endpoint not answering, attempt preserved"; exit 0; }
  echo "SUBMIT (remaining before: $rem)"
  out=$(airace endpoint --task llm --url "$URL" 2>&1); echo "$out"
  echo "$out" | grep -oE 'Lượt còn lại: *[0-9]+' | grep -oE '[0-9]+' >"$REM_FILE" || true
  [ "$i" -lt "$N" ] && sleep "$INTERVAL"
done
airace list | tee "$DIR/airace-list-last.txt"
