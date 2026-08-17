#!/usr/bin/env bash
# airace submission driver for round-2. Runs ON the contest VM (airace is TTY-only there,
# not remotely drivable). Install: crontab -e -> `0 * * * * /bin/bash <repo>/scripts/vm/submit-cron.sh`
#   submit-cron.sh                   one guarded submission (what cron calls hourly)
#   submit-cron.sh --submissions 3   run 3 submissions sequentially, waiting for each
#                                    grading run to finish before the next
set -u
URL="http://10.10.1.117:8000"          # this VM's internal IP (ip -4 addr show | grep 10.10)
DAILY_QUOTA=30
N=1
[ "${1:-}" = "--submissions" ] && N="${2:?usage: submit-cron.sh [--submissions N]}"
DIR="$HOME/submit-log"; mkdir -p "$DIR"
LOG="$DIR/submit-$(date +%F).log"
REM_FILE="$DIR/remaining-$(date +%F)"  # freshest "Lượt còn lại" seen today
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) invocation, submissions=$N"
exec 9>"$DIR/.lock"; flock -n 9 || { echo "SKIP: another invocation is running"; exit 0; }

remaining() {   # autodetect attempts left today
  # 1) freshest: "Lượt còn lại: NN" cached from the last registration output today
  if [ -s "$REM_FILE" ]; then cat "$REM_FILE"; return; fi
  # 2) derive: quota minus today's registrations counted from airace list ("[17/08 07:14] ...")
  local used; used=$(airace list 2>/dev/null | grep -cF "[$(date +%d/%m) ") || used=0
  echo $((DAILY_QUOTA - used))
}

grading_busy() { airace list 2>&1 | tee "$DIR/airace-list-last.txt" | grep -qE 'đang chạy|đang chờ'; }

submit_once() {
  local rem; rem=$(remaining)
  [ "$rem" -le 0 ] && { echo "STOP: no attempts left today"; return 1; }
  # a dead endpoint still burns an attempt (submission-CLI-README) -- one cheap curl
  # protects the budget
  curl -fsm 10 "$URL/v1/models" >/dev/null \
    || { echo "STOP: endpoint not answering, attempt preserved"; return 1; }
  echo "SUBMIT (remaining before: $rem)"
  local out; out=$(airace endpoint --task llm --url "$URL" 2>&1); echo "$out"
  echo "$out" | grep -oE 'Lượt còn lại: *[0-9]+' | grep -oE '[0-9]+' >"$REM_FILE" || true
}

for i in $(seq 1 "$N"); do
  if grading_busy; then
    if [ "$N" -eq 1 ]; then echo "SKIP: a submission is still being graded"; exit 0; fi
    echo "WAIT: grading in flight..."   # batch mode: poll up to 30 min (a run's max), then proceed
    for _ in $(seq 1 15); do sleep 120; grading_busy || break; done
  fi
  submit_once || exit 0
done
airace list | tee "$DIR/airace-list-last.txt"
