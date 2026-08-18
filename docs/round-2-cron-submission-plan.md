# Round-2 cron submission automation

## Context

Round-2 submissions are endpoint registrations made on the VM with the organizer CLI:
`airace endpoint --task llm --url http://10.10.1.117:8000` (see
`round-2/submission-CLI-README.md`; note this VM's IP is **10.10.1.117** — the `.107` in
older docs examples is stale). Each grading run takes ~20-30 min; the daily budget is 30
attempts, and **a registration consumes an attempt even if the endpoint is dead**. Scores
are read back with `airace list` ("chấm: xong — ĐIỂM …" / "đang chạy" / "đang chờ" / "lỗi").

Decision: the server runs continuously and is managed manually — the automation only
**submits on a schedule** via cron. It autodetects the remaining daily attempts and stops
at 0; `--submissions N` runs N submissions, 3 min apart. No config changes, no restarts,
no failure orchestration (a `lỗi` result is just logged; the next tick proceeds normally).

airace cannot be driven remotely (interactive-TTY SSH only), so the script lives in the
repo and is installed by hand on the VM.

## The script: `scripts/vm/submit-cron.sh`

- `submit-cron.sh` — one guarded submission (what cron calls hourly).
- `submit-cron.sh --submissions N` — N submissions, sleeping 3 min between each.
  Submissions **queue on the judge** — this does not wait for a prior run to finish
  grading before firing the next one.

Guards, all purely budget-protective:

1. **Remaining-attempt autodetection**: freshest source is airace's own `Lượt còn lại: NN`
   line cached from the last registration today (`~/submit-log/remaining-YYYY-MM-DD`);
   before any submission has run, it derives the count from `airace list` (registrations
   are stamped `[dd/mm hh:mm]` — count today's, subtract from 30). Manual submissions are
   counted automatically either way. Date-stamped cache ⇒ resets with the daily quota.
2. `curl /v1/models` pre-flight — a dead endpoint still burns an attempt, so never
   register one.

Logs accumulate in `~/submit-log/submit-YYYY-MM-DD.log`; the latest `airace list`
snapshot (with all scores) is always in `~/submit-log/airace-list-last.txt`.

## Install (on the VM)

```sh
git pull
chmod +x scripts/vm/submit-cron.sh
command -v airace              # if cron's PATH won't find it, hardcode this path in the script
bash scripts/vm/submit-cron.sh # one manual run to verify end-to-end
crontab -e                     # add:
# 0 * * * * /bin/bash <repo>/scripts/vm/submit-cron.sh
```

Optionally restrict to waking hours (`0 6-23 * * *`) to bias attempts toward hours when
someone can react. If airace refuses to run without a TTY under cron, wrap its calls as
`script -qec "airace …" /dev/null`.

## Verification

- Manual run prints `SUBMIT (remaining before: NN)`; `airace list` shows the new
  submission; `~/submit-log/` contains the day's log + cached remaining count.
- With the server stopped: `STOP: endpoint not answering, attempt preserved`.
- `--submissions 2` shows two `SUBMIT` lines, 3 min apart, with no wait for grading
  in between.
- Quota exhausted: `STOP: no attempts left today`, no airace registration attempted.
- Next day: the date-suffixed remaining-file rolls over automatically; no reset job needed.
