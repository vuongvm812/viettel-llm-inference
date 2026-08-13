#!/usr/bin/env bash
# Move a file onto a machine whose ONLY input channel is a console clipboard (paste-in works,
# copy-out does not, no ssh, no egress) -- the rung-0 scenario in docs/ci/case-3-console-only.md.
#
# Packs a file into numbered chunk files, each a STANDALONE paste-able shell snippet that appends
# base64 to a staging file on the target. No receiver script, no pre-existing state: chunk 001
# resets the staging file, the last chunk decodes, gunzips, and sha256-verifies against a hash
# baked in at pack time. A truncated or skipped paste -- this channel's signature failure -- is
# caught twice: a line-count check before decode names the missing chunk's worth of lines, and
# the hash check catches everything else. The verdict is a loud PASTE OK / PASTE CORRUPT, never a
# plausible half-file.
#
#   scripts/ci/paste-pack.sh <file> <target-path-on-vps> [outdir]
#   git diff | scripts/ci/paste-pack.sh --stdin <target-path-on-vps> [outdir]
#   scripts/ci/paste-pack.sh --probe [outdir]     # measure the console's real paste limit
#
#   VTL_PASTE_LIMIT   max bytes per chunk (default 8000 -- deliberately conservative; measure
#                     the real limit with --probe on day 0 and re-pack bigger)
#
# Pack side runs on macOS or Linux (gzip, base64, fold, split -- all POSIX-ish; sha256 via
# sha256sum or shasum). Receiver side needs only what every Ubuntu image has: sh, cat, wc,
# base64, gunzip, sha256sum.
set -euo pipefail

LIMIT="${VTL_PASTE_LIMIT:-8000}"
STAGE=/tmp/px.b64   # staging path on the target; fixed on purpose (chunks must be standalone)

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# ---------------------------------------------------------------------------------------------
# --probe: emit self-verifying chunks of increasing size. Paste them in order; the largest one
# that prints its own expected length is the console's usable limit. A truncated paste either
# breaks the shell syntax (unterminated quote) or prints a shorter length -- both are visible.
if [ "${1:-}" = "--probe" ]; then
  outdir="${2:-paste-probe}"; mkdir -p "$outdir"
  for size in 2000 4000 8000 16000 32000 64000 128000 256000; do
    f="$outdir/probe-$size.sh"
    fill=$((size - 60))   # ~60 bytes of wrapper; exactness doesn't matter, self-report does
    {
      printf 'X='
      printf 'a%.0s' $(seq 1 "$fill")
      printf '; echo "PROBE %s got ${#X} want %s"\n' "$size" "$fill"
    } > "$f"
  done
  echo "probe chunks in $outdir/ -- paste smallest to largest; the last one whose 'got' equals"
  echo "'want' is the console's usable paste size. Set VTL_PASTE_LIMIT to ~90% of it."
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# pack
if [ "${1:-}" = "--stdin" ]; then
  [ $# -ge 2 ] || { echo "usage: ... | paste-pack.sh --stdin <target-path> [outdir]" >&2; exit 2; }
  TARGET="$2"; OUTDIR="${3:-paste-out}"
  SRC="$(mktemp)"; cat > "$SRC"; CLEAN_SRC=1
else
  [ $# -ge 2 ] || { echo "usage: paste-pack.sh <file> <target-path> [outdir] | --stdin | --probe" >&2; exit 2; }
  SRC="$1"; TARGET="$2"; OUTDIR="${3:-paste-out}"; CLEAN_SRC=0
  [ -f "$SRC" ] || { echo "paste-pack: no such file: $SRC" >&2; exit 2; }
fi
mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/chunk-*.sh "$OUTDIR"/assemble.sh

HASH=$(sha256 "$SRC")
B64="$(mktemp)"
# fold to fixed-width lines: BSD base64 emits ONE line, and the chunker + the receiver's
# line-count integrity check both need line-oriented content.
gzip -c -9 "$SRC" | base64 | tr -d '\n' | fold -w 76 > "$B64"
printf '\n' >> "$B64"   # trailing newline so wc -l counts the last line
TOTAL_LINES=$(wc -l < "$B64" | tr -d ' ')

# lines per chunk: chunk byte cost = lines*77 + wrapper (~70 bytes). Stay under LIMIT.
LPC=$(( (LIMIT - 100) / 77 )); [ "$LPC" -ge 1 ] || { echo "paste-pack: VTL_PASTE_LIMIT too small" >&2; exit 2; }
split -l "$LPC" "$B64" "$OUTDIR/part-"

n=0
for part in "$OUTDIR"/part-*; do
  n=$((n+1)); id=$(printf '%03d' "$n"); chunk="$OUTDIR/chunk-$id.sh"
  {
    # chunk 001 resets the staging file -- stale state from an aborted earlier pack otherwise
    # corrupts silently, and the whole design exists to make corruption loud.
    [ "$n" -eq 1 ] && printf 'rm -f %s\n' "$STAGE"
    printf "cat >> %s <<'PXEOF'\n" "$STAGE"
    cat "$part"
    printf 'PXEOF\n'
    printf 'echo "chunk %s: $(wc -l < %s) lines staged"\n' "$id" "$STAGE"
  } > "$chunk"
  rm -f "$part"
done

# the assemble snippet: line-count gate first (names the shortfall), then decode, then hash gate.
cat > "$OUTDIR/assemble.sh" <<EOF
L=\$(wc -l < $STAGE | tr -d ' '); [ "\$L" = "$TOTAL_LINES" ] || { echo "PASTE CORRUPT: \$L of $TOTAL_LINES lines staged -- a chunk is missing or truncated"; exit 1; }
mkdir -p "\$(dirname '$TARGET')"
base64 -d < $STAGE | gunzip > '$TARGET' || { echo "PASTE CORRUPT: decode failed"; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then H=\$(sha256sum '$TARGET' | cut -d' ' -f1); else H=\$(shasum -a 256 '$TARGET' | cut -d' ' -f1); fi
[ "\$H" = "$HASH" ] && echo "PASTE OK $TARGET (\$(wc -c < '$TARGET') bytes)" || { echo "PASTE CORRUPT: sha256 mismatch"; exit 1; }
rm -f $STAGE
EOF

[ "$CLEAN_SRC" = 1 ] && rm -f "$SRC"; rm -f "$B64"
echo "packed -> $OUTDIR/: $n chunk(s) + assemble.sh (chunk size <= ${LIMIT}B, $TOTAL_LINES b64 lines, sha256 $HASH)"
echo "paste chunk-001..chunk-$(printf '%03d' "$n") in order, then assemble.sh"
