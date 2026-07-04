#!/usr/bin/env bash
# Resolve a research-paper input to a local file path for Claude's Read tool.
# Accepts: arXiv id, arXiv abs/pdf URL, generic PDF/publisher URL, or a local path.
# Prints exactly one line to stdout: the resolved local file path. Errors go to stderr.
set -euo pipefail

err() { echo "fetch_paper: $*" >&2; exit 1; }

[ $# -eq 1 ] || err "usage: fetch_paper.sh <arxiv-id|arxiv-url|pdf-url|path>"
input="$1"

# Local file: echo as-is.
if [ -f "$input" ]; then
  echo "$input"
  exit 0
fi

url="$input"

# Bare arXiv id (e.g. 2301.01234 or 2301.01234v2) -> pdf URL.
if [[ "$input" =~ ^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$ ]]; then
  url="https://arxiv.org/pdf/${input}.pdf"
# arXiv abs URL -> pdf URL.
elif [[ "$input" =~ ^https?://arxiv\.org/abs/ ]]; then
  url="${input/\/abs\//\/pdf\/}"
  url="${url%.pdf}.pdf"
fi

case "$url" in
  http://*|https://*) : ;;
  *) err "not a local file and not a URL: $input" ;;
esac

command -v curl >/dev/null 2>&1 || err "curl is required but not found"

tmp="$(mktemp -t paper.XXXXXX)"
ctype="$(curl -sL -A "Mozilla/5.0" -w '%{content_type}' -o "$tmp" "$url")" \
  || { rm -f "$tmp"; err "download failed: $url"; }

[ -s "$tmp" ] || { rm -f "$tmp"; err "downloaded empty file from: $url"; }

# Give PDFs a .pdf extension so Read detects them; otherwise keep as .html.
case "$ctype" in
  *pdf*) out="${tmp}.pdf" ;;
  *)     out="${tmp}.html" ;;
esac
mv "$tmp" "$out"
echo "$out"
