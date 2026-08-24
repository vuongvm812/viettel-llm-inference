#!/usr/bin/env python3
"""``bash -n`` the shell heredocs embedded in a Dockerfile's ``RUN <<'MARKER' bash`` blocks.

Those blocks are real shell scripts that only ever execute inside a long, expensive image
build (`make vllm-fork VLLM_RS_PGO=1` for the BOLT ones), so a syntax error in them is
discovered minutes-to-hours later, on a build host, in a layer that also costs a PGO
rebuild. This is the cheap insurance: extract each named block and hand it to `bash -n`.

Extraction is deliberately narrow rather than a general Dockerfile parser -- the block must
be named on the command line, its opener must match ``<<'MARKER'`` and its terminator must be
the marker alone on a line at column 0, which is exactly how docker's heredoc syntax requires
it to be written. Anything else is reported as an error rather than silently skipped, so this
cannot quietly stop checking when the Dockerfile is edited.

Usage:  check_dockerfile_heredocs.py <dockerfile> <MARKER> [<MARKER> ...]
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile


def extract(lines: list[str], marker: str) -> tuple[str, int]:
    """The body of ``<<'marker'`` and the 1-based line it starts on. Raises if not found."""
    opener = re.compile(r"<<'%s'(\s|$)" % re.escape(marker))
    start = None
    for i, line in enumerate(lines):
        if start is None:
            if opener.search(line):
                start = i + 1
            continue
        if line.rstrip("\n") == marker:
            return "\n".join(lines[start:i]), start + 1
    if start is None:
        raise SystemExit(f"heredoc check: no `<<'{marker}'` opener found")
    raise SystemExit(f"heredoc check: `<<'{marker}'` is never terminated by a bare `{marker}` line")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        raise SystemExit(f"usage: {argv[0]} <dockerfile> <MARKER> [<MARKER> ...]")
    path, markers = argv[1], argv[2:]
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rc = 0
    for marker in markers:
        body, first_line = extract(lines, marker)
        if not body.strip():
            print(f"heredoc check: {marker} is empty", file=sys.stderr)
            rc = 1
            continue
        # A leading blank run keeps `bash -n`'s reported line numbers aligned with the
        # Dockerfile's, so an error message points at the file you actually edit.
        padded = "\n" * (first_line - 1) + body + "\n"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".sh", prefix=f"{marker}-", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(padded)
                tmp = fh.name
            proc = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        finally:
            if tmp:
                os.unlink(tmp)
        if proc.returncode != 0:
            print(f"heredoc check: {path} :: {marker} FAILED bash -n", file=sys.stderr)
            sys.stderr.write(proc.stderr.replace(tmp or "", f"{path}({marker})"))
            rc = 1
        else:
            print(f"bash -n {path} :: {marker} (lines {first_line}-{first_line + body.count(chr(10))}): ok")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
