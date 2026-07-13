#!/bin/bash
set -e

python3 -m vllm.entrypoints.openai.api_server "$@" &
PID=$!

until python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; do
  sleep 2
done

python3 /opt/vtl/warmup.py || true
touch /opt/vtl/warmed

wait $PID
