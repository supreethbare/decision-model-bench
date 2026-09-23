#!/bin/zsh
# Run both local models back to back (not in parallel, so latencies are not skewed).
# Starts the Open-Jev server, runs it, then stops it again.
set -e
cd "${0:A:h}"
export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1

./.venv/bin/python run.py --model needle

OJ=${OPENJEV_DIR:-../Open-Jev}   # set OPENJEV_DIR if your checkout lives elsewhere
$OJ/.venv/bin/python -m jev.server --checkpoint $OJ/models/Open-Jev-2B/package/checkpoint \
  --device mps --max-length 4096 --batch-size 16 --no-prefix-cache > results/openjev_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q '"url"' results/openjev_server.log 2>/dev/null; do sleep 2; done

./.venv/bin/python run.py --model openjev
