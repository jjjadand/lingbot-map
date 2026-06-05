#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-/tmp/lingbot_realtime.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file: $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid"
  fi
  echo "Stopped realtime service PID $pid"
else
  echo "PID $pid is not running"
fi

rm -f "$PID_FILE"
