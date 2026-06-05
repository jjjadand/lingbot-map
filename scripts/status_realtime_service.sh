#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-/tmp/lingbot_realtime.pid}"
LOG_FILE="${LOG_FILE:-/tmp/lingbot_realtime.log}"
PORT="${PORT:-18087}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Realtime service running"
    echo "  PID : $pid"
  else
    echo "Realtime service not running (stale PID file)"
  fi
else
  echo "Realtime service not running"
fi

echo "  Port: $PORT"
echo "  Log : $LOG_FILE"

if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | grep ":$PORT" || true
fi
