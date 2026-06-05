#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
PID_FILE="${PID_FILE:-/tmp/lingbot_realtime.pid}"
LOG_FILE="${LOG_FILE:-/tmp/lingbot_realtime.log}"
PORT="${PORT:-18087}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing python at $VENV_PY" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Realtime service already running with PID $old_pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

mkdir -p "$ROOT_DIR/.cache/torch_extensions"

cd "$ROOT_DIR"
export FLASHINFER_WORKSPACE_BASE="$ROOT_DIR/.cache"
export TORCH_EXTENSIONS_DIR="$ROOT_DIR/.cache/torch_extensions"

setsid "$VENV_PY" -u demo_realtime.py \
  --model_path "$ROOT_DIR/lingbot-map.pt" \
  --video_device /dev/video0 \
  --server_ip 0.0.0.0 \
  --port "$PORT" \
  --image_width 640 \
  --image_height 360 \
  --fps 10 \
  --capture_fps 10 \
  --pixel_format MJPG \
  --camera_num_iterations 4 \
  --num_scale_frames 4 \
  --conf_threshold 10 \
  --downsample_factor 2 \
  --export_glb \
  --export_npz \
  >"$LOG_FILE" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Started realtime service"
echo "  PID : $pid"
echo "  Port: $PORT"
echo "  Log : $LOG_FILE"
