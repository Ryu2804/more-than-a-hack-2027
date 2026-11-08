#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"

cmd="${1:-status}"
case "$cmd" in
  start)
    hours="${2:-8}"
    if [ -f "$DIR/loop.pid" ] && kill -0 "$(cat "$DIR/loop.pid")" 2>/dev/null; then
      echo "loop already running (pid $(cat "$DIR/loop.pid"))"
      exit 0
    fi
    rm -f "$DIR/STOP"
    nohup "$PY" "$DIR/loop.py" --start "$hours" > "$DIR/loop.log" 2>&1 &
    echo $! > "$DIR/loop.pid"
    echo "started loop pid $(cat "$DIR/loop.pid") hours=$hours"
    ;;
  status)
    "$PY" "$DIR/loop.py" --status
    if [ -f "$DIR/loop.pid" ]; then echo "pid: $(cat "$DIR/loop.pid")"; fi
    ;;
  log)
    tail -n 60 "$DIR/loop.log" 2>/dev/null || true
    ;;
  report)
    "$PY" "$DIR/loop.py" --report
    ;;
  stop)
    "$PY" "$DIR/loop.py" --stop
    ;;
  *)
    echo "usage: $0 {start [hours]|status|log|report|stop}"
    exit 1
    ;;
esac
