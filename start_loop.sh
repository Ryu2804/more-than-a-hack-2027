#!/usr/bin/env bash
# Start/stop helper for the spellF1 overnight loop.
#   ./start_loop.sh start [hours]   run the queue in the background (default 8 h budget)
#   ./start_loop.sh status          print queue state
#   ./start_loop.sh log             last 40 log lines
#   ./start_loop.sh report          the morning report
#   ./start_loop.sh stop            stop the running loop
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=experiments/012_spellf1/out/nightly
mkdir -p "$OUT"
case "${1:-start}" in
  start)
    nohup "$PY" experiments/012_spellf1/nightly.py --hours "${2:-8}" >> "$OUT/nohup.log" 2>&1 &
    echo "loop started, pid $!"
    echo "follow with: ./start_loop.sh log"
    ;;
  status) "$PY" experiments/012_spellf1/nightly.py --status ;;
  log) tail -n 40 "$OUT/nightly.log" ;;
  report) cat "$OUT/MORNING_REPORT.md" ;;
  stop) pkill -f "experiments/012_spellf1/nightly.py" && echo "stopped" || echo "not running" ;;
  *) echo "usage: $0 {start [hours]|status|log|report|stop}"; exit 2 ;;
esac
