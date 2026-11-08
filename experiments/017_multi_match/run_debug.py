import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import sandbox as sb
W = "/tmp/work_multi"
print(sb.ship(W))
print(sb.run_bg(W, "python -u remote_pipeline.py --out /tmp/work_multi/debug/out --games-dir /tmp/work_multi/games --sk 2 --folds 3 --seeds 1 --steps 40", "debug.log"))
ok, out = sb.wait(W, "ALL DONE", "debug.log", timeout=1500, every=10)
print(ok); print(out[-1500:])
