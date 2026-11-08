"""Internal boundary-timing check (no ground truth): do independent evidence sources agree on WHEN possessions start?

Compares possession-start times of two decodes (default: ball-only vs shape-only TCN from 008) against a circular-shift null.
Usage: python timing_check.py [csv_a] [csv_b]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT), str(ROOT / "starter")]
import common
from games import half_video_blocks

OUT = ROOT / "experiments" / "008_tcn_shape" / "out"
A = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "v1_ball.csv"
B = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "v2_tcn_only.csv"
FPS = 25
blocks = half_video_blocks("BAR-ATH")
_, Bl, _ = common.load_atd_arrays(); vis = np.isfinite(Bl[:, 0])

def starts(path):
    lab = pd.read_csv(path)["label"].values
    out = []
    for _, a, b in blocks:
        seg = lab[a:b + 1]
        for t in ("H", "A"):
            s, _ = common.runs(seg == t)
            out += [(a + int(x), t) for x in s]
    return sorted(out)

sa, sb = starts(A), starts(B)
def nearest(sa, sb):
    d = []
    for f, t in sa:
        c = [abs(g - f) for g, u in sb if u == t]
        d.append(min(c) / FPS if c else np.nan)
    return np.array(d)

d = nearest(sa, sb)
rng = np.random.default_rng(0)
span = {h: (a, b) for h, a, b in blocks}
def shifted(sb, off):
    out = []
    for g, u in sb:
        h = 1 if g <= blocks[0][2] else 2; a, b = span[h]
        out.append((a + (g - a + int(off * FPS)) % (b - a + 1), u))
    return out
null = np.concatenate([nearest(sa, shifted(sb, rng.uniform(60, 600))) for _ in range(50)])

# is the start inside a ball gap?  (within 1 s of a frame with no ball)
in_gap = np.array([(~vis[max(0, f - FPS): f + FPS + 1]).any() for f, _ in sa])
print(f"A={A.name} starts: {len(sa)} | B={B.name} starts: {len(sb)}")
print("tolerance   observed   null(shift)   lift")
for tol in (1, 2, 5, 10):
    o, n = np.mean(d <= tol), np.mean(null <= tol)
    print(f"  {tol:>2d} s      {o:6.1%}      {n:6.1%}     {o / max(n, 1e-9):.1f}x")
print(f"median |dt| observed {np.nanmedian(d):.1f} s | null {np.nanmedian(null):.1f} s")
labA = pd.read_csv(A)["label"].values
after_D = np.array([labA[f - 1] == "D" for f, _ in sa])      # same D overlay in both decodes => identical start by construction
for name, m in (("start right after D (shared overlay)", after_D), ("start NOT after D (H<->A change)", ~after_D)):
    print(f"{name:38s} n={int(m.sum()):3d}  within 1 s: {np.mean(d[m] <= 1):.1%}  within 2 s: {np.mean(d[m] <= 2):.1%}  within 5 s: {np.mean(d[m] <= 5):.1%}")
sel = ~after_D
print("H<->A changes only, vs shift null:")
for tol in (1, 2, 5, 10):
    print(f"  {tol:>2d} s  observed {np.mean(d[sel] <= tol):6.1%}  null {np.mean(null <= tol):6.1%}")
for name, m in (("H<->A change near a ball gap", in_gap & sel), ("H<->A change with ball visible", ~in_gap & sel)):
    print(f"{name:25s} n={int(m.sum()):3d}  within 2 s: {np.mean(d[m] <= 2):.1%}  within 5 s: {np.mean(d[m] <= 5):.1%}")
