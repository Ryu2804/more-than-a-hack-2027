"""Which true possessions do we lose? Decode held-out matches with the adopted settings and report, per true-spell duration bin, the share of true spells
that have a predicted spell of the same team with IoU >= 0.3, plus predicted-spell duration share. Sandbox job. Usage: spell_hist.py GAMES PREDS OUT"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote_eval as re_, decoder as dc, common
FPS = 25
games, preds, out = sys.argv[1:4]
CFG = {**re_.DEFAULT, "merge": 1.0, "kappa": 0.05, "rate_scale": 1.2}
BINS = [0, 2, 4, 8, 15, 30, 1e9]
rows = {"gt": np.zeros(len(BINS) - 1), "hit": np.zeros(len(BINS) - 1), "pred": np.zeros(len(BINS) - 1)}
def spells(lab, blocks):
    o = []
    for _, a, b in blocks:
        seg = lab[a:b + 1]
        for t in ("H", "A"):
            s, e = common.runs(seg == t); o += [(a + x, a + y, t) for x, y in zip(s, e)]
    return o
for f in sorted(Path(preds).glob("*.npz")):
    z, p = np.load(Path(games) / f.name), np.load(f)
    y, half, dist, ix = z["y"].astype(int), z["half"].astype(int), z["dist"], z["ix"]; n = len(y)
    blocks = [(h, int(np.flatnonzero(half == h).min()), int(np.flatnonzero(half == h).max())) for h in (1, 2) if (half == h).any()]
    graded = np.zeros(n, bool)
    for _, a, b in blocks: graded[a:b + 1] = True
    P = {k: re_.interp_full(n, ix, p[k], blocks) for k in ("pdead_real", "pteam_real")}
    lp_h, lp_a = dc.ball_evidence(dist, CFG["gate_r"], CFG["tau"]); h_ev, a_ev = dc.evidence(lp_h, lp_a, P["pteam_real"], CFG["kappa"])
    dead = dc.dead_mask(P["pdead_real"], blocks, CFG["dead_thr"], CFG["min_dead_s"], CFG["shift"])
    make = lambda lam: dc.decode(n, blocks, h_ev, a_ev, lam, dead)
    lam = dc.calibrate_lambda(make, blocks, float(p["train_rate"]) * CFG["rate_scale"] * graded.sum() / FPS / 60)
    lab = re_.merge_short(make(lam), blocks, CFG["merge"])
    gt = np.array([re_.NAMES[v] for v in y]); gs, ps = spells(gt, blocks), spells(lab, blocks)
    for a, b, t in gs:
        k = np.searchsorted(BINS, (b - a) / FPS, side="right") - 1; rows["gt"][k] += 1
        for c, d, u in ps:
            if u == t and d > a and c < b and (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) >= 0.3: rows["hit"][k] += 1; break
    for a, b, t in ps: rows["pred"][np.searchsorted(BINS, (b - a) / FPS, side="right") - 1] += 1
json.dump({k: v.tolist() for k, v in rows.items()} | {"bins": BINS[:-1] + ["inf"]}, open(out, "w"))
print("HIST DONE", {k: v.tolist() for k, v in rows.items()})
