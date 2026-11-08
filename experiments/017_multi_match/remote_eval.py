"""Sandbox job: score the full decoder (decoder.py) with held-out predictions on every other match, per config.

    python remote_eval.py --games-dir G --preds-dir P --configs '[{"gate_r":0.35,"team":"real"}, ...]' [--out result.json]
Scores with the challenge's metric code (possession_metrics.py). Labels are the datasets' own derivations, not the challenge GT.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import common
import decoder as dc
import possession_metrics as pm

FPS = 25
NAMES = {-1: "X", 0: "H", 1: "A", 2: "D"}
DEFAULT = dict(gate_r=0.35, kappa=0.02, lam="cal", dead_thr=0.5, min_dead_s=2.0, shift=0.0, tau=3.0, team="real", dead_src="real", rate_scale=1.0, merge=0.0)


def interp_full(n, ix, vals, blocks):
    out = np.full(n, np.nan)
    for _, a, b in blocks:
        m = (ix >= a) & (ix <= b)
        if m.sum() > 1:
            out[a:b + 1] = np.interp(np.arange(a, b + 1), ix[m], vals[m])
    return out


def merge_short(lab, blocks, min_s):
    out = lab.copy()
    for _, a, b in blocks:
        seg = out[a:b + 1]
        s, e = common.runs((seg == "H") | (seg == "A"))
        for si, ei in zip(s, e):
            sub = seg[si:ei]
            ch = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            starts = np.r_[0, ch]; ends = np.r_[ch, len(sub)]
            for k in range(1, len(starts)):
                if (ends[k] - starts[k]) < min_s * FPS:
                    sub[starts[k]:ends[k]] = sub[starts[k] - 1]
            seg[si:ei] = sub
    return out


def baseline_labels(dist, n, window=50):
    dH, dA = dist[:, 0], dist[:, 1]
    ok = np.isfinite(dH) & np.isfinite(dA)
    codes = np.full(n, -1, np.int8); codes[ok & (dH < dA)] = 0; codes[ok & (dH >= dA)] = 1
    h = np.concatenate(([0], np.cumsum(codes == 0))); a = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(n); lo = np.maximum(0, idx - window); hi = np.minimum(n, idx + window + 1)
    hh, aa = h[hi] - h[lo], a[hi] - a[lo]
    lab = np.full(n, "D", dtype="<U1"); seen = (hh + aa) > 0
    lab[seen] = np.where(hh[seen] >= aa[seen], "H", "A")
    return lab


def evaluate(args):
    name, gpath, ppath, configs = args
    z, p = np.load(gpath), np.load(ppath)
    y, half, dist = z["y"].astype(int), z["half"].astype(int), z["dist"]
    n = len(y)
    blocks = [(h, int(np.flatnonzero(half == h).min()), int(np.flatnonzero(half == h).max())) for h in (1, 2) if (half == h).any()]
    graded = np.zeros(n, bool)
    for _, a, b in blocks: graded[a:b + 1] = True
    gt = pm.label_codes(np.array([NAMES[int(v)] for v in y]))
    gmin = graded.sum() / FPS / 60
    ix = z["ix"]
    P = {k: interp_full(n, ix, p[k], blocks) for k in ("pdead_real", "pdead_hidden", "pteam_real", "pteam_hidden")}
    def score(lab):
        lab = np.where(graded, lab, "X")
        r = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)
        t25 = r["transitions"].get(25, r["transitions"].get("25", {}))
        return dict(macro_f1=r["macro_f1"], spell_f1=r["spell_f1_mean"], gt_spells=r["n_gt_spells"], pred_spells=r["n_pred_spells"], alive_dead_f1=r["alive_dead_f1"],
                    team_acc=r["team_acc_both_alive"], trans_f1=t25.get("f1", float("nan")), dur=r["mean_spell_duration_s"], gt_dur=r["gt_mean_spell_duration_s"])
    res = dict(game=name, dead_share_gt=float(np.mean(y[graded] == 2)), baseline=score(baseline_labels(dist, n)), configs=[])
    for c in configs:
        cfg = {**DEFAULT, **c}
        lp_h, lp_a = dc.ball_evidence(dist, cfg["gate_r"], cfg["tau"])
        h_ev, a_ev = dc.evidence(lp_h, lp_a, P[f"pteam_{cfg['team']}"], cfg["kappa"])
        dead = dc.dead_mask(P[f"pdead_{cfg['dead_src']}"], blocks, cfg["dead_thr"], cfg["min_dead_s"], cfg["shift"])
        make = lambda lam: dc.decode(n, blocks, h_ev, a_ev, lam, dead)
        lam = cfg["lam"]
        if lam == "cal":
            lam = dc.calibrate_lambda(make, blocks, float(p["train_rate"]) * cfg["rate_scale"] * gmin)   # calibrated without the merge
        lab = make(lam)
        if cfg["merge"]:
            lab = merge_short(lab, blocks, cfg["merge"])                                                 # merge applied afterwards (validated order)
        r = score(lab); r.update(lam=float(lam), dead_pred=float(dead[graded].mean()))
        res["configs"].append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", required=True); ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--configs", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    configs = json.loads(a.configs)
    pdir = Path(a.preds_dir)
    jobs = [(f.stem, Path(a.games_dir) / f"{f.stem}.npz", f, configs) for f in sorted(pdir.glob("*.npz"))]
    with Pool(min(10, len(jobs))) as pool:
        results = pool.map(evaluate, jobs)
    text = json.dumps(dict(configs=configs, results=results))
    if a.out: Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
