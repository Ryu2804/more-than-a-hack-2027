"""Decoder-side ablations on the held-out predictions (analysis only, no training)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import harness as H
import decoder as dc
import features as ft
import other_games as og
import common

CFG = {"team": "real", "gate_r": 0.35}          # current best configuration (adopted by the rule from sweep 1)
_extra = {}

def extra(g):
    if g not in _extra:
        d = og.load(g); F = ft.build(d["P"], d["B"], d["T"], d["half"])
        _extra[g] = dict(c=F["ball"][0][:, 0].astype(float), obs=F["ball"][0][:, 1] > 0, P=d["P"], B=d["B"], T=d["T"])
    return _extra[g]

def merge_short(lab, blocks, min_s):
    """Merge H/A runs shorter than min_s into the previous run (post-processing)."""
    out = lab.copy()
    for _, a, b in blocks:
        seg = out[a:b + 1]
        s, e = common.runs((seg == "H") | (seg == "A"))
        for si, ei in zip(s, e):
            sub = seg[si:ei]
            ch = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            starts = np.r_[0, ch]; ends = np.r_[ch, len(sub)]
            for k in range(1, len(starts)):
                if (ends[k] - starts[k]) < min_s * 25:
                    sub[starts[k]:ends[k]] = sub[starts[k] - 1]
            seg[si:ei] = sub
    return out

def run(variant, games=(1, 2, 3)):
    rows = []
    for g in games:
        G = H.game(g); X = extra(g); cfg = {**H.DEFAULT, **CFG}
        lp_h, lp_a = dc.ball_evidence(G["dist"], cfg["gate_r"], cfg["tau"])
        v = variant
        if v.get("soft"):
            k = v["soft"]; gate = np.isfinite(G["dist"][:, 0]) & np.isfinite(G["dist"][:, 1]) & (np.minimum(G["dist"][:, 0], G["dist"][:, 1]) <= cfg["gate_r"] * v.get("soft_gate", 1.0))
            sh = np.where(gate & X["obs"], -np.logaddexp(0, -k * X["c"]), 0.0); sa = np.where(gate & X["obs"], -np.logaddexp(0, k * X["c"]), 0.0)
            lp_h, lp_a = (sh, sa) if v.get("mode") == "soft_only" else (lp_h + sh, lp_a + sa)
        h_ev, a_ev = dc.evidence(lp_h, lp_a, G["pteam_real"], cfg["kappa"])
        pdead = G["pdead_real"] if v.get("dead", "real") == "real" else (0.5 * (G["pdead_real"] + G["pdead_hidden"]) if v.get("dead") == "mean" else G["pdead_hidden"])
        dead = dc.dead_mask(pdead, G["blocks"], cfg["dead_thr"], cfg["min_dead_s"], cfg["shift"])
        make = lambda lam: dc.decode(G["n"], G["blocks"], h_ev, a_ev, lam, dead)
        lam = dc.calibrate_lambda(make, G["blocks"], G["train_rate"] * G["graded_min"])
        lab = make(lam)
        if v.get("merge"):
            lab = merge_short(lab, G["blocks"], v["merge"])
        r = H.score(G, lab); r["lam"] = lam; rows.append(r)
    return rows

if __name__ == "__main__":
    ref = run({}); rs = H.summarise(ref); rps = [r["spell_f1"] for r in ref]
    print(f"REF  spellF1 {rs['spell_f1']:.3f} {[round(x,3) for x in rps]} macro {rs['macro_f1']:.3f} team {rs['team_acc']:.3f} trans {rs['trans_f1_25']:.3f}")
    variants = {"nearest+soft k=2": {"soft": 2.0}, "nearest+soft k=4": {"soft": 4.0}, "nearest+soft k=8": {"soft": 8.0},
                "soft only k=4": {"soft": 4.0, "mode": "soft_only"}, "soft only k=8, gate x3": {"soft": 8.0, "mode": "soft_only", "soft_gate": 3.0},
                "dead: mean(real,hidden)": {"dead": "mean"}, "dead: hidden": {"dead": "hidden"},
                "merge runs < 0.5 s": {"merge": 0.5}, "merge runs < 1 s": {"merge": 1.0}, "merge runs < 2 s": {"merge": 2.0}}
    for name, v in variants.items():
        rows = run(v); s = H.summarise(rows); ps = [r["spell_f1"] for r in rows]
        print(f"{name:28s} spellF1 {s['spell_f1']:.3f} ({s['spell_f1']-rs['spell_f1']:+.3f}, better {sum(p>q for p,q in zip(ps,rps))}/3) macro {s['macro_f1']:.3f} team {s['team_acc']:.3f} trans {s['trans_f1_25']:.3f} spells {[r['pred_spells'] for r in rows]}")
