"""Offline validation of the WHOLE pipeline on other matches (Metrica sample games 1-3, public).

For each held-out game: predictions come from a model trained only on the other two games (experiments/015 train_model.py --folds);
ball evidence + decoding (experiments/015 decoder.py, the same code the submission uses); lambda calibrated to the OTHER games'
possession rate; scored with the challenge's own metric code (evaluation/possession_metrics.py) against that game's
event-derived labels (experiments/other_games.py). Labels are our derivation, not the challenge's ground truth.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "evaluation", ROOT / "starter", ROOT / "experiments", ROOT / "experiments" / "015_final_pipeline"):
    sys.path.insert(0, str(p))

import common  # noqa: E402
import decoder as dc  # noqa: E402
import other_games as og  # noqa: E402
import possession_metrics as pm  # noqa: E402
from baseline import smooth  # noqa: E402

FPS = 25
FOLDS = Path(common.CACHE) / "folds"
NAMES = {-1: "X", 0: "H", 1: "A", 2: "D"}
_CACHE = {}


def interp_full(n, ix, vals, blocks):
    out = np.full(n, np.nan)
    for _, a, b in blocks:
        m = (ix >= a) & (ix <= b)
        out[a:b + 1] = np.interp(np.arange(a, b + 1), ix[m], vals[m])
    return out


def game(g, fold_dir=None):
    key = (g, str(fold_dir))
    if key in _CACHE:
        return _CACHE[key]
    d = og.load(g)
    P, B, T, half, y = d["P"], d["B"], d["T"], d["half"], d["y"]
    n = len(B)
    blocks = [(h, int(np.flatnonzero(half == h).min()), int(np.flatnonzero(half == h).max())) for h in (1, 2)]
    z = np.load((fold_dir or FOLDS) / f"fold_game{g}.npz")
    out = dict(g=g, n=n, blocks=blocks, y=y, half=half, dist=dc.nearest_distances(P, B, T),
               train_rate=float(z["train_rate"]), train_dead=float(z["train_dead_share"]),
               pdead_real=interp_full(n, z["ix"], z["pdead_real"], blocks), pdead_hidden=interp_full(n, z["ix"], z["pdead_hidden"], blocks),
               pteam_hidden=interp_full(n, z["ix"], z["pteam_hidden"], blocks), pteam_real=interp_full(n, z["ix"], z["pteam_real"], blocks))
    out["gt"] = pm.label_codes(np.array([NAMES[int(v)] for v in y]))
    out["graded_min"] = sum(b - a + 1 for _, a, b in blocks) / FPS / 60
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    out["graded"] = graded
    _CACHE[key] = out
    return out


def score(G, labels):
    lab = np.where(G["graded"], labels, "X")
    r = pm.evaluate(G["gt"], pm.label_codes(lab), bootstrap_n=0)
    t = r["transitions"]
    t25 = t.get(25, t.get("25", {})) if isinstance(t, dict) else {}
    return dict(macro_f1=r["macro_f1"], spell_f1=r["spell_f1_mean"], gt_spells=r["n_gt_spells"], pred_spells=r["n_pred_spells"],
                alive_dead_f1=r["alive_dead_f1"], team_acc=r["team_acc_both_alive"], dur=r["mean_spell_duration_s"], gt_dur=r["gt_mean_spell_duration_s"],
                trans_f1_25=t25.get("f1", float("nan")) if isinstance(t25, dict) else float("nan"))


def baseline_labels(G, window=50):
    dH, dA = G["dist"][:, 0], G["dist"][:, 1]
    ok = np.isfinite(dH) & np.isfinite(dA)
    codes = np.full(G["n"], -1, np.int8)
    codes[ok & (dH < dA)] = 0; codes[ok & (dH >= dA)] = 1
    return smooth(codes, window)


DEFAULT = dict(gate_r=1.0, kappa=0.02, lam="cal", dead_thr=0.5, min_dead_s=2.0, shift=0.0, tau=3.0, team="hidden", dead_src="real", rate_scale=1.0)


def pipeline_labels(G, cfg):
    cfg = {**DEFAULT, **cfg}
    lp_h, lp_a = dc.ball_evidence(G["dist"], cfg["gate_r"], cfg["tau"])
    h_ev, a_ev = dc.evidence(lp_h, lp_a, G[f"pteam_{cfg['team']}"], cfg["kappa"])
    dead = dc.dead_mask(G[f"pdead_{cfg['dead_src']}"], G["blocks"], cfg["dead_thr"], cfg["min_dead_s"], cfg["shift"])
    make = lambda lam: dc.decode(G["n"], G["blocks"], h_ev, a_ev, lam, dead)
    lam = cfg["lam"]
    if lam == "cal":
        lam = dc.calibrate_lambda(make, G["blocks"], G["train_rate"] * cfg["rate_scale"] * G["graded_min"])
    return make(lam), lam, dead


def run(cfg, games=(1, 2, 3)):
    rows = []
    for g in games:
        G = game(g)
        lab, lam, dead = pipeline_labels(G, cfg)
        r = score(G, lab); r.update(game=g, lam=lam, dead_share=float(dead[G["graded"]].mean())); rows.append(r)
    return rows


def summarise(rows):
    keys = ("macro_f1", "spell_f1", "alive_dead_f1", "team_acc", "trans_f1_25")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


if __name__ == "__main__":
    import json
    print("game | baseline_smooth50: macroF1 spellF1 poss(gt) | pipeline: macroF1 spellF1 poss(gt) lam D-share")
    base, pipe = [], []
    for g in (1, 2, 3):
        G = game(g)
        b = score(G, baseline_labels(G)); base.append(b)
        p = run({}, (g,))[0]; pipe.append(p)
        print(f"{g}    | {b['macro_f1']:.3f} {b['spell_f1']:.3f} {b['pred_spells']}({b['gt_spells']})   | {p['macro_f1']:.3f} {p['spell_f1']:.3f} {p['pred_spells']}({p['gt_spells']}) {p['lam']:.1f} {p['dead_share']:.1%}")
    print("mean baseline", {k: round(v, 3) for k, v in summarise(base).items()})
    print("mean pipeline", {k: round(v, 3) for k, v in summarise(pipe).items()})
