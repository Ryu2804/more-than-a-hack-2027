"""One command regenerates the submission from the provided data.

    python experiments/015_final_pipeline/run.py [--out experiments/015_final_pipeline/out/submission.csv]

Inputs: data/tactical/BAR-ATH_tactical_atd.{xml,txt} (Smart Tagging and video are NOT used).
Model: weights/ holds a 3-seed ensemble of a two-head TCN (dead ball, team) trained ONLY on other matches (public Metrica sample games and
SkillCorner open data) by train_model.py / experiments/017_multi_match on a marimo/molab sandbox. `--retrain` retrains locally on the Metrica games.

Decoding (decoder.py, the same code validated on held-out matches in experiments/016 and 017):
  1. p(dead) (ball channels as tracked), prior-shifted, > 0.5 for >= 2 s -> D;
  2. ball evidence (nearest-player team, only within GATE_R metres) + the TCN team probability (kappa) -> two-state Viterbi;
     switch cost lambda calibrated so the number of possessions equals the training matches' possession rate; H/A runs shorter than
     MERGE_S seconds are merged into the previous run;
  3. X outside the two graded halves, D before the second-half kick-off.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "starter", ROOT / "evaluation", ROOT / "experiments", HERE):
    sys.path.insert(0, str(p))

def _weights_dir():
    a = sys.argv
    return Path(a[a.index("--weights") + 1]) if "--weights" in a else HERE / "weights_nometrica"


_meta_path = _weights_dir() / "meta.npz"
if "--retrain" not in sys.argv and _meta_path.exists() and "use_pp" in np.load(_meta_path).files and bool(np.load(_meta_path)["use_pp"]):
    os.environ["USE_PP"] = "1"          # the weights were trained with the per-player set encoder: features.py must build it too

import common  # noqa: E402
import decoder as dc  # noqa: E402
import features as ft  # noqa: E402
import possession_metrics as pm  # noqa: E402
import train_model as tm  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, anchor_video_frame, half_video_blocks, source_frames  # noqa: E402

FPS, STRIDE = 25, 5
GATE_R, DEAD_THR, MIN_DEAD_S, KAPPA, MERGE_S, RATE_SCALE = 0.35, 0.5, 2.0, 0.05, 1.0, 1.0   # chosen on held-out matches (experiments/016, 017); KAPPA 0.05 and MERGE_S 1.0 hold with true spells <2 s or <4 s excluded; RATE_SCALE 1.0 = the unscaled training-match rate (1.2 helped only when short spells count)
TARGET_DEAD_SHARE = 0.407       # La Liga effective playing time 59.3% (Football Observatory / CIES): ball not in play ~40.7% of match time
WEIGHTS = HERE / "weights_nometrica"      # 3-seed ensemble, per-player set encoder, trained on 27 other matches: SkillCorner open data (MIT) + IDSSE/DFL (CC BY 4.0); no Metrica (experiments/017, 020)


def merge_short(lab, blocks, min_s):
    """Merge H/A runs shorter than min_s seconds into the previous run (removes sliver flips)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "submission.csv"))
    ap.add_argument("--retrain", action="store_true", help="retrain locally on the Metrica games instead of loading weights/")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--no-shift", action="store_true", help="disable the dead-ball prior shift")
    ap.add_argument("--weights", default=None, help="folder with tcn_seed*.pt, meta.npz, training_report.json (default: ./weights)")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.retrain:
        games = tm.build_games()
        mu, sd = tm.standardiser(games)
        rate = float(np.mean([G["poss_per_min"] for G in games.values()]))
        train_dead = float(np.mean([G["dead_share"] for G in games.values()]))
        inputs = tm.make_inputs(mu, sd)
        nets = [tm.train(games, inputs, s, dev) for s in tm.SEEDS]
    else:
        WD = Path(args.weights) if args.weights else WEIGHTS
        meta = np.load(WD / "meta.npz")
        mu, sd, rate = meta["mu"], meta["sd"], float(meta["rate_mean"])
        rep = json.loads((WD / "training_report.json").read_text())
        if "poss_per_min" in rep:               # multi-source report: balance the sources
            def balanced(d):
                by = {}
                for name, v in d.items():
                    by.setdefault(name.split("_")[0], []).append(v)
                return float(np.mean([np.mean(v) for v in by.values()]))
            rate, train_dead = balanced(rep["poss_per_min"]), balanced(rep["dead_share"])
        else:
            train_dead = float(meta["dead_share_mean"]) if "dead_share_mean" in meta.files else float(np.mean([g["dead_share"] for g in rep["games"].values()]))
        dil = tuple(int(x) for x in meta["dil"]) if "dil" in meta.files else tm.DIL
        inputs = tm.make_inputs(mu, sd)
        nets = []
        for s in range(sum(1 for _ in WD.glob("tcn_seed*.pt"))):
            net = tm.TCN2(tm.NCH, dil=dil, ch=int(meta["ch"]) if "ch" in meta.files else None).to(dev)
            net.load_state_dict(torch.load(WD / f"tcn_seed{s}.pt", map_location=dev)); nets.append(net.eval())
    logit = lambda p: np.log(p / (1 - p))
    shift = 0.0 if args.no_shift else float(logit(TARGET_DEAD_SHARE) - logit(train_dead))

    n_frames = source_frames("tactical")
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n_frames, bool); half = np.zeros(n_frames, int)
    for h, a, b in blocks:
        graded[a:b + 1] = True; half[a:b + 1] = h
    graded_min = graded.sum() / FPS / 60
    kickoff2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"]) + 1
    assert kickoff2 == 113742
    second_first = [a for h, a, b in blocks if h == 2][0]

    P, B, T = common.load_atd_arrays()
    F = ft.build(P[..., :2], B[:, :2], T, half)
    p_dead = np.full(n_frames, np.nan); p_team = np.full(n_frames, 0.5)
    for h, a, b in blocks:
        ix = np.arange(a, b + 1, STRIDE); full = np.arange(a, b + 1)
        pl, bl_, vd = F["player"][:, ix], F["ball"][:, ix], F["pvalid"][:, ix]
        if ft.USE_PP:
            pp, ppm = ft.player_set(P[..., :2], B[:, :2], T, half, ix)
            out = [tm.predict(n, inputs, pl, bl_, vd, False, dev, pp, ppm) for n in nets]
        else:
            out = [tm.predict(n, inputs, pl, bl_, vd, False, dev) for n in nets]      # ball channels as tracked (validated: better than hiding them)
        p_dead[a:b + 1] = np.interp(full, ix, np.mean([o[0] for o in out], 0))
        p_team[a:b + 1] = np.interp(full, ix, np.mean([o[1] for o in out], 0))

    dead = dc.dead_mask(p_dead, blocks, DEAD_THR, MIN_DEAD_S, shift)
    dist = dc.nearest_distances(P[..., :2], B[:, :2], T)
    lp_h, lp_a = dc.ball_evidence(dist, GATE_R)
    h_ev, a_ev = dc.evidence(lp_h, lp_a, p_team, KAPPA)
    make = lambda lam: dc.decode(n_frames, blocks, h_ev, a_ev, lam, dead, forced_dead=[(second_first, kickoff2)])
    target = rate * RATE_SCALE * graded_min
    lam_star = dc.calibrate_lambda(make, blocks, target)           # lambda is calibrated WITHOUT the merge (validated order) ...
    lab = merge_short(make(lam_star), blocks, MERGE_S)              # ... and slivers are merged afterwards

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_submission(args.out, lab)
    problems = pm.validate_submission(pm.read_label_csv(args.out), n_frames)
    print(f"wrote {args.out}")
    print(f"dead-ball prior shift {shift:+.3f} (training dead share {train_dead:.1%} -> target {TARGET_DEAD_SHARE:.1%})")
    print(f"possession rate {rate:.2f}/min -> target {target:.0f}; lambda* {lam_star:.2f} -> {dc.n_possessions(lab, blocks)} possessions | "
          f"D {np.mean(lab[graded] == 'D'):.1%} H {np.mean(lab[graded] == 'H'):.1%} A {np.mean(lab[graded] == 'A'):.1%} of graded frames")
    print("validation:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
