#!/usr/bin/env python3
"""IDSSE possession harness: spellF1 of a team decoder with dead given.

Why oracle dead: the published baseline's D rule is "ball not visible", but in IDSSE the ball is
missing only while play continues (P(dead | missing) = 0), so that rule cannot transfer. Handing the
GT dead state isolates the component under study, the possession/team decoder. Dead detection is
measured separately (C03a, C03c). This is recorded in RUN_RULES.md.

Features here are tracking only: player positions/mask/team ids and the ball position. BallStatus and
BallPossession are labels and never enter the prediction.

Usage:
  python harness.py --matches J03WMX J03WOH --windows 0 25 50 125
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PARSED = os.path.join(ROOT, "..", "data", "external", "idsee", "parsed")
DEV = ["J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ"]
CONFIRM = ["J03WN1", "J03WR9"]
FPS = 25


def load(match, parsed_dir=PARSED):
    """Load one parsed match and drop referee tracks (not players, break the two-team mapping)."""
    npz = {k: v for k, v in np.load(os.path.join(parsed_dir, match + ".npz"),
                                    allow_pickle=False).items()}
    keep = np.array([str(t).lower() != "referee" for t in npz["team_ids"]], dtype=bool)
    for key in ("player_x", "player_y", "player_speed", "player_mask"):
        npz[key] = npz[key][keep]
    npz["person_ids"] = npz["person_ids"][keep]
    npz["team_ids"] = npz["team_ids"][keep]
    return npz


def gt_codes(npz):
    """X where ungraded, D where BallStatus 0, H/A where alive with possession 1/2."""
    status, possession = npz["ball_status"], npz["ball_possession"]
    codes = np.full(len(status), 3, dtype=np.int64)
    codes[status == 0] = 2
    alive = status == 1
    codes[alive & (possession == 1)] = 0
    codes[alive & (possession == 2)] = 1
    return codes


def sides(npz):
    teams = npz["team_ids"]
    uniq = np.array(sorted(set(teams.tolist())), dtype=teams.dtype)
    if len(uniq) != 2:
        return None
    return np.where(teams == uniq[0], 0, 1).astype(np.int8)


def nearest_team(npz):
    """Team side of the nearest observed player per frame; -1 when nobody can be assigned."""
    bx, by = npz["ball_x"], npz["ball_y"]
    px, py, played = npz["player_x"], npz["player_y"], npz["player_mask"]
    finite = np.isfinite(bx) & np.isfinite(by)
    distance = np.hypot(px - bx[None, :], py - by[None, :])
    distance = np.where(played & finite[None, :], distance, np.inf)
    index = np.argmin(distance, axis=0)
    answered = np.isfinite(distance[index, np.arange(distance.shape[1])])
    side = sides(npz)
    out = np.full(len(bx), -1, dtype=np.int8)
    if side is not None:
        out[answered] = side[index[answered]]
    return out, answered


def smooth_team(team, window):
    """Centred majority over +/- window frames; -1 when the window has no observation."""
    if window <= 0:
        return team.copy()
    n = len(team)
    home = (team == 0).astype(np.int64)
    away = (team == 1).astype(np.int64)
    seen = (team >= 0).astype(np.int64)
    edges = np.arange(n)
    lo = np.maximum(0, edges - window)
    hi = np.minimum(n, edges + window + 1)
    cumulative = lambda a: np.concatenate(([0], np.cumsum(a)))  # noqa: E731
    ch, ca, cs = cumulative(home), cumulative(away), cumulative(seen)
    h, a, s = ch[hi] - ch[lo], ca[hi] - ca[lo], cs[hi] - cs[lo]
    return np.where((h > a) & (s > 0), 0, np.where((a > h) & (s > 0), 1, -1)).astype(np.int8)


def predict_oracle_dead(gt, team):
    """Dead where GT is dead, else the team; unknown team becomes D (a wrong possession, not X)."""
    return np.where(gt == 2, 2, np.where(team == 0, 0, np.where(team == 1, 1, 2))).astype(np.int64)


def score(gt, pred):
    runs = pm.valid_runs(gt)
    spells = pm.spell_scores(gt, pred, runs)
    stats = pm.possession_stats(gt, runs)
    return {
        "spell_f1": spells["mean_f1"],
        "per_iou": {str(k): v for k, v in spells["per_iou"].items()},
        "n_gt": spells["n_gt"],
        "n_pred": spells["n_pred"],
        "gt_spells": stats["n_spells"],
        "gt_mean_duration_s": stats["mean_duration_s"],
    }


def team_agreement(gt, team):
    use = (gt <= 1) & (team >= 0)
    return float((team[use] == gt[use]).mean()) if use.any() else 0.0


def evaluate_match(match, windows, parsed_dir=PARSED, flip_fixed=None):
    npz = load(match, parsed_dir)
    gt = gt_codes(npz)
    team, answered = nearest_team(npz)
    agreement = team_agreement(gt, team)
    flipped = (flip_fixed if flip_fixed is not None else agreement < 0.5)
    if flipped:
        team = np.where(team >= 0, 1 - team, -1).astype(np.int8)
    out = {
        "match": match,
        "frames": int(len(gt)),
        "graded": int((gt != 3).sum()),
        "alive": int((gt <= 1).sum()),
        "dead": int((gt == 2).sum()),
        "spells": int(len(pm.possession_spells(gt, pm.valid_runs(gt)))),
        "nearest_agreement_raw": agreement,
        "flipped": bool(flipped),
        "answered_fraction": float(answered.mean()),
        "windows": {},
    }
    for window in windows:
        pred = predict_oracle_dead(gt, smooth_team(team, window))
        out["windows"][str(window)] = score(gt, pred)
    return out


def paired_ci(values, reps=2000, seed=20261107):
    """CI90 of the mean of per-match paired differences (matches are the unit here)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [5, 95])
    return float(values.mean()), [float(low), float(high)]


def family(matches, windows, parsed_dir=PARSED):
    per_match = [evaluate_match(m, windows, parsed_dir) for m in matches]
    table = {}
    for window in windows:
        scores = [row["windows"][str(window)]["spell_f1"] for row in per_match]
        table[str(window)] = {
            "mean_spell_f1": float(np.mean(scores)),
            "per_match": dict(zip(matches, [float(s) for s in scores])),
        }
    for a, b in zip(windows, windows[1:]):
        deltas = [row["windows"][str(b)]["spell_f1"] - row["windows"][str(a)]["spell_f1"]
                  for row in per_match]
        mean, ci = paired_ci(deltas)
        table[str(b)]["delta_vs_{}".format(a)] = {"mean": mean, "ci90": ci}
    return {"matches": per_match, "table": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", nargs="*", default=DEV)
    parser.add_argument("--windows", type=int, nargs="*", default=[0, 25, 50, 125])
    parser.add_argument("--parsed-dir", default=PARSED)
    parser.add_argument("--out", default=os.path.join(HERE, "out", "harness.json"))
    args = parser.parse_args()
    result = family(args.matches, args.windows, args.parsed_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    for window, row in result["table"].items():
        print("w", window, "mean", round(row["mean_spell_f1"], 4))
    for window in args.windows[1:]:
        row = result["table"][str(window)]
        key = "delta_vs_{}".format(args.windows[args.windows.index(window) - 1])
        print("w", window, "delta", round(row[key]["mean"], 4),
              [round(v, 4) for v in row[key]["ci90"]])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
