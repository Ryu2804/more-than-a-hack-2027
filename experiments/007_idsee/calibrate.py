#!/usr/bin/env python3
"""007 calibration: do IDSSE labels reproduce the published baseline ordering?

IDSSE gives per-frame BallStatus (0 = ball not in play) and BallPossession (1 = one team, 2 = the
other). With those as labels, the published baselines (raw nearest player, majority vote over 25 /
50 / 125 frames) should rank in a known order - raw worst, smooth50 best of the three, smooth125
between. If that order does not come out, the labels are not usable as a proxy and every later
IDSSE number is inconclusive.

This script only checks the convention and the ordering. It selects nothing and writes nothing back
to the dataset.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import baseline as bl  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D = 0, 1, 2


def nearest_team(npz):
    """Team of the nearest observed player for each frame; -1 where no ball or no player."""
    ball_x, ball_y = npz["ball_x"], npz["ball_y"]
    player_x, player_y = npz["player_x"], npz["player_y"]
    mask = npz["player_mask"]
    team_ids, person_ids = npz["team_ids"], npz["person_ids"]

    teams = np.unique(team_ids)
    side = np.zeros(len(person_ids), dtype=np.int8)
    side[team_ids == teams[1]] = 1

    dx = player_x - ball_x[None, :]
    dy = player_y - ball_y[None, :]
    distance = np.hypot(dx, dy)
    distance = np.where(mask, distance, np.inf)
    nearest = np.argmin(distance, axis=0)
    best = distance[nearest, np.arange(distance.shape[1])]
    out = np.full(distance.shape[1], -1, dtype=np.int8)
    usable = np.isfinite(best) & np.isfinite(ball_x) & np.isfinite(ball_y)
    out[usable] = side[nearest[usable]]
    return out, {
        "teams": teams.tolist(),
        "players": len(person_ids),
        "frames_with_ball_and_player": int(usable.sum()),
    }


def build_labels(npz):
    status = npz["ball_status"]
    possession = npz["ball_possession"]
    labels = np.full(len(status), D, dtype=np.int8)
    alive = status == 1
    labels[alive & (possession == 1)] = H
    labels[alive & (possession == 2)] = A
    return labels


def majority(codes, window):
    if window <= 0:
        return np.where(codes == H, "H", np.where(codes == A, "A", "D"))
    home = np.concatenate(([0], np.cumsum(codes == H)))
    away = np.concatenate(([0], np.cumsum(codes == A)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home_count = home[high] - home[low]
    away_count = away[high] - away[low]
    out = np.full(len(codes), "D", dtype="<U1")
    out[home_count >= away_count] = "H"
    out[away_count > home_count] = "A"
    alive = (home_count + away_count) == 0
    out[alive] = "D"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", default=os.path.join(ROOT, "..", "data", "external", "idsee", "parsed"))
    parser.add_argument("--match", default="J03WOH")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = os.path.join(args.parsed, "{}.npz".format(args.match))
    npz = np.load(path, allow_pickle=True)
    gt = build_labels(npz)
    obs, info = nearest_team(npz)

    agreement = float((obs[obs >= 0] == gt[obs >= 0]).mean())
    swapped = agreement < 0.5
    if swapped:
        obs = np.where(obs < 0, -1, 1 - obs).astype(np.int8)
        agreement = 1.0 - agreement
        gt = np.where(gt == D, D, 1 - gt).astype(np.int8)

    graded = gt != D
    result = {
        "match": args.match,
        "frames": int(len(gt)),
        "graded_frames": int(graded.sum()),
        "dead_fraction": float((gt == D).mean()),
        "possession_labels": int((gt != D).sum()),
        "nearest_team_agreement": agreement,
        "team_mapping_swapped": bool(swapped),
        "players": info["players"],
        "frames_with_ball_and_player": info["frames_with_ball_and_player"],
        "baselines": {},
    }

    names = (("raw", 0), ("smooth25", 25), ("smooth50", 50), ("smooth125", 125))
    for name, window in names:
        codes = pm.label_codes(majority(obs, window))
        metrics = pm.evaluate(gt, codes, bootstrap_n=0)
        result["baselines"][name] = {
            "spell_f1_mean": metrics["spell_f1_mean"],
            "macro_f1": metrics["macro_f1"],
            "n_pred_spells": metrics["n_pred_spells"],
            "n_gt_spells": metrics["n_gt_spells"],
            "mean_spell_duration_s": metrics["mean_spell_duration_s"],
            "gt_mean_spell_duration_s": metrics["gt_mean_spell_duration_s"],
            "fragmentation": metrics["fragmentation"],
            "alive_dead_f1": metrics["alive_dead_f1"],
        }
        print("{}: spellF1 {:.4f} macroF1 {:.4f} pred {} gt {}".format(
            name, metrics["spell_f1_mean"], metrics["macro_f1"],
            metrics["n_pred_spells"], metrics["n_gt_spells"]), flush=True)

    values = [result["baselines"][name]["spell_f1_mean"] for name, _ in names]
    order = np.argsort(values)
    names_only = [names[i][0] for i in order]
    expected = ["raw", "smooth25", "smooth125", "smooth50"]
    result["ordering"] = names_only
    result["ordering_matches_published"] = names_only == expected
    result["note"] = "ordering check only; no parameter was chosen from it"
    print("ordering", names_only, "expected", expected)

    out = args.out or os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calibration_{}.json".format(args.match)), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
