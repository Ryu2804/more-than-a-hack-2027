#!/usr/bin/env python3
"""011 phase 1: how good could ownership be if the ball position were perfect?

The video lane only pays off if knowing the ball position well is enough to name the owning team.
IDSSE gives an oracle for that: its ball is tracked on every graded frame, and BallPossession names
the team in control (1 = home, 2 = away) whenever BallStatus == 1.

So this is an upper bound on any ball-detection work: with the true ball, how accurate can a simple
owner rule be, and how much of the match can it answer at all? Rules are scored on team accuracy
over answered alive frames, plus coverage. Nothing is tuned on BAR-ATH and no candidate is chosen.

Caveat recorded in the report: the dataset paper does not say how 'control' is computed, so a
nearest-player rule may be partly circular with the label. That is why the interesting part is the
comparison BETWEEN rules and the coverage/accuracy trade-off, not the absolute ceiling.
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

KMH_TO_MS = 1.0 / 3.6
DEV = ["J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ"]
CONFIRM = ["J03WN1", "J03WR9"]


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def sides(npz):
    """Map the two arbitrary team ids to H/A by whichever choice beats chance on the ball."""
    teams = npz["team_ids"]
    uniq = sorted(set(teams.tolist()))
    if len(uniq) != 2:
        return None
    mapped = np.where(teams == uniq[0], 0, 1)
    return mapped


def rules(npz):
    px, py = npz["player_x"], npz["player_y"]
    mask = npz["player_mask"]
    speed = npz["player_speed"] * KMH_TO_MS
    bx, by = npz["ball_x"], npz["ball_y"]
    ball_speed = npz["ball_speed"] * KMH_TO_MS
    status = npz["ball_status"]
    possession = npz["ball_possession"]

    team = sides(npz)
    if team is None:
        return None

    alive = status == 1
    truth = np.where(possession == 1, 0, np.where(possession == 2, 1, -1))

    dx = px - bx[None, :]
    dy = py - by[None, :]
    distance = np.hypot(dx, dy)
    distance = np.where(mask, distance, np.inf)

    order = np.argsort(distance, axis=0)
    nearest_slot = order[0]
    nearest_distance = distance[nearest_slot, np.arange(distance.shape[1])]
    nearest_team = team[nearest_slot]

    top3 = order[:3]
    top3_team = team[top3]
    votes = top3_team.sum(axis=0)
    three_vote = np.where(votes >= 2, 1, 0).astype(np.int8)

    out = {}

    def score(name, predicted, answered, carry=None):
        pred = predicted.copy()
        ok = answered.copy() & alive & (truth >= 0)
        if carry is not None:
            filled = pred.copy()
            last = -1
            gap = 0
            for i in range(len(pred)):
                if answered[i]:
                    last = pred[i]
                    gap = 0
                elif last >= 0 and gap < carry:
                    filled[i] = last
                    gap += 1
                else:
                    last = -1 if gap >= carry else last
                    gap += 1
            pred = filled
            ok = alive & (truth >= 0)
        correct = int((pred[ok] == truth[ok]).sum())
        total = int(ok.sum())
        return {
            "accuracy": correct / total if total else None,
            "answered_frames": total,
            "coverage": float(total / max(1, int((alive & (truth >= 0)).sum()))) if carry is None else 1.0,
        }

    out["nearest"] = score("nearest", nearest_team, np.ones(len(bx), dtype=bool))
    for metres in (1.0, 2.0, 3.0, 5.0):
        answered = nearest_distance <= metres
        out["gate_{:g}m".format(metres)] = score("gate", nearest_team, answered)
    out["three_nearest_vote"] = score("three", three_vote, np.ones(len(bx), dtype=bool))
    out["flight_neutral"] = score("flight", nearest_team, ball_speed <= 8.0)
    for frames in (25, 50, 125):
        out["carry_{}".format(frames)] = score(
            "carry", nearest_team, nearest_distance <= 2.0, carry=frames)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", default=os.path.join(ROOT, "..", "data", "external", "idsee", "parsed"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    result = {"dev": {}, "confirm": {}, "rules": []}
    for split, matches in (("dev", DEV), ("confirm", CONFIRM)):
        for match in matches:
            path = os.path.join(args.parsed_dir, match + ".npz")
            if not os.path.isfile(path):
                continue
            table = rules(load(path))
            if table is None:
                continue
            # Convention check only: flip when the nearest rule is worse than chance.
            if table["nearest"]["accuracy"] is not None and table["nearest"]["accuracy"] < 0.5:
                flipped = {}
                for key, row in table.items():
                    flipped[key] = dict(row)
                    if row["accuracy"] is not None:
                        flipped[key]["accuracy"] = 1.0 - row["accuracy"]
                table = flipped
                table["flipped"] = True
            result[split][match] = table
            print(split, match, "nearest", round(table["nearest"]["accuracy"], 3),
                  "gate3", round(table["gate_3m"]["accuracy"], 3),
                  "cov3", round(table["gate_3m"]["coverage"], 3),
                  "carry50", round(table["carry_50"]["accuracy"], 3))

    if result["dev"]:
        result["rules"] = sorted(result["dev"][next(iter(result["dev"]))].keys())
    with open(os.path.join(args.out_dir, "phase1.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
