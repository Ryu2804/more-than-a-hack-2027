#!/usr/bin/env python3
"""009: how good is nearest-player ownership, and when is it trustworthy?

IDSSE publishes per-frame BallPossession (1/2) on frames where the ball is active. That is real
truth about who has the ball on a different match, which BAR-ATH does not give us. The question is
not "can we predict possession" (nearest player is the trivial predictor) but "in which regimes is
the trivial predictor right", because BAR-ATH has to decide when to trust the nearest player and
when to stay silent.

Outputs accuracy and coverage per bin of nearest-player distance and of ball speed, plus the same
for a short majority vote. No thresholds are chosen here; the table is the decision material.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

KMH_TO_MS = 1.0 / 3.6
DIST_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 1e9)
SPEED_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)


def ownership_fields(npz):
    frames = npz["frames"]
    ball_status = npz["ball_status"]
    possession = npz["ball_possession"]
    px, py, ps = npz["player_x"], npz["player_y"], npz["player_speed"]
    mask = npz["player_mask"]
    team_ids = npz["team_ids"]
    sides = np.zeros(len(team_ids), dtype=np.int8)
    order = {tid: index for index, tid in enumerate(sorted(set(team_ids.tolist())))}
    for index, tid in enumerate(team_ids):
        sides[index] = 0 if order[tid] == 0 else 1

    n = len(frames)
    bx, by = npz["ball_x"], npz["ball_y"]
    speed_ms = npz["ball_speed"] * KMH_TO_MS
    nearest_team = np.full(n, -1, dtype=np.int8)
    nearest_dist = np.full(n, np.nan, dtype=np.float32)

    for t in range(n):
        observed = mask[:, t]
        if not observed.any():
            continue
        dx = px[observed, t] - bx[t]
        dy = py[observed, t] - by[t]
        distance = np.hypot(dx, dy)
        best = int(np.argmin(distance))
        nearest_dist[t] = float(distance[best])
        nearest_team[t] = sides[observed][best]

    truth = np.full(n, -1, dtype=np.int8)
    truth[(ball_status == 1) & (possession == 1)] = 0
    truth[(ball_status == 1) & (possession == 2)] = 1
    valid = truth >= 0
    return nearest_team, nearest_dist, speed_ms, truth, valid


def binned(values, edges):
    index = np.digitize(values, edges) - 1
    return np.clip(index, 0, len(edges) - 2)


def table(nearest_team, nearest_dist, speed_ms, truth, valid, edges, values, label):
    index = binned(values, edges)
    rows = []
    for b in range(len(edges) - 1):
        sel = valid & (index == b)
        n = int(sel.sum())
        rows.append({
            "bin": "[{}, {})".format(edges[b], edges[b + 1] if edges[b + 1] < 1e8 else "inf"),
            "frames": n,
            "share": float(n / max(1, valid.sum())),
            "accuracy": float((nearest_team[sel] == truth[sel]).mean()) if n else None,
        })
    return {label: rows}


def majority(codes, window):
    out = codes.copy()
    valid = codes >= 0
    for shift in range(1, window + 1):
        pass
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", default=os.path.join(ROOT, "..", "data", "external", "idsee", "parsed"))
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = args.matches or sorted(
        f[:-4] for f in os.listdir(args.parsed_dir) if f.endswith(".npz"))
    result = {}
    for name in names:
        path = os.path.join(args.parsed_dir, name + ".npz")
        if not os.path.isfile(path):
            continue
        with np.load(path) as npz:
            nearest_team, nearest_dist, speed_ms, truth, valid = ownership_fields(npz)
        rows = {"overall_accuracy": float((nearest_team[valid] == truth[valid]).mean()),
                "frames": int(valid.sum())}
        rows.update(table(nearest_team, nearest_dist, speed_ms, truth, valid,
                          DIST_EDGES, nearest_dist, "by_distance"))
        rows.update(table(nearest_team, nearest_dist, speed_ms, truth, valid,
                          SPEED_EDGES, speed_ms, "by_ball_speed"))

        trust_dist = 3.0
        trust_speed = 4.0
        trusted = valid & (nearest_dist <= trust_dist) & (speed_ms <= trust_speed)
        rows["trusted_gate"] = {
            "rule": "distance <= {} m and ball speed <= {} m/s".format(trust_dist, trust_speed),
            "coverage": float(trusted.sum() / max(1, valid.sum())),
            "accuracy": float((nearest_team[trusted] == truth[trusted]).mean()) if trusted.any() else None,
        }
        result[name] = rows
        print(name, "overall", round(rows["overall_accuracy"], 3),
              "trusted", round(rows["trusted_gate"]["accuracy"], 3),
              "coverage", round(rows["trusted_gate"]["coverage"], 3))

    with open(os.path.join(args.out_dir, "ownership.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
