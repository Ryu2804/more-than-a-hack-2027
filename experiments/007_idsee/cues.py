#!/usr/bin/env python3
"""007 cue scoring: dead-ball cues vs BallStatus on IDSSE (external matches, other games).

BallStatus == 0 is the DFL annotation "ball not in play". It is per frame and automatic in the
sense that it comes with the dataset, not from us, so it is a legitimate external label.

Each cue is computed only from inputs a BAR-ATH submission could also use (ball position, ball
speed, player positions, player speed, and the absence of the ball). No possession label, no
BallPossession, no hand work.

Output: precision / recall / F1 per cue per match, plus speed distributions that settle the unit of
the ball-speed field before any threshold is trusted.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

FPS = 25
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
MARGIN = 0.5
KMH_TO_MS = 1.0 / 3.6   # the file stores km/h; every threshold below is in m/s


def runs_of(flags):
    flags = np.asarray(flags, dtype=bool)
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def run_mask(flags, min_len):
    out = np.zeros(len(flags), dtype=bool)
    for start, stop in runs_of(flags):
        if stop - start >= min_len:
            out[start:stop] = True
    return out


def score(pred, dead, valid):
    pred = pred & valid
    tp = int((pred & dead).sum())
    fp = int((pred & ~dead & valid).sum())
    fn = int((~pred & dead & valid).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def nearest_player(npz):
    """Distance and speed of the nearest observed player, per frame."""
    px, py, ps, mask = npz["player_x"], npz["player_y"], npz["player_speed"], npz["player_mask"]
    bx, by = npz["ball_x"], npz["ball_y"]
    n = px.shape[1]
    nearest_distance = np.full(n, np.nan, dtype=np.float32)
    nearest_speed = np.full(n, np.nan, dtype=np.float32)
    for t in range(n):
        if not np.isfinite(bx[t]):
            continue
        observed = mask[:, t]
        if not observed.any():
            continue
        dx = px[observed, t] - bx[t]
        dy = py[observed, t] - by[t]
        distance = np.hypot(dx, dy)
        best = int(np.argmin(distance))
        nearest_distance[t] = distance[best]
        nearest_speed[t] = ps[observed, t][best]
    return nearest_distance, nearest_speed


def cues(npz):
    status = npz["ball_status"]
    valid = status >= 0
    dead = valid & (status == 0)

    bx, by, bs = npz["ball_x"], npz["ball_y"], npz["ball_speed"]
    present = np.isfinite(bx) & np.isfinite(by)
    missing = ~present

    nearest_distance, nearest_speed = nearest_player(npz)

    out = {}
    out["long_missing_125"] = run_mask(missing, 125)
    out["missing_any"] = missing

    outside = present & ((np.abs(bx) > PITCH_LENGTH / 2 + MARGIN) |
                         (np.abs(by) > PITCH_WIDTH / 2 + MARGIN))
    out["out_of_pitch_15"] = run_mask(outside, 15)

    speed_ms = bs * KMH_TO_MS
    nearest_ms = nearest_speed * KMH_TO_MS
    for limit in (0.5, 1.0, 2.0, 3.0):
        tag = str(limit).replace(".", "p")
        slow_ball = present & np.isfinite(speed_ms) & (speed_ms < limit)
        out["ball_slow_{}_25".format(tag)] = run_mask(slow_ball, 25)
    for limit in (0.5, 1.0, 2.0):
        tag = str(limit).replace(".", "p")
        slow_nearest = present & np.isfinite(nearest_ms) & (nearest_ms < limit)
        out["nearest_slow_{}_25".format(tag)] = run_mask(slow_nearest, 25)

    tight = present & ((np.abs(bx) > PITCH_LENGTH / 2 + 0.1) |
                       (np.abs(by) > PITCH_WIDTH / 2 + 0.1))
    out["out_of_pitch_01_15"] = run_mask(tight, 15)

    crowded = present & (nearest_distance <= 2.0) & np.isfinite(nearest_ms) & (nearest_ms < 1.0)
    out["nearest_slow_within_2m_25"] = run_mask(crowded, 25)

    base = out["long_missing_125"] | out["out_of_pitch_15"]
    out["missing_or_out"] = base
    out["out_or_slow2"] = out["out_of_pitch_15"] | out["nearest_slow_2p0_25"]
    out["out_or_ballslow2"] = out["out_of_pitch_15"] | out["ball_slow_2p0_25"]
    out["missing_or_out_or_slow"] = base | out["nearest_slow_2p0_25"]
    out["everything"] = (base | out["nearest_slow_2p0_25"] | out["ball_slow_2p0_25"])
    return out, dead, valid, {
        "ball_present_fraction": float(present.mean()),
        "dead_fraction": float(dead.sum() / max(1, valid.sum())),
        "ball_speed_median_present": float(np.nanmedian(bs[present])) if present.any() else None,
        "ball_speed_p95_present": float(np.nanpercentile(bs[present], 95)) if present.any() else None,
        "player_speed_median": float(np.nanmedian(npz["player_speed"][npz["player_mask"]])),
        "nearest_speed_median": float(np.nanmedian(nearest_speed)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", default=os.path.join(ROOT, "..", "data", "external", "idsee", "parsed"))
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.parsed_dir) if f.endswith(".npz"))
    if args.matches:
        files = [f for f in files if any(m in f for m in args.matches)]
    if not files:
        raise SystemExit("no parsed matches in {}".format(args.parsed_dir))

    result = {}
    for name in files:
        match = os.path.splitext(name)[0]
        npz = np.load(os.path.join(args.parsed_dir, name))
        masks, dead, valid, stats = cues(npz)
        rows = {key: score(mask, dead, valid) for key, mask in masks.items()}
        result[match] = {"stats": stats, "cues": rows}
        print(match, "dead", round(stats["dead_fraction"], 3),
              "ball present", round(stats["ball_present_fraction"], 3),
              "best", max(rows.items(), key=lambda kv: kv[1]["f1"])[0],
              round(max(row["f1"] for row in rows.values()), 3))

    with open(os.path.join(args.out_dir, "cues.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
