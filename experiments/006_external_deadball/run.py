#!/usr/bin/env python3
"""006: external dead-ball benchmark on Metrica sample games (other match, allowed data).

The BAR-ATH run has no dead-ball ground truth except the 5-minute public window, and the earlier
review showed dead-vs-alive confusion dominates the frame errors. Metrica publishes tracking plus an
event stream for two different matches, so the dead-ball evidence rules can be scored honestly on a
game where we did not design them.

Dead interval here = union of event rows that take the ball out of play (BALL OUT, SHOT OFF
TARGET-OUT). The tracking file has no in-play flag; the only ball signal is NaN coordinates.

Rules scored at frame level: ball-NaN runs of a given minimum length, with and without a restart
event nearby. Output is precision / recall / F1 plus the run-length structure, per game.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

GAMES = ("Sample_Game_1", "Sample_Game_2")
DEAD_TYPES = ("BALL OUT", "SHOT OFF TARGET-OUT")
RESTART_TYPES = ("SET PIECE", "CORNER KICK", "THROW IN", "GOAL KICK", "FREE KICK", "KICK OFF", "GOAL")
FPS = 25


def runs_of(flags):
    flags = np.asarray(flags, dtype=bool)
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def load_ball(path):
    table = pd.read_csv(path, skiprows=3, header=None)
    if table.shape[1] < 31:
        raise SystemExit("unexpected column count {} in {}".format(table.shape[1], path))
    frames = table.iloc[:, 1].to_numpy(dtype=np.int64)
    x = pd.to_numeric(table.iloc[:, -2], errors="coerce").to_numpy()
    y = pd.to_numeric(table.iloc[:, -1], errors="coerce").to_numpy()
    present = np.isfinite(x) & np.isfinite(y)
    n = int(frames.max()) + 1
    out = np.zeros(n, dtype=bool)
    out[frames[present]] = True
    return out


def load_events(path):
    table = pd.read_csv(path)
    table["Type"] = table["Type"].astype(str).str.strip().str.upper()
    table["Subtype"] = table["Subtype"].astype(str).str.strip().str.upper()
    dead = np.zeros(0, dtype=bool)
    returns = {}

    def intervals(mask, frame_max):
        spans = []
        for _, row in table[mask].iterrows():
            start = int(row["Start Frame"])
            end = int(row["End Frame"]) if pd.notna(row["End Frame"]) else start
            if end < start:
                start, end = end, start
            spans.append((max(0, start - 1), min(frame_max, end)))
        return spans

    return table, intervals


def score(pred, dead):
    tp = int((pred & dead).sum())
    fp = int((pred & ~dead).sum())
    fn = int((~pred & dead).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def analyse_game(folder):
    name = os.path.basename(folder)
    tracking = os.path.join(folder, "{}_RawTrackingData_Home_Team.csv".format(name))
    events_path = os.path.join(folder, "{}_RawEventsData.csv".format(name))
    present = load_ball(tracking)
    n = len(present)
    missing = ~present

    table = pd.read_csv(events_path)
    types = table["Type"].astype(str).str.strip().str.upper()
    starts = np.sort(table["Start Frame"].astype(int).to_numpy())

    # A death event only marks the instant the ball leaves play. The stoppage runs until the next
    # event of any type (the restart). Capped at 30 s so a missing restart does not swallow play.
    dead_mask = np.zeros(n, dtype=bool)
    for _, row in table[types.isin(DEAD_TYPES)].iterrows():
        start = int(row["Start Frame"])
        end = int(row["End Frame"]) if pd.notna(row["End Frame"]) else start
        later = starts[starts > max(start, end)]
        stop = int(later[0]) if later.size and later[0] - end < 750 else end + 250
        dead_mask[max(0, min(start, end) - 1):min(n, stop)] = True

    restart_frames = set()
    for _, row in table[types.isin(RESTART_TYPES)].iterrows():
        restart_frames.add(int(row["Start Frame"]))

    runs = runs_of(missing)
    lengths = np.array([b - a for a, b in runs], dtype=np.int64)
    rules = {}
    for min_len in (25, 75, 125, 250, 375):
        pred = np.zeros(n, dtype=bool)
        for start, stop in runs:
            if stop - start >= min_len:
                pred[start:stop] = True
        rules["nan_run_{}".format(min_len)] = score(pred, dead_mask)

    pred = np.zeros(n, dtype=bool)
    for start, stop in runs:
        if stop - start < 125:
            continue
        near = any(start - 75 <= frame <= stop + 75 for frame in restart_frames)
        if near:
            pred[start:stop] = True
    rules["nan_run_125_near_restart"] = score(pred, dead_mask)

    return {
        "frames": int(n),
        "ball_missing_fraction": float(missing.mean()),
        "dead_fraction": float(dead_mask.mean()),
        "missing_given_dead": float(missing[dead_mask].mean()) if dead_mask.any() else 0.0,
        "dead_given_missing": float(dead_mask[missing].mean()) if missing.any() else 0.0,
        "missing_runs": int(len(lengths)),
        "missing_runs_ge_125": int((lengths >= 125).sum()),
        "missing_run_median_s": float(np.median(lengths) / FPS) if len(lengths) else 0.0,
        "missing_run_max_s": float(lengths.max() / FPS) if len(lengths) else 0.0,
        "rules": rules,
    }


def _spans(subset, n):
    spans = []
    for _, row in subset.iterrows():
        start = int(row["Start Frame"])
        end = int(row["End Frame"]) if pd.notna(row["End Frame"]) else start
        if end < start:
            start, end = end, start
        spans.append((max(0, start - 1), min(n, max(start + 1, end))))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "..", "data", "external", "metrica"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    result = {}
    for game in GAMES:
        folder = os.path.join(args.data_dir, game)
        if not os.path.isdir(folder):
            result[game] = {"available": False}
            continue
        result[game] = analyse_game(folder)
        row = result[game]
        best = row["rules"]["nan_run_125"]
        print(game, "frames", row["frames"],
              "dead", round(row["dead_fraction"], 3),
              "missing", round(row["ball_missing_fraction"], 3),
              "P(miss|dead)", round(row["missing_given_dead"], 3),
              "P(dead|miss)", round(row["dead_given_missing"], 3),
              "run125 P/R/F1", round(best["precision"], 3), round(best["recall"], 3), round(best["f1"], 3))

    with open(os.path.join(args.out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
