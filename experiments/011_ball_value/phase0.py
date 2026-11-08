#!/usr/bin/env python3
"""011 phase 0: is the BAR-ATH cue failure the cue's fault or the DP's?

The 008 transfer test put the frozen external cue into the missing-aware DP and got a dead share of
0.85-0.91, which is implausible. Two candidates: the cue itself already flags almost everything, or
the cue is moderate and the DP's hard dead override inflates it. This separates the two.

No ground truth is used. Numbers are structural.
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m2 = _load(os.path.join(ROOT, "experiments", "002_possession_dp", "run.py"), "m2")
FPS = m2.FPS
H, A, D = m2.H, m2.A, m2.D
PITCH_LENGTH, PITCH_WIDTH = m2.PITCH_LENGTH, m2.PITCH_WIDTH

OUT_MARGIN = 0.1
OUT_RUN = 15
SLOW_SPEED = 2.0
SLOW_RUN = 25


def load(data_dir, n_frames):
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(m2.GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(m2.GAME))
    team = m2.bl.team_of_each_track(atd_xml)

    obs = np.full(n_frames, -1, dtype=np.int8)
    present = np.zeros(n_frames, dtype=bool)
    x = np.full(n_frames, np.nan, dtype=np.float32)
    y = np.full(n_frames, np.nan, dtype=np.float32)
    nearest_speed = np.full(n_frames, np.nan, dtype=np.float32)

    with open(atd_txt) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            ball = parts[2]
            if not ball or ball[0] == "N":
                continue
            try:
                frame = int(parts[0]) - 1
                coords = ball.split(",")
                if frame < 0 or frame >= n_frames:
                    continue
                bx = float(coords[0]) * PITCH_LENGTH
                by = float(coords[1]) * PITCH_WIDTH
            except (ValueError, IndexError):
                continue
            x[frame] = bx
            y[frame] = by
            present[frame] = True
            best = None
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    first = group.find(",")
                    second = group.find(",", first + 1)
                    third = group.find(",", second + 1)
                    px = float(group[:first]) * PITCH_LENGTH
                    py = float(group[first + 1:second]) * PITCH_WIDTH
                    sp = float(group[second + 1:third if third > 0 else None])
                except (ValueError, IndexError):
                    continue
                distance = (px - bx) ** 2 + (py - by) ** 2
                if best is None or distance < best[0]:
                    best = (distance, sp, int(team[slot]))
            if best is not None:
                obs[frame] = best[2]
                nearest_speed[frame] = best[1]
    return obs, present, x, y, nearest_speed


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


def main():
    data_dir = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data")
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    n_frames = m2.source_frames("tactical", m2.GAME)
    graded = m2.graded_mask(n_frames)

    obs, present, x, y, nearest_speed = load(data_dir, n_frames)

    outside = present & ((np.abs(x - PITCH_LENGTH / 2) > PITCH_LENGTH / 2 + OUT_MARGIN) |
                         (np.abs(y - PITCH_WIDTH / 2) > PITCH_WIDTH / 2 + OUT_MARGIN))
    out_cue = run_mask(outside & graded, OUT_RUN)
    slow_cue = run_mask((nearest_speed < SLOW_SPEED) & present & graded, SLOW_RUN)
    cue = (out_cue | slow_cue) & graded

    cue_only = obs.copy()
    cue_only[cue] = D

    graded_n = int(graded.sum())
    cue_share = float(cue.sum() / graded_n)
    cue_d_share = float((cue_only == D)[graded].mean())
    run_lengths = [(b - a) / FPS for a, b in runs_of(cue_only == D)]
    dp_path = os.path.join(ROOT, "experiments", "008_barath_motion_cue", "out", "results.json")
    dp = None
    if os.path.isfile(dp_path):
        dp = json.load(open(dp_path))

    result = {
        "graded_frames": graded_n,
        "cue_frames": int(cue.sum()),
        "cue_share_of_graded": cue_share,
        "out_cue_frames": int(out_cue.sum()),
        "slow_cue_frames": int(slow_cue.sum()),
        "cue_only_dead_share": cue_d_share,
        "cue_only_dead_runs": len(run_lengths),
        "cue_only_dead_run_median_s": float(np.median(run_lengths)) if run_lengths else 0.0,
        "cue_only_dead_run_p90_s": float(np.percentile(run_lengths, 90)) if run_lengths else 0.0,
        "dp8_dead_share": None if dp is None else dp.get("arms", {}).get("motion_only", {}).get("structure", {}).get("dead_fraction_graded"),
        "nearest_speed_median": float(np.nanmedian(nearest_speed)),
        "nearest_speed_below_2_fraction_of_present": float(np.nanmean(nearest_speed[present] < SLOW_SPEED)),
    }
    with open(os.path.join(out_dir, "phase0.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    for key, value in result.items():
        print("{:44s} {}".format(key, value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
