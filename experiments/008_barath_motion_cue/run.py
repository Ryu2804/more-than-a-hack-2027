#!/usr/bin/env python3
"""008: the externally validated dead-ball cue, applied to BAR-ATH without ground truth.

Experiment 007 froze a cue on IDSSE dev matches (four matches, F1 0.80-0.87 against BallStatus):

    out_of_pitch:  ball more than 0.1 m outside the pitch for at least 15 frames
    near_slow:     nearest player slower than 2.0 m/s for at least 25 frames
    union of the two

IDSSE cannot validate BAR-ATH's dominant evidence (ball absence) because its ball is always
tracked, but it does validate geometry and motion. BAR-ATH has both: ATD positions and per-track
speed, homography-derived pitch geometry. Neither has been used as dead evidence before.

This script builds the cue from the tactical ATD, compares it with the existing `final` evidence,
decodes with the same missing-aware DP, and reports GT-free structure plus the public-window
numbers as a diagnostic only. Nothing is selected here.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dp2 = _load(os.path.join(ROOT, "experiments", "002_possession_dp", "run.py"), "dp2")
pm = dp2.pm

GAME = dp2.GAME
H, A, D = dp2.H, dp2.A, dp2.D
FPS = dp2.FPS
PITCH_LENGTH, PITCH_WIDTH = dp2.PITCH_LENGTH, dp2.PITCH_WIDTH

OUT_MARGIN = 0.1
OUT_RUN = 15
SLOW_SPEED = 2.0
SLOW_RUN = 25


def load_ball_and_nearest(data_dir, n_frames):
    """Ball position, nearest-player distance and nearest-player provided speed per frame."""
    import xml.etree.ElementTree as ET
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME))
    team = dp2.bl.team_of_each_track(atd_xml)

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    nearest_d = np.full(n_frames, np.nan, dtype=np.float32)
    nearest_s = np.full(n_frames, np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)

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
                bx = float(coords[0]) * PITCH_LENGTH
                by = float(coords[1]) * PITCH_WIDTH
            except (ValueError, IndexError):
                continue
            if frame < 0 or frame >= n_frames:
                continue
            ball_x[frame], ball_y[frame] = bx, by
            present[frame] = True
            best_d, best_s = None, np.nan
            for movement in parts[1].split(";"):
                if not movement or movement[0] == "N":
                    continue
                try:
                    first = movement.find(",")
                    second = movement.find(",", first + 1)
                    px = float(movement[:first]) * PITCH_LENGTH
                    py = float(movement[first + 1:second]) * PITCH_WIDTH
                except (ValueError, IndexError):
                    continue
                distance = float(np.hypot(px - bx, py - by))
                if best_d is None or distance < best_d:
                    best_d = distance
                    try:
                        best_s = float(movement[second + 1:].split(",")[0])
                    except (ValueError, IndexError):
                        best_s = np.nan
            nearest_d[frame] = best_d if best_d is not None else np.nan
            nearest_s[frame] = best_s
    return {"ball_x": ball_x, "ball_y": ball_y, "nearest_d": nearest_d,
            "nearest_s": nearest_s, "present": present}


def masks_from_atd(data, n_frames):
    graded = dp2.graded_mask(n_frames)
    out = np.zeros(n_frames, dtype=bool)
    ball_x, ball_y = data["ball_x"], data["ball_y"]
    outside = data["present"] & (
        (np.abs(ball_x) > PITCH_LENGTH / 2 + OUT_MARGIN) |
        (np.abs(ball_y) > PITCH_WIDTH / 2 + OUT_MARGIN))
    out |= dp2.run_length_mask(outside, OUT_RUN)

    slow = np.isfinite(data["nearest_s"]) & (data["nearest_s"] < SLOW_SPEED)
    out |= dp2.run_length_mask(slow, SLOW_RUN)
    return out & graded


def structure(codes, n_frames):
    runs = dp2.full_match_run_list(n_frames)
    spells = pm.possession_spells(codes, runs)
    durations = [(stop - start) / FPS for start, stop, _code in spells]
    graded = dp2.graded_mask(n_frames)
    return {
        "possessions": len(durations),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "dead_fraction": float((codes[graded] == D).mean()),
        "switches_per_min": float(max(0, len(spells) - 1) / max(1e-9, (codes[graded] != D).sum() / FPS / 60.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n_frames = dp2.source_frames("tactical", GAME)
    data = load_ball_and_nearest(args.data_dir, n_frames)
    motion = masks_from_atd(data, n_frames)

    restart = dp2.parse_set_piece_starts(args.data_dir, n_frames)
    evidence = dp2.build_evidence(
        n_frames, data["present"], data["ball_x"], data["ball_y"],
        np.full(n_frames, np.nan, dtype=np.float32), restart)
    final = evidence["final"]
    nearest_speed_present = np.isfinite(data["nearest_s"][data["present"]])

    new_cue = final | motion
    overlaps = {
        "motion_frames": int(motion.sum()),
        "final_frames": int(final.sum()),
        "union_frames": int(new_cue.sum()),
        "added_frames": int((new_cue & ~final).sum()),
        "removed_frames": 0,
        "motion_inside_final_fraction": float((motion & final).sum() / max(1, motion.sum())),
        "nearest_speed_availability": float(nearest_speed_present.mean()) if nearest_speed_present.size else 0.0,
    }

    obs = dp2.load_observations(args.data_dir, n_frames)[0]
    arms = {}

    def pick_lambda(mask):
        runs = dp2.full_match_run_list(n_frames)
        rows = []
        for lam in (4.0, 8.0, 16.0, 25.0, 32.0, 50.0, 64.0):
            trial = dp2.dp_full(obs, mask, lam, n_frames)
            rows.append((lam, len(pm.possession_spells(trial, runs)), trial))
        inside = [row for row in rows if dp2.COUNT_BAND[0] <= row[1] <= dp2.COUNT_BAND[1]]
        pool = inside or rows
        best = min(pool, key=lambda row: abs(row[1] - dp2.COUNT_CENTRE))
        return best[0], [{"lambda": row[0], "possessions": row[1]} for row in rows]

    def decode(mask, name):
        lam, grid = pick_lambda(mask)
        codes = dp2.dp_full(obs, mask, lam, n_frames)
        arms[name] = {"lambda": lam, "lambda_grid": grid,
                      "structure": structure(codes, n_frames), "codes": codes}
        return codes

    log = {}
    for name, mask in (("evidence_final", final), ("motion", motion), ("union", new_cue)):
        log[name] = int(mask.sum())
        decode(mask, name)

    gt_codes = pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))))
    for name, arm in arms.items():
        arm["public"] = dp2.metrics_for(gt_codes, arm.pop("codes"), args.bootstrap)

    result = {
        "n_frames": int(n_frames),
        "constants": {"out_margin_m": OUT_MARGIN, "out_run": OUT_RUN,
                      "slow_speed_ms": SLOW_SPEED, "slow_run": SLOW_RUN},
        "evidence_frames": log,
        "overlaps": overlaps,
        "arms": arms,
        "note": ("cue frozen on IDSSE dev matches (F1 0.80-0.87); applied here without tuning. "
                 "Public numbers are diagnostic only and were not used to choose anything."),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=float)

    print("motion frames {} ({} new, {:.0%} already in final)".format(
        overlaps["motion_frames"], overlaps["added_frames"], overlaps["motion_inside_final_fraction"]))
    for name, arm in arms.items():
        s = arm["structure"]
        print("{:<16} lambda {:>4} poss {:>4} dur {:>5.1f} dead {:.3f} public {:.4f}".format(
            name, arm["lambda"], s["possessions"], s["mean_duration_s"], s["dead_fraction"],
            arm["public"]["spell_f1_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
