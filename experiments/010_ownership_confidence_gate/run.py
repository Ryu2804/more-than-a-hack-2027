#!/usr/bin/env python3
"""010: ownership-confidence gate for the DP emission (BAR-ATH).

Experiment 009 measured, on the IDSSE external matches, that the trivial nearest-player ownership
is only ~0.77 accurate overall but ~0.95 accurate when the nearest player is within 3 m and the ball
is slower than 4 m/s, which covers about 29% of alive frames. This experiment applies exactly that
gate to BAR-ATH: trust the nearest player's team only inside the trusted regime, leave every other
frame uninformative, and let the missing-aware DP carry the state through.

The gate thresholds come from the external measurement, not from BAR-ATH ground truth. Public
window numbers are reported as a diagnostic only; nothing is selected here.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

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
bl, pm = dp2.bl, dp2.pm
GAME = dp2.GAME
H, A, D = dp2.H, dp2.A, dp2.D
FPS = dp2.FPS
PITCH_LENGTH, PITCH_WIDTH = dp2.PITCH_LENGTH, dp2.PITCH_WIDTH

GATE_DISTANCE = 3.0   # metres, from 009 trusted regime
GATE_SPEED = 4.0      # m/s, from 009 trusted regime


def load_gate_inputs(data_dir, n_frames):
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME))
    team_of_track = bl.team_of_each_track(atd_xml)
    obs = np.full(n_frames, -1, dtype=np.int8)
    nearest = np.full(n_frames, np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)

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
                if len(coords) > 2 and coords[2] != "NaN":
                    ball_speed[frame] = float(coords[2])
            except (ValueError, IndexError):
                continue
            ball_x[frame], ball_y[frame] = bx, by
            present[frame] = True
            best = None
            best_team = -1
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    first = group.find(",")
                    second = group.find(",", first + 1)
                    px = float(group[:first]) * PITCH_LENGTH
                    py = float(group[first + 1:second if second > 0 else None]) * PITCH_WIDTH
                except (ValueError, IndexError):
                    continue
                distance = float(np.hypot(px - bx, py - by))
                if best is None or distance < best:
                    best = distance
                    best_team = int(team_of_track[slot])
            if best is not None:
                obs[frame] = best_team
                nearest[frame] = best
    return obs, nearest, present, ball_x, ball_y, ball_speed


def gated_obs(obs, nearest, present, speed):
    out = obs.copy()
    trusted = present & np.isfinite(nearest) & (nearest <= GATE_DISTANCE)
    moving = np.isfinite(speed) & (speed > GATE_SPEED)
    out[~trusted] = -1
    out[moving] = -1
    return out, trusted, moving


def structure(codes, n_frames):
    graded = dp2.graded_mask(n_frames)
    runs = dp2.full_match_run_list(n_frames)
    spells = pm.possession_spells(codes, runs)
    durations = [float(item[1] - item[0]) / FPS for item in spells]
    return {
        "possessions": len(spells),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "dead_fraction_graded": float((graded & (codes == D)).sum() / max(1, graded.sum())),
    }


def pick_lambda(obs, dead, n_frames):
    rows = []
    for lam in dp2.LAMBDA_GRID:
        codes = dp2.dp_full(obs, dead, lam, n_frames)
        stats = structure(codes, n_frames)
        rows.append({"lambda": lam, **stats})
    inside = [r for r in rows if dp2.COUNT_BAND[0] <= r["possessions"] <= dp2.COUNT_BAND[1]]
    pool = inside or rows
    best = min(pool, key=lambda r: abs(r["possessions"] - dp2.COUNT_CENTRE))
    return best["lambda"], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    n_frames = dp2.source_frames("tactical", GAME)
    obs, nearest, present, ball_x, ball_y, ball_speed = load_gate_inputs(args.data_dir, n_frames)
    speed = dp2.ball_speed_series(present, ball_x, ball_y, ball_speed, FPS)
    restart = dp2.parse_set_piece_starts(args.data_dir, n_frames)
    evidence = dp2.build_evidence(n_frames, present, ball_x, ball_y, ball_speed, restart)

    obs_gated, trusted, moving = gated_obs(obs, nearest, present, speed)
    graded = dp2.graded_mask(n_frames)
    coverage = float((trusted & graded).sum() / max(1, (present & graded).sum()))

    arms = {}
    for name, source in (("dp_atd_nearest", obs), ("dp_gated", obs_gated)):
        lam, rows = pick_lambda(source, evidence["final"], n_frames)
        codes = dp2.dp_full(source, evidence["final"], lam, n_frames)
        arms[name] = {"lambda": lam, "grid": rows, "structure": structure(codes, n_frames),
                      "codes": codes}

    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt",
                                                       "{}_possession_gt_public.csv".format(GAME))))
    result = {
        "gate": {"distance_m": GATE_DISTANCE, "ball_speed_ms": GATE_SPEED},
        "trusted_coverage_of_present_graded": coverage,
        "moving_frames": int(moving.sum()),
        "arms": {},
        "runtime_s": time.time() - started,
    }
    for name, data in arms.items():
        metrics = dp2.metrics_for(gt, data["codes"], args.bootstrap)
        result["arms"][name] = {
            "lambda": data["lambda"], "structure": data["structure"],
            "public_diagnostic": metrics, "grid": data["grid"],
        }

    reference = arms["dp_atd_nearest"]["codes"]
    result["paired_gated_vs_atd"] = dp2.paired_spell_bootstrap(
        gt, arms["dp_gated"]["codes"], reference, 1000, 0)

    with open(os.path.join(args.out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=float)

    for name, data in result["arms"].items():
        print(name, "lambda", data["lambda"], "possessions", data["structure"]["possessions"],
              "dur", round(data["structure"]["mean_duration_s"], 1),
              "dead", round(data["structure"]["dead_fraction_graded"], 3),
              "spellF1", round(data["public_diagnostic"]["spell_f1_mean"], 4))
    print("trusted coverage", round(coverage, 3),
          "paired delta", round(result["paired_gated_vs_atd"]["delta_mean"], 4),
          "[", round(result["paired_gated_vs_atd"]["delta_low"], 4), ",",
          round(result["paired_gated_vs_atd"]["delta_high"], 4), "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
