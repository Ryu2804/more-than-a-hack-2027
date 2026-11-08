#!/usr/bin/env python3
"""Phase 0b: is the BAR-ATH failure the cue or the decision layer?

Phase 0 showed the frozen motion cue alone marks 51% of graded frames dead, while experiment 008's
decoder turned the same cue into 85% dead. This script keeps the cue fixed and changes only how the
decoder consumes it:

  hard   - the 002 behaviour: D=0 cost on cue frames, H/A=1 (what 008 did)
  soft   - D cheaper on cue frames but not forced (D=0, H/A=w) with w in {0.3, 0.6}
  capped - hard override, but runs longer than CAP seconds are truncated to the cue's own run

Everything is GT-free: lambda still comes from the possession-count band, and the public window is
printed as a diagnostic only. No candidate is selected here.
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as p0
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
FPS = dp2.FPS

OUT_MARGIN = 0.1
OUT_RUN = 15
SLOW = 2.0
SLOW_RUN = 25
CAP_SECONDS = 30.0
SOFT_WEIGHTS = (0.3, 0.6)


def load_inputs(data_dir, n_frames):
    """ball position, presence, ball speed, nearest player speed and nearest team."""
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(dp2.GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(dp2.GAME))
    team = dp2.bl.team_of_each_track(atd_xml)
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    present = np.zeros(n_frames, dtype=bool)
    obs = np.full(n_frames, -1, dtype=np.int8)
    near_speed = np.full(n_frames, np.nan)

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
                x = float(coords[0]) * dp2.PITCH_LENGTH
                y = float(coords[1]) * dp2.PITCH_WIDTH
            except (ValueError, IndexError):
                continue
            if not (0 <= frame < n_frames):
                continue
            bx[frame], by[frame], present[frame] = x, y, True
            best = None
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    first = group.find(",")
                    second = group.find(",", first + 1)
                    third = group.find(",", second + 1)
                    px = float(group[:first]) * dp2.PITCH_LENGTH
                    py = float(group[first + 1:second]) * dp2.PITCH_WIDTH
                    ps = float(group[second + 1:third]) if third > 0 else float("nan")
                except (ValueError, IndexError):
                    continue
                distance = (px - x) ** 2 + (py - y) ** 2
                if best is None or distance < best[0]:
                    best = (distance, int(team[slot]), ps)
            if best is not None:
                obs[frame] = best[1]
                near_speed[frame] = best[2]
    return bx, by, present, obs, near_speed


def motion_cue(bx, by, near_speed, graded, present=None):
    if present is None:
        present = np.isfinite(bx)
    cx = bx - dp2.PITCH_LENGTH / 2.0
    cy = by - dp2.PITCH_WIDTH / 2.0
    outside = graded & present & ((np.abs(cx) > 52.5 + OUT_MARGIN) | (np.abs(cy) > 34.0 + OUT_MARGIN))
    out = dp2.run_length_mask(outside, OUT_RUN)
    slow = graded & np.isfinite(near_speed) & (near_speed < SLOW)
    return out | dp2.run_length_mask(slow, SLOW_RUN)


def costs(obs, cue, mode, weight):
    n = len(obs)
    cost = np.ones((3, n), dtype=np.float32)
    seen = obs >= 0
    idx = np.flatnonzero(seen)
    cost[obs[idx], idx] = 0.0
    if cue is not None and cue.any():
        dead_idx = np.flatnonzero(cue)
        cost[2, dead_idx] = 0.0
        cost[0, dead_idx] = weight if mode == "soft" else 1.0
        cost[1, dead_idx] = weight if mode == "soft" else 1.0
    return cost


def capping(cue, cap_frames):
    out = np.zeros_like(cue)
    for start, stop in dp2.runs_of(cue):
        out[start:min(stop, start + cap_frames)] = True
    return out


def structure(codes, n_frames):
    graded = dp2.graded_mask(n_frames)
    runs = dp2.full_match_run_list(n_frames)
    spells = pm.possession_spells(codes, runs)
    durations = [float(item[1] - item[0]) / FPS for item in spells]
    return {
        "possessions": len(durations),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "dead_fraction_graded": float((graded & (codes == 2)).sum() / max(1, graded.sum())),
    }


def decode(cost, switch_penalty, n_frames):
    """Viterbi over a per-frame 3xN cost matrix, one pass per half."""
    out = np.full(n_frames, 2, dtype=np.int8)
    for _, start, end in dp2.half_video_blocks(dp2.GAME):
        block = cost[:, start:end + 1]
        n = block.shape[1]
        dp = np.empty((n, 3), dtype=np.float32)
        back = np.empty((n, 3), dtype=np.int8)
        dp[0] = block[:, 0]
        back[0] = np.arange(3, dtype=np.int8)
        for t in range(1, n):
            prev = dp[t - 1]
            order = np.argsort(prev)
            best, second = int(order[0]), int(order[1])
            for state in range(3):
                same = prev[state]
                if best != state:
                    switch, source = prev[best] + switch_penalty, best
                else:
                    switch, source = prev[second] + switch_penalty, second
                if same <= switch:
                    dp[t, state] = same + block[state, t]
                    back[t, state] = state
                else:
                    dp[t, state] = switch + block[state, t]
                    back[t, state] = source
        labels = np.empty(n, dtype=np.int8)
        labels[-1] = int(np.argmin(dp[-1]))
        for t in range(n - 1, 0, -1):
            labels[t - 1] = back[t, labels[t]]
        out[start:end + 1] = labels
    return out


def pick_lambda(cost, n_frames):
    rows = []
    for lam in dp2.LAMBDA_GRID:
        codes = decode(cost, lam, n_frames)
        rows.append({"lambda": lam, **structure(codes, n_frames)})
    inside = [r for r in rows if dp2.COUNT_BAND[0] <= r["possessions"] <= dp2.COUNT_BAND[1]]
    pool = inside or rows
    best = min(pool, key=lambda r: abs(r["possessions"] - dp2.COUNT_CENTRE))
    return best["lambda"], rows


def main():
    data_dir = os.path.join(ROOT, "..", "data")
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    n_frames = dp2.source_frames("tactical", dp2.GAME)
    graded = dp2.graded_mask(n_frames)

    obs, present, bx, by, near_speed = p0.load(data_dir, n_frames)
    cue = motion_cue(bx, by, near_speed, graded, present)
    print("cue frames", int(cue.sum()), "of graded", int(graded.sum()), flush=True)

    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(dp2.GAME))))
    arms = {}
    plans = [
        ("hard", "hard", 1.0, cue),
        ("soft_0p3", "soft", 0.3, cue),
        ("soft_0p6", "soft", 0.6, cue),
        ("capped_30s", "hard", 1.0, capping(cue, int(CAP_SECONDS * FPS))),
    ]
    for name, mode, weight, mask in plans:
        cost = costs(obs, mask, mode, weight)
        lam, grid = pick_lambda(cost, n_frames)
        codes = decode(cost, lam, n_frames)
        metrics = dp2.metrics_for(gt, codes, 1000)
        arms[name] = {
            "lambda": lam,
            "structure": structure(codes, n_frames),
            "public": {k: metrics[k] for k in ("spell_f1_mean", "spell_bootstrap_low",
                                               "spell_bootstrap_high", "macro_f1", "n_pred_spells")},
            "grid": grid,
        }
        print(name, "lambda", lam, arms[name]["structure"], "public", round(metrics["spell_f1_mean"], 4), flush=True)

    with open(os.path.join(out_dir, "phase0b.json"), "w") as handle:
        json.dump({"cue_frames": int(cue.sum()), "arms": arms}, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
