#!/usr/bin/env python3
"""Small, reproducible real-data smoke test for the possession DP idea.

This is an experiment, not a submission pipeline. It answers one narrow question on the public
5-minute ground-truth slice:

    Does a missing-aware DP with simple dead-ball evidence beat baseline_smooth50 on spellF1?

The public slice is used only for pipeline alignment / a smoke test. It is far too small (17
possessions) to select a final model or to claim a full-match improvement. The script therefore
prints and stores confidence intervals, and the report must treat every comparison as
exploratory.

Inputs are the gated match assets under data/ (see DATA.md), never committed or redistributed.
Outputs are metrics, label counts and optional submissions; no raw tracking data is copied.

Usage:
    .venv/bin/python experiments/possession_dp_smoke.py \
        --data-dir ../data \
        --bootstrap 1000 \
        --write-submissions

The script is deterministic apart from timing and bootstrap draws. Set PYTHONHASHSEED=0 for an
extra reproducibility check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import baseline as bl  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAMES, half_video_blocks, source_frames  # noqa: E402

GAME = "BAR-ATH"
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_observations(data_dir: str, n_frames: int):
    """Return per-frame nearest-player team, ball position, presence and provided speed."""
    atd_txt = os.path.join(data_dir, "tactical", f"{GAME}_tactical_atd.txt")
    atd_xml = os.path.join(data_dir, "tactical", f"{GAME}_tactical_atd.xml")
    if not os.path.isfile(atd_txt) or not os.path.isfile(atd_xml):
        raise SystemExit(
            "missing tactical ATD assets under {}\n"
            "Expected {} and {}\n"
            "Data is gated; see DATA.md and point --data-dir at the bundle.".format(
                data_dir, atd_txt, atd_xml
            )
        )

    team = bl.team_of_each_track(atd_xml)
    pitch_length, pitch_width = GAMES[GAME]["pitch"]
    obs = np.full(n_frames, -1, dtype=np.int8)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
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
                x_norm = float(coords[0])
                y_norm = float(coords[1])
                if len(coords) > 2 and coords[2] != "NaN":
                    ball_speed[frame] = float(coords[2])
            except (ValueError, IndexError):
                continue
            if frame < 0 or frame >= n_frames:
                continue

            ball_x[frame] = x_norm * pitch_length
            ball_y[frame] = y_norm * pitch_width
            best_distance = None
            best_team = -1
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    comma = group.index(",")
                    px = float(group[:comma]) * pitch_length
                    py = float(group[comma + 1:group.index(",", comma + 1)]) * pitch_width
                except (ValueError, IndexError):
                    continue
                dx = px - ball_x[frame]
                dy = py - ball_y[frame]
                distance = dx * dx + dy * dy
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_team = int(team[slot])
            obs[frame] = best_team
            present[frame] = True

    return obs, present, ball_x, ball_y, ball_speed


def run_length_mask(flags: np.ndarray, min_length: int) -> np.ndarray:
    """True on every frame of a run of at least `min_length` consecutive True values.

    A centred minimum_filter1d would only keep the middle of a long run; this marks the whole
    run, which is what a dead-ball evidence mask needs.
    """
    flags = np.asarray(flags, dtype=bool)
    if min_length <= 1:
        return flags.copy()
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    out = np.zeros_like(flags)
    for start, stop in zip(starts, stops):
        if stop - start >= min_length:
            out[start:stop] = True
    return out


def dead_evidence(
    data_dir: str,
    n_frames: int,
    present: np.ndarray,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_speed: np.ndarray,
):
    """Three cheap, interpretable dead-ball candidates.

    long_missing: no ball for >= 5 s. Strong signal, but also occlusion.
    low_speed: ball present and below 1 m/s for >= 1 s. Weak alone; kept as an ablation.
    set_piece: first 5 s of each Smart Tagging SET PIECES instance. The full 35 s instance is
        not dead-ball time; only the restart window is plausibly dead.
    """
    missing = ~present
    long_missing = run_length_mask(missing, 125)

    speed = np.full(n_frames, np.nan, dtype=np.float32)
    speed[present] = ball_speed[present]
    present_idx = np.flatnonzero(present)
    if len(present_idx) > 1:
        step = np.hypot(np.diff(ball_x[present_idx]), np.diff(ball_y[present_idx])) * GAMES[GAME][
            "fps"
        ]
        for k, frame in enumerate(present_idx[1:]):
            if np.isnan(speed[frame]):
                speed[frame] = step[k]
    valid = ~np.isnan(speed)
    smoothed = np.copy(speed)
    if valid.sum() >= 5:
        smoothed[valid] = median_filter(speed[valid], size=5, mode="nearest")
    low_speed = present & np.isfinite(smoothed) & (smoothed < 1.0)
    low_speed = run_length_mask(low_speed, 25)

    set_piece = np.zeros(n_frames, dtype=bool)
    tag_path = os.path.join(data_dir, "tactical", f"{GAME}_tactical_smart_tagging.xml")
    if os.path.isfile(tag_path):
        root = ET.parse(tag_path).getroot()
        for instance in root.iter("instance"):
            if instance.findtext("code", "").strip().upper() != "SET PIECES":
                continue
            start = instance.findtext("start")
            if start is None:
                continue
            start_frame = max(0, int(round(float(start) * GAMES[GAME]["fps"])))
            end_frame = min(n_frames, start_frame + 125)
            if end_frame > start_frame:
                set_piece[start_frame:end_frame] = True

    return {
        "long_missing": long_missing,
        "low_speed": low_speed,
        "set_piece": set_piece,
        "all": long_missing | low_speed | set_piece,
    }


def dp_half(obs_half: np.ndarray, dead_half: np.ndarray | None, switch_penalty: float):
    """Viterbi DP over H/A/D for one half.

    Emission cost:
      - observed frame, no dead evidence: 0 for nearest team, 1 for the other team, 1 for D.
      - missing frame: 0 for every state (the DP may keep the last state or switch; the switch
        penalty decides).
      - dead evidence: 0 for D, 1 for H/A. This overrides a present ball on purpose: the
        evidence says the ball is not in play, not that the nearest player owns it.
    """
    n = len(obs_half)
    cost = np.zeros((n, 3), dtype=np.float32)
    observed = obs_half >= 0
    if observed.any():
        idx = np.flatnonzero(observed)
        cost[idx, obs_half[idx]] = 0.0
        cost[idx, 1 - obs_half[idx]] = 1.0
        cost[idx, 2] = 1.0
    if dead_half is not None:
        idx = np.flatnonzero(dead_half)
        cost[idx, 2] = 0.0
        cost[idx, 0] = 1.0
        cost[idx, 1] = 1.0

    dp = np.empty((n, 3), dtype=np.float32)
    back = np.empty((n, 3), dtype=np.int8)
    dp[0] = cost[0]
    back[0] = np.arange(3, dtype=np.int8)
    for t in range(1, n):
        prev = dp[t - 1]
        order = np.argsort(prev)
        best_state, second_state = int(order[0]), int(order[1])
        for state in range(3):
            same = prev[state]
            if best_state != state:
                switch = prev[best_state] + switch_penalty
                source = best_state
            else:
                switch = prev[second_state] + switch_penalty
                source = second_state
            if same <= switch:
                dp[t, state] = same + cost[t, state]
                back[t, state] = state
            else:
                dp[t, state] = switch + cost[t, state]
                back[t, state] = source

    labels = np.empty(n, dtype=np.int8)
    labels[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def dp_full(obs: np.ndarray, dead: np.ndarray | None, switch_penalty: float, n_frames: int):
    """Run the DP per half; frames outside the two halves are D (dead / halftime)."""
    out = np.full(n_frames, D, dtype=np.int8)
    for _, start, end in half_video_blocks(GAME):
        out[start:end + 1] = dp_half(
            obs[start:end + 1],
            dead[start:end + 1] if dead is not None else None,
            switch_penalty,
        )
    return out


def labels_for(codes: np.ndarray) -> np.ndarray:
    return LETTERS[codes]


def metrics_for(gt_codes, codes: np.ndarray, bootstrap_n: int):
    result = pm.evaluate(gt_codes, codes, bootstrap_n=bootstrap_n)
    compact = {
        "spell_f1_mean": result["spell_f1_mean"],
        "spell_bootstrap_low": result["spell_bootstrap_low"],
        "spell_bootstrap_high": result["spell_bootstrap_high"],
        "macro_f1": result["macro_f1"],
        "bootstrap_low": result["bootstrap_low"],
        "bootstrap_high": result["bootstrap_high"],
        "n_gt_spells": result["n_gt_spells"],
        "n_pred_spells": result["n_pred_spells"],
        "spell_count_error": result["spell_count_error"],
        "mean_spell_duration_s": result["mean_spell_duration_s"],
        "gt_mean_spell_duration_s": result["gt_mean_spell_duration_s"],
        "fragmentation": result["fragmentation"],
        "alive_dead_f1": result["alive_dead_f1"],
        "n_abstain_on_graded": result["n_abstain_on_graded"],
        "per_class": result["per_class"],
        "bootstrap_n": result["bootstrap_n"],
    }
    compact["pred_counts"] = {
        "H": int((codes == H).sum()),
        "A": int((codes == A).sum()),
        "D": int((codes == D).sum()),
    }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MTAH_DATA_DIR")
        or os.path.abspath(os.path.join(ROOT, "..", "data")),
        help="folder containing tactical/ and broadcast/ (default: ../data or MTAH_DATA_DIR)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.abspath(
            os.path.join(
                ROOT, "..", "research_loop", "riset_methodology", "experiments",
                "real_data_dp_smoke",
            )
        ),
        help="output folder for results.json, results.csv and optional submissions/",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap draws per arm")
    parser.add_argument("--seed", type=int, default=0, help="numpy seed for bootstrap draws")
    parser.add_argument(
        "--write-submissions",
        action="store_true",
        help="write frame,label CSVs under <out-dir>/submissions/ (prefixed smoke_)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    started = time.time()
    started_utc = utc_now()
    os.makedirs(args.out_dir, exist_ok=True)

    n_frames = source_frames("tactical", GAME)
    obs, present, ball_x, ball_y, ball_speed = load_observations(args.data_dir, n_frames)
    evidence = dead_evidence(args.data_dir, n_frames, present, ball_x, ball_y, ball_speed)

    gt_path = os.path.join(
        ROOT, "evaluation", "gt", f"{GAME}_possession_gt_public.csv"
    )
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))

    arms = {}
    definitions = {}

    # Baselines: the two mandatory reference points, plus smooth125 as a diagnostic only.
    t0 = time.time()
    raw_labels = bl.smooth(obs, 0)
    arms["baseline_raw"] = pm.label_codes(raw_labels)
    definitions["baseline_raw"] = {
        "family": "baseline",
        "description": "nearest player to the ball, D where the ball is missing",
        "window_frames": 0,
        "dead_evidence": "missing only",
    }
    runtime_raw = time.time() - t0

    t0 = time.time()
    smooth50_labels = bl.smooth(obs, 50)
    arms["baseline_smooth50"] = pm.label_codes(smooth50_labels)
    definitions["baseline_smooth50"] = {
        "family": "baseline",
        "description": "majority vote over +/-50 frames (2 s), the published floor",
        "window_frames": 50,
        "dead_evidence": "majority only",
    }
    runtime_smooth50 = time.time() - t0

    t0 = time.time()
    smooth125_labels = bl.smooth(obs, 125)
    arms["baseline_smooth125"] = pm.label_codes(smooth125_labels)
    definitions["baseline_smooth125"] = {
        "family": "baseline",
        "description": "majority vote over +/-125 frames (5 s), diagnostic only",
        "window_frames": 125,
        "dead_evidence": "majority only",
    }
    runtime_smooth125 = time.time() - t0

    runtimes = {
        "baseline_raw": runtime_raw,
        "baseline_smooth50": runtime_smooth50,
        "baseline_smooth125": runtime_smooth125,
    }

    # Fairness check: smoothing plus a hard dead-evidence mask. If this matches the DP, the DP's
    # gain is not coming from temporal inference but from the dead evidence itself.
    def smooth_with_dead(window, dead, label):
        t0 = time.time()
        labels = bl.smooth(obs, window).copy()
        if dead is not None:
            labels[dead] = "D"
        arms[label] = pm.label_codes(labels)
        definitions[label] = {
            "family": "baseline_dead_postprocess",
            "description": (
                "majority vote over +/-{} frames, then force D on the dead-evidence mask"
                .format(window)
            ),
            "window_frames": window,
            "dead_evidence": "all" if dead is evidence["all"] else "long_missing",
        }
        runtimes[label] = time.time() - t0

    smooth_with_dead(50, evidence["all"], "smooth50_dead_all")
    smooth_with_dead(125, evidence["all"], "smooth125_dead_all")
    smooth_with_dead(125, evidence["long_missing"], "smooth125_dead_long")

    # DP arms: one dead-evidence component per arm at lambda=50, then lambda sensitivity.
    main_lambda = 50.0
    dp_specs = [
        ("dp50_none", 50.0, None, "missing-aware DP, no dead evidence"),
        ("dp50_long_missing", 50.0, evidence["long_missing"], "D on missing runs >= 5 s"),
        ("dp50_low_speed", 50.0, evidence["low_speed"], "D on low-speed runs >= 1 s"),
        ("dp50_set_piece", 50.0, evidence["set_piece"], "D on first 5 s of SET PIECES"),
        ("dp50_all", 50.0, evidence["all"], "union of the three evidence masks"),
        ("dp25_all", 25.0, evidence["all"], "lambda sensitivity (1 s switch penalty)"),
        ("dp16_all", 16.0, evidence["all"], "lambda sensitivity (0.64 s switch penalty)"),
        ("dp64_all", 64.0, evidence["all"], "lambda sensitivity (2.56 s switch penalty)"),
    ]
    for name, penalty, dead, description in dp_specs:
        t0 = time.time()
        arms[name] = dp_full(obs, dead, penalty, n_frames)
        runtimes[name] = time.time() - t0
        definitions[name] = {
            "family": "missing_aware_dp",
            "description": description,
            "switch_penalty": penalty,
            "dead_evidence": (
                "none" if dead is None else
                "long_missing" if name.endswith("long_missing") else
                "low_speed" if name.endswith("low_speed") else
                "set_piece" if name.endswith("set_piece") else
                "all"
            ),
        }

    results = {}
    for name, codes in arms.items():
        results[name] = metrics_for(gt_codes, codes, args.bootstrap)
        results[name]["runtime_s"] = runtimes[name]
        results[name]["definition"] = definitions[name]

    # Official scorer parity: if submissions were written, run the repo's scorer on them.
    official = None
    if args.write_submissions:
        sub_dir = os.path.join(args.out_dir, "submissions")
        os.makedirs(sub_dir, exist_ok=True)
        for name, codes in arms.items():
            bl.write_submission(
                os.path.join(sub_dir, f"smoke_{name}.csv"), labels_for(codes)
            )
        official = {
            "note": (
                "Run separately: .venv/bin/python evaluation/run_evaluation.py "
                "--results <out-dir>/submissions --bootstrap 1000"
            ),
            "submission_dir": sub_dir,
        }

    data_files = [
        os.path.join(args.data_dir, "tactical", f"{GAME}_tactical_atd.txt"),
        os.path.join(args.data_dir, "tactical", f"{GAME}_tactical_atd.xml"),
        os.path.join(args.data_dir, "tactical", f"{GAME}_tactical_smart_tagging.xml"),
        gt_path,
    ]
    payload = {
        "experiment": "real_data_dp_smoke",
        "status": "exploratory_public_window_smoke_test_not_full_match_evidence",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "runtime_total_s": round(time.time() - started, 3),
        "command": " ".join([sys.executable] + sys.argv),
        "seed": args.seed,
        "bootstrap_n": args.bootstrap,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": getattr(__import__("pandas"), "__version__", None),
            "scipy": __import__("scipy").__version__,
        },
        "public_window": {
            "video_frames": [11185, 18684],
            "graded_frames": 7500,
            "gt_possessions": int(results["baseline_raw"]["n_gt_spells"]),
            "gt_mean_duration_s": results["baseline_raw"]["gt_mean_spell_duration_s"],
            "warning": "5-minute slice; 17 possessions; intervals are wide; never tune on this",
        },
        "inputs": {
            path: {"sha256": sha256(path), "bytes": os.path.getsize(path)}
            for path in data_files
            if os.path.isfile(path)
        },
        "evidence_frame_counts": {
            key: int(mask.sum()) for key, mask in evidence.items()
        },
        "ball_present_frames": int(present.sum()),
        "arms": results,
        "official_scorer": official,
        "interpretation": {
            "primary_metric": "spell_f1_mean",
            "decision_rule": (
                "A gap is treated as exploratory unless the bootstrap intervals separate and "
                "the effect exceeds the public-window noise; no arm is selected for submission "
                "from this slice."
            ),
            "bytetrack_botsort": (
                "Not run: the bundle has point tracking only, no video frames and no bounding-box "
                "detections. ByteTrack/BoT-SORT need bbox + confidence and a detector."
            ),
        },
    }

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    csv_path = os.path.join(args.out_dir, "results.csv")
    columns = [
        "arm", "family", "spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high",
        "macro_f1", "n_pred_spells", "mean_spell_duration_s", "fragmentation",
        "alive_dead_f1", "f1_H", "f1_A", "f1_D", "pred_H", "pred_A", "pred_D",
        "runtime_s",
    ]
    def fmt(value, digits):
        return "" if value is None else f"{value:.{digits}f}"

    with open(csv_path, "w") as handle:
        handle.write(",".join(columns) + "\n")
        for name, row in results.items():
            per_class = row["per_class"]
            counts = row["pred_counts"]
            values = [
                name,
                row["definition"]["family"],
                fmt(row["spell_f1_mean"], 6),
                fmt(row["spell_bootstrap_low"], 6),
                fmt(row["spell_bootstrap_high"], 6),
                fmt(row["macro_f1"], 6),
                str(row["n_pred_spells"]),
                fmt(row["mean_spell_duration_s"], 3),
                fmt(row["fragmentation"], 3),
                fmt(row["alive_dead_f1"], 4),
                fmt(per_class["H"]["f1"], 4),
                fmt(per_class["A"]["f1"], 4),
                fmt(per_class["D"]["f1"], 4),
                str(counts["H"]),
                str(counts["A"]),
                str(counts["D"]),
                fmt(row["runtime_s"], 3),
            ]
            handle.write(",".join(values) + "\n")

    print("wrote", json_path)
    print("wrote", csv_path)
    print(
        "baseline_raw spellF1={:.4f}  baseline_smooth50 spellF1={:.4f}  dp50_all spellF1={:.4f}".format(
            results["baseline_raw"]["spell_f1_mean"],
            results["baseline_smooth50"]["spell_f1_mean"],
            results["dp50_all"]["spell_f1_mean"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
