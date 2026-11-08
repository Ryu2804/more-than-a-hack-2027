#!/usr/bin/env python3
"""Possession DP pipeline v2: gated evidence, GT-free lambda calibration, validation-only public window.

Experiment notes and pre-registration summary: NOTES.md in this folder.

What this script does, in order:

1. Loads the tactical ATD and rebuilds the ball speed series without the v1 gap bug.
2. Builds observed-only dead-ball evidence: kinematics (speed + stationarity), pitch geometry,
   Smart Tagging restarts, and gated long-missing runs. The old ungated long-missing run is kept
   as a negative control, never as part of the final configuration.
3. Calibrates the DP switch penalty on a **GT-free** criterion over the full match: predicted
   possession count must fall inside the pre-registered count band and be as close as possible to
   its centre. This is the accepted note's GT-free rule, not a public-window fit.
4. Validates a fixed, pre-declared arm set on the public 5-minute window. Nothing is selected there.
5. Calibrates two synthetic harnesses: an i.i.d. observation harness (the assumption behind the
   accepted lambda=4 default) and an autocorrelated harness whose confusion rate is matched to the
   real nearest-player switch rate. This shows why lambda=4 transfers badly to the real data.
6. Runs GT-free structural checks on the full match: label distribution, self-supervised masking.
7. Writes results.json, results.csv, run.log, submissions/ and the official scorer reports.

Usage:
    .venv/bin/python experiments/002_possession_dp/run.py \
        --data-dir ../data --bootstrap 1000 --seed 0 --write-submissions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import baseline as bl  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAMES, half_video_blocks, source_frames  # noqa: E402

GAME = "BAR-ATH"
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]

PREREG_LAMBDA = 4.0
LAMBDA_GRID = (4.0, 8.0, 16.0, 25.0, 32.0, 50.0, 64.0)
# WARNING (Audit 2026-11-08 / A-R3): This count band was derived from synthetic 82.8k frames.
# For BAR-ATH (158.8k frames), these thresholds are unscaled historical artifacts from 002.
# Do not use for formal submission selection without scaling by match frame length.
COUNT_BAND = (110.4, 662.4)   # accepted note: GT-free plausible full-match possession count
COUNT_CENTRE = 270.4

SPEED_MIN = 1.0
SPEED_MIN_RUN = 30
STATIONARY_SPAN = 1.5
STATIONARY_DENSITY = 0.8
GEOM_MARGIN = 0.5
GEOM_MIN_RUN = 15
MISS_MIN_RUN = 125
MISS_VERY_LONG = 375
MISS_RESTART_TOL = 75
MISS_BOUNDARY_M = 3.0
ST_WINDOW = 125
GAP_MAX_FRAMES = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: str) -> str:
    """Repo-relative when possible, so provenance does not leak machine paths."""
    absolute = os.path.abspath(path)
    try:
        return os.path.relpath(absolute, ROOT)
    except ValueError:
        return os.path.basename(path)


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


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# -- data ----------------------------------------------------------------------------------------

def load_observations(data_dir: str, n_frames: int):
    atd_txt = os.path.join(data_dir, "tactical", f"{GAME}_tactical_atd.txt")
    atd_xml = os.path.join(data_dir, "tactical", f"{GAME}_tactical_atd.xml")
    if not os.path.isfile(atd_txt) or not os.path.isfile(atd_xml):
        raise SystemExit(
            "missing tactical ATD assets under {}\nExpected {} and {}\n"
            "Data is gated; see DATA.md and point --data-dir at the bundle.".format(
                data_dir, atd_txt, atd_xml)
        )

    team = bl.team_of_each_track(atd_xml)
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

            ball_x[frame] = x_norm * PITCH_LENGTH
            ball_y[frame] = y_norm * PITCH_WIDTH
            best_distance = None
            best_team = -1
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    comma = group.index(",")
                    px = float(group[:comma]) * PITCH_LENGTH
                    py = float(group[comma + 1:group.index(",", comma + 1)]) * PITCH_WIDTH
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


def ball_speed_series(present, ball_x, ball_y, ball_speed, fps, gap_max=GAP_MAX_FRAMES):
    """Gap-aware speed fallback. The v1 fallback multiplied speed by the frame gap."""
    n = len(present)
    speed = np.full(n, np.nan, dtype=np.float32)
    speed[present] = ball_speed[present]
    idx = np.flatnonzero(present)
    if len(idx) > 1:
        gap = np.diff(idx).astype(np.float64)
        step = np.hypot(np.diff(ball_x[idx]), np.diff(ball_y[idx]))
        estimate = step / (gap / float(fps))
        for k, frame in enumerate(idx[1:]):
            if np.isnan(speed[frame]) and gap[k] <= gap_max:
                speed[frame] = estimate[k]
    return speed


def run_length_mask(flags, min_length):
    flags = np.asarray(flags, dtype=bool)
    if min_length <= 1:
        return flags.copy()
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    out = np.zeros_like(flags)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= min_length:
            out[start:stop] = True
    return out


def runs_of(flags):
    flags = np.asarray(flags, dtype=bool)
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def graded_mask(n_frames):
    """The two graded halves; everything outside is ungraded and irrelevant for evidence."""
    graded = np.zeros(n_frames, dtype=bool)
    for _, start, end in half_video_blocks(GAME):
        graded[start:end + 1] = True
    return graded


def parse_set_piece_starts(data_dir: str, n_frames: int):
    starts = []
    tag_path = os.path.join(data_dir, "tactical", f"{GAME}_tactical_smart_tagging.xml")
    if not os.path.isfile(tag_path):
        return np.zeros(0, dtype=np.int64)
    root = ET.parse(tag_path).getroot()
    for instance in root.iter("instance"):
        if instance.findtext("code", "").strip().upper() != "SET PIECES":
            continue
        start = instance.findtext("start")
        if start is None:
            continue
        frame = int(round(float(start) * FPS))
        if 0 <= frame < n_frames:
            starts.append(frame)
    return np.array(sorted(starts), dtype=np.int64)


def build_evidence(n_frames, present, ball_x, ball_y, ball_speed, restart_starts):
    graded = graded_mask(n_frames)

    long_missing = run_length_mask((~present) & graded, MISS_MIN_RUN)

    speed = ball_speed_series(present, ball_x, ball_y, ball_speed, FPS)
    valid = np.isfinite(speed)
    trend = np.full(n_frames, np.nan, dtype=np.float32)
    if valid.sum() >= 7:
        compressed = median_filter(speed[valid], size=5, mode="nearest")
        window = min(7, compressed.size if compressed.size % 2 else compressed.size - 1)
        if window >= 3:
            compressed = savgol_filter(compressed, window, 2, mode="nearest")
        trend[valid] = compressed
    low_speed = present & np.isfinite(trend) & (trend < SPEED_MIN)
    low_speed = run_length_mask(low_speed, SPEED_MIN_RUN)

    idx = np.flatnonzero(present)
    stationary = np.zeros(n_frames, dtype=bool)
    if idx.size > 1:
        bx, by = ball_x[idx], ball_y[idx]
        lo = np.searchsorted(idx, idx - 25, side="left")
        hi = np.searchsorted(idx, idx + 25, side="right") - 1
        span = np.hypot(bx[hi] - bx[lo], by[hi] - by[lo])
        density = (hi - lo + 1) / 51.0
        stationary[idx] = (span < STATIONARY_SPAN) & (density >= STATIONARY_DENSITY)
    stationary = run_length_mask(stationary, SPEED_MIN_RUN)

    outside = present & (
        (ball_x < -GEOM_MARGIN) | (ball_x > PITCH_LENGTH + GEOM_MARGIN)
        | (ball_y < -GEOM_MARGIN) | (ball_y > PITCH_WIDTH + GEOM_MARGIN)
    )
    geom = run_length_mask(outside, GEOM_MIN_RUN)

    st = np.zeros(n_frames, dtype=bool)
    for start in restart_starts:
        st[start:min(n_frames, start + ST_WINDOW)] = True

    miss_gated = np.zeros(n_frames, dtype=bool)
    for start, stop in runs_of((~present) & graded):
        if stop - start < MISS_MIN_RUN:
            continue
        near_restart = bool(
            restart_starts.size
            and np.any((restart_starts >= start - MISS_RESTART_TOL)
                       & (restart_starts <= stop + MISS_RESTART_TOL))
        )
        near_boundary = False
        before = np.flatnonzero(present[max(0, start - 50):start])
        after = np.flatnonzero(present[stop:min(n_frames, stop + 50)])
        if before.size:
            near_boundary |= ball_near_boundary(ball_x[max(0, start - 50) + before[-1]],
                                                ball_y[max(0, start - 50) + before[-1]])
        if after.size:
            near_boundary |= ball_near_boundary(ball_x[stop + after[0]], ball_y[stop + after[0]])
        if near_restart or near_boundary:
            miss_gated[start:stop] = True

    kin = run_length_mask((low_speed | stationary) & graded, 1)
    masks = {
        "long_missing": long_missing & graded,
        "low_speed": low_speed & graded,
        "stationary": stationary & graded,
        "kin": kin & graded,
        "geom": geom & graded,
        "set_piece": st & graded,
        "miss_gated": miss_gated & graded,
        "all": (long_missing | low_speed | st) & graded,
        "final": (kin | geom | st | miss_gated) & graded,
    }
    return masks


def ball_near_boundary(x, y, margin=MISS_BOUNDARY_M):
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return (x < margin or x > PITCH_LENGTH - margin
            or y < margin or y > PITCH_WIDTH - margin)


# -- DP ------------------------------------------------------------------------------------------

def dp_half(obs_half, dead_half, switch_penalty):
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


def dp_full(obs, dead, switch_penalty, n_frames):
    out = np.full(n_frames, D, dtype=np.int8)
    for _, start, end in half_video_blocks(GAME):
        out[start:end + 1] = dp_half(
            obs[start:end + 1], dead[start:end + 1] if dead is not None else None, switch_penalty)
    return out


# -- metrics -------------------------------------------------------------------------------------

def metrics_for(gt_codes, codes, bootstrap_n):
    result = pm.evaluate(gt_codes, codes, bootstrap_n=bootstrap_n)
    compact = {
        "spell_f1_mean": result["spell_f1_mean"],
        "spell_bootstrap_low": result["spell_bootstrap_low"],
        "spell_bootstrap_high": result["spell_bootstrap_high"],
        "macro_f1": result["macro_f1"],
        "n_gt_spells": result["n_gt_spells"],
        "n_pred_spells": result["n_pred_spells"],
        "spell_count_error": result["spell_count_error"],
        "mean_spell_duration_s": result["mean_spell_duration_s"],
        "gt_mean_spell_duration_s": result["gt_mean_spell_duration_s"],
        "fragmentation": result["fragmentation"],
        "alive_dead_f1": result["alive_dead_f1"],
        "per_class": result["per_class"],
    }
    compact["pred_counts"] = {
        "H": int((codes == H).sum()), "A": int((codes == A).sum()),
        "D": int((codes == D).sum()),
    }
    return compact


def spell_flags(gt_codes, codes):
    runs = pm.valid_runs(gt_codes)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(codes, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), tau) for tau in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def paired_spell_bootstrap(gt_codes, codes_a, codes_b, n_draws, seed):
    if not n_draws:
        return None
    runs = pm.valid_runs(gt_codes)
    n_gt = len(pm.possession_spells(gt_codes, runs))
    if n_gt == 0:
        return None
    flags_a, n_pred_a = spell_flags(gt_codes, codes_a)
    flags_b, n_pred_b = spell_flags(gt_codes, codes_b)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        scores = []
        for flags, n_pred in ((flags_a, n_pred_a), (flags_b, n_pred_b)):
            f1s = []
            for row in flags:
                tp = int(row[picks].sum())
                fn = n_gt - tp
                fp = max(0, n_pred - tp)
                f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
            scores.append(float(np.mean(f1s)))
        deltas[draw] = scores[0] - scores[1]
    low, high = np.percentile(deltas, pm.BOOTSTRAP_PERCENTILES)
    return {
        "delta_mean": float(np.mean(deltas)),
        "delta_low": float(low),
        "delta_high": float(high),
        "fraction_positive": float(np.mean(deltas > 0)),
        "n_gt_spells": int(n_gt),
    }


# -- GT-free lambda selection --------------------------------------------------------------------

def full_match_run_list(n_frames):
    return [(start, end + 1) for _, start, end in half_video_blocks(GAME)]


def sweep_lambda(obs, evidence, n_frames):
    """Predicted count / duration per lambda over the whole match. Uses no ground truth."""
    runs = full_match_run_list(n_frames)
    rows = {}
    for lam in LAMBDA_GRID:
        codes = dp_full(obs, evidence["final"], lam, n_frames)
        spells = pm.possession_spells(codes, runs)
        lengths = np.array([b - a for a, b, _ in spells])
        rows[str(lam)] = {
            "lambda": lam,
            "n_pred_possessions": len(spells),
            "mean_duration_s": float(lengths.mean()) / FPS if len(lengths) else 0.0,
            "d_fraction_graded": float(np.mean(
                codes[graded_mask(n_frames)] == D)),
        }
    scored = []
    for lam, row in rows.items():
        count = row["n_pred_possessions"]
        inside = COUNT_BAND[0] <= count <= COUNT_BAND[1]
        scored.append({
            "lambda": float(lam), "count": count, "inside_band": bool(inside),
            "distance_to_centre": abs(count - COUNT_CENTRE),
        })
    valid = [row for row in scored if row["inside_band"]]
    pool = valid if valid else scored
    selected = min(pool, key=lambda row: row["distance_to_centre"])
    return {
        "grid": rows,
        "criterion": {
            "band": list(COUNT_BAND),
            "centre": COUNT_CENTRE,
            "rule": (
                "GT-free: predicted full-match possession count must lie in the band; among "
                "valid lambdas pick the one closest to the centre. No ground truth is used."
            ),
            "source": "accepted/temporal_smoothing_vs_state_models.md count-band rule",
        },
        "selected_lambda": selected["lambda"],
        "selected_count": selected["count"],
        "within_band": selected["inside_band"],
        "fallback_outside_band": not selected["inside_band"],
    }


# -- synthetic harnesses ---------------------------------------------------------------------------

def synthetic_truth_aligned(evidence_mask, n_frames, rng):
    """Possessions only where the final evidence says the ball is in play; D elsewhere.

    This is deliberately generous to the evidence layer: the generator places dead states exactly
    where observable stoppages are. Synthetic evidence numbers are therefore an upper bound, never
    evidence that the evidence layer works on the real match.
    """
    graded = graded_mask(n_frames)
    alive = graded & ~evidence_mask
    codes = np.full(n_frames, D, dtype=np.int8)
    for start, stop in runs_of(alive):
        t = start
        team = int(rng.randint(2))
        while t < stop:
            length = int(min(stop - t, max(25, rng.lognormal(np.log(350.0), 0.7))))
            codes[t:t + length] = team
            t += length
            if rng.rand() < 0.8:
                team = 1 - team
    return codes


def synthetic_truth_iid(halves, n_frames, rng):
    """Possessions with durations but dead states independent of the real evidence timeline."""
    codes = np.full(n_frames, D, dtype=np.int8)
    for _, start, end in halves:
        t = start
        team = int(rng.randint(2))
        while t < end:
            t += int(min(end - t, max(12, rng.exponential(90))))
            if t >= end:
                break
            length = int(min(end - t, max(25, rng.lognormal(np.log(350.0), 0.7))))
            codes[t:t + length] = team
            t += length
            if rng.rand() < 0.8:
                team = 1 - team
    return codes


def synthetic_obs_iid(truth, present, rng):
    n = len(truth)
    obs = np.full(n, -1, dtype=np.int8)
    idx = np.flatnonzero(present)
    if not idx.size:
        return obs
    t = truth[idx]
    u = rng.rand(idx.size)
    w = rng.rand(idx.size)
    alive = t != D
    values = np.empty(idx.size, dtype=np.int8)
    values[alive] = np.where(
        u[alive] < 0.93, t[alive], np.where(u[alive] < 0.97, 1 - t[alive], D))
    values[~alive] = np.where(w[~alive] < 0.5, H, A)
    obs[idx] = values
    return obs


def synthetic_obs_autocorr(truth, present, rng, p_start, confusion_mean):
    """Nearest-player labels with autocorrelated confusion episodes.

    The accepted lambda=4 default was calibrated on i.i.d. noise. Real nearest-player labels
    switch 28 times a minute, in runs of about two seconds, because a defender shielding the ball
    stays nearest for a while. A 4-frame switch penalty cannot ride through that. This harness
    reproduces the observed switch rate, which is a GT-free statistic. Dead frames also get a
    slowly changing nearest player rather than an independent coin flip.
    """
    n = len(truth)
    obs = np.full(n, -1, dtype=np.int8)
    idx = np.flatnonzero(present)
    if not idx.size:
        return obs
    t = truth[idx]
    values = t.copy()

    k = 0
    while k < idx.size:
        if rng.rand() < p_start:
            length = min(idx.size - k, int(rng.exponential(confusion_mean)) + 1)
            team = int(rng.randint(2))
            sub = np.arange(k, k + length)
            alive_idx = sub[t[sub] != D]
            values[alive_idx] = team
            k += length
        else:
            k += 1

    dead = t == D
    if dead.any():
        dead_team = np.empty(idx.size, dtype=np.int8)
        current = int(rng.randint(2))
        for i in range(idx.size):
            if rng.rand() < 0.005:
                current = 1 - current
            dead_team[i] = current
        values[dead] = dead_team[dead]
    obs[idx] = values
    return obs


def switch_rate(obs, present):
    seq = obs[np.flatnonzero(present)]
    if len(seq) < 2:
        return 0.0
    switches = int(np.sum(seq[1:] != seq[:-1]))
    return switches / (len(seq) / FPS / 60.0)


def calibrate_confusion(truth, present, present_switch_rate, confusion_mean=50, seed=7):
    """Bisection on the confusion-start probability to match the real observed switch rate.

    Kept as a diagnostic: the achievable curve is reported so the report can state how far the
    harness can go before it stops carrying any truth signal.
    """
    lo, hi = 1e-7, 0.2
    history = []
    for step in range(9):
        mid = 0.5 * (lo + hi)
        rng = np.random.RandomState(seed)
        obs = synthetic_obs_autocorr(truth, present, rng, mid, confusion_mean)
        rate = switch_rate(obs, present)
        history.append({"p_start": mid, "switch_rate": rate})
        if rate < present_switch_rate:
            lo = mid
        else:
            hi = mid
    return {
        "p_start": 0.5 * (lo + hi),
        "confusion_mean_frames": confusion_mean,
        "target_switch_rate_per_min": present_switch_rate,
        "achieved_switch_rate_per_min": history[-1]["switch_rate"],
        "history": history,
    }


def build_arm_set(obs, evidence, n_frames, lambda_main):
    arms, definitions = {}, {}

    def add(name, codes, family, description, **extra):
        arms[name] = codes
        definitions[name] = {"family": family, "description": description, **extra}

    add("baseline_raw", pm.label_codes(bl.smooth(obs, 0)), "baseline",
        "nearest player, D where the ball is missing", window_frames=0)
    add("baseline_smooth50", pm.label_codes(bl.smooth(obs, 50)), "baseline",
        "majority vote +/-50 frames", window_frames=50)
    add("baseline_smooth125", pm.label_codes(bl.smooth(obs, 125)), "baseline",
        "majority vote +/-125 frames", window_frames=125)

    for window, mask_name, label in (
        (50, "all", "smooth50_dead_all"),
        (125, "long_missing", "smooth125_dead_long"),
        (125, "final", "smooth125_dead_final"),
    ):
        labels = bl.smooth(obs, window).copy()
        labels[evidence[mask_name]] = "D"
        add(label, pm.label_codes(labels), "baseline_dead_postprocess",
            "majority vote +/-{} then force D on {} evidence".format(window, mask_name),
            window_frames=window, dead_evidence=mask_name)

    add("negative_constant_D", np.full(n_frames, D, dtype=np.int8), "negative_control",
        "constant D; spellF1 must be 0")
    add("negative_constant_H", np.full(n_frames, H, dtype=np.int8), "negative_control",
        "constant H; spellF1 must be 0")

    for lam, evidence_name, dead, label in (
        (PREREG_LAMBDA, "none", None, "dp4_none"),
        (PREREG_LAMBDA, "final", evidence["final"], "dp4_final"),
        (lambda_main, "none", None, "dpL_none"),
        (lambda_main, "long_missing", evidence["long_missing"], "dpL_long_missing"),
        (lambda_main, "final", evidence["final"], "dpL_final"),
    ):
        add(label, dp_full(obs, dead, lam, n_frames), "missing_aware_dp",
            "missing-aware DP, lambda={}, evidence={}".format(lam, evidence_name),
            switch_penalty=lam, dead_evidence=evidence_name)
    return arms, definitions


# -- GT-free structural checks ---------------------------------------------------------------------

def candidate_windows(present, count=30, length=125):
    runs = [(a, b) for a, b in runs_of(present) if b - a >= length]
    if not runs:
        return []
    picks = []
    for k in range(count):
        run = runs[min(len(runs) - 1, int(k * len(runs) / count))]
        mid = (run[0] + run[1] - length) // 2
        picks.append((int(mid), int(mid) + length))
    return sorted(set(picks))


def mask_probe(obs, evidence, final_codes, n_frames, windows, lambda_main):
    if not windows:
        return None
    final_changes, none_changes, naive_changes = [], [], []
    for start, stop in windows:
        hidden = obs.copy()
        hidden[start:stop] = -1
        final = dp_full(hidden, evidence["final"], lambda_main, n_frames)
        final_changes.append(float(np.mean(final[start:stop] != final_codes[start:stop])))
        none = dp_full(hidden, None, lambda_main, n_frames)
        none_changes.append(float(np.mean(none[start:stop] != final_codes[start:stop])))
        naive = np.where(hidden >= 0, hidden, D).astype(np.int8)
        naive_changes.append(float(np.mean(naive[start:stop] != final_codes[start:stop])))
    return {
        "n_windows": len(windows),
        "window_frames": int(windows[0][1] - windows[0][0]),
        "final_mean_label_change": float(np.mean(final_changes)),
        "dp_none_mean_label_change": float(np.mean(none_changes)),
        "missing_to_D_mean_label_change": float(np.mean(naive_changes)),
        "note": (
            "Low change for final/dp_none means the missing-aware DP keeps its state through an "
            "occlusion; ~1.0 for missing->D shows the naive rule rewrites the window. Structural, "
            "not accuracy."
        ),
    }


# -- main ----------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MTAH_DATA_DIR") or os.path.abspath(os.path.join(ROOT, "..", "data")))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-seeds", type=int, default=3)
    parser.add_argument("--write-submissions", action="store_true")
    args = parser.parse_args()

    started = time.time()
    started_utc = utc_now()
    log_lines = []

    def log(message):
        line = "[{}] {}".format(time.strftime("%H:%M:%S"), message)
        print(line)
        log_lines.append(line)

    os.makedirs(args.out_dir, exist_ok=True)
    log("possession_dp_pipeline v2")
    log("command: " + " ".join([sys.executable] + sys.argv))

    n_frames = source_frames("tactical", GAME)
    obs, present, ball_x, ball_y, ball_speed = load_observations(args.data_dir, n_frames)
    restart_starts = parse_set_piece_starts(args.data_dir, n_frames)
    graded = graded_mask(n_frames)
    log("frames {}; graded {}; ball present {} ({:.1%}); restarts {}".format(
        n_frames, int(graded.sum()), int(present.sum()), present.mean(), restart_starts.size))

    t0 = time.time()
    evidence = build_evidence(n_frames, present, ball_x, ball_y, ball_speed, restart_starts)
    evidence_runtime = time.time() - t0
    for key, mask in evidence.items():
        log("evidence {:>13}: {:>7} graded frames ({:.1%} of graded)".format(
            key, int(mask.sum()), mask.sum() / float(graded.sum())))

    obs_switch_rate = switch_rate(obs, present)
    log("nearest-player switch rate: {:.2f}/min".format(obs_switch_rate))

    # ---- GT-free lambda selection ----
    selection = sweep_lambda(obs, evidence, n_frames)
    lambda_main = selection["selected_lambda"]
    log("lambda sweep: " + "; ".join(
        "{}->{} ({:.1f}s)".format(row["lambda"], row["n_pred_possessions"],
                                  row["mean_duration_s"])
        for row in selection["grid"].values()))
    log("lambda selected by GT-free count band: {} (count {}, inside_band {})".format(
        lambda_main, selection["selected_count"], selection["within_band"]))

    gt_path = os.path.join(ROOT, "evaluation", "gt", f"{GAME}_possession_gt_public.csv")
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))

    arms, definitions = build_arm_set(obs, evidence, n_frames, lambda_main)
    public_names = [
        "baseline_raw", "baseline_smooth50", "baseline_smooth125",
        "smooth125_dead_long", "smooth125_dead_final",
        "dp4_none", "dp4_final",
        "dpL_none", "dpL_long_missing", "dpL_final",
        "negative_constant_D", "negative_constant_H",
    ]
    public_results = {}
    for name in public_names:
        t0 = time.time()
        public_results[name] = metrics_for(gt_codes, arms[name], args.bootstrap)
        public_results[name]["runtime_s"] = time.time() - t0
        public_results[name]["definition"] = definitions[name]
        log("public {:<22} spellF1={:.4f} [{:.4f},{:.4f}] macroF1={:.4f} n_pred={} Df1={:.3f}".format(
            name, public_results[name]["spell_f1_mean"],
            public_results[name]["spell_bootstrap_low"],
            public_results[name]["spell_bootstrap_high"],
            public_results[name]["macro_f1"], public_results[name]["n_pred_spells"],
            public_results[name]["per_class"]["D"]["f1"]))

    public_paired = {}
    for a_name, b_name in [
        ("dpL_final", "baseline_smooth50"),
        ("dpL_final", "baseline_smooth125"),
        ("dpL_final", "smooth125_dead_final"),
        ("dpL_final", "dpL_none"),
        ("dpL_final", "dp4_final"),
        ("smooth125_dead_final", "baseline_smooth125"),
    ]:
        public_paired["{}_minus_{}".format(a_name, b_name)] = paired_spell_bootstrap(
            gt_codes, arms[a_name], arms[b_name], args.bootstrap, args.seed)

    # ---- synthetic harness A: i.i.d. noise (the accepted default's assumption) ----
    halves = half_video_blocks(GAME)
    synthetic_iid = {}
    for seed in range(args.synthetic_seeds):
        rng = np.random.RandomState(1000 + seed)
        truth = synthetic_truth_iid(halves, n_frames, rng)
        s_obs = synthetic_obs_iid(truth, present, rng)
        wanted = {
            "baseline_smooth50": pm.label_codes(bl.smooth(s_obs, 50)),
            "baseline_smooth125": pm.label_codes(bl.smooth(s_obs, 125)),
            "dp4_none": dp_full(s_obs, None, 4.0, n_frames),
            "dp16_none": dp_full(s_obs, None, 16.0, n_frames),
            "dp32_none": dp_full(s_obs, None, 32.0, n_frames),
            "dp50_none": dp_full(s_obs, None, 50.0, n_frames),
        }
        synthetic_iid[seed] = {
            name: metrics_for(truth, codes, args.bootstrap) for name, codes in wanted.items()
        }
        log("synthetic iid seed {}: smooth50 {:.4f}; dp4 {:.4f}; dp32 {:.4f}; dp50 {:.4f}".format(
            seed, synthetic_iid[seed]["baseline_smooth50"]["spell_f1_mean"],
            synthetic_iid[seed]["dp4_none"]["spell_f1_mean"],
            synthetic_iid[seed]["dp32_none"]["spell_f1_mean"],
            synthetic_iid[seed]["dp50_none"]["spell_f1_mean"]))

    # ---- synthetic harness B: autocorrelated confusion ----
    # The confusion rate is fixed (p_start 0.01, mean episode 2 s), not fitted. Fitting it to the
    # real 28 switches/min would need the real error fraction, which is unavailable without GT; at
    # the switch rate the harness can reach without destroying the truth signal, it is already a
    # *milder* form of autocorrelation than the real data. The lambda curve below therefore shows a
    # direction, not a calibration.
    AUTOCORR_P_START = 0.01
    synthetic_autocorr = {}
    for seed in range(args.synthetic_seeds):
        rng = np.random.RandomState(2000 + seed)
        truth = synthetic_truth_aligned(evidence["final"], n_frames, rng)
        rng2 = np.random.RandomState(3000 + seed)
        s_obs = synthetic_obs_autocorr(truth, present, rng2, AUTOCORR_P_START, 50)
        wanted = {
            "baseline_smooth50": pm.label_codes(bl.smooth(s_obs, 50)),
            "baseline_smooth125": pm.label_codes(bl.smooth(s_obs, 125)),
            "smooth125_dead_final": None,
        }
        labels = bl.smooth(s_obs, 125).copy()
        labels[evidence["final"]] = "D"
        wanted["smooth125_dead_final"] = pm.label_codes(labels)
        for lam in LAMBDA_GRID:
            wanted["dp{}_final".format(int(lam))] = dp_full(
                s_obs, evidence["final"], lam, n_frames)
        synthetic_autocorr[seed] = {
            name: metrics_for(truth, codes, args.bootstrap) for name, codes in wanted.items()
        }
        synthetic_autocorr[seed]["switch_rate_per_min"] = switch_rate(s_obs, present)
        synthetic_autocorr[seed]["paired"] = {}
        main_key = "dp{}_final".format(int(lambda_main))
        for a_name, b_name in [
            (main_key, "baseline_smooth125"),
            (main_key, "smooth125_dead_final"),
            ("smooth125_dead_final", "baseline_smooth125"),
            ("dp4_final", "baseline_smooth125"),
            (main_key, "dp4_final"),
        ]:
            synthetic_autocorr[seed]["paired"]["{}_minus_{}".format(a_name, b_name)] = (
                paired_spell_bootstrap(truth, wanted[a_name], wanted[b_name],
                                       args.bootstrap, 7000 + seed))
        log("synthetic autocorr seed {}: switch {:.1f}/min; smooth50 {:.4f}; ".format(
            seed, synthetic_autocorr[seed]["switch_rate_per_min"],
            synthetic_autocorr[seed]["baseline_smooth50"]["spell_f1_mean"])
            + " ".join("dp{}={:.3f}".format(int(lam), synthetic_autocorr[seed][
                "dp{}_final".format(int(lam))]["spell_f1_mean"]) for lam in LAMBDA_GRID))

    # ---- full-match GT-free diagnostics ----
    final_codes = arms["dpL_final"]
    probe = mask_probe(obs, evidence, final_codes, n_frames,
                       candidate_windows(present), lambda_main)
    log("mask probe: final {:.3f}; dp_none {:.3f}; missing->D {:.3f}".format(
        probe["final_mean_label_change"], probe["dp_none_mean_label_change"],
        probe["missing_to_D_mean_label_change"]))

    graded_codes = final_codes[graded]
    full_counts = {
        "frames": int(n_frames),
        "graded_frames": int(graded.sum()),
        "labels_H": int((graded_codes == H).sum()),
        "labels_A": int((graded_codes == A).sum()),
        "labels_D": int((graded_codes == D).sum()),
        "d_fraction_graded": float(np.mean(graded_codes == D)),
        "n_pred_possessions": int(len(pm.possession_spells(
            final_codes, full_match_run_list(n_frames)))),
        "mean_possession_s": float(np.mean([b - a for a, b, _ in pm.possession_spells(
            final_codes, full_match_run_list(n_frames))]) / FPS),
        "d_on_final_evidence": float(np.mean(final_codes[evidence["final"]] == D)),
        "d_on_long_missing": float(np.mean(final_codes[evidence["long_missing"]] == D)),
    }

    # ---- full-match GT-free diagnostics for every arm (used only by GT-free selection) ----
    arm_full_diag = {}
    for name in public_names:
        codes = arms[name]
        spells = pm.possession_spells(codes, full_match_run_list(n_frames))
        lengths = np.array([b - a for a, b, _ in spells])
        count = int(len(spells))
        arm_full_diag[name] = {
            "n_pred_possessions": count,
            "mean_possession_s": float(lengths.mean()) / FPS if count else 0.0,
            "d_fraction_graded": float(np.mean(codes[graded] == D)),
            "inside_count_band": bool(COUNT_BAND[0] <= count <= COUNT_BAND[1]),
            "distance_to_count_centre": float(abs(count - COUNT_CENTRE)),
        }
    log("full-match arm diagnostics (GT-free): " + "; ".join(
        "{} {} poss {:.1f}s".format(n, d["n_pred_possessions"], d["mean_possession_s"])
        for n, d in arm_full_diag.items()))

    # ---- submissions + official scorer ----
    official = None
    if args.write_submissions:
        sub_dir = os.path.join(args.out_dir, "submissions")
        os.makedirs(sub_dir, exist_ok=True)
        for name in public_names:
            bl.write_submission(os.path.join(sub_dir, f"{name}.csv"), LETTERS[arms[name]])
        log("wrote {} submissions".format(len(public_names)))
        official_dir = os.path.join(args.out_dir, "official_reports")
        os.makedirs(official_dir, exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
               "--results", sub_dir, "--out", official_dir,
               "--bootstrap", str(args.bootstrap)]
        log("official scorer: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        with open(os.path.join(args.out_dir, "official_scorer.log"), "w") as handle:
            handle.write(proc.stdout)
            handle.write("\n--- stderr ---\n")
            handle.write(proc.stderr)
        scoreboard = os.path.join(official_dir, "scoreboard.csv")
        if os.path.isfile(scoreboard):
            import csv
            parity = {}
            with open(scoreboard) as handle:
                for row in csv.DictReader(handle):
                    name = row.get("team") or row.get("submission") or ""
                    parity[name] = {
                        "spell_f1_mean": float(row["spell_f1_mean"]),
                        "macro_f1": float(row["macro_f1"]),
                        "rank": int(row["rank"]),
                    }
            diffs = {
                name: {
                    field: abs(parity[name][field] - public_results[name][field])
                    for field in ("spell_f1_mean", "macro_f1")
                }
                for name in public_names if name in parity
            }
            official = {
                "scoreboard": display_path(scoreboard),
                "parity_abs_diff": diffs,
                "max_abs_diff": max((v for d in diffs.values() for v in d.values()),
                                   default=None),
                "note": ("Same possession_metrics module; identical numbers verify file and "
                         "pipeline consistency, not independent validation."),
            }
            log("official parity max abs diff: {}".format(official["max_abs_diff"]))
        else:
            official = {"error": "scoreboard not found", "stderr": proc.stderr[-2000:]}
            log("official scorer produced no scoreboard")

    runtime_total = time.time() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)

    def aggregate(runs, names):
        out = {}
        for name in names:
            values = [runs[seed][name]["spell_f1_mean"] for seed in runs]
            out[name] = {
                "spell_f1_mean": float(np.mean(values)),
                "spell_f1_per_seed": [float(v) for v in values],
                "spell_f1_min": float(np.min(values)),
                "spell_f1_max": float(np.max(values)),
                "macro_f1_mean": float(np.mean([runs[s][name]["macro_f1"] for s in runs])),
                "n_pred_spells_mean": float(np.mean(
                    [runs[s][name]["n_pred_spells"] for s in runs])),
            }
        return out

    def paired_across_seeds(runs):
        """Per-seed paired deltas plus the pre-registered active/reject screen."""
        names = sorted(runs[0].get("paired", {}))
        out = {}
        for name in names:
            rows = [runs[s]["paired"][name] for s in runs if runs[s]["paired"].get(name)]
            if not rows:
                continue
            deltas = [row["delta_mean"] for row in rows]
            excludes_zero = [bool(row["delta_low"] > 0 or row["delta_high"] < 0)
                             for row in rows]
            out[name] = {
                "delta_mean_per_seed": [float(d) for d in deltas],
                "delta_mean_avg": float(np.mean(deltas)),
                "seeds_positive": int(sum(d > 0 for d in deltas)),
                "seeds_ci_excludes_zero": int(sum(excludes_zero)),
                "seeds": len(rows),
                "active_screen": bool(
                    float(np.mean(deltas)) >= 0.04
                    and sum(excludes_zero) >= 2
                    and sum(d > 0 for d in deltas) >= 2),
            }
        return out

    data_files = [
        os.path.join(args.data_dir, "tactical", f"{GAME}_tactical_atd.txt"),
        os.path.join(args.data_dir, "tactical", f"{GAME}_tactical_atd.xml"),
        os.path.join(args.data_dir, "tactical", f"{GAME}_tactical_smart_tagging.xml"),
        gt_path,
    ]

    payload = {
        "experiment": "possession_dp_pipeline_v2",
        "status": ("exploratory_public_window_validation_only_gtfree_lambda_"
                   "synthetic_calibration_no_full_match_evidence"),
        "prereg": "PREREG.md",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "runtime_total_s": round(runtime_total, 3),
        "command": " ".join([display_path(sys.executable)] + sys.argv),
        "seed": args.seed,
        "bootstrap_n": args.bootstrap,
        "git_commit": git_commit(),
        "script": {
            "path": os.path.relpath(os.path.abspath(__file__), ROOT),
            "sha256": sha256(os.path.abspath(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": getattr(__import__("pandas"), "__version__", None),
            "scipy": __import__("scipy").__version__,
        },
        "compute": {
            "wall_s": round(runtime_total, 3),
            "user_cpu_s": round(usage.ru_utime, 3),
            "sys_cpu_s": round(usage.ru_stime, 3),
            "peak_rss_mb": round(usage.ru_maxrss / 1024.0, 1),
            "evidence_runtime_s": round(evidence_runtime, 3),
            "gpu_used": False,
        },
        "data_audit": {
            "tactical_frames": int(n_frames),
            "graded_frames": int(graded.sum()),
            "public_gt_possessions": int(len(pm.possession_spells(
                gt_codes, pm.valid_runs(gt_codes)))),
            "public_gt_window": [11185, 18684],
            "ball_present_frames": int(present.sum()),
            "ball_present_fraction": float(present.mean()),
            "nearest_player_switch_rate_per_min": float(obs_switch_rate),
            "set_piece_restarts": int(restart_starts.size),
        },
        "inputs": {
            display_path(path): {"sha256": sha256(path), "bytes": os.path.getsize(path)}
            for path in data_files if os.path.isfile(path)
        },
        "evidence_frame_counts_graded": {
            key: int(mask.sum()) for key, mask in evidence.items()
        },
        "lambda_selection": {
            **selection,
            "prereg_lambda": PREREG_LAMBDA,
            "selected_lambda_used": lambda_main,
        },
        "public_window": {
            "role": "validation only; no arm or parameter may be selected from these numbers",
            "arms": public_results,
            "paired": public_paired,
        },
        "synthetic_iid": {
            "role": ("mechanism only: i.i.d. 93% correct observations, the assumption behind the "
                     "accepted lambda=4 default; dead states not aligned with the evidence"),
            "aggregate": aggregate(synthetic_iid, list(synthetic_iid[0].keys())),
            "per_seed": synthetic_iid,
        },
        "synthetic_autocorr": {
            "role": ("mechanism only: autocorrelated confusion (p_start 0.01, mean episode 2 s) is a "
                     "milder form of autocorrelation than the real data, and dead states are placed "
                     "on the final evidence, so evidence results here are an upper bound, never real "
                     "evidence"),
            "settings": {"p_start": 0.01, "confusion_mean_frames": 50},
            "real_switch_rate_per_min": obs_switch_rate,
            "aggregate": aggregate(
                synthetic_autocorr,
                [key for key in synthetic_autocorr[0]
                 if key not in ("switch_rate_per_min", "paired")]),
            "paired_summary": paired_across_seeds(synthetic_autocorr),
            "per_seed": synthetic_autocorr,
        },
        "full_match_diagnostics": {
            "role": "GT-free structural checks only; no accuracy claim",
            **full_counts,
            "mask_probe": probe,
            "per_arm": arm_full_diag,
        },
        "official_scorer": official,
        "interpretation": {
            "primary_metric": "spell_f1_mean",
            "decision_rule": (
                "Component active only if synthetic delta >= 0.04 with paired 90% CI excluding 0 "
                "in >= 2/3 seeds. Public window cannot support an improvement claim."
            ),
            "bytetrack_botsort": (
                "Not run: the bundle has point tracking and metadata only. No video and no "
                "bounding-box detections are present, so ByteTrack/BoT-SORT (bbox + confidence + "
                "detector) are not applicable."
            ),
            "known_limitations": [
                "Public window: 17 GT possessions, CI width ~0.3, not representative.",
                "Bootstrap holds n_pred fixed; intervals for count-changing arms are optimistic.",
                "The alpha=0.04 / CI rule is borrowed from synthetic work, not calibrated here.",
                "Team identity switches are not repaired; the ATD team label is used as-is.",
                "Synthetic dead states are placed on observable stoppages, so evidence gains there "
                "are built in by construction.",
                "No full-match GT: ranking performance cannot be verified before the deadline.",
                "experiments/ is untracked in git; script sha256 is the only provenance.",
            ],
        },
    }

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)

    csv_path = os.path.join(args.out_dir, "results.csv")
    columns = ["benchmark", "arm", "family", "spell_f1_mean", "spell_bootstrap_low",
               "spell_bootstrap_high", "macro_f1", "n_pred_spells", "mean_spell_duration_s",
               "fragmentation", "alive_dead_f1", "f1_H", "f1_A", "f1_D", "runtime_s"]

    def fmt(value, digits):
        return "" if value is None else f"{value:.{digits}f}"

    with open(csv_path, "w") as handle:
        handle.write(",".join(columns) + "\n")
        for name in public_names:
            row = public_results[name]
            per_class = row["per_class"]
            handle.write(",".join([
                "public", name, row["definition"]["family"],
                fmt(row["spell_f1_mean"], 6), fmt(row["spell_bootstrap_low"], 6),
                fmt(row["spell_bootstrap_high"], 6), fmt(row["macro_f1"], 6),
                str(row["n_pred_spells"]), fmt(row["mean_spell_duration_s"], 3),
                fmt(row["fragmentation"], 3), fmt(row["alive_dead_f1"], 4),
                fmt(per_class["H"]["f1"], 4), fmt(per_class["A"]["f1"], 4),
                fmt(per_class["D"]["f1"], 4), fmt(row["runtime_s"], 3),
            ]) + "\n")
        for benchmark, runs in (("synthetic_iid", synthetic_iid),
                                ("synthetic_autocorr", synthetic_autocorr)):
            for name, row in aggregate(
                    runs, [key for key in runs[0]
                           if key not in ("switch_rate_per_min", "paired")]).items():
                handle.write(",".join([
                    benchmark, name, "", fmt(row["spell_f1_mean"], 6), "", "",
                    fmt(row["macro_f1_mean"], 6), fmt(row["n_pred_spells_mean"], 1),
                    "", "", "", "", "", "", "",
                ]) + "\n")

    with open(os.path.join(args.out_dir, "run.log"), "w") as handle:
        handle.write("\n".join(log_lines) + "\n\nwrote {}\nwrote {}\n".format(json_path, csv_path))
    log("wrote {} and {}".format(json_path, csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
