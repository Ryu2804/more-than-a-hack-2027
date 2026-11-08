#!/usr/bin/env python3
"""011 phase 2b: does the C03c detector spec survive BAR-ATH geometry?

C03c measured the value of a recovered ball on IDSSE and left a spec: position noise <= 2 m and
per-frame detection p = 1.0. BAR-ATH has no full-match dead/alive ground truth, so cue F1 cannot be
computed there. What can be tested is structural: with 2 m of position noise, does the frozen cue
`out_or_slow2` change? If the cue mask and the nearest player are almost invariant, the spec is
compatible with BAR-ATH geometry. This does not prove that F1 transfers.

Arms: ball_true and ball_noisy_s{0.5,1,2,3} (N(0, sigma) per axis, p = 1). No masking; p was
measured on IDSSE. Referee/team mapping is not needed, so only player positions and speeds are kept.

Outputs: cue IoU, cue frame and run ratios, nearest-player identity agreement, and the change in
nearest-player speed. IoU and agreement get a block bootstrap CI90 (250-frame blocks, 400 reps).

No public-window numbers are read or written. No candidate is selected.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

FPS = 25
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
MARGIN = 0.1
SLOW_MS = 2.0
OUT_RUN = 15
SLOW_RUN = 25
SIGMAS = (0.5, 1.0, 2.0, 3.0)
REPLICATES = 3
BLOCK = 250
BOOTSTRAP = 400
SEED = 20261108
MAX_SLOTS = 64
CHUNK = 50000


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def missing_runs(present):
    return np.array([stop - start for start, stop in runs_of(~present)], dtype=np.int64)


def build_mask(n, lengths, target_frames, rng):
    """Non-overlapping runs with lengths drawn from `lengths`, covering ~target_frames."""
    chosen = []
    total = 0
    while total < target_frames:
        length = int(rng.choice(lengths))
        if length <= 0:
            continue
        chosen.append(min(length, n))
        total += chosen[-1]
    if total > target_frames:
        chosen[-1] -= total - target_frames
        if chosen[-1] <= 0:
            chosen.pop()
    total = sum(chosen)
    gaps = rng.exponential(size=len(chosen) + 1)
    gaps = gaps / gaps.sum() * max(0, n - total)
    gaps = np.floor(gaps).astype(np.int64)
    gaps[0] += max(0, n - total) - int(gaps.sum())
    mask = np.zeros(n, dtype=bool)
    position = int(gaps[0])
    for index, length in enumerate(chosen):
        end = min(n, position + length)
        mask[position:end] = True
        position = end + int(gaps[index + 1])
    return mask


def load_atd(data_dir, n_frames, game):
    """Ball and all player slots from the tactical ATD, in metres, corner origin."""
    path = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(game))
    px = np.full((MAX_SLOTS, n_frames), np.nan, dtype=np.float32)
    py = np.full((MAX_SLOTS, n_frames), np.nan, dtype=np.float32)
    ps = np.full((MAX_SLOTS, n_frames), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    max_slot = 0

    with open(path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                continue
            if frame < 0 or frame >= n_frames:
                continue
            ball = parts[2]
            if not ball or ball[0] == "N":
                continue
            try:
                coords = ball.split(",")
                bx[frame] = float(coords[0]) * PITCH_LENGTH
                by[frame] = float(coords[1]) * PITCH_WIDTH
            except (ValueError, IndexError):
                continue
            present[frame] = True
            groups = parts[1].split(";")
            max_slot = max(max_slot, len(groups))
            for slot, group in enumerate(groups):
                if slot >= MAX_SLOTS or not group or group[0] == "N":
                    continue
                fields = group.split(",")
                if len(fields) < 3:
                    continue
                try:
                    px[slot, frame] = float(fields[0]) * PITCH_LENGTH
                    py[slot, frame] = float(fields[1]) * PITCH_WIDTH
                    ps[slot, frame] = float(fields[2])
                except ValueError:
                    continue
    return (bx, by, present, px[:max_slot], py[:max_slot], ps[:max_slot], max_slot)


def nearest(px, py, speed, ball_x, ball_y):
    """Nearest observed player per frame; -1 where the ball or every player is missing."""
    n = len(ball_x)
    index = np.full(n, -1, dtype=np.int16)
    best_speed = np.full(n, np.nan, dtype=np.float32)
    finite_ball = np.isfinite(ball_x) & np.isfinite(ball_y)
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        dx = px[:, start:stop] - ball_x[start:stop][None, :]
        dy = py[:, start:stop] - ball_y[start:stop][None, :]
        distance = np.hypot(dx, dy)
        usable = np.isfinite(distance) & finite_ball[None, start:stop]
        distance = np.where(usable, distance, np.inf)
        local = np.argmin(distance, axis=0)
        best = distance[local, np.arange(stop - start)]
        answered = np.isfinite(best)
        index[start:stop][answered] = local[answered]
        best_speed[start:stop][answered] = speed[local[answered], np.arange(stop - start)[answered]]
    return index, best_speed


def cue_of(ball_x, ball_y, present, index, speed, graded):
    answered = index >= 0
    outside = (present &
               ((ball_x < -MARGIN) | (ball_x > PITCH_LENGTH + MARGIN) |
                (ball_y < -MARGIN) | (ball_y > PITCH_WIDTH + MARGIN)))
    out_cue = run_mask(outside & graded, OUT_RUN)
    slow = answered & np.isfinite(speed) & (speed < SLOW_MS)
    slow_cue = run_mask(slow & graded, SLOW_RUN)
    return (out_cue | slow_cue) & graded, answered


def block_counts(cue_arm, cue_base, block_ids):
    """Per-block tp/fp/fn for arm vs base; block_ids is the graded-ordinal block index."""
    n_blocks = int(block_ids.max()) + 1
    tp = np.zeros(n_blocks, dtype=np.float64)
    fp = np.zeros(n_blocks, dtype=np.float64)
    fn = np.zeros(n_blocks, dtype=np.float64)
    np.add.at(tp, block_ids[cue_arm & cue_base], 1)
    np.add.at(fp, block_ids[cue_arm & ~cue_base], 1)
    np.add.at(fn, block_ids[~cue_arm & cue_base], 1)
    return tp, fp, fn


def iou_from_blocks(tp, fp, fn, takes=None):
    if takes is not None:
        tp, fp, fn = tp[takes], fp[takes], fn[takes]
    tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    return float(tp / max(1.0, tp + fp + fn))


def accuracy_from_blocks(correct, total, takes=None):
    if takes is not None:
        correct, total = correct[takes], total[takes]
    return float(correct.sum() / max(1.0, total.sum()))


def bootstrap_iou(tp, fp, fn, reps, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        takes = rng.integers(0, len(tp), size=len(tp))
        values.append(iou_from_blocks(tp, fp, fn, takes))
    return [float(v) for v in np.percentile(values, [5, 95])]


def bootstrap_accuracy(correct, total, reps, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        takes = rng.integers(0, len(total), size=len(total))
        values.append(accuracy_from_blocks(correct, total, takes))
    return [float(v) for v in np.percentile(values, [5, 95])]


def main():
    started = time.time()
    data_dir = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data")
    m2 = _load(os.path.join(ROOT, "experiments", "002_possession_dp", "run.py"), "m2_phase2b")
    n_frames = m2.source_frames("tactical", m2.GAME)
    graded = m2.graded_mask(n_frames)
    graded_positions = np.flatnonzero(graded)

    bx, by, present, px, py, ps, slots = load_atd(data_dir, n_frames, m2.GAME)
    index_true, speed_true = nearest(px, py, ps, bx, by)
    cue_true, answered_true = cue_of(bx, by, present, index_true, speed_true, graded)

    block_ids = np.arange(len(graded_positions)) // BLOCK
    n_blocks = int(block_ids.max()) + 1
    block_of_frame = np.full(n_frames, -1, dtype=np.int32)
    block_of_frame[graded_positions] = block_ids
    cue_true_graded = cue_true[graded_positions]

    result = {
        "meta": {
            "match": m2.GAME,
            "frames": int(n_frames),
            "graded_frames": int(graded.sum()),
            "ball_present_frames": int(present.sum()),
            "player_slots": int(slots),
            "sigmas": list(SIGMAS),
            "replicates": REPLICATES,
            "block_frames": BLOCK,
            "bootstrap_reps": BOOTSTRAP,
            "p": 1.0,
        },
        "bootstrap": BOOTSTRAP,
        "arms": {
            "ball_true": {
                "cue_frames": int(cue_true.sum()),
                "cue_runs": int(len(runs_of(cue_true))),
                "answered_frames": int(answered_true.sum()),
            }
        },
    }

    # Common random numbers across sigma: the same draw for every sigma per replicate.
    draws = []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED + 7919 * rep)
        draws.append((rng.standard_normal(n_frames), rng.standard_normal(n_frames)))

    # Secondary, exposure-matched sensitivity: noise only where a BAR-ATH-like mask removes the ball.
    lengths = missing_runs(present)
    target = int(round(float((~present).mean()) * n_frames))
    masks = []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED + 104729 * rep)
        masks.append(build_mask(n_frames, lengths, target, rng))
    result["meta"]["mask_target_fraction"] = float((~present).mean())

    for exposure in ("all", "mask"):
        for sigma in SIGMAS:
            ious, accs, frames, run_counts = [], [], [], []
            speed_deltas = []
            tp_sum = np.zeros(n_blocks, dtype=np.float64)
            fp_sum = np.zeros(n_blocks, dtype=np.float64)
            fn_sum = np.zeros(n_blocks, dtype=np.float64)
            acc_correct = np.zeros(n_blocks, dtype=np.float64)
            acc_total = np.zeros(n_blocks, dtype=np.float64)

            for rep, (ex, ey) in enumerate(draws):
                noisy_x = bx + (sigma * ex).astype(np.float32)
                noisy_y = by + (sigma * ey).astype(np.float32)
                if exposure == "mask":
                    m = masks[rep]
                    noisy_x = np.where(m, noisy_x, bx)
                    noisy_y = np.where(m, noisy_y, by)
                index, speed = nearest(px, py, ps, noisy_x, noisy_y)
                cue, answered = cue_of(noisy_x, noisy_y, present, index, speed, graded)

                use = answered & answered_true
                frame_use = np.flatnonzero(use & graded)
                same = index[frame_use] == index_true[frame_use]
                bidx = block_of_frame[frame_use]
                correct = np.zeros(n_blocks, dtype=np.float64)
                total = np.zeros(n_blocks, dtype=np.float64)
                np.add.at(total, bidx, 1)
                np.add.at(correct, bidx[same], 1)

                tp, fp, fn = block_counts(cue[graded_positions], cue_true_graded, block_ids)
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                acc_correct += correct
                acc_total += total

                ious.append(iou_from_blocks(tp, fp, fn))
                accs.append(accuracy_from_blocks(correct, total))
                frames.append(int(cue.sum()) / max(1, int(cue_true.sum())))
                run_counts.append(len(runs_of(cue)) / max(1, len(runs_of(cue_true))))
                delta = np.abs(speed[use] - speed_true[use])
                delta = delta[np.isfinite(delta)]
                if len(delta):
                    speed_deltas.append(delta)

            tp_sum /= REPLICATES
            fp_sum /= REPLICATES
            fn_sum /= REPLICATES
            acc_correct /= REPLICATES
            acc_total /= REPLICATES
            name = ("ball_noisy_s{:g}" if exposure == "all" else "ball_masked_noisy_s{:g}").format(sigma)
            speed_delta = np.concatenate(speed_deltas) if speed_deltas else np.array([0.0])
            offset = 0 if exposure == "all" else 500
            result["arms"][name] = {
                "exposure": exposure,
                "cue_frames": int(round(np.mean(frames) * int(cue_true.sum()))),
                "cue_runs": int(round(np.mean(run_counts) * len(runs_of(cue_true)))),
                "cue_iou": float(np.mean(ious)),
                "cue_iou_ci90": bootstrap_iou(tp_sum, fp_sum, fn_sum, BOOTSTRAP,
                                              SEED + offset + int(sigma * 10)),
                "identity_agreement": float(np.mean(accs)),
                "identity_ci90": bootstrap_accuracy(acc_correct, acc_total, BOOTSTRAP,
                                                    SEED + 100 + offset + int(sigma * 10)),
                "cue_frame_ratio": float(np.mean(frames)),
                "cue_run_ratio": float(np.mean(run_counts)),
                "cue_frame_ratio_range": [float(np.min(frames)), float(np.max(frames))],
                "cue_run_ratio_range": [float(np.min(run_counts)), float(np.max(run_counts))],
                "nearest_speed_abs_delta_median": float(np.median(speed_delta)),
                "nearest_speed_abs_delta_p90": float(np.percentile(speed_delta, 90)),
                "replicates": REPLICATES,
            }
            print(name, "iou", round(float(np.mean(ious)), 4),
                  "identity", round(float(np.mean(accs)), 4),
                  "frame_ratio", round(float(np.mean(frames)), 4),
                  "run_ratio", round(float(np.mean(run_counts)), 4), flush=True)

    s2 = result["arms"]["ball_noisy_s2"]
    checks = {
        "iou_ok": s2["cue_iou"] >= 0.95,
        "identity_ok": s2["identity_agreement"] >= 0.95,
        "frame_ratio_ok": abs(s2["cue_frame_ratio"] - 1.0) <= 0.05,
        "run_ratio_ok": abs(s2["cue_run_ratio"] - 1.0) <= 0.10,
    }
    fail = (s2["cue_iou"] < 0.90 or s2["identity_agreement"] < 0.90
            or abs(s2["cue_run_ratio"] - 1.0) > 0.25)
    result["selection"] = {
        "sigma_m": 2.0,
        "checks": checks,
        "supported": bool(all(checks.values())),
        "not_supported": bool(fail),
        "inconclusive": bool(not all(checks.values()) and not fail),
    }
    # Post-hoc sensitivity, not part of the pre-registered verdict: noise only on a BAR-ATH-like mask.
    s2m = result["arms"]["ball_masked_noisy_s2"]
    result["sensitivity_exposure_matched"] = {
        "arm": "ball_masked_noisy_s2",
        "note": "pasca-hoc; noise hanya pada mask 43% seperti bola hilang BAR-ATH, bukan semua frame",
        "cue_iou": s2m["cue_iou"],
        "identity_agreement": s2m["identity_agreement"],
        "cue_frame_ratio": s2m["cue_frame_ratio"],
        "cue_run_ratio": s2m["cue_run_ratio"],
    }
    result["meta"]["exposure_note"] = (
        "primary arms noise every frame with a ball; ball_masked_noisy_* is a post-hoc sensitivity "
        "with noise only on a synthetic BAR-ATH-like missing mask")
    result["runtime_s"] = time.time() - started

    out_path = os.path.join(HERE, "out", "phase2b.json")
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("wrote", out_path, "runtime", round(result["runtime_s"], 1), "s")
    print("selection", result["selection"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
