#!/usr/bin/env python3
"""Migrate farras HGB+Viterbi pipeline to per-match IDSSE predictions (best-effort)."""

import os
import sys
import time
from pathlib import Path

import numpy as np

FARRAS = "/tmp/opencode/farras-worktree/experiments/exp_farras"
sys.path.insert(0, os.path.join(FARRAS, "wave1"))
sys.path.insert(0, os.path.join(FARRAS, "wave2"))
sys.path.insert(0, "/tmp/opencode/jeremy-worktree/experiments/exp_1")
sys.path.insert(0, "/tmp/opencode/jeremy-worktree/experiments/exp_1/judge")
os.environ.setdefault("MTAH_REPO", "/home/rasyaarya/lomba/more_than_a_hack/more-than-a-hack-2027")

import data as fdata  # noqa: E402
import support as fsupport  # noqa: E402
import exp1_common as jc  # noqa: E402
import experiment_data as jd  # noqa: E402

# load w08 method
spec = __import__("importlib.util", fromlist=["x"]).spec_from_file_location("w08", os.path.join(FARRAS, "wave2", "methods", "w08_symmetric_boost.py"))
w08 = __import__("importlib.util", fromlist=["x"]).module_from_spec(spec)
spec.loader.exec_module(w08)

OUT = Path(os.environ.get("FARRAS_OUT", "/tmp/opencode/farras_migration_out"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pred").mkdir(exist_ok=True)
(OUT / "gt").mkdir(exist_ok=True)

MATCH_IDS = list(jc.MATCH_IDS)
LABELS = {0: "H", 1: "A", 2: "D", 3: "X"}


def period_blocks(period):
    return fdata.period_blocks(period)


def load_match(name):
    m = jc.load_degraded(name)
    # team mapping from rasya npz
    import numpy as np
    side = np.asarray(m.teams[:m.players.shape[1]], dtype=np.int8)
    # normalize to 0/1
    uniq = np.unique(side)
    if len(uniq) == 2:
        side = np.where(side == uniq[0], 0, 1).astype(np.int8)
    else:
        side = None
    return m, side


def convert(match, team_side):
    players_m = match.players[:, :, :2].astype(np.float32)  # meters
    ball_m = match.ball.astype(np.float32)
    pitch = np.array([105.0, 68.0], dtype=np.float32)
    players = players_m / pitch[None, None, :]
    ball = ball_m / pitch[None, :]
    half = match.half
    labels = match.labels.astype(np.int8)
    y25 = labels.copy()
    y25[y25 == 3] = -1

    blocks25 = period_blocks(half)
    sampled = np.concatenate([np.arange(a, b, 5) for a, b in blocks25]) if blocks25 else np.array([], dtype=int)
    players_s = players[sampled]
    ball_s = ball[sampled]
    # distances in meters
    dist = np.linalg.norm((players_s - ball_s[:, None, :]) * np.array([105.3, 68.0]), axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    n_slots = players_s.shape[1]
    if team_side is None:
        dhome = np.full(len(sampled), 30.0, dtype=np.float32)
        daway = np.full(len(sampled), 30.0, dtype=np.float32)
    else:
        home_slots = np.flatnonzero(team_side == 0)
        away_slots = np.flatnonzero(team_side == 1)
        dhome = dist[:, home_slots].min(axis=1) if len(home_slots) else np.full(len(sampled), 30.0)
        daway = dist[:, away_slots].min(axis=1) if len(away_slots) else np.full(len(sampled), 30.0)
    distances = np.column_stack([dhome, daway]).astype(np.float32)

    blocks = []
    offset = 0
    for a, b in blocks25:
        size = len(range(a, b, 5))
        blocks.append((offset, offset + size))
        offset += size

    X = fdata.features(players_s, ball_s, distances, blocks)
    Xocc = X.copy()
    y = y25[sampled].astype(np.int8)

    return dict(X=X, X_occluded=Xocc, y=y, y25=y25, sampled=sampled,
                blocks=np.array(blocks), blocks25=np.array(blocks25), n_frames=len(half), fps=5)


def write_csv(path, codes):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label"])
        for i, c in enumerate(codes):
            w.writerow([i, LABELS[int(c)]])


def training(datasets):
    xs, ys, blocks, offset = [], [], [], 0
    for dataset in datasets:
        for key in ("X", "X_occluded"):
            xs.append(dataset[key])
            ys.append(dataset["y"])
            blocks.extend((int(a) + offset, int(b) + offset) for a, b in dataset["blocks"])
            offset += len(dataset[key])
    return dict(X=np.concatenate(xs), y=np.concatenate(ys), blocks=blocks, fps=5)


def prediction_input(dataset, regime="X"):
    return dict(X=dataset[regime].copy(), blocks=[tuple(map(int, b)) for b in dataset["blocks"]], fps=5)


def checked_predict(module, train, test, config):
    model = module.fit(train, dict(config))
    pred = np.asarray(module.predict(model, test, dict(config)))
    if pred.shape != (len(test["X"]),) or not np.isin(pred, [0, 1, 2]).all():
        raise ValueError("Invalid prediction")
    return pred.astype(np.int8)


def expand(prediction, data):
    return fdata.expand(prediction, data)


CONFIG = {"name": "cost6", "switch_cost": 6}


def main():
    datasets = {}
    matches = {}
    for name in MATCH_IDS:
        m, side = load_match(name)
        matches[name] = m
        datasets[name] = convert(m, side)
        print("converted", name, "frames", len(m.labels), "samples", len(datasets[name]["X"]), flush=True)

    for target in MATCH_IDS:
        out_pred = OUT / "pred" / f"{target}_pred.csv"
        if out_pred.exists():
            print(target, "already done", flush=True)
            continue
        t0 = time.time()
        train_names = [n for n in MATCH_IDS if n != target][:2]
        train_ds = [datasets[n] for n in train_names]
        target_ds = datasets[target]
        train = training(train_ds)
        pred5 = checked_predict(w08, train, prediction_input(target_ds, "X"), CONFIG)
        full = expand(pred5, target_ds)
        # align to rasya frame range
        m = matches[target]
        base = 10000
        n_frames = int(m.frame.max()) - base + 1
        aligned = np.full(n_frames, 3, dtype=np.int8)
        # full is aligned to m.frame order (25fps, includes separator)
        for idx, frame_id in enumerate(m.frame):
            if base <= frame_id < base + n_frames:
                aligned[int(frame_id) - base] = full[idx]
        write_csv(out_pred, aligned)
        print(target, "done in", round(time.time() - t0, 1), flush=True)

    print("all farras predictions written", flush=True)


if __name__ == "__main__":
    main()