#!/usr/bin/env python3
"""007: parse IDSSE positions XML into compact npz.

IDSSE (Bassek et al. 2025) ships one XML per match with one FrameSet per tracked person and one
FrameSet for the ball. Every frame carries the player position in metres (pitch 105x68, origin at
the centre), speed in m/s and an observation mask M. The ball FrameSet additionally carries
BallStatus (0 = inactive, 1 = active) and BallPossession (1 = home, 2 = away), which are the
labels this run needs. Video is not part of the dataset.

Nothing here trains on labels; this only converts the published XML into arrays so cycles can be
cheap. Output goes under data/external/idsee/parsed/ (gitignored).
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DATA = os.path.join(ROOT, "..", "data", "external", "idsee")


def parse_positions(path):
    """Return arrays for one match. Ball FrameSet is the one that carries BallStatus."""
    sections = {}
    person_order = []
    person_meta = {}
    ball = {"n": [], "x": [], "y": [], "s": [], "status": [], "possession": []}
    saw_ball = False

    context = ET.iterparse(path, events=("end",))
    for _, element in context:
        if element.tag != "FrameSet":
            continue
        person = element.get("PersonId")
        team = element.get("TeamId")
        section = element.get("GameSection")
        frames = list(element)
        if not frames:
            element.clear()
            continue
        first = frames[0]
        is_ball = first.get("BallStatus") is not None
        if is_ball:
            saw_ball = True
            for frame in frames:
                ball["n"].append(int(frame.get("N")))
                ball["x"].append(float(frame.get("X")))
                ball["y"].append(float(frame.get("Y")))
                ball["s"].append(float(frame.get("S")))
                ball["status"].append(int(frame.get("BallStatus")))
                ball["possession"].append(int(frame.get("BallPossession")))
        else:
            key = (person, section)
            if key not in person_meta:
                person_order.append(key)
                person_meta[key] = {"person": person, "team": team,
                                    "n": [], "x": [], "y": [], "s": [], "m": []}
            meta = person_meta[key]
            for frame in frames:
                meta["n"].append(int(frame.get("N")))
                meta["x"].append(float(frame.get("X")))
                meta["y"].append(float(frame.get("Y")))
                meta["s"].append(float(frame.get("S")))
                meta["m"].append(int(frame.get("M")))
        element.clear()

    if not saw_ball:
        raise SystemExit("no ball FrameSet found in {}".format(path))

    n_min = min(min(v["n"]) for v in person_meta.values())
    n_max = max(max(v["n"]) for v in person_meta.values())
    n_max = max(n_max, max(ball["n"]))
    count = n_max - n_min + 1
    frames = np.arange(n_min, n_min + count, dtype=np.int32)

    ball_x = np.full(count, np.nan, dtype=np.float32)
    ball_y = np.full(count, np.nan, dtype=np.float32)
    ball_s = np.full(count, np.nan, dtype=np.float32)
    ball_status = np.full(count, -1, dtype=np.int8)
    ball_possession = np.full(count, -1, dtype=np.int8)
    idx = np.array(ball["n"], dtype=np.int64) - n_min
    ball_x[idx] = ball["x"]
    ball_y[idx] = ball["y"]
    ball_s[idx] = ball["s"]
    ball_status[idx] = ball["status"]
    ball_possession[idx] = ball["possession"]

    people = [v for v in person_meta.values()]
    p_count = len(people)
    player_x = np.full((p_count, count), np.nan, dtype=np.float32)
    player_y = np.full((p_count, count), np.nan, dtype=np.float32)
    player_s = np.full((p_count, count), np.nan, dtype=np.float32)
    player_m = np.zeros((p_count, count), dtype=bool)
    person_ids = []
    team_ids = []
    sections_out = []
    for k, meta in enumerate(people):
        ids = np.array(meta["n"], dtype=np.int64) - n_min
        player_x[k, ids] = meta["x"]
        player_y[k, ids] = meta["y"]
        player_s[k, ids] = meta["s"]
        player_m[k, ids] = np.array(meta["m"], dtype=bool)
        person_ids.append(meta["person"])
        team_ids.append(meta["team"])

    return {
        "frames": frames,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed": ball_s,
        "ball_status": ball_status,
        "ball_possession": ball_possession,
        "player_x": player_x,
        "player_y": player_y,
        "player_speed": player_s,
        "player_mask": player_m,
        "person_ids": np.array(person_ids),
        "team_ids": np.array(team_ids),
    }


def summarise(arrays):
    status = arrays["ball_status"]
    graded = status >= 0
    dead = status == 0
    return {
        "frames": int(len(status)),
        "graded_frames": int(graded.sum()),
        "dead_frames": int(dead.sum()),
        "dead_fraction": float(dead.sum() / max(1, graded.sum())),
        "ball_present_fraction": float(np.isfinite(arrays["ball_x"]).mean()),
        "possessions": int(np.sum(np.diff(arrays["ball_possession"]) != 0)),
        "players": int(len(arrays["person_ids"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--match", action="append", default=None)
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "parsed")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.data_dir) if "positions_raw_observed" in f)
    if args.match:
        files = [f for f in files if any(m in f for m in args.match)]

    index = {}
    for name in files:
        path = os.path.join(args.data_dir, name)
        match_id = name.split("DFL-MAT-")[1].split(".")[0]
        target = os.path.join(out_dir, match_id + ".npz")
        if os.path.isfile(target):
            print("skip", match_id, "(parsed)", flush=True)
            with np.load(target) as cached:
                index[match_id] = summarise(cached)
            continue
        started = time.time()
        arrays = parse_positions(path)
        np.savez_compressed(target, **arrays)
        index[match_id] = dict(summarise(arrays), seconds=round(time.time() - started, 1))
        print(match_id, index[match_id], flush=True)

    with open(os.path.join(out_dir, "index.json"), "w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
