"""Shared helpers for experiments. Each experiment owns experiments/<id>_<name>/ and writes only to its out/."""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
CACHE = os.path.join(HERE, "_cache")  # gitignored; parsed tracking so reruns skip the ~1 min parse


def cached(name, build):
    """Return the numpy array cached as _cache/<name>.npy, computing it with build() on first use."""
    path = os.path.join(CACHE, name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(CACHE, exist_ok=True)
    arr = build()
    np.save(path, arr)
    return arr


def save(exp_file, name, labels):
    """Write out/<name>.csv next to the experiment's run.py (frame,label, one row per frame)."""
    from baseline import write_submission
    out = os.path.join(os.path.dirname(os.path.abspath(exp_file)), "out")
    return write_submission(os.path.join(out, name + ".csv"), labels)


def score(exp_file, *extra):
    """Validate and score everything in the experiment's out/ on the public GT -> out/reports/."""
    out = os.path.join(os.path.dirname(os.path.abspath(exp_file)), "out")
    run = os.path.join(ROOT, "evaluation", "run_evaluation.py")
    subprocess.run([sys.executable, run, "--results", out, "--out", os.path.join(out, "reports"),
                    *extra], check=False)


def load_atd_arrays():
    """Whole tactical ATD as arrays, cached: players (N,46,3) [x,y,speed m/s], ball (N,3), team (46,) 0=H 1=A.

    x,y stay normalised 0-1; row i is video frame i. Missing = NaN.
    """
    from baseline import team_of_each_track
    from games import asset_paths, source_frames
    paths = asset_paths()
    n = source_frames("tactical")

    def build():
        players = np.full((n, 46, 3), np.nan, np.float32)
        ball = np.full((n, 3), np.nan, np.float32)
        with open(paths["atd_raw"]) as f:
            for line in f:
                p = line.split(":")
                if len(p) != 3:
                    continue
                i = int(p[0]) - 1
                players[i] = np.array(p[1].replace(";", ",").split(","), dtype=np.float32).reshape(46, 3)
                ball[i] = np.array(p[2].split(","), dtype=np.float32)
        return np.concatenate([players.reshape(n, -1), ball], axis=1)

    flat = cached("atd_full", build)
    return flat[:, :-3].reshape(n, 46, 3), flat[:, -3:], team_of_each_track(paths["atd_meta"])
