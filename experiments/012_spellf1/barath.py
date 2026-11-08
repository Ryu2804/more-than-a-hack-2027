#!/usr/bin/env python3
"""C06: transfer the frozen C05 touch-hold decoder to BAR-ATH.

C05 selected r = 0.8 m, minlen = 30 frames on IDSSE leave-one-match-out. That configuration is
frozen here: nothing is tuned on BAR-ATH. The only BAR-ATH labels available are the public 5-minute
window (frames 11185-18684), so it is a sanity veto, never a selector.

Dead is not predicted by C05 (IDSSE used oracle dead). Three dead arms are reported:
`missing` (ball not tracked, the published baseline rule), `cue` (out_or_slow2 from 011), and
`oracle` (public GT D inside the window; a ceiling, not deployable).

No submission CSV is written and no window number is used to choose anything.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "games"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))
import baseline as B  # noqa: E402
import phase2b as P2B  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402

GAME = "BAR-ATH"
FPS = 25
R_M = 0.8
MINLEN = 30
WINDOW = (11185, 18684)
SMOOTH_W = 50
CHUNK = 50000


class _FrameCounter:
    """Minimal stand-in for the repo's games.source_frames signature."""

    def __call__(self, kind, game):
        import games
        return games.source_frames(kind, game)


def source_frames(kind, game):
    import games
    return games.source_frames(kind, game)


def load_context(data_dir):
    n = source_frames("tactical", GAME)
    bx, by, present, px, py, ps, _max_slot = P2B.load_atd(data_dir, n, GAME)
    team_slots = B.team_of_each_track(
        os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME)))

    dmin = np.full(n, np.nan, dtype=np.float32)
    index = np.full(n, -1, dtype=np.int16)
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        dx = px[:, start:stop] - bx[start:stop][None, :]
        dy = py[:, start:stop] - by[start:stop][None, :]
        distance = np.hypot(dx, dy)
        usable = np.isfinite(distance) & present[None, start:stop]
        distance = np.where(usable, distance, np.inf)
        best = distance.argmin(axis=0)
        value = distance[best, np.arange(stop - start)]
        ok = np.isfinite(value)
        index[start:stop] = np.where(ok, best, -1)
        dmin[start:stop] = np.where(ok, value, np.nan)

    team = np.full(n, -1, dtype=np.int8)
    ok = index >= 0
    team[ok] = team_slots[index[ok]]

    speed = np.full(n, np.nan, dtype=np.float32)
    ok2 = (index >= 0) & np.isfinite(ps[index, np.arange(n)])
    speed[ok2] = ps[index[ok2], np.arange(n)[ok2]] / 3.6

    return {
        "frames": n,
        "present": present,
        "dmin": dmin,
        "index": index,
        "team": team,
        "speed": speed,
        "ball_x": bx,
        "ball_y": by,
    }


def smooth_codes(codes, window):
    """Published baseline smoothing: centred majority over H/A, D where neither appears."""
    if window <= 0:
        return codes.copy()
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    low = np.maximum(0, idx - window)
    high = np.minimum(len(codes), idx + window + 1)
    h = home[high] - home[low]
    a = away[high] - away[low]
    out = np.full(len(codes), -1, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], 0, 1)
    return out


def public_gt(data_dir):
    path = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
    codes = np.full(199319, -1, dtype=np.int8)
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            code = {"H": 0, "A": 1, "D": 2, "X": -1}.get(label, -1)
            codes[int(frame)] = code
    return codes


def score_window(gt, pred, low, high):
    gt_w = gt[low:high + 1]
    pred_w = pred[low:high + 1]
    if np.any(gt_w < 0):
        raise ValueError("unaligned window: {} X frames".format(int((gt_w < 0).sum())))
    runs = pm.valid_runs(gt_w)
    scores = pm.spell_scores(gt_w, np.where(pred_w < 0, 2, pred_w), runs)
    low_ci, high_ci = pm.bootstrap_spell_f1(gt_w, np.where(pred_w < 0, 2, pred_w), runs, 2000)
    return {
        "spell_f1": float(scores["mean_f1"]),
        "spell_f1_ci90": [low_ci, high_ci],
        "n_gt": int(scores["n_gt"]),
        "n_pred": int(scores["n_pred"]),
    }


def main():
    data_dir = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data")
    context = load_context(data_dir)
    gt = public_gt(data_dir)
    n = context["frames"]
    low, high = WINDOW
    graded = np.zeros(n, dtype=bool)
    graded[low:high + 1] = True

    missing_dead = ~context["present"]
    cue, answered = P2B.cue_of(context["ball_x"], context["ball_y"], context["present"],
                               context["index"], context["speed"], graded)

    held = T.hold(context["dmin"], context["team"], R_M, MINLEN)
    nearest_codes = np.where(context["team"] >= 0, context["team"], -1).astype(np.int8)
    smooth = smooth_codes(nearest_codes, SMOOTH_W)

    arms = {}
    combos = [
        ("nearest_raw_deadmissing", nearest_codes, missing_dead),
        ("nearest_smooth50_deadmissing", smooth, missing_dead),
        ("nearest_smooth50_deadcue", smooth, cue),
        ("touch_hold_r0.8_m30_deadmissing", held, missing_dead),
        ("touch_hold_r0.8_m30_deadcue", held, cue),
        ("touch_hold_r0.8_m30_deadoracle", held, None),
    ]
    for name, codes, dead in combos:
        if dead is None:
            dead_arm = (gt == 2)
        else:
            dead_arm = dead & graded
        pred = np.where(dead_arm, 2, np.where(codes >= 0, codes, 2)).astype(np.int8)
        arms[name] = score_window(gt, pred, low, high)
        arms[name]["dead_rule"] = ("oracle (public GT, ceiling only)" if dead is None else
                                   ("ball missing" if dead is missing_dead else "out_or_slow2"))
        print(name, round(arms[name]["spell_f1"], 4), "n_pred", arms[name]["n_pred"],
              flush=True)

    out = {
        "game": GAME,
        "window": list(WINDOW),
        "config": {"r_m": R_M, "minlen": MINLEN, "smooth_window": SMOOTH_W,
                   "frozen_from": "C05 IDSSE LOSO"},
        "arms": arms,
        "window_note": "5-minute sanity only; config frozen before scoring; never a selector",
        "ball_present_fraction": float(context["present"].mean()),
        "touch_frames": float(np.mean((context["dmin"] < R_M) & (context["team"] >= 0))),
    }
    path = os.path.join(HERE, "out", "barath_transfer.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
