#!/usr/bin/env python3
"""005: use jersey colour to decide who owns the ball, not the static track label.

004 established that the video separates the two kits cleanly (Fisher median 6.3, colour agreement
0.88) while the ATD team labels flip 28 times a minute because a track slot is not a person. This
experiment turns the colour check into an observation:

  obs_colour   team of the nearest player, taken from the jersey colour of that player's torso
               instead of from the track's static team label; falls back to the ATD label when the
               colour is ambiguous.

Then the same missing-aware DP as 002/003 is run on that observation and compared, GT-free, on
structure plus the public window as a correctness check. Switch penalty is chosen per arm by the
possession-count band, never by the public score.

No hand labels, no private ground truth, no external data about this match.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import median_filter

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
m4 = _load(os.path.join(ROOT, "experiments", "004_video_team_check", "run.py"), "m4")
pm = dp2.pm
bl = dp2.bl
GAME = dp2.GAME
H, A, D = dp2.H, dp2.A, dp2.D
FPS = dp2.FPS

COLOUR_MARGIN = 0.06   # required lead of the winning centroid over the losing one
NEAR_PLAYERS = 3       # candidates considered around the ball
NEAR_METRES = 12.0     # ignore players further than this from the ball
LAMBDA_GRID = dp2.LAMBDA_GRID


def load_ball_and_players(data_dir, n_frames):
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME))
    team_of_track = bl.team_of_each_track(atd_xml)
    players = {}
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
                x = float(coords[0])
                y = float(coords[1])
                if frame < 0 or frame >= n_frames:
                    continue
                if len(coords) > 2 and coords[2] != "NaN":
                    ball_speed[frame] = float(coords[2])
            except (ValueError, IndexError):
                continue
            ball_x[frame] = x * dp2.PITCH_LENGTH
            ball_y[frame] = y * dp2.PITCH_WIDTH
            present[frame] = True
            group = []
            for slot, chunk in enumerate(parts[1].split(";")):
                if not chunk or chunk[0] == "N":
                    continue
                try:
                    first = chunk.find(",")
                    second = chunk.find(",", first + 1)
                    px = float(chunk[:first])
                    py = float(chunk[first + 1:second if second > 0 else None])
                except (ValueError, IndexError):
                    continue
                group.append((slot, int(team_of_track[slot]), px, py))
            players[frame] = group
    return players, ball_x, ball_y, ball_speed, present


def colour_team(frame, matrix, x, y, centroids):
    """Colour-implied team for one player box, or None when the colour is ambiguous."""
    rows = m4.jersey_features(frame, matrix, [(0, -1, x, y)])
    if not rows:
        return None
    feature = np.array(rows[0][2:5])
    distance_home = float(np.linalg.norm(feature - centroids["home"]))
    distance_away = float(np.linalg.norm(feature - centroids["away"]))
    margin = abs(distance_home - distance_away)
    if margin < COLOUR_MARGIN:
        return None
    return H if distance_home < distance_away else A



def nearest_slots(players, ball_x, ball_y, n_frames):
    """Slot of the nearest player to the ball, per frame (no distance limit; 002 parity)."""
    out = np.full(n_frames, -1, dtype=np.int16)
    for frame, group in players.items():
        if not group:
            continue
        best = None
        for slot, team, px, py in group:
            d = (px * dp2.PITCH_LENGTH - ball_x[frame]) ** 2 + (py * dp2.PITCH_WIDTH - ball_y[frame]) ** 2
            if best is None or d < best[0]:
                best = (d, slot)
        out[frame] = best[1]
    return out


def load_track_repair(path, centroids, n_frames):
    """Per slot, a smoothed colour-implied team over time, or None when 004 has not run."""
    if not os.path.isfile(path):
        return None
    data = np.load(path)["data"]
    frames = data[:, 0].astype(np.int64)
    slots = data[:, 1].astype(np.int64)
    feature = data[:, 3:6].astype(np.float64)
    d_home = np.linalg.norm(feature - centroids["home"], axis=1)
    d_away = np.linalg.norm(feature - centroids["away"], axis=1)
    implied = np.where(d_home <= d_away, H, A).astype(np.int8)
    out = {}
    for slot in np.unique(slots):
        mask = slots == slot
        order = np.argsort(frames[mask])
        f = frames[mask][order]
        values = implied[mask][order]
        smooth = median_filter(values, size=15, mode="nearest")
        arr = np.full(n_frames, -1, dtype=np.int8)
        arr[f] = smooth
        last = np.maximum.accumulate(np.where(arr >= 0, np.arange(n_frames), -1))
        last[last < 0] = 0
        out[int(slot)] = arr[last]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    parser.add_argument("--centroids", default=os.path.join(ROOT, "out", "004_video_team_check", "results.json"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--features", default=os.path.join(ROOT, "out", "004_video_team_check", "features.npz"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    with open(args.centroids) as handle:
        centroids = {key: np.array(value) for key, value in json.load(handle)["team_centroids"].items()}

    n_frames = dp2.source_frames("tactical", GAME)
    matrices, valid = m4.load_homography(os.path.join(args.data_dir, "tactical", "{}_tactical_homography.csv".format(GAME)))
    players, ball_x, ball_y, ball_speed, present = load_ball_and_players(args.data_dir, n_frames)

    cache_path = os.path.join(args.out_dir, "obs_cache.npz")
    if os.path.isfile(cache_path) and not args.force:
        cached = np.load(cache_path)
        obs_colour, obs_atd = cached["colour"], cached["atd"]
        stats = {key: int(cached[key]) for key in
                 ("frames_with_ball", "frames_decoded", "changed", "voted")}
        print("loaded cached observations ({} ball frames)".format(stats["frames_with_ball"]), flush=True)
    else:
        frames = [f for f in range(0, n_frames, args.stride) if f in players and present[f]]
        wanted = set(frames)
        print("colour pass over {} frames with the ball present".format(len(wanted)), flush=True)
        obs_colour = np.full(n_frames, -1, dtype=np.int8)
        obs_atd = np.full(n_frames, -1, dtype=np.int8)
        changed = 0
        voted = 0

        capture = cv2.VideoCapture(os.path.join(args.data_dir, "tactical", "{}_tactical.mp4".format(GAME)))
        if not capture.isOpened():
            raise SystemExit("cannot open the tactical video")
        last = max(wanted) if wanted else -1
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                bx, by = ball_x[index], ball_y[index]
                group = players[index]
                ranked = sorted(group, key=lambda p: (p[2] * dp2.PITCH_LENGTH - bx) ** 2
                                + (p[3] * dp2.PITCH_WIDTH - by) ** 2)[:NEAR_PLAYERS]
                ranked = [p for p in ranked if np.hypot(p[2] * dp2.PITCH_LENGTH - bx,
                                                        p[3] * dp2.PITCH_WIDTH - by) <= NEAR_METRES]
                if ranked:
                    obs_atd[index] = ranked[0][1]
                    votes = []
                    for slot, team, px, py in ranked[:2]:
                        implied = colour_team(frame, matrices[index], px, py, centroids)
                        votes.append((implied, team, float(np.hypot(
                            px * dp2.PITCH_LENGTH - bx, py * dp2.PITCH_WIDTH - by))))
                    top = votes[0][0]
                    if top is None and len(votes) > 1:
                        top = votes[1][0]
                        if top is not None:
                            voted += 1
                    if top is not None:
                        obs_colour[index] = top
                    else:
                        obs_colour[index] = votes[0][1]
                    if top is not None and top != votes[0][1]:
                        changed += 1
            index += 1
            if index % 20000 == 0:
                print("  decoded {} frames".format(index), flush=True)
            if index > last:
                break
        capture.release()
        stats = {"frames_with_ball": len(wanted), "frames_decoded": index,
                 "changed": changed, "voted": voted}
        np.savez_compressed(cache_path, colour=obs_colour, atd=obs_atd, **stats)
        print("cached observations to {}".format(cache_path), flush=True)
    print("decoded {} frames".format(stats["frames_decoded"]), flush=True)

    restart_starts = dp2.parse_set_piece_starts(args.data_dir, n_frames)
    evidence = dp2.build_evidence(n_frames, present, ball_x, ball_y, ball_speed, restart_starts)
    gt_codes = pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))))

    def switch_rate(obs):
        seq = obs[np.flatnonzero(obs >= 0)]
        if len(seq) < 2:
            return 0.0
        return float(np.sum(seq[1:] != seq[:-1])) / (len(seq) / FPS / 60.0)

    def hard_costs(obs):
        cost = np.ones((3, len(obs)), dtype=np.float32)
        idx = np.flatnonzero(obs >= 0)
        cost[obs[idx], idx] = 0.0
        return cost

    runs = dp2.full_match_run_list(n_frames)

    def structure(codes):
        # Official possession definition (pm.possession_spells), not raw alive runs: 002 showed the
        # two differ by ~2.6x and only the official count matches COUNT_CENTRE.
        spells = pm.possession_spells(codes, runs)
        durations = [(stop - start) / FPS for start, stop, _ in spells]
        return {"n_possessions": len(spells),
                "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
                "dead_fraction_graded": float(((codes == D) & dp2.graded_mask(n_frames)).sum() / dp2.graded_mask(n_frames).sum())}

    def pick(obs):
        rows = []
        for lam in LAMBDA_GRID:
            codes = dp2.dp_full(obs, evidence["final"], lam, n_frames)
            rows.append({"lambda": lam, **structure(codes)})
        inside = [r for r in rows if dp2.COUNT_BAND[0] <= r["n_possessions"] <= dp2.COUNT_BAND[1]]
        pool = inside or rows
        best = min(pool, key=lambda r: abs(r["n_possessions"] - dp2.COUNT_CENTRE))
        return best["lambda"], rows, bool(inside)

    nearest_slot = nearest_slots(players, ball_x, ball_y, n_frames)
    arms = {}
    obs_atd_002, present_002, _, _, _ = dp2.load_observations(args.data_dir, n_frames)
    arms["dp_atd_nearest"] = obs_atd_002
    arms["dp_colour_obs"] = obs_colour

    repaired = load_track_repair(args.features, centroids, n_frames)
    if repaired is not None:
        obs_repaired = obs_atd_002.copy()
        for frame in np.flatnonzero(obs_atd_002 >= 0):
            slot = nearest_slot[frame]
            if slot >= 0 and repaired[slot][frame] >= 0:
                obs_repaired[frame] = repaired[slot][frame]
        arms["dp_track_repaired"] = obs_repaired

        obs_gated_track = obs_atd_002.copy()
        for frame in np.flatnonzero(obs_atd_002 >= 0):
            slot = nearest_slot[frame]
            if slot >= 0 and repaired[slot][frame] >= 0 and repaired[slot][frame] != obs_atd_002[frame]:
                obs_gated_track[frame] = -1
        arms["dp_colour_gated_track"] = obs_gated_track

    # Colour is noisy per frame (149 switches/min vs 28.4). A safer use than replacing the label is
    # to drop the frames where the two sources disagree, leaving the DP to interpolate.
    disagree = (obs_colour >= 0) & (obs_atd_002 >= 0) & (obs_colour != obs_atd_002)
    obs_gated = obs_atd_002.copy()
    obs_gated[disagree] = -1
    arms["dp_colour_gated"] = obs_gated

    results = {}
    reference_codes = None
    for name, obs in arms.items():
        lam, rows, inside = pick(obs)
        codes = dp2.dp_full(obs, evidence["final"], lam, n_frames)
        metrics = dp2.metrics_for(gt_codes, codes, args.bootstrap)
        if reference_codes is None:
            reference_codes = codes
        paired = None if name == "dp_atd_nearest" else dp2.paired_spell_bootstrap(
            gt_codes, codes, reference_codes, 1000, args.seed if hasattr(args, "seed") else 0)
        results[name] = {
            "lambda": lam, "inside_band": inside, "grid": rows, "paired_vs_atd": paired,
            "switch_rate_per_min": switch_rate(obs),
            "public": metrics,
            "structure": structure(codes),
        }
        print("{}: lambda {} spells {} dur {:.1f} spellF1 {:.4f} switches/min {:.1f}".format(
            name, lam, metrics["n_pred_spells"], metrics["mean_spell_duration_s"],
            metrics["spell_f1_mean"], results[name]["switch_rate_per_min"]), flush=True)

    output = {
        "frames_with_ball": stats["frames_with_ball"],
        "frames_decoded": stats["frames_decoded"],
        "colour_changed_vs_atd": stats["changed"],
        "colour_changed_fraction": stats["changed"] / max(1, stats["frames_with_ball"]),
        "colour_from_second_player": stats["voted"],
        "colour_margin": COLOUR_MARGIN,
        "centroids": {key: value.tolist() for key, value in centroids.items()},
        "arms": results,
        "runtime_s": time.time() - started,
        "rules": {"public_window": "correctness check only", "switches": "GT-free count band"},
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print("colour observation differs from ATD nearest on {:.1f}% of ball frames".format(
        100.0 * output["colour_changed_fraction"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
