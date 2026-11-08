#!/usr/bin/env python3
"""004: does the video support the ATD team labels?

The bottleneck found in 002/003 is the per-frame ownership signal: the nearest player flips team
28 times a minute, and nothing in the tracking tells us whether a slot is really the player it says
it is. The video can answer that without any hand labels and without a detector:

  * ATD positions + homography give a torso box in pixels for every player in every frame;
  * the jersey colours inside that box give a feature vector per player (blue / red / white);
  * k-means is not needed: two team centroids per frame come from the ATD labels themselves, so
    the check is "does the colour of this player sit closer to its own team than to the other".

Everything here is algorithmic. No frame is inspected by a person, no label is edited by hand, and
the public ground truth is not used at all.

Outputs are diagnostics: separability, per-frame colour agreement, and a per-track table of slots
whose colour drifts away from their ATD team (candidate identity switches). Nothing is promoted to
a submission.
"""

import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import baseline as bl  # noqa: E402
from games import GAMES, half_video_blocks, source_frames  # noqa: E402

GAME = "BAR-ATH"
H, A, D = 0, 1, 2
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]


def load_homography(path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    matrices = np.stack([
        np.stack([table[col] for col in ("h00", "h01", "h02")], axis=1),
        np.stack([table[col] for col in ("h10", "h11", "h12")], axis=1),
        np.stack([table[col] for col in ("h20", "h21", "h22")], axis=1),
    ], axis=1).astype(np.float64)
    return matrices, table["valid"].astype(bool)


def project(matrix, x, y):
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator)


def load_players(data_dir, frames_wanted):
    """frame -> list of (slot, team, x, y) in normalised pitch coordinates."""
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME))
    team_of_track = bl.team_of_each_track(atd_xml)
    out = {}
    with open(atd_txt) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                continue
            if frame not in frames_wanted:
                continue
            players = []
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    first = group.find(",")
                    second = group.find(",", first + 1)
                    x = float(group[:first])
                    y = float(group[first + 1:second if second > 0 else None])
                except (ValueError, IndexError):
                    continue
                players.append((slot, int(team_of_track[slot]), x, y))
            if players:
                out[frame] = players
    return out


def jersey_features(frame, matrix, players, min_draw_px=5):
    """Per player: (blue, red, white, saturation) over the torso box, plus the box used."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rows = []
    for slot, team, x, y in players:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        fx, fy = project(matrix, x, y)
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        top_x, top_y = project(matrix, x, np.clip(y + 0.02, 0, 1))
        bottom_x, bottom_y = project(matrix, x, np.clip(y - 0.02, 0, 1))
        scale = float(np.hypot(top_x - bottom_x, top_y - bottom_y)) / (0.04 * PITCH_WIDTH)
        if not np.isfinite(scale) or scale <= 0:
            continue
        box_height = 0.7 * scale
        box_width = 0.56 * scale
        if box_height < min_draw_px:
            continue
        y0 = int(round(fy - 1.6 * scale))
        y1 = int(round(fy - 0.9 * scale))
        x0 = int(round(fx - box_width / 2))
        x1 = int(round(fx + box_width / 2))
        y0, y1 = max(0, y0), min(height, max(y0 + 2, y1))
        x0, x1 = max(0, x0), min(width, max(x0 + 2, x1))
        if y1 - y0 < 3 or x1 - x0 < 3:
            continue

        patch = hsv[y0:y1, x0:x1]
        hue = patch[:, :, 0].astype(np.int16)
        sat = patch[:, :, 1].astype(np.int16)
        val = patch[:, :, 2].astype(np.int16)

        green = (hue >= 35) & (hue <= 95) & (sat > 60)
        keep = ~green
        if keep.sum() < 6:
            continue
        hue, sat, val = hue[keep], sat[keep], val[keep]
        blue = ((hue >= 90) & (hue <= 140) & (sat > 60)).mean()
        red = (((hue <= 12) | (hue >= 168)) & (sat > 90)).mean()
        white = ((sat < 45) & (val > 150)).mean()
        rows.append((slot, team, float(blue), float(red), float(white), float(sat.mean())))
    return rows


def frame_agreement(rows):
    if len(rows) < 4:
        return None
    arr = np.array([[r[2], r[3], r[4]] for r in rows])
    teams = np.array([r[1] for r in rows])

    def centroid(mask):
        return arr[mask].mean(axis=0) if mask.any() else None

    home = centroid(teams == H)
    away = centroid(teams == A)
    if home is None or away is None:
        return None
    separation = float(np.linalg.norm(home - away))
    spread = 0.5 * (float(arr[teams == H].std(axis=0).mean()) +
                    float(arr[teams == A].std(axis=0).mean()))
    scale = separation / spread if spread > 1e-6 else float("inf")
    d_home = np.linalg.norm(arr - home, axis=1)
    d_away = np.linalg.norm(arr - away, axis=1)
    assigned = np.where(d_home <= d_away, H, A)
    agreement = float((assigned == teams).mean())
    return {"separation": separation, "fisher": scale, "agreement": agreement, "n": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--save-visuals", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    homography_path = os.path.join(args.data_dir, "tactical", "{}_tactical_homography.csv".format(GAME))
    matrices, valid = load_homography(homography_path)
    n_frames = source_frames("tactical", GAME)
    graded = np.zeros(n_frames, dtype=bool)
    for _, start, end in half_video_blocks(GAME):
        graded[start:end + 1] = True

    sampled = [f for f in range(0, n_frames, args.step) if graded[f] and valid[f]]
    if args.limit_frames:
        sampled = sampled[:args.limit_frames]
    wanted = set(sampled)
    print("sampling {} frames (step {}), homography valid {:.1f}%".format(
        len(sampled), args.step, 100.0 * valid.mean()), flush=True)

    players_by_frame = load_players(args.data_dir, wanted)
    print("ATD rows for {} sampled frames".format(len(players_by_frame)), flush=True)

    video_path = os.path.join(args.data_dir, "tactical", "{}_tactical.mp4".format(GAME))
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit("cannot open {}".format(video_path))

    per_frame = []
    raw_rows = []
    track_sum = {}
    track_n = {}
    visuals = []
    last_wanted = max(wanted) if wanted else -1
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in players_by_frame:
            rows = jersey_features(frame, matrices[frame_index], players_by_frame[frame_index])
            if rows:
                stats = frame_agreement(rows)
                if stats:
                    stats["frame"] = frame_index
                    per_frame.append(stats)
                for slot, team, blue, red, white, sat in rows:
                    raw_rows.append((frame_index, slot, team, blue, red, white, sat))
                    key = (slot, team)
                    acc = track_sum.setdefault(key, np.zeros(4))
                    acc += np.array([blue, red, white, sat])
                    track_n[key] = track_n.get(key, 0) + 1
                if args.save_visuals and len(visuals) < 12 and len(per_frame) % 50 == 0:
                    canvas = frame.copy()
                    for slot, team, x, y in players_by_frame[frame_index]:
                        fx, fy = project(matrices[frame_index], x, y)
                        if np.isfinite(fx) and np.isfinite(fy):
                            colour = (255, 120, 0) if team == H else (0, 0, 255)
                            cv2.circle(canvas, (int(fx), int(fy)), 5, colour, -1)
                    visuals.append(canvas)
        frame_index += 1
        if frame_index % 20000 == 0:
            print("  decoded {} frames, {} sampled".format(frame_index, len(per_frame)), flush=True)
        if frame_index > last_wanted:
            break
    capture.release()
    print("decoded {} of {} frames".format(frame_index, n_frames), flush=True)

    if not per_frame:
        raise SystemExit("no usable frames")

    agreements = np.array([row["agreement"] for row in per_frame])
    fishers = np.array([row["fisher"] for row in per_frame if np.isfinite(row["fisher"])])
    tracks = []
    for (slot, team), total in sorted(track_sum.items()):
        mean = total / track_n[(slot, team)]
        tracks.append({
            "slot": slot, "atd_team": int(team), "frames": int(track_n[(slot, team)]),
            "blue": float(mean[0]), "red": float(mean[1]), "white": float(mean[2]),
            "saturation": float(mean[3]),
        })

    home_tracks = [t for t in tracks if t["atd_team"] == H]
    away_tracks = [t for t in tracks if t["atd_team"] == A]
    home_mean = np.mean([[t["blue"], t["red"], t["white"]] for t in home_tracks], axis=0) if home_tracks else None
    away_mean = np.mean([[t["blue"], t["red"], t["white"]] for t in away_tracks], axis=0) if away_tracks else None
    suspect = []
    if home_mean is not None and away_mean is not None:
        for track in tracks:
            feature = np.array([track["blue"], track["red"], track["white"]])
            own = home_mean if track["atd_team"] == H else away_mean
            other = away_mean if track["atd_team"] == H else home_mean
            track["own_distance"] = float(np.linalg.norm(feature - own))
            track["other_distance"] = float(np.linalg.norm(feature - other))
            matches_own = track["own_distance"] <= track["other_distance"]
            track["colour_team"] = int(track["atd_team"] if matches_own else 1 - track["atd_team"])
            track["suspect"] = not matches_own
            if track["suspect"]:
                suspect.append(track)

    # Per-track disagreement runs: a sustained block where the jersey colour belongs to the other
    # team is the signature of a slot that changed player (or a wrong static team label).
    runs_by_track = {}
    if raw_rows:
        data = np.array(raw_rows, dtype=np.float32)
        np.savez_compressed(os.path.join(args.out_dir, "features.npz"),
                            data=data, columns=np.array(["frame", "slot", "team", "blue", "red", "white", "sat"]))
        home_vec, away_vec = home_mean, away_mean
        if home_vec is not None and away_vec is not None:
            distance_home = np.linalg.norm(data[:, 3:6] - np.array(home_vec), axis=1)
            distance_away = np.linalg.norm(data[:, 3:6] - np.array(away_vec), axis=1)
            implies = np.where(distance_home <= distance_away, H, A)
            data_team = data[:, 2].astype(np.int8)
            agree = implies == data_team
            for slot in np.unique(data[:, 1]).astype(int):
                mask = data[:, 1] == slot
                frames = data[mask, 0]
                ok = agree[mask]
                longest = 0
                current = 0
                for value in ok:
                    current = 0 if value else current + 1
                    longest = max(longest, current)
                runs_by_track[str(slot)] = {
                    "atd_team": int(data_team[mask][0]),
                    "frames": int(mask.sum()),
                    "disagree_fraction": float(1.0 - ok.mean()),
                    "longest_disagreement_s": float(longest * args.step / FPS),
                }

    if args.save_visuals:
        for index, canvas in enumerate(visuals):
            cv2.imwrite(os.path.join(args.out_dir, "projection_{:02d}.png".format(index)), canvas)

    result = {
        "step": args.step,
        "frames_sampled": len(sampled),
        "frames_with_features": len(per_frame),
        "frames_decoded": frame_index,
        "agreement_mean": float(agreements.mean()),
        "agreement_median": float(np.median(agreements)),
        "agreement_p10": float(np.percentile(agreements, 10)),
        "agreement_p90": float(np.percentile(agreements, 90)),
        "fisher_median": float(np.median(fishers)) if len(fishers) else None,
        "separability_median": float(np.median([row["separation"] for row in per_frame])),
        "team_centroids": {"home": None if home_mean is None else home_mean.tolist(),
                           "away": None if away_mean is None else away_mean.tolist()},
        "tracks": tracks,
        "suspect_tracks": suspect,
        "per_track_disagreement": runs_by_track,
        "note": "colour agreement is an algorithmic check of the ATD team labels; no hand labels used",
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print("agreement mean {:.3f} median {:.3f} p10 {:.3f}".format(
        result["agreement_mean"], result["agreement_median"], result["agreement_p10"]))
    print("fisher median {}".format(result["fisher_median"]))
    print("suspect tracks {}/{}".format(len(suspect), len(tracks)))
    if runs_by_track:
        worst = sorted(runs_by_track.items(), key=lambda kv: -kv[1]["longest_disagreement_s"])[:8]
        print("longest colour disagreement runs (s):",
              ", ".join("{}:{:.1f}".format(k, v["longest_disagreement_s"]) for k, v in worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
