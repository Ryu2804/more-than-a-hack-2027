#!/usr/bin/env python3
"""011 phase 2: value of information of the ball under BAR-ATH-like missingness.

The BAR-ATH ball is absent on ~43% of frames. A video detector would only pay off if the ball it
recovers is accurate often enough for the frozen cue `out_or_slow2` to keep working. IDSSE gives an
oracle: its ball is tracked and BallStatus is the dead/alive label.

Treatment (pre-registered in cycles/C03/PREREG.md and PREREG_PHASE2_ADDENDUM.md):
- remove the ball on runs whose lengths are sampled from the BAR-ATH missing-run distribution,
- `ball_true`: oracle upper bound, nothing removed,
- `ball_masked`: removed, no recovery,
- `ball_recovered`: on masked frames, detected with probability p and placed at truth + N(0, sigma)
  per axis; no interpolation.

Two readings of `p`, both reported:
- `frame` (primary): p is a per-frame detection probability, independent between frames.
- `_run` (secondary sensitivity): p is the probability that an entire masked run is recovered.
  This is what a detector plus a gap-filling tracker approximates.

Metrics: F1 of `out_or_slow2` against BallStatus, and nearest-player ownership accuracy on alive
frames. Paired delta vs `ball_masked`, CI90 from a stratified block bootstrap (250-frame blocks).

Referee tracks are dropped (not players). No public-window numbers are read or written. No candidate
is selected for submission.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

FPS = 25
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
MARGIN = 0.1
KMH_TO_MS = 1.0 / 3.6
OUT_RUN = 15
SLOW_MS = 2.0
SLOW_RUN = 25
SIGMAS = (0.5, 1.0, 2.0, 3.0)
PS = (1.0, 0.8, 0.6, 0.4)
REPLICATES = 3
BLOCK = 250
BOOTSTRAP = 400
SEED = 20261107
DEV = ["J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ"]
CONFIRM = ["J03WN1", "J03WR9"]
MODES = ("frame", "run")


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


def barath_missing(data_dir):
    """Missing-run lengths and missing fraction of the BAR-ATH ball track."""
    m2 = _load(os.path.join(ROOT, "experiments", "002_possession_dp", "run.py"), "m2_phase2")
    n = m2.source_frames("tactical", m2.GAME)
    present = np.zeros(n, dtype=bool)
    path = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(m2.GAME))
    with open(path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            ball = parts[2]
            if not ball or ball[0] == "N":
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                continue
            if 0 <= frame < n:
                present[frame] = True
    lengths = np.array([stop - start for start, stop in runs_of(~present)], dtype=np.int64)
    return {
        "frames": int(n),
        "ball_present_fraction": float(present.mean()),
        "missing_fraction": float(1.0 - present.mean()),
        "missing_runs": int(len(lengths)),
        "run_lengths": lengths,
        "run_median_frames": float(np.median(lengths)) if len(lengths) else 0.0,
        "run_p90_frames": float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
        "run_max_frames": int(lengths.max()) if len(lengths) else 0,
    }


def build_mask(n, lengths, target_frames, rng):
    """Non-overlapping runs with lengths drawn from `lengths`, covering ~target_frames."""
    chosen = []
    total = 0
    while total < target_frames:
        length = int(rng.choice(lengths))
        if length <= 0:
            continue
        length = min(length, n)
        chosen.append(length)
        total += length
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


def run_gate(mask, rng):
    """One uniform per masked run, tiled to frames: common random numbers for the `run` mode."""
    gate = np.ones(len(mask), dtype=np.float64)
    for start, stop in runs_of(mask):
        gate[start:stop] = rng.random()
    return gate


def nearest(npz, ball_x, ball_y):
    """Nearest observed player per frame and its speed in m/s; -1 index where unanswerable."""
    px, py, ps, played = npz["player_x"], npz["player_y"], npz["player_speed"], npz["player_mask"]
    dx = px - ball_x[None, :]
    dy = py - ball_y[None, :]
    distance = np.hypot(dx, dy)
    finite_ball = np.isfinite(ball_x) & np.isfinite(ball_y)
    distance = np.where(played & finite_ball[None, :], distance, np.inf)
    index = np.argmin(distance, axis=0)
    best = distance[index, np.arange(distance.shape[1])]
    answered = np.isfinite(best)
    speed = ps[index, np.arange(distance.shape[1])].astype(np.float32) * KMH_TO_MS
    return answered, index, speed


def observe(npz, mask, sigma, p_detect, gate, noise_eps):
    """Ball observation for one arm. Common random numbers across p and sigma."""
    bx, by = npz["ball_x"], npz["ball_y"]
    truth_present = np.isfinite(bx) & np.isfinite(by)
    recovered = mask & truth_present & (gate < p_detect)
    obs_present = truth_present & (~mask | recovered)
    out_x = np.where(obs_present, bx, np.nan).astype(np.float32)
    out_y = np.where(obs_present, by, np.nan).astype(np.float32)
    if recovered.any():
        out_x[recovered] += (noise_eps[0][recovered] * sigma).astype(np.float32)
        out_y[recovered] += (noise_eps[1][recovered] * sigma).astype(np.float32)
    return out_x, out_y


def cue_masks(npz, ball_x, ball_y):
    answered, index, nearest_speed = nearest(npz, ball_x, ball_y)
    outside = answered & ((np.abs(ball_x) > PITCH_LENGTH / 2.0 + MARGIN) |
                          (np.abs(ball_y) > PITCH_WIDTH / 2.0 + MARGIN))
    out_cue = run_mask(outside, OUT_RUN)
    slow = answered & np.isfinite(nearest_speed) & (nearest_speed < SLOW_MS)
    slow_cue = run_mask(slow, SLOW_RUN)
    return out_cue | slow_cue, answered, index


def frame_counts(npz, cue, answered, index, team):
    """Per-frame cue tp/fp/fn and ownership correct/total, returned as per-block sums."""
    status = npz["ball_status"]
    possession = npz["ball_possession"]
    valid = status >= 0
    dead = valid & (status == 0)
    pred = cue & valid
    tp = pred & dead
    fp = pred & ~dead & valid
    fn = ~pred & dead & valid

    alive = status == 1
    truth = np.where(possession == 1, 0, np.where(possession == 2, 1, -1))
    graded_alive = alive & (truth >= 0)
    answered_alive = answered & graded_alive
    nearest_team = np.full(len(status), -1, dtype=np.int8)
    nearest_team[answered] = team[index[answered]]
    correct = answered_alive & (nearest_team == truth)

    n = len(status)
    blocks = int(np.ceil(n / float(BLOCK)))
    cue_blocks = np.zeros((blocks, 3), dtype=np.int64)
    own_blocks = np.zeros((blocks, 2), dtype=np.int64)
    idx = np.arange(n) // BLOCK
    np.add.at(cue_blocks, (idx[tp], 0), 1)
    np.add.at(cue_blocks, (idx[fp], 1), 1)
    np.add.at(cue_blocks, (idx[fn], 2), 1)
    np.add.at(own_blocks, (idx[correct], 0), 1)
    np.add.at(own_blocks, (idx[answered_alive], 1), 1)
    return cue_blocks, own_blocks


def totals(cue_blocks, own_blocks):
    tp, fp, fn = (int(cue_blocks[:, 0].sum()), int(cue_blocks[:, 1].sum()),
                  int(cue_blocks[:, 2].sum()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    correct = int(own_blocks[:, 0].sum())
    answered = int(own_blocks[:, 1].sum())
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ownership_accuracy": correct / max(1, answered),
        "ownership_answered_alive": answered,
        "ownership_correct": correct,
    }


def sides(npz):
    teams = npz["team_ids"]
    uniq = sorted(set(teams.tolist()))
    if len(uniq) != 2:
        return None
    return np.where(teams == uniq[0], 0, 1).astype(np.int8)


def drop_referees(npz):
    """Remove referee tracks: they are not players, and leaving them breaks the side mapping."""
    team_ids = npz["team_ids"]
    keep = np.array([str(t).lower() != "referee" for t in team_ids], dtype=bool)
    if keep.all():
        return npz, 0
    cleaned = dict(npz)
    for key in ("player_x", "player_y", "player_speed", "player_mask"):
        cleaned[key] = npz[key][keep]
    cleaned["person_ids"] = npz["person_ids"][keep]
    cleaned["team_ids"] = team_ids[keep]
    return cleaned, int((~keep).sum())


def ownership_flip(npz, team, ball_x, ball_y):
    """Convention check on ball_true only; returns the team vector to use for every arm."""
    answered, index, _ = nearest(npz, ball_x, ball_y)
    status, possession = npz["ball_status"], npz["ball_possession"]
    alive = status == 1
    truth = np.where(possession == 1, 0, np.where(possession == 2, 1, -1))
    use = answered & alive & (truth >= 0)
    if not use.any():
        return team, False
    nearest_team = team[index[use]]
    agreement = float((nearest_team == truth[use]).mean())
    if agreement < 0.5:
        return (1 - team).astype(np.int8), True
    return team, False


def bootstrap_once(cue_parts, own_parts, takes):
    """Pooled F1 and ownership accuracy from the pre-drawn block resample."""
    pool = np.concatenate([blocks[take] for blocks, take in zip(cue_parts, takes)], axis=0)
    tp, fp, fn = (int(pool[:, 0].sum()), int(pool[:, 1].sum()), int(pool[:, 2].sum()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    own = np.concatenate([blocks[take] for blocks, take in zip(own_parts, takes)], axis=0)
    accuracy = float(own[:, 0].sum() / max(1, own[:, 1].sum()))
    return f1, accuracy


def bootstrap_delta(cue_arm, cue_base, own_arm, own_base, reps, seed):
    rng = np.random.default_rng(seed)
    deltas, f1s, base_f1s, accs = [], [], [], []
    for _ in range(reps):
        takes = [rng.integers(0, len(blocks), size=len(blocks)) for blocks in cue_arm]
        f1, accuracy = bootstrap_once(cue_arm, own_arm, takes)
        base_f1, base_accuracy = bootstrap_once(cue_base, own_base, takes)
        deltas.append(f1 - base_f1)
        f1s.append(f1)
        base_f1s.append(base_f1)
        accs.append(accuracy - base_accuracy)
    low, high = np.percentile(deltas, [5, 95])
    return {
        "delta_f1_mean": float(np.mean(deltas)),
        "delta_f1_ci90": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "f1_mean": float(np.mean(f1s)),
        "baseline_f1_mean": float(np.mean(base_f1s)),
        "delta_ownership_mean": float(np.mean(accs)),
        "delta_ownership_ci90": [float(v) for v in np.percentile(accs, [5, 95])],
        "reps": reps,
    }


def arm_name(sigma, p_detect, mode="frame"):
    suffix = "" if mode == "frame" else "_run"
    return "ball_recovered_s{:g}_p{:g}{}".format(sigma, p_detect, suffix)


def _by_match(items, matches, replicates):
    """Group a flat list (rep-major, then match) into one array per match."""
    out = []
    for match_index in range(len(matches)):
        parts = [items[rep * len(matches) + match_index] for rep in range(replicates)]
        out.append(np.concatenate(parts, axis=0))
    return out


def run_split(names, parsed_dir, lengths, target_frames, sigmas, ps, replicates,
              out_path, modes=MODES):
    started = time.time()
    result = {
        "meta": {
            "split": names,
            "sigmas": list(sigmas),
            "ps": list(ps),
            "modes": list(modes),
            "replicates": replicates,
            "block_frames": BLOCK,
            "bootstrap_reps": BOOTSTRAP,
            "mask_target_fraction": target_frames,
        },
        "matches": {},
        "arms": {},
        "paired": {},
    }
    cue_blocks = {}
    own_blocks = {}
    mean_totals = {}

    for match in names:
        path = os.path.join(parsed_dir, match + ".npz")
        if not os.path.isfile(path):
            result["matches"][match] = {"available": False}
            continue
        npz = {k: v for k, v in np.load(path, allow_pickle=False).items()}
        npz, dropped_referees = drop_referees(npz)
        n = len(npz["ball_status"])
        team = sides(npz)
        bx, by = npz["ball_x"], npz["ball_y"]
        if team is None:
            team_fixed, flipped = None, False
        else:
            team_fixed, flipped = ownership_flip(npz, team, bx, by)
        result["matches"][match] = {
            "available": True,
            "frames": int(n),
            "referees_dropped": dropped_referees,
            "ownership_evaluable": team_fixed is not None,
            "team_mapping_flipped": bool(flipped),
        }

        for rep in range(replicates):
            seed = SEED + 1000003 * rep + (zlib.crc32(match.encode()) % 100000)
            rng = np.random.default_rng(seed)
            mask = build_mask(n, lengths, int(round(target_frames * n)), rng)
            detect_u = rng.random(n)
            run_u = run_gate(mask, rng)
            noise_eps = (rng.standard_normal(n), rng.standard_normal(n))

            arms = [("ball_true", None, None, None), ("ball_masked", 0.0, 0.0, detect_u)]
            for mode in modes:
                gate = detect_u if mode == "frame" else run_u
                for sigma in sigmas:
                    for p_detect in ps:
                        arms.append((arm_name(sigma, p_detect, mode), sigma, p_detect, gate))

            for name, sigma, p_detect, gate in arms:
                if name == "ball_true":
                    obs_x, obs_y = bx, by
                else:
                    obs_x, obs_y = observe(npz, mask, sigma, p_detect, gate, noise_eps)
                cue, answered, index = cue_masks(npz, obs_x, obs_y)
                index_team = np.zeros_like(index) if team_fixed is None else team_fixed
                cb, ob = frame_counts(npz, cue, answered, index, index_team)
                cue_blocks[(name, match, rep)] = cb
                own_blocks[(name, match, rep)] = ob
                mean_totals.setdefault((name, match), []).append(totals(cb, ob))

        print(match, "done", flush=True)

    arm_names = ["ball_true", "ball_masked"]
    for mode in modes:
        arm_names += [arm_name(s, p, mode) for s in sigmas for p in ps]
    for name in arm_names:
        per_match = {}
        for match in names:
            rows = mean_totals.get((name, match))
            if rows:
                per_match[match] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        result["arms"][name] = {"per_match": per_match}

    for mode in modes:
        for sigma in sigmas:
            for p_detect in ps:
                name = arm_name(sigma, p_detect, mode)
                present_matches = [m for m in names if (name, m, 0) in cue_blocks]
                if not present_matches:
                    continue
                cue_arm, cue_base, own_arm, own_base = [], [], [], []
                for rep in range(replicates):
                    for match in present_matches:
                        cue_arm.append(cue_blocks[(name, match, rep)])
                        cue_base.append(cue_blocks[("ball_masked", match, rep)])
                        own_arm.append(own_blocks[(name, match, rep)])
                        own_base.append(own_blocks[("ball_masked", match, rep)])
                cue_arm = _by_match(cue_arm, present_matches, replicates)
                cue_base = _by_match(cue_base, present_matches, replicates)
                own_arm = _by_match(own_arm, present_matches, replicates)
                own_base = _by_match(own_base, present_matches, replicates)
                row = bootstrap_delta(cue_arm, cue_base, own_arm, own_base, BOOTSTRAP,
                                      SEED + 7 + (0 if mode == "frame" else 5000)
                                      + int(sigma * 10) * 100 + int(p_detect * 10))
                row["matches"] = present_matches
                result["paired"][name] = row

    result["runtime_s"] = time.time() - started
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def candidate_selection(result, mode="frame", max_sigma=2.0, min_p=0.6):
    """Worst passing combination: largest sigma, then smallest p."""
    passing = []
    for sigma in result["meta"]["sigmas"]:
        for p_detect in result["meta"]["ps"]:
            if sigma > max_sigma or p_detect < min_p:
                continue
            name = arm_name(sigma, p_detect, mode)
            row = result["paired"].get(name)
            if row is None:
                continue
            arm = result["arms"][name]["per_match"]
            if not arm:
                continue
            precision = float(np.mean([v["precision"] for v in arm.values()]))
            recall = float(np.mean([v["recall"] for v in arm.values()]))
            low, high = row["delta_f1_ci90"]
            ok = (row["delta_f1_mean"] >= 0.04 and low > 0
                  and precision >= 0.60 and recall >= 0.70)
            passing.append({
                "arm": name, "sigma": sigma, "p": p_detect,
                "delta_f1": row["delta_f1_mean"], "ci90": [low, high],
                "f1_mean": float(np.mean([v["f1"] for v in arm.values()])),
                "precision_mean": precision, "recall_mean": recall,
                "passes": bool(ok),
            })
    passing.sort(key=lambda r: (r["sigma"], -r["p"]), reverse=True)
    winners = [r for r in passing if r["passes"]]
    return {"mode": mode, "grid": passing, "worst_passing": winners[0] if winners else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir",
                        default=os.path.join(ROOT, "..", "data", "external", "idsee", "parsed"))
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or
                        os.path.join(ROOT, "..", "data"))
    parser.add_argument("--split", choices=("dev", "confirm", "all"), default="dev")
    parser.add_argument("--sigma", type=float, nargs="*", default=None)
    parser.add_argument("--p", type=float, nargs="*", default=None)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    missing = barath_missing(args.data_dir)
    lengths = missing.pop("run_lengths")
    target_frames = missing["missing_fraction"]
    print("BAR-ATH missing fraction", round(target_frames, 4),
          "runs", missing["missing_runs"],
          "median", missing["run_median_frames"], flush=True)

    if args.split == "dev":
        names, sigmas, ps = DEV, SIGMAS, PS
    elif args.split == "confirm":
        names = CONFIRM
        sigmas = tuple(args.sigma) if args.sigma else SIGMAS
        ps = tuple(args.p) if args.p else PS
    else:
        names, sigmas, ps = DEV + CONFIRM, SIGMAS, PS

    out_path = args.out or os.path.join(HERE, "out", "phase2_{}.json".format(args.split))
    result = run_split(names, args.parsed_dir, lengths, target_frames,
                       tuple(sigmas), tuple(ps), args.replicates, out_path)
    result["barath_missing"] = missing
    result["selection"] = candidate_selection(result, "frame")
    result["selection_run"] = candidate_selection(result, "run")

    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("wrote", out_path, "runtime", round(result["runtime_s"], 1), "s", flush=True)
    for name, row in result["paired"].items():
        if name.endswith("_run"):
            continue
        print(name, "delta", round(row["delta_f1_mean"], 4),
              "ci", [round(v, 4) for v in row["delta_f1_ci90"]], flush=True)
    winner = result["selection"]["worst_passing"]
    winner_run = result["selection_run"]["worst_passing"]
    print("worst passing frame", winner["arm"] if winner else None, flush=True)
    print("worst passing run  ", winner_run["arm"] if winner_run else None, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
