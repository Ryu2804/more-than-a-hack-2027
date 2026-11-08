#!/usr/bin/env python3
"""011 phase 2C: does a short gap bridge remove the p = 1.0 requirement of C03c?

Same mask, same per-frame detection as C03c, but frames whose detection gap is at most K frames are
filled by linear interpolation between the last and the first observed position. Anchors outside the
mask are true positions; anchors inside it carry the arm's position noise. No smoothing, no velocity
extrapolation, no use of the hidden true position inside a gap.

Pre-registered in cycles/C03/PREREG_PHASE2C_ADDENDUM.md. Metrics and bootstrap are inherited from
phase2.py. No public-window numbers are read or written, no candidate is chosen.
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase2 as p2  # noqa: E402

ROOT = p2.ROOT
KS = (5, 15, 25)
SIGMAS = (0.5, 1.0, 2.0)
PS = (1.0, 0.8, 0.6, 0.4)
REPLICATES = 3
SEED = p2.SEED


def bridge_observe(npz, mask, sigma, p_detect, gate, noise_eps, k_max):
    """Observed ball after gap bridging. Returns x, y, filled mask, anchor mask."""
    bx, by = npz["ball_x"], npz["ball_y"]
    truth = np.isfinite(bx) & np.isfinite(by)
    detected = mask & truth & (gate < p_detect)
    anchor = truth & (~mask | detected)
    x = np.where(anchor, bx, np.nan).astype(np.float64)
    y = np.where(anchor, by, np.nan).astype(np.float64)
    if detected.any():
        x[detected] += noise_eps[0][detected] * sigma
        y[detected] += noise_eps[1][detected] * sigma

    n = len(x)
    index = np.arange(n)
    previous = np.maximum.accumulate(np.where(anchor, index, -1))
    following = np.minimum.accumulate(np.where(anchor, index, n)[::-1])[::-1]
    need = mask & ~anchor & truth
    ok = need & (previous >= 0) & (following < n)
    gap = following - previous - 1
    ok &= (gap >= 1) & (gap <= k_max)
    if ok.any():
        position = index[ok]
        before = previous[ok]
        after = following[ok]
        weight = (position - before) / (after - before)
        x[ok] = (1.0 - weight) * x[before] + weight * x[after]
        y[ok] = (1.0 - weight) * y[before] + weight * y[after]
    filled = anchor | ok
    return x.astype(np.float32), y.astype(np.float32), filled, anchor


def arm_name(k_max, sigma, p_detect):
    return "ball_bridged_k{:g}_s{:g}_p{:g}".format(k_max, sigma, p_detect)


def run_split(names, parsed_dir, lengths, target_frames, ks, sigmas, ps, replicates, out_path):
    started = time.time()
    result = {
        "meta": {
            "split": names,
            "ks": list(ks),
            "sigmas": list(sigmas),
            "ps": list(ps),
            "replicates": replicates,
            "block_frames": p2.BLOCK,
            "bootstrap_reps": p2.BOOTSTRAP,
            "mask_target_fraction": target_frames,
        },
        "matches": {},
        "arms": {},
        "paired": {},
        "coverage": {},
    }
    cue_blocks, own_blocks, mean_totals = {}, {}, {}
    coverage = {}

    for match in names:
        path = os.path.join(parsed_dir, match + ".npz")
        if not os.path.isfile(path):
            result["matches"][match] = {"available": False}
            continue
        npz = {k: v for k, v in np.load(path, allow_pickle=False).items()}
        npz, dropped = p2.drop_referees(npz)
        n = len(npz["ball_status"])
        team = p2.sides(npz)
        bx, by = npz["ball_x"], npz["ball_y"]
        if team is None:
            team_fixed, flipped = None, False
        else:
            team_fixed, flipped = p2.ownership_flip(npz, team, bx, by)
        result["matches"][match] = {
            "available": True,
            "frames": int(n),
            "referees_dropped": dropped,
            "ownership_evaluable": team_fixed is not None,
            "team_mapping_flipped": bool(flipped),
        }

        for rep in range(replicates):
            seed = SEED + 1000003 * rep + (zlib.crc32(match.encode()) % 100000)
            rng = np.random.default_rng(seed)
            mask = p2.build_mask(n, lengths, int(round(target_frames * n)), rng)
            detect_u = rng.random(n)
            noise_eps = (rng.standard_normal(n), rng.standard_normal(n))
            masked_total = int(mask.sum())
            if masked_total:
                coverage.setdefault(match, []).append({})

            arms = [("ball_true", None, None), ("ball_masked", 0.0, 0.0)]
            for k_max in ks:
                for sigma in sigmas:
                    for p_detect in ps:
                        arms.append((arm_name(k_max, sigma, p_detect), sigma, p_detect))

            for name, sigma, p_detect in arms:
                if name == "ball_true":
                    obs_x, obs_y, filled = bx, by, np.ones(n, dtype=bool)
                else:
                    obs_x, obs_y, filled, _ = bridge_observe(
                        npz, mask, sigma, p_detect, detect_u, noise_eps,
                        0.0 if name == "ball_masked" else _k_of(name))
                cue, answered, index = p2.cue_masks(npz, obs_x, obs_y)
                index_team = np.zeros_like(index) if team_fixed is None else team_fixed
                cb, ob = p2.frame_counts(npz, cue, answered, index, index_team)
                cue_blocks[(name, match, rep)] = cb
                own_blocks[(name, match, rep)] = ob
                mean_totals.setdefault((name, match), []).append(p2.totals(cb, ob))
                if masked_total and name != "ball_true":
                    coverage[match][-1].setdefault(name, []).append(
                        float(filled[mask].mean()))

        print(match, "done", flush=True)

    arm_names = ["ball_true", "ball_masked"]
    arm_names += [arm_name(k, s, p) for k in ks for s in sigmas for p in ps]
    for name in arm_names:
        per_match = {}
        for match in names:
            rows = mean_totals.get((name, match))
            if rows:
                per_match[match] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        result["arms"][name] = {"per_match": per_match}

    for match in names:
        rows = coverage.get(match)
        if not rows:
            continue
        result["coverage"][match] = {}
        for name in arm_names:
            values = [r[name] for r in rows if name in r]
            if values:
                result["coverage"][match][name] = float(np.mean(values))

    for k_max in ks:
        for sigma in sigmas:
            for p_detect in ps:
                name = arm_name(k_max, sigma, p_detect)
                present = [m for m in names if (name, m, 0) in cue_blocks]
                if not present:
                    continue
                cue_arm, cue_base, own_arm, own_base = [], [], [], []
                for rep in range(replicates):
                    for match in present:
                        cue_arm.append(cue_blocks[(name, match, rep)])
                        cue_base.append(cue_blocks[("ball_masked", match, rep)])
                        own_arm.append(own_blocks[(name, match, rep)])
                        own_base.append(own_blocks[("ball_masked", match, rep)])
                row = p2.bootstrap_delta(
                    p2._by_match(cue_arm, present, replicates),
                    p2._by_match(cue_base, present, replicates),
                    p2._by_match(own_arm, present, replicates),
                    p2._by_match(own_base, present, replicates),
                    p2.BOOTSTRAP,
                    SEED + 11 + int(k_max) * 1000 + int(sigma * 10) * 100 + int(p_detect * 10))
                row["matches"] = present
                result["paired"][name] = row

    result["runtime_s"] = time.time() - started
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def _k_of(name):
    return float(name.split("_k")[1].split("_")[0])


def candidate_selection(result, max_sigma=2.0, min_p=0.6, max_k=25):
    """Weakest passing bridge: largest gap, then smallest p, then largest sigma."""
    passing = []
    for k_max in result["meta"]["ks"]:
        if k_max > max_k:
            continue
        for sigma in result["meta"]["sigmas"]:
            if sigma > max_sigma:
                continue
            for p_detect in result["meta"]["ps"]:
                if p_detect < min_p:
                    continue
                name = arm_name(k_max, sigma, p_detect)
                row = result["paired"].get(name)
                arm = result["arms"].get(name, {}).get("per_match")
                if row is None or not arm:
                    continue
                precision = float(np.mean([v["precision"] for v in arm.values()]))
                recall = float(np.mean([v["recall"] for v in arm.values()]))
                low, high = row["delta_f1_ci90"]
                ok = (row["delta_f1_mean"] >= 0.04 and low > 0
                      and precision >= 0.60 and recall >= 0.70)
                passing.append({
                    "arm": name, "k": k_max, "sigma": sigma, "p": p_detect,
                    "delta_f1": row["delta_f1_mean"], "ci90": [low, high],
                    "f1_mean": float(np.mean([v["f1"] for v in arm.values()])),
                    "precision_mean": precision, "recall_mean": recall,
                    "passes": bool(ok),
                })
    passing.sort(key=lambda r: (r["k"], -r["p"], r["sigma"]), reverse=True)
    winners = [r for r in passing if r["passes"]]
    return {"grid": passing, "worst_passing": winners[0] if winners else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir",
                        default=os.path.join(ROOT, "..", "data", "external", "idsee", "parsed"))
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or
                        os.path.join(ROOT, "..", "data"))
    parser.add_argument("--split", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--k", type=float, nargs="*", default=None)
    parser.add_argument("--sigma", type=float, nargs="*", default=None)
    parser.add_argument("--p", type=float, nargs="*", default=None)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    missing = p2.barath_missing(args.data_dir)
    lengths = missing.pop("run_lengths")
    target_frames = missing["missing_fraction"]
    print("BAR-ATH missing fraction", round(target_frames, 4),
          "runs", missing["missing_runs"], flush=True)

    if args.split == "dev":
        names = p2.DEV
        ks, sigmas, ps = KS, SIGMAS, PS
    else:
        names = p2.CONFIRM
        ks = tuple(args.k) if args.k else (KS[-1],)
        sigmas = tuple(args.sigma) if args.sigma else (2.0,)
        ps = tuple(args.p) if args.p else (0.6,)

    out_path = args.out or os.path.join(HERE, "out", "phase2c_{}.json".format(args.split))
    result = run_split(names, args.parsed_dir, lengths, target_frames,
                       tuple(ks), tuple(sigmas), tuple(ps), args.replicates, out_path)
    result["barath_missing"] = missing
    result["selection"] = candidate_selection(result)

    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("wrote", out_path, "runtime", round(result["runtime_s"], 1), "s", flush=True)
    for name, row in result["paired"].items():
        print(name, "delta", round(row["delta_f1_mean"], 4),
              "ci", [round(v, 4) for v in row["delta_f1_ci90"]], flush=True)
    winner = result["selection"]["worst_passing"]
    print("worst passing", winner["arm"] if winner else None, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
