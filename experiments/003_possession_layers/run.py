#!/usr/bin/env python3
"""003: extra possession layers, each run under its own best GT-free conditions.

Layers under test (all beyond what 002 did):

  own_soft     probabilistic ownership from per-team minimum distance instead of a hard
               nearest-player vote; low-confidence frames become uninformative.
  own_flight   same, plus a ball-in-flight rule (fast ball and no player close) so a pass is
               not forced into a possession.
  st_team      Smart Tagging Team tag on every phase as a weak emission prior, not only restarts.
  id_fix       minimum-dwell repair of short opposite-team runs (identity-switch symptom).
  hsmm         duration-aware Viterbi (bucketed semi-Markov) with a football duration prior.

Rules being respected: no hand labels, no private ground truth, no external data about this
match, no redistribution. The public 5-minute window is a correctness check, never a selector.
Every switch penalty is chosen by the GT-free possession-count band, separately per arm family,
so each layer is judged with its own best calibration rather than a shared constant.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

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
bl = dp2.bl
pm = dp2.pm
GAMES = dp2.GAMES
GAME = dp2.GAME
H, A, D = dp2.H, dp2.A, dp2.D
FPS = dp2.FPS
PITCH_LENGTH, PITCH_WIDTH = dp2.PITCH_LENGTH, dp2.PITCH_WIDTH
LETTERS = dp2.LETTERS

# Ownership model: logistic in the difference of per-team minimum distance.
OWN_SIGMA = 2.0          # metres; distance difference that maps to ~0.73 confidence
OWN_CONF = 0.62          # below this the frame carries no usable team information
FLIGHT_SPEED = 12.0      # m/s; faster than a sprinting player
FLIGHT_DIST = 5.0        # metres; nobody that close means the ball is travelling
ST_BIAS = 0.15           # weak additive cost against the opposite team inside a tagged phase
ID_MINDWELL = 15         # frames; opposite-team runs shorter than this are suspect
HSMM_MU = {H: 300, A: 300, D: 150}          # expected run length in frames
HSMM_EDGES = (0, 25, 75, 150, 300, 600)     # bucket starts, in frames
HSMM_KAPPA = 0.08        # hazard weight once a run passes its expected length
HSMM_KAPPA_GRID = (0.02, 0.05, 0.08, 0.15, 0.30, 1.00, 3.00)

TEAM_TAG = {"Barcelona": H, "Atletic Club": A}

LAMBDA_GRID = (4.0, 8.0, 16.0, 25.0, 32.0, 50.0, 64.0)
COUNT_BAND = dp2.COUNT_BAND
COUNT_CENTRE = dp2.COUNT_CENTRE
COUNT_TOLERANCE = 5.0   # possessions; structural counts cannot separate a few-frame tie


# -- data ------------------------------------------------------------------------------------------

def load_team_distances(data_dir, n_frames):
    """Per frame: minimum distance to a player of each team, nearest distance, nearest team.

    Same parsing contract as 002 (tactical ATD), but keeps the two team minima instead of only the
    winner, which is what the soft ownership layer needs.
    """
    atd_txt = os.path.join(data_dir, "tactical", "{}_tactical_atd.txt".format(GAME))
    atd_xml = os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME))
    if not os.path.isfile(atd_txt):
        raise SystemExit("missing {} (data is gated)".format(atd_txt))
    team_of_track = bl.team_of_each_track(atd_xml)

    d_home = np.full(n_frames, np.inf, dtype=np.float32)
    d_away = np.full(n_frames, np.inf, dtype=np.float32)
    nearest = np.full(n_frames, np.inf, dtype=np.float32)
    nearest_team = np.full(n_frames, -1, dtype=np.int8)
    present = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)

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
                if frame < 0 or frame >= n_frames:
                    continue
                if len(coords) > 2 and coords[2] != "NaN":
                    ball_speed[frame] = float(coords[2])
            except (ValueError, IndexError):
                continue

            bx = x_norm * PITCH_LENGTH
            by = y_norm * PITCH_WIDTH
            ball_x[frame] = bx
            ball_y[frame] = by
            present[frame] = True

            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    first = group.find(",")
                    second = group.find(",", first + 1)
                    px = float(group[:first]) * PITCH_LENGTH
                    py = float(group[first + 1:second if second > 0 else None]) * PITCH_WIDTH
                except (ValueError, IndexError):
                    continue
                distance = float(np.hypot(px - bx, py - by))
                side = int(team_of_track[slot]) if slot < len(team_of_track) else H
                if side == H:
                    if distance < d_home[frame]:
                        d_home[frame] = distance
                else:
                    if distance < d_away[frame]:
                        d_away[frame] = distance
                if distance < nearest[frame]:
                    nearest[frame] = distance
                    nearest_team[frame] = side
    return {
        "d_home": d_home, "d_away": d_away, "nearest": nearest,
        "nearest_team": nearest_team, "present": present,
        "ball_x": ball_x, "ball_y": ball_y, "ball_speed": ball_speed,
    }


def load_smart_team(data_dir, n_frames):
    """Team tag of every Smart Tagging phase, as an int code per frame (-1 unknown)."""
    import xml.etree.ElementTree as ET
    path = os.path.join(data_dir, "tactical", "{}_tactical_smart_tagging.xml".format(GAME))
    tags = np.full(n_frames, -1, dtype=np.int8)
    if not os.path.isfile(path):
        return tags
    for instance in ET.parse(path).getroot().iter("instance"):
        code = (instance.findtext("code") or "").upper()
        if code.startswith("DEFENDING"):
            # The Team tag names the side defending, which is the opposite of possession.
            continue
        team = None
        for label in instance.findall("label"):
            if label.findtext("group") == "Team":
                team = TEAM_TAG.get(label.findtext("text"))
        if team is None:
            continue
        start = int(round(float(instance.findtext("start")) * FPS))
        stop = int(round(float(instance.findtext("end")) * FPS))
        start = max(0, min(n_frames, start))
        stop = max(0, min(n_frames, stop))
        tags[start:stop] = team
    return tags


# -- emission costs --------------------------------------------------------------------------------

def hard_costs(obs):
    """002 semantics: 0 for the winning team, 1 for the other team and for D."""
    n = len(obs)
    cost = np.ones((3, n), dtype=np.float32)
    observed = obs >= 0
    if observed.any():
        idx = np.flatnonzero(observed)
        cost[obs[idx], idx] = 0.0
    return cost


def soft_costs(dists, flight=False, st_team=None):
    """Logistic ownership cost; uninformative where the two minima are close (or the ball flies)."""
    d_home, d_away = dists["d_home"], dists["d_away"]
    n = len(d_home)
    cost = np.ones((3, n), dtype=np.float32)
    present = dists["present"]
    p_home = 1.0 / (1.0 + np.exp(-(d_away - d_home) / OWN_SIGMA))
    p_home = np.clip(p_home, 0.0, 1.0)
    cost[H, present] = 1.0 - p_home[present]
    cost[A, present] = p_home[present]

    confident = present & (np.maximum(p_home, 1.0 - p_home) >= OWN_CONF)
    neutral = present & ~confident
    if flight:
        speed = dists.get("speed")
        if speed is None:
            speed = dists["ball_speed"]
        flying = present & (speed > FLIGHT_SPEED) & (dists["nearest"] > FLIGHT_DIST)
        neutral |= flying
    cost[:, neutral] = 0.5

    if st_team is not None:
        tagged = st_team >= 0
        idx = np.flatnonzero(tagged)
        if idx.size:
            team = st_team[idx]
            cost[1 - team, idx] += ST_BIAS
    return cost


def with_dead(cost, dead):
    if dead is None:
        return cost
    out = cost.copy()
    idx = np.flatnonzero(dead)
    out[2, idx] = 0.0
    out[0, idx] = 1.0
    out[1, idx] = 1.0
    return out


# -- decoders --------------------------------------------------------------------------------------

def viterbi(cost, dead, switch_penalty, n_frames):
    cost = with_dead(cost, dead)
    out = np.full(n_frames, D, dtype=np.int8)
    for _, start, end in dp2.half_video_blocks(GAME):
        out[start:end + 1] = _viterbi_slice(cost[:, start:end + 1], switch_penalty)
    return out


def _viterbi_slice(cost, switch_penalty):
    n = cost.shape[1]
    dp = np.empty((n, 3), dtype=np.float32)
    back = np.empty((n, 3), dtype=np.int8)
    dp[0] = cost[:, 0]
    back[0] = np.arange(3, dtype=np.int8)
    for t in range(1, n):
        prev = dp[t - 1]
        order = np.argsort(prev)
        best, second = int(order[0]), int(order[1])
        for state in range(3):
            same = prev[state]
            if best != state:
                switch = prev[best] + switch_penalty
                source = best
            else:
                switch = prev[second] + switch_penalty
                source = second
            if same <= switch:
                dp[t, state] = same + cost[state, t]
                back[t, state] = state
            else:
                dp[t, state] = switch + cost[state, t]
                back[t, state] = source
    labels = np.empty(n, dtype=np.int8)
    labels[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def duration_prior(cost, dead, switch_penalty, n_frames, kappa, passes=3):
    """Iterated duration prior: a soft semi-Markov that actually bites.

    An explicit-duration Viterbi needs the run length in the state, and a bucketed version only
    helps if the bucket is forced to advance. Here the duration enters through reweighting: decode,
    charge the frames of every run that overshoots its expected length, decode again. The penalty
    grows with the overshoot, so the second decode prefers switching where the first one stretched
    a possession. Cheap (three decodes) and honest about being an approximation.
    """
    base = with_dead(cost, dead)
    extra = np.zeros_like(base)
    codes = None
    for _ in range(passes):
        codes = viterbi(base + extra, None, switch_penalty, n_frames)
        extra = np.zeros_like(base)
        for label in (H, A, D):
            mu = HSMM_MU[label]
            for start, stop in dp2.runs_of(codes == label):
                length = stop - start
                if length <= mu:
                    continue
                extra[label, start + mu:stop] += kappa
    return codes


def merge_short_runs(codes, min_len=ID_MINDWELL, passes=2):
    """Merge short opposite-team runs into the surrounding label; isolated runs become D."""
    out = codes.copy()
    for _ in range(passes):
        for label in (H, A):
            for start, stop in dp2.runs_of(out == label):
                if stop - start >= min_len:
                    continue
                left = out[start - 1] if start > 0 else D
                right = out[stop] if stop < len(out) else D
                if left == right and left != label:
                    out[start:stop] = left
                elif left == D and right in (H, A):
                    out[start:stop] = right
                elif right == D and left in (H, A):
                    out[start:stop] = left
                elif left in (H, A) and right in (H, A) and left != right:
                    left_len = _run_length(out, start - 1)
                    right_len = _run_length(out, stop)
                    out[start:stop] = left if left_len >= right_len else right
    return out


def _run_length(codes, index):
    if index < 0 or index >= len(codes):
        return 0
    label = codes[index]
    left = index
    while left > 0 and codes[left - 1] == label:
        left -= 1
    right = index
    while right + 1 < len(codes) and codes[right + 1] == label:
        right += 1
    return right - left + 1


# -- diagnostics -----------------------------------------------------------------------------------

def structure(codes, n_frames):
    graded = dp2.graded_mask(n_frames)
    spells = pm.possession_spells(codes, dp2.full_match_run_list(n_frames))
    durations = [float(stop - start) / FPS for start, stop, _ in spells]
    alive = int((graded & (codes != D)).sum())
    return {
        "n_possessions": len(durations),
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "min_duration_s": float(np.min(durations)) if durations else 0.0,
        "dead_fraction_graded": float((graded & (codes == D)).sum() / max(1, graded.sum())),
        "switches_per_min": float(max(0, len(spells) - 1) / max(1e-9, alive / FPS / 60.0)),
    }


def pick_lambda(build_cost, dead, n_frames):
    """GT-free: choose the switch penalty inside the accepted possession-count band."""
    rows = []
    for lam in LAMBDA_GRID:
        codes = viterbi(build_cost(), dead, lam, n_frames)
        stats = structure(codes, n_frames)
        rows.append({"lambda": lam, **stats})
    inside = [row for row in rows if COUNT_BAND[0] <= row["n_possessions"] <= COUNT_BAND[1]]
    pool = inside or rows
    best = min(abs(row["n_possessions"] - COUNT_CENTRE) for row in pool)
    # Structural counts cannot separate near-ties, so take the weakest penalty that is within
    # COUNT_TOLERANCE of the best one: no extra assumption unless the count asks for it.
    near = [row for row in pool if abs(row["n_possessions"] - COUNT_CENTRE) <= best + COUNT_TOLERANCE]
    chosen = min(near, key=lambda row: row["lambda"])
    return chosen["lambda"], rows, bool(inside)


# -- synthetic harness -----------------------------------------------------------------------------

def synthetic_series(seed, n_frames):
    rng = np.random.RandomState(seed)
    halves = [(source, start, end) for source, start, end in dp2.half_video_blocks(GAME)]
    truth = dp2.synthetic_truth_iid(halves, n_frames, rng)
    obs = dp2.synthetic_obs_autocorr(truth, _present, rng, 0.01, 50)
    return truth, obs


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-seeds", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    started = time.time()

    log_lines = []

    def log(message):
        line = "{} {}".format(datetime.now(timezone.utc).strftime("%H:%M:%S"), message)
        print(line, flush=True)
        log_lines.append(line)

    n_frames = dp2.source_frames("tactical", GAME)
    log("loading tactical tracking")
    dists = load_team_distances(args.data_dir, n_frames)
    obs = dists["nearest_team"].copy()
    obs[~dists["present"]] = -1

    global _present
    _present = dists["present"]

    restart_starts = dp2.parse_set_piece_starts(args.data_dir, n_frames)
    evidence = dp2.build_evidence(
        n_frames, dists["present"], dists["ball_x"], dists["ball_y"],
        dists["ball_speed"], restart_starts)
    dists["speed"] = dp2.ball_speed_series(
        dists["present"], dists["ball_x"], dists["ball_y"], dists["ball_speed"], FPS)
    st_team = load_smart_team(args.data_dir, n_frames)
    log("evidence final covers {} graded frames; ST team tag covers {} frames".format(
        int(evidence["final"].sum()), int((st_team >= 0).sum())))

    gt_raw = pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME)))
    gt_codes = pm.label_codes(gt_raw)
    if len(gt_codes) != n_frames:
        raise SystemExit("public GT length mismatch")

    arms, definitions = {}, {}

    def add(name, codes, family, description):
        arms[name] = codes
        definitions[name] = {"family": family, "description": description,
                             "structure": structure(codes, n_frames)}

    log("baselines")
    add("base_smooth125", pm.label_codes(bl.smooth(obs, 125)), "baseline",
        "majority vote +/-125, D where missing")
    labels = bl.smooth(obs, 125).copy()
    labels[evidence["final"]] = "D"
    add("ev_smooth125", pm.label_codes(labels), "baseline_dead_postprocess",
        "majority vote +/-125 plus hard D on final evidence")

    log("hard DP with final evidence (reference arm)")
    lam_ref, ref_rows, ref_inside = pick_lambda(lambda: hard_costs(obs), evidence["final"], n_frames)
    log("  lambda {} (band {})".format(lam_ref, ref_inside))
    add("dp_ev", viterbi(hard_costs(obs), evidence["final"], lam_ref, n_frames),
        "missing_aware_dp", "hard nearest-team emissions, final evidence")

    log("identity repair on the observation layer")
    obs_repaired = obs.copy()
    missing = ~dists["present"]
    obs_repaired[missing] = D
    obs_repaired = merge_short_runs(obs_repaired)
    obs_repaired[missing] = -1
    lam_id, id_rows, id_inside = pick_lambda(
        lambda: hard_costs(obs_repaired), evidence["final"], n_frames)
    log("  lambda {} (band {}), {} frames changed".format(
        lam_id, id_inside, int((obs_repaired != obs).sum())))
    add("id_obs_dp", viterbi(hard_costs(obs_repaired), evidence["final"], lam_id, n_frames),
        "identity", "short opposite-team observation runs merged before the DP")

    log("soft ownership")
    lam_soft, soft_rows, soft_inside = pick_lambda(
        lambda: soft_costs(dists), evidence["final"], n_frames)
    log("  lambda {} (band {})".format(lam_soft, soft_inside))
    add("own_soft", viterbi(soft_costs(dists), evidence["final"], lam_soft, n_frames),
        "ownership", "logistic ownership, low-confidence frames uninformative")

    log("soft ownership plus ball-in-flight rule")
    lam_flight, flight_rows, flight_inside = pick_lambda(
        lambda: soft_costs(dists, flight=True), evidence["final"], n_frames)
    log("  lambda {} (band {})".format(lam_flight, flight_inside))
    add("own_flight", viterbi(soft_costs(dists, flight=True), evidence["final"], lam_flight, n_frames),
        "ownership", "soft ownership plus pass/in-flight neutrality")

    log("adding Smart Tagging team prior")
    lam_st, st_rows, st_inside = pick_lambda(
        lambda: soft_costs(dists, flight=True, st_team=st_team), evidence["final"], n_frames)
    log("  lambda {} (band {})".format(lam_st, st_inside))
    st_codes = viterbi(soft_costs(dists, flight=True, st_team=st_team), evidence["final"], lam_st, n_frames)
    add("own_flight_st", st_codes, "ownership_st", "plus weak Team tag prior on all phases")

    log("minimum-dwell repair")
    add("own_flight_st_id", merge_short_runs(st_codes), "ownership_st_id",
        "plus merge of opposite-team runs shorter than 15 frames")

    log("duration-aware HSMM, strength calibrated on the GT-free count band")
    hsmm_cost = soft_costs(dists, flight=True, st_team=st_team)
    kappa_rows = []
    for kappa in HSMM_KAPPA_GRID:
        codes = duration_prior(hsmm_cost, evidence["final"], lam_st, n_frames, kappa)
        kappa_rows.append({"kappa": kappa, **structure(codes, n_frames)})
    inside = [row for row in kappa_rows if COUNT_BAND[0] <= row["n_possessions"] <= COUNT_BAND[1]]
    pool = inside or kappa_rows
    best = min(abs(row["n_possessions"] - COUNT_CENTRE) for row in pool)
    near = [row for row in pool if abs(row["n_possessions"] - COUNT_CENTRE) <= best + COUNT_TOLERANCE]
    kappa_best = min(near, key=lambda row: row["kappa"])["kappa"]
    log("  kappa {} (band {})".format(kappa_best, bool(inside)))
    hsmm_codes = duration_prior(hsmm_cost, evidence["final"], lam_st, n_frames, kappa_best)
    add("hsmm", hsmm_codes, "hsmm", "duration prior via iterative reweighting")

    add("negative_constant_D", np.full(n_frames, D, dtype=np.int8), "negative_control", "constant D")
    add("negative_constant_H", np.full(n_frames, H, dtype=np.int8), "negative_control", "constant H")

    log("public-window metrics (diagnostic only)")
    public = {}
    for name, codes in arms.items():
        public[name] = dp2.metrics_for(gt_codes, codes, args.bootstrap)

    log("synthetic mechanism test (i.i.d. truth, autocorrelated confusion)")
    synthetic = {}
    for index in range(args.synthetic_seeds):
        seed = 1000 + index
        truth, syn_obs = synthetic_series(seed, n_frames)
        runs = {}
        for name, lam in (("dp_ev", lam_ref), ("hsmm", lam_st)):
            codes = viterbi(hard_costs(syn_obs), evidence["final"], lam, n_frames)
            runs[name] = {"codes": codes, "metrics": dp2.metrics_for(truth, codes, 0)}
        codes = viterbi(hard_costs(syn_obs), evidence["final"], lam_ref, n_frames)
        repaired = merge_short_runs(codes)
        runs["dp_ev_id"] = {"codes": repaired, "metrics": dp2.metrics_for(truth, repaired, 0)}
        paired = {}
        base_codes = runs["dp_ev"]["codes"]
        for name in ("hsmm", "dp_ev_id"):
            paired[name] = dp2.paired_spell_bootstrap(
                truth, runs[name]["codes"], base_codes, 400, seed)
        synthetic[str(seed)] = {
            name: {"spell_f1_mean": value["metrics"]["spell_f1_mean"],
                   "macro_f1": value["metrics"]["macro_f1"],
                   "n_pred_spells": value["metrics"]["n_pred_spells"]}
            for name, value in runs.items()
        }
        synthetic[str(seed)]["paired"] = paired

    log("broadcast cross-check (descriptive, timelines are not frame-synced)")
    broadcast = broadcast_check(args.data_dir)

    finished = time.time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    gt_spells = pm.possession_spells(gt_codes, dp2.full_match_run_list(n_frames))
    inputs = {
        "tactical_atd_txt": os.path.join(args.data_dir, "tactical", "{}_tactical_atd.txt".format(GAME)),
        "tactical_atd_xml": os.path.join(args.data_dir, "tactical", "{}_tactical_atd.xml".format(GAME)),
        "smart_tagging_xml": os.path.join(args.data_dir, "tactical", "{}_tactical_smart_tagging.xml".format(GAME)),
        "public_gt": os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME)),
    }
    result = {
        "command": " ".join([".venv/bin/python"] + sys.argv),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "git_commit": git_commit(),
        "inputs_sha256": {name: sha256(path) for name, path in inputs.items()},
        "data_audit": {
            "frames": n_frames,
            "graded_frames": int(dp2.graded_mask(n_frames).sum()),
            "ball_present_frames": int(dists["present"].sum()),
            "public_gt_graded_frames": int(np.sum(gt_codes != 3)),
            "public_gt_possessions": len(gt_spells),
            "public_gt_mean_duration_s": float(np.mean([b - a for a, b, _ in gt_spells])) / FPS,
        },
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "runtime_s": finished - started,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "constants": {
            "own_sigma": OWN_SIGMA, "own_conf": OWN_CONF,
            "flight_speed": FLIGHT_SPEED, "flight_dist": FLIGHT_DIST,
            "st_bias": ST_BIAS, "id_mindwell": ID_MINDWELL,
            "hsmm_mu": HSMM_MU, "hsmm_edges": HSMM_EDGES, "hsmm_kappa": HSMM_KAPPA,
        },
        "evidence_graded_frames": {name: int(mask.sum()) for name, mask in evidence.items()},
        "lambda_selection": {
            "id_obs_dp": {"lambda": lam_id, "inside_band": id_inside, "grid": id_rows},
            "dp_ev": {"lambda": lam_ref, "inside_band": ref_inside, "grid": ref_rows},
            "own_soft": {"lambda": lam_soft, "inside_band": soft_inside, "grid": soft_rows},
            "own_flight": {"lambda": lam_flight, "inside_band": flight_inside, "grid": flight_rows},
            "own_flight_st": {"lambda": lam_st, "inside_band": st_inside, "grid": st_rows},
        },
        "arms": {name: {"definition": definitions[name], "public": public[name]} for name in arms},
        "hsmm_calibration": {"grid": kappa_rows, "selected_kappa": kappa_best,
                             "inside_band": bool(inside), "lambda": lam_st},
        "synthetic": synthetic,
        "broadcast": broadcast,
        "compute": {
            "peak_rss_mb": usage.ru_maxrss / 1024.0,
            "user_cpu_s": usage.ru_utime,
            "sys_cpu_s": usage.ru_stime,
        },
        "rules": {
            "public_window": "diagnostic only; no arm selected on it",
            "switches": "each lambda chosen by the GT-free possession-count band",
            "full_match_gt": "not used, not available",
        },
    }

    with open(os.path.join(args.out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=float)

    with open(os.path.join(args.out_dir, "results.csv"), "w") as handle:
        handle.write("arm,family,spell_f1_mean,ci_low,ci_high,macro_f1,n_pred,duration_s,fragmentation,dead_fraction,possessions_full_match\n")
        for name in arms:
            row = public[name]
            handle.write("{},{},{:.6f},{:.6f},{:.6f},{:.6f},{},{:.4f},{:.4f},{:.4f},{}\n".format(
                name, definitions[name]["family"], row["spell_f1_mean"],
                row["spell_bootstrap_low"], row["spell_bootstrap_high"], row["macro_f1"],
                row["n_pred_spells"], row["mean_spell_duration_s"], row["fragmentation"],
                definitions[name]["structure"]["dead_fraction_graded"],
                definitions[name]["structure"]["n_possessions"]))

    with open(os.path.join(args.out_dir, "run.log"), "w") as handle:
        handle.write("\n".join(log_lines) + "\n")
    log("wrote results to {}".format(args.out_dir))
    return 0


def broadcast_check(data_dir):
    """Run the same structure on broadcast tracking, without frame-level alignment."""
    path = os.path.join(data_dir, "broadcast", "{}_broadcast_atd.txt".format(GAME))
    if not os.path.isfile(path):
        return {"available": False}
    frames = 0
    present = 0
    switches = 0
    previous = None
    with open(path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frames += 1
            ball = parts[2]
            if not ball or ball[0] == "N":
                continue
            present += 1
            if previous is not None and ball[:12] != previous:
                switches += 1
            previous = ball[:12]
    return {
        "available": True,
        "records": frames,
        "ball_present": present,
        "presence_fraction": present / max(1, frames),
        "raw_ball_string_changes": switches,
        "note": "broadcast is not frame-synced to the tactical video; used as an independent presence check only",
    }


if __name__ == "__main__":
    sys.exit(main())
