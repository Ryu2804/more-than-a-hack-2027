#!/usr/bin/env python3
"""Sanity test: ball-free dead detector (trained on IDSSE) applied to BAR-ATH 5-minute window.

Background: the ball-based cue out_or_slow2 scores 0.05 on this window (it is not deployable there).
The ball-free logistic detector uses only player collective motion, so it can in principle transfer.

Design: train on all 5 IDSSE dev matches (frozen features/thresholds, no BAR-ATH tuning), predict
dead on BAR-ATH frames, plug into touch_hold (r 0.8, minlen 30) + merge 150, score the public
5-minute window with the official scorer. This window is a sanity veto only (17 possessions), never
a selector. No submission CSV is written.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))
import ballfree_dead as BD  # noqa: E402
import barath as B  # noqa: E402
import baseline as B0  # noqa: E402
import deployable as D  # noqa: E402
import harness as H  # noqa: E402
import phase2b as P2B  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402

GAME = "BAR-ATH"
R_M, MINLEN, MERGE_K = 0.8, 30, 150
WINDOW = (11185, 18684)
CHUNK = 50000


def features_from_speeds(speeds, home_mask, away_mask, n_visible):
    """Same 11 ball-free features as ballfree_dead.features, from (slots x frames) speeds."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(speeds, axis=0)
        mean = np.nanmean(speeds, axis=0)
        std = np.nanstd(speeds, axis=0)
        frac_slow = (speeds < 0.5).sum(axis=0) / np.maximum(1, n_visible)
        frac_fast = (speeds > 3.0).sum(axis=0) / np.maximum(1, n_visible)
        med_home = np.nanmedian(np.where(home_mask[:, None], speeds, np.nan), axis=0)
        med_away = np.nanmedian(np.where(away_mask[:, None], speeds, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    med_home = np.where(np.isfinite(med_home), med_home, 0.0)
    med_away = np.where(np.isfinite(med_away), med_away, 0.0)
    diff = np.abs(med_home - med_away)
    def roll(x, w):
        out = np.zeros_like(x)
        c = np.cumsum(np.r_[0.0, x])
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            out[i] = (c[i + 1] - c[lo]) / (i - lo + 1)
        return out
    X = np.column_stack([med, mean, std, frac_slow, frac_fast, n_visible,
                         med_home, med_away, diff, roll(med, 25), roll(med, 100)])
    return X


def load_atd_all(data_dir):
    n = B.source_frames("tactical", GAME)
    bx, by, present, px, py, ps, max_slot = P2B.load_atd(data_dir, n, GAME)
    team_slots = np.asarray(B0.team_of_each_track(
        os.path.join(data_dir, "tactical", "{}_tactical_atd.xml".format(GAME))))[:max_slot]
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
    return n, bx, by, present, px, py, ps, max_slot, team_slots, dmin, index, team


def score_pred(gt, pred, low, high):
    gt_w = gt[low:high + 1]
    pred_w = pred[low:high + 1]
    runs = pm.valid_runs(gt_w)
    scores = pm.spell_scores(gt_w, np.where(pred_w < 0, 2, pred_w), runs)
    low_ci, high_ci = pm.bootstrap_spell_f1(gt_w, np.where(pred_w < 0, 2, pred_w), runs, 2000)
    return {"spell_f1": float(scores["mean_f1"]), "spell_f1_ci90": [low_ci, high_ci],
            "n_gt": int(scores["n_gt"]), "n_pred": int(scores["n_pred"])}


def main():
    data_dir = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data")

    # ---- train on all 5 IDSSE dev matches ----
    X_parts, y_parts = [], []
    for name in H.DEV:
        npz = H.load(name)
        X, _ = BD.features(npz)
        status = npz["ball_status"]
        g = status >= 0
        X_parts.append(X[g])
        y_parts.append((status[g] == 0).astype(np.int64))
    Xtr = np.vstack(X_parts)
    ytr = np.concatenate(y_parts)
    w, mean, std = BD.train_logreg(Xtr, ytr, l2=1e-2)
    print(f"trained on {Xtr.shape[0]} IDSSE graded frames, dead rate {ytr.mean():.3f}", flush=True)

    # ---- scale-free variant: z-score features per match before training ----
    # ATD speeds are systematically ~3x lower than IDSEE (median 0.44 vs 1.43 m/s) and n_visible
    # differs (10.5 vs 22), so absolute features saturate the logistic on BAR-ATH. This variant
    # tests whether the signal survives when each match is scaled onto its own feature distribution.
    # BAR-ATH normalization uses full-match statistics only (no window labels, no window tuning).
    Xz_parts = []
    for name in H.DEV:
        npz = H.load(name)
        X, _ = BD.features(npz)
        g = npz["ball_status"] >= 0
        mu = X[g].mean(axis=0)
        sd = np.where(X[g].std(axis=0) < 1e-8, 1.0, X[g].std(axis=0))
        Xz_parts.append((X[g] - mu) / sd)
    Xztr = np.vstack(Xz_parts)
    wz, mean_z, std_z = BD.train_logreg(Xztr, ytr, l2=1e-2)
    print("trained z-score variant", flush=True)

    # ---- BAR-ATH ----
    n, bx, by, present, px, py, ps, max_slot, team_slots, dmin, index, team = load_atd_all(data_dir)
    gt = B.public_gt(data_dir)
    low, high = WINDOW
    graded = np.zeros(n, dtype=bool)
    graded[low:high + 1] = True

    speeds = (ps.astype(np.float64) / 3.6)
    home = team_slots == 0
    away = team_slots == 1
    n_visible = np.isfinite(speeds).sum(axis=0).astype(np.float64)
    Xb = features_from_speeds(speeds, home, away, n_visible)
    p = BD.predict(w, mean, std, Xb)
    dead_bf = (p > 0.5) & graded

    # scale-free prediction: BAR-ATH features normalized with full-match statistics
    mu_b = Xb.mean(axis=0)
    sd_b = np.where(Xb.std(axis=0) < 1e-8, 1.0, Xb.std(axis=0))
    Xb_z = (Xb - mu_b) / sd_b
    p_z = BD.predict(wz, mean_z, std_z, Xb_z)
    dead_bf_z = (p_z > 0.5) & graded
    print(f"ball-free raw: flagged {dead_bf[graded].mean():.3f} mean p {p[graded].mean():.3f}; "
          f"z-score: flagged {dead_bf_z[graded].mean():.3f} mean p {p_z[graded].mean():.3f}",
          flush=True)

    held = T.hold(dmin, team, R_M, MINLEN)
    arms = {}

    def arm_pred(codes, dead_arm):
        return np.where(dead_arm, 2, np.where(codes >= 0, codes, 2)).astype(np.int8)

    # primary: touch_hold + ball-free dead (no merge, same as barath.py arms)
    pred = arm_pred(held, dead_bf)
    arms["touch_hold_ballfree_dead"] = score_pred(gt, pred, low, high)
    # scale-free variant of the same detector
    arms["touch_hold_ballfree_zscore"] = score_pred(
        gt, arm_pred(held, dead_bf_z), low, high)
    arms["touch_hold_ballfree_zscore_merge150"] = score_pred(
        gt, D.merge_short_dead(arm_pred(held, dead_bf_z), MERGE_K), low, high)
    # variant with the IDSSE-proven merge of short D runs (k=150)
    pred_m = D.merge_short_dead(pred, MERGE_K)
    arms["touch_hold_ballfree_dead_merge150"] = score_pred(gt, pred_m, low, high)
    # references computed on the same context for direct comparability
    arms["touch_hold_deadmissing"] = score_pred(gt, arm_pred(held, (~present) & graded), low, high)
    arms["touch_hold_deadoracle"] = score_pred(gt, arm_pred(held, (gt == 2) & graded), low, high)

    # dead quality of ball-free vs public GT D inside the window
    ytrue = (gt[low:high + 1] == 2).astype(int)
    yhat = dead_bf[low:high + 1].astype(int)
    tp = int(((ytrue == 1) & (yhat == 1)).sum())
    fp = int(((ytrue == 0) & (yhat == 1)).sum())
    fn = int(((ytrue == 1) & (yhat == 0)).sum())
    dead_q = {"precision": tp / (tp + fp) if tp + fp else 0.0,
              "recall": tp / (tp + fn) if tp + fn else 0.0,
              "flagged": float(yhat.mean())}

    out = {
        "game": GAME,
        "window": list(WINDOW),
        "method": ("ball-free logistic dead detector trained on 5 IDSSE dev matches "
                   "(11 player-motion features, no ball) + touch_hold r0.8/minlen30"),
        "arms": arms,
        "dead_quality_in_window": dead_q,
        "window_note": "5-minute sanity only; frozen IDSSE-trained detector; never a selector",
    }
    path = os.path.join(HERE, "out", "barath_ballfree.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    for k, v in arms.items():
        print(k, round(v["spell_f1"], 4), "n_pred", v["n_pred"],
              "CI90", [round(x, 4) for x in v["spell_f1_ci90"]], flush=True)
    print("dead quality in window: P {:.3f} R {:.3f} flagged {:.3f}".format(
        dead_q["precision"], dead_q["recall"], dead_q["flagged"]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
