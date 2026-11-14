#!/usr/bin/env python3
"""Migrate Ryu TCN + Viterbi pipeline to per-match IDSSE predictions (best-effort)."""

import os
import sys
import json
import time
import math
from pathlib import Path

RYU = "/tmp/opencode/ryu-worktree"
os.environ["USE_PP"] = "1"
os.environ["TOUCH_FEATS"] = "0"
for p in (RYU, RYU + "/experiments", RYU + "/experiments/015_final_pipeline"):
    sys.path.insert(0, p)
sys.path.insert(0, "/tmp/opencode/jeremy-worktree/experiments/exp_1")

import numpy as np
import torch

import common as rcommon  # ryu common
import decoder as rdecoder
import features as rfeatures
import train_model as rtrain

# jeremy helpers
import exp1_common as jc

OUT = Path("/tmp/opencode/ryu_migration_out")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pred").mkdir(exist_ok=True)
MATCH_IDS = list(jc.MATCH_IDS)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


WD = Path(RYU) / "experiments" / "015_final_pipeline" / "weights_nometrica"
meta = np.load(WD / "meta.npz")
mu, sd = meta["mu"], meta["sd"]
dil = tuple(int(x) for x in meta["dil"])
rate_mean = float(meta["rate_mean"])
train_dead = float(meta["dead_share_mean"])
rep = json.loads((WD / "training_report.json").read_text())
ch = int(meta["ch"]) if "ch" in meta.files else 64

nets = []
for s in range(sum(1 for _ in WD.glob("tcn_seed*.pt"))):
    net = rtrain.TCN2(rtrain.NCH, ch=ch, dil=dil).to("cpu")
    net.load_state_dict(torch.load(WD / f"tcn_seed{s}.pt", map_location="cpu"))
    nets.append(net.eval())

# ryu uses USE_PP from env; ensure features flag matches
rfeatures.USE_PP = True


def load_match(name):
    return jc.load_degraded(name)


def build_arrays(match):
    # P (N, K, 3): x,y normalized, speed
    players_m = match.players  # (N,K,3)
    n, k, _ = players_m.shape
    pitch = np.array([105.0, 68.0], dtype=np.float32)
    P = players_m.copy().astype(np.float32)
    P[:, :, 0] = players_m[:, :, 0] / pitch[0]
    P[:, :, 1] = players_m[:, :, 1] / pitch[1]
    # speed keep
    B = np.full((n, 3), np.nan, dtype=np.float32)
    B[:, 0] = match.ball[:, 0] / pitch[0]
    B[:, 1] = match.ball[:, 1] / pitch[1]
    B[:, 2] = np.nan
    T = np.asarray(match.teams[:k], dtype=np.int8)
    # normalize team to 0/1
    uniq = np.unique(T)
    if len(uniq) == 2:
        T = np.where(T == uniq[0], 0, 1).astype(np.int8)
    half = match.half.astype(np.int8)
    return P, B, T, half, n


def half_blocks(half):
    blocks = []
    for h in (1, 2):
        s, e = rcommon.runs(half == h)
        for si, ei in zip(s, e):
            blocks.append((h, int(si), int(ei - 1)))
    return blocks


def merge_short(lab, blocks, min_s):
    out = lab.copy()
    FPS = 25
    for _, a, b in blocks:
        seg = out[a:b + 1]
        s, e = rcommon.runs((seg == "H") | (seg == "A"))
        for si, ei in zip(s, e):
            sub = seg[si:ei]
            ch = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            starts = np.r_[0, ch]; ends = np.r_[ch, len(sub)]
            for k in range(1, len(starts)):
                if (ends[k] - starts[k]) < min_s * FPS:
                    sub[starts[k]:ends[k]] = sub[starts[k] - 1]
            seg[si:ei] = sub
        out[a:b + 1] = seg
    return out


def run_one(match):
    P, B, T, half, n = build_arrays(match)
    blocks = half_blocks(half)
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    graded_min = graded.sum() / 25.0 / 60.0

    inputs = rtrain.make_inputs(mu, sd)
    F = rfeatures.build(P, B[:, :2], T, half)
    p_dead = np.full(n, np.nan, dtype=np.float64)
    p_team = np.full(n, 0.5, dtype=np.float64)
    for _, a, b in blocks:
        ix = np.arange(a, b + 1, 5)
        if len(ix) < 2:
            continue
        pl = F["player"][:, ix]
        bl = F["ball"][:, ix]
        vd = F["pvalid"][:, ix]
        if rfeatures.USE_PP:
            pp, ppm = rfeatures.player_set(P, B[:, :2], T, half, ix)
            out = [rtrain.predict(net, inputs, pl, bl, vd, False, "cpu", pp, ppm) for net in nets]
        else:
            out = [rtrain.predict(net, inputs, pl, bl, vd, False, "cpu") for net in nets]
        p_dead[a:b + 1] = np.interp(np.arange(a, b + 1), ix, np.mean([o[0] for o in out], 0))
        p_team[a:b + 1] = np.interp(np.arange(a, b + 1), ix, np.mean([o[1] for o in out], 0))

    shift = logit(0.407) - logit(train_dead)
    dead = rdecoder.dead_mask(p_dead, blocks, 0.5, 2.0, shift)
    dist = rdecoder.nearest_distances(P[..., :2], B[:, :2], T)
    lp_h, lp_a = rdecoder.ball_evidence(dist, 0.35)
    h_ev, a_ev = rdecoder.evidence(lp_h, lp_a, p_team, 0.05)

    target = rate_mean * 1.0 * graded_min
    make = lambda lam: rdecoder.decode(n, blocks, h_ev, a_ev, lam, dead, forced_dead=())
    lam_star = rdecoder.calibrate_lambda(make, blocks, target)
    lab = merge_short(make(lam_star), blocks, 1.0)
    return lab


def write_csv(path, codes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("frame,label\n")
        for i, c in enumerate(codes):
            f.write(f"{i},{c}\n")


def main():
    for name in MATCH_IDS:
        out_pred = OUT / "pred" / f"{name}_pred.csv"
        if out_pred.exists():
            print(name, "already done", flush=True)
            continue
        t0 = time.time()
        m = load_match(name)
        lab = run_one(m)
        # align to rasya frame range
        base = 10000
        n_frames = int(m.frame.max()) - base + 1
        aligned = np.full(n_frames, "X", dtype="<U1")
        for idx, frame_id in enumerate(m.frame):
            if base <= frame_id < base + n_frames:
                aligned[int(frame_id) - base] = lab[idx]
        # write
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        with out_pred.open("w") as f:
            f.write("frame,label\n")
            for i, c in enumerate(aligned):
                f.write(f"{i},{c}\n")
        print(name, "done in", round(time.time() - t0, 1), flush=True)
    print("all ryu predictions written", flush=True)


if __name__ == "__main__":
    main()