"""ryu's committed final pipeline applied zero-shot to the frozen proxy (reference arm).

Source: origin/ryu commit bd90a9c, ``experiments/015_final_pipeline/run.py`` with
``weights_new/`` (3-seed two-head TCN trained by ryu on 3 Metrica + 20 SkillCorner games).
The committed ``run.py`` only runs with ``weights_new`` (its ``weights/`` branch fails on a
missing ``dead_share`` field), so that is the pipeline reproduced here. ryu's modules
(features, decoder, train_model, common) are imported verbatim from ``reference/ryu``;
only ``merge_short`` is copied, because ``run.py`` imports BAR-ATH-only modules at top level.
Every constant is ryu's; nothing is fitted on the proxy.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from exp1_common import ROOT, Match

RYU_DIR = ROOT / "reference" / "ryu"
WEIGHTS = RYU_DIR / "weights_new"
FPS, STRIDE = 25, 5
GATE_R, DEAD_THR, MIN_DEAD_S, KAPPA, MERGE_S = 0.35, 0.5, 2.0, 0.02, 0.5  # ryu run.py
TARGET_DEAD_SHARE = 0.407  # ryu run.py (La Liga effective playing time)
RYU_PITCH = np.array([105.3, 68.0])  # ryu features normalise by BAR-ATH pitch size
CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def _modules():
    if str(RYU_DIR) not in sys.path:
        sys.path.insert(0, str(RYU_DIR))
    import decoder as dc  # noqa: E402  (ryu, verbatim)
    import features as ft  # noqa: E402
    import train_model as tm  # noqa: E402
    import common as ryu_common  # noqa: E402

    return dc, ft, tm, ryu_common


def merge_short(lab, blocks, min_s, runs):
    """Verbatim from ryu run.py: merge H/A runs shorter than min_s seconds into the previous run."""
    out = lab.copy()
    for _, a, b in blocks:
        seg = out[a:b + 1]
        s, e = runs((seg == "H") | (seg == "A"))
        for si, ei in zip(s, e):
            sub = seg[si:ei]
            ch = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            starts = np.r_[0, ch]
            ends = np.r_[ch, len(sub)]
            for k in range(1, len(starts)):
                if (ends[k] - starts[k]) < min_s * FPS:
                    sub[starts[k]:ends[k]] = sub[starts[k] - 1]
            seg[si:ei] = sub
    return out


def _balanced(values: dict) -> float:
    by_source: dict[str, list[float]] = {}
    for name, value in values.items():
        by_source.setdefault(name.split("_")[0], []).append(value)
    return float(np.mean([np.mean(v) for v in by_source.values()]))


def load_ensemble(device: str):
    """Mirror of run.py's weights_new branch: ensemble, inputs, rate and dead-ball prior shift."""
    _, _, tm, _ = _modules()
    meta = np.load(WEIGHTS / "meta.npz")
    report = json.loads((WEIGHTS / "training_report.json").read_text())
    rate, train_dead = _balanced(report["poss_per_min"]), _balanced(report["dead_share"])
    dilation = tuple(int(x) for x in meta["dil"])
    inputs = tm.make_inputs(meta["mu"], meta["sd"])
    nets = []
    for seed in range(sum(1 for _ in WEIGHTS.glob("tcn_seed*.pt"))):
        net = tm.TCN2(tm.NCH, dil=dilation).to(device)
        net.load_state_dict(torch.load(WEIGHTS / f"tcn_seed{seed}.pt", map_location=device))
        nets.append(net.eval())
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    shift = float(logit(TARGET_DEAD_SHARE) - logit(train_dead))
    return nets, inputs, rate, shift


def ryu_arm(match: Match, ensemble, device: str) -> tuple[np.ndarray, dict]:
    dc, ft, tm, ryu_common = _modules()
    nets, inputs, rate, shift = ensemble
    n = len(match.half)
    players = match.players[..., :2] / RYU_PITCH
    ball = match.ball / RYU_PITCH
    half = match.half.astype(int)
    blocks = [(h, int(np.flatnonzero(half == h)[0]), int(np.flatnonzero(half == h)[-1])) for h in (1, 2)]
    graded_min = float((half > 0).sum()) / FPS / 60
    feats = ft.build(players, ball, match.teams, half)
    p_dead, p_team = np.full(n, np.nan), np.full(n, 0.5)
    with torch.no_grad():
        for _, a, b in blocks:
            ix, full = np.arange(a, b + 1, STRIDE), np.arange(a, b + 1)
            pl, bl, vd = feats["player"][:, ix], feats["ball"][:, ix], feats["pvalid"][:, ix]
            out = [tm.predict(net, inputs, pl, bl, vd, False, device) for net in nets]
            p_dead[a:b + 1] = np.interp(full, ix, np.mean([o[0] for o in out], 0))
            p_team[a:b + 1] = np.interp(full, ix, np.mean([o[1] for o in out], 0))
    dead = dc.dead_mask(p_dead, blocks, DEAD_THR, MIN_DEAD_S, shift)
    dist = dc.nearest_distances(players, ball, match.teams)
    lp_h, lp_a = dc.ball_evidence(dist, GATE_R)
    h_ev, a_ev = dc.evidence(lp_h, lp_a, p_team, KAPPA)

    def make(lam):
        return dc.decode(n, blocks, h_ev, a_ev, lam, dead)

    target = rate * graded_min
    lam_star = dc.calibrate_lambda(make, blocks, target)
    labels = merge_short(make(lam_star), blocks, MERGE_S, ryu_common.runs)
    codes = np.vectorize(CODES.get)(labels).astype("int8")
    info = {"lambda": lam_star, "target_possessions": target, "rate_per_min": rate, "dead_shift": shift}
    return codes, info
