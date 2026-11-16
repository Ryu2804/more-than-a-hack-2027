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


def team_distances():
    """Nearest-player distance to the ball in metres for each team, cached. (N,2) [home, away]; NaN if unknown."""
    def build():
        P, B, T = load_atd_arrays()
        n = len(B)
        out = np.full((n, 2), np.nan, np.float32)
        L, W = 105.3, 68.0
        for a in range(0, n, 20000):
            b = min(n, a + 20000)
            d = np.hypot(P[a:b, :, 0] * L - (B[a:b, 0] * L)[:, None],
                         P[a:b, :, 1] * W - (B[a:b, 1] * W)[:, None])
            d = np.where(np.isnan(d), np.inf, d)      # absent player = infinitely far (np.min would propagate NaN)
            for t in (0, 1):
                dt = np.where((T == t)[None], d, np.inf).min(1)
                out[a:b, t] = np.where(np.isfinite(dt), dt, np.nan)
        return out
    return cached("team_dist", build)


def viterbi_2state(lp_h, lp_a, switch_cost):
    """Most likely H/A path. lp_* are per-frame log-emissions (0 where there is no evidence, so the
    state simply holds). switch_cost is in nats. Returns int8 0=H, 1=A."""
    n = len(lp_h)
    em = np.stack([lp_h, lp_a], 1).astype(np.float64)
    back = np.zeros((n, 2), np.int8)
    s0, s1 = em[0]
    for t in range(1, n):
        a0, a1 = s0, s1
        # to state 0: stay (a0) or switch from 1 (a1 - cost)
        if a0 >= a1 - switch_cost:
            b0, p0 = a0, 0
        else:
            b0, p0 = a1 - switch_cost, 1
        if a1 >= a0 - switch_cost:
            b1, p1 = a1, 1
        else:
            b1, p1 = a0 - switch_cost, 0
        back[t, 0], back[t, 1] = p0, p1
        s0, s1 = b0 + em[t, 0], b1 + em[t, 1]
    path = np.zeros(n, np.int8)
    path[-1] = 0 if s0 >= s1 else 1
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(mask):
    """(start, end_exclusive) of each True run in a boolean array."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def viterbi_k(em, cost):
    """K-state Viterbi. em (n,K) log-emissions; cost (K,K) nats for switching i->j (zero diagonal). Returns int path."""
    n, K = em.shape
    score = em[0].astype(np.float64).copy()
    back = np.zeros((n, K), np.int8)
    ar = np.arange(K)
    for t in range(1, n):
        cand = score[:, None] - cost
        bi = cand.argmax(0)
        score = cand[bi, ar] + em[t]
        back[t] = bi
    path = np.zeros(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def soft_control(sigma):
    """Soft ball control per frame: c=(S_H-S_A)/(S_H+S_A+eps) in [-1,1] and a mask of frames that carry evidence
    (ball visible and some player within a few sigma). S_team = sum over that team's players of exp(-d/sigma)."""
    def build():
        P, B, T = load_atd_arrays()
        n = len(B)
        out = np.zeros((n, 2), np.float32)
        L, W = 105.3, 68.0
        for a in range(0, n, 20000):
            b = min(n, a + 20000)
            d = np.hypot(P[a:b, :, 0] * L - (B[a:b, 0] * L)[:, None], P[a:b, :, 1] * W - (B[a:b, 1] * W)[:, None])
            w = np.where(np.isnan(d), 0.0, np.exp(-np.nan_to_num(d, nan=1e9) / sigma))
            out[a:b, 0] = (w * (T == 0)[None]).sum(1)
            out[a:b, 1] = (w * (T == 1)[None]).sum(1)
        return out
    S = cached("soft_%g" % sigma, build)
    tot = S[:, 0] + S[:, 1]
    c = (S[:, 0] - S[:, 1]) / (tot + 0.05)
    obs = np.isfinite(load_atd_arrays()[1][:, 0]) & (tot >= 0.05)
    return c.astype(np.float64), obs


def hmm2_em(c, obs, blocks, n_iter=40, tol=1e-4, alpha=1.0):
    """Unsupervised 2-state (H=+m, A=-m) Gaussian HMM, symmetric, shared sd, learned switch prob q.
    Frames without evidence carry likelihood 1 (state just evolves). blocks: [(a,b)] inclusive, decoded separately.
    Returns dict(q, m, s, ll_hist)."""
    xs = [c[a:b + 1] for a, b in blocks]
    os_ = [obs[a:b + 1] for a, b in blocks]
    allx = np.concatenate([x[o] for x, o in zip(xs, os_)])
    q, m, s = 0.002, float(np.median(np.abs(allx))) or 0.3, float(allx.std()) * 0.7
    hist = []
    for it in range(n_iter):
        sw = tot = 0.0
        num_m = den_m = 0.0
        keep = []
        ll = 0.0
        for x, o in zip(xs, os_):
            n = len(x)
            g = np.exp(-0.5 * ((x[:, None] - np.array([m, -m])[None]) / s) ** 2) / (s * 2.5066282746)
            lik = np.where(o[:, None], np.maximum(g, 1e-300) ** alpha, 1.0)   # alpha<1: power likelihood (tempering)
            lh, la = lik[:, 0].tolist(), lik[:, 1].tolist()
            p = 1.0 - q
            ah, aa = [0.0] * n, [0.0] * n
            cs = [0.0] * n
            h, a_ = 0.5 * lh[0], 0.5 * la[0]
            z = h + a_; cs[0] = z; ah[0], aa[0] = h / z, a_ / z
            for t in range(1, n):
                h = (ah[t - 1] * p + aa[t - 1] * q) * lh[t]
                a_ = (aa[t - 1] * p + ah[t - 1] * q) * la[t]
                z = h + a_; cs[t] = z; ah[t], aa[t] = h / z, a_ / z
            bh, ba = [1.0] * n, [1.0] * n
            for t in range(n - 2, -1, -1):
                bh[t] = (p * lh[t + 1] * bh[t + 1] + q * la[t + 1] * ba[t + 1]) / cs[t + 1]
                ba[t] = (p * la[t + 1] * ba[t + 1] + q * lh[t + 1] * bh[t + 1]) / cs[t + 1]
            ah_, aa_, bh_, ba_ = map(np.array, (ah, aa, bh, ba))
            gh, ga = ah_ * bh_, aa_ * ba_
            # expected switches
            sw += float(np.sum((ah_[:-1] * q * np.array(la[1:]) * ba_[1:] + aa_[:-1] * q * np.array(lh[1:]) * bh_[1:]) / np.array(cs[1:])))
            tot += n - 1
            ob = o
            num_m += float(np.sum((gh[ob] - ga[ob]) * x[ob])); den_m += float(np.sum(gh[ob] + ga[ob]))
            keep.append((gh[ob], ga[ob], x[ob]))
            ll += float(np.sum(np.log(cs)))
        q_new = min(max(sw / tot, 1e-5), 0.2)
        m_new = num_m / den_m
        res = sum(float(np.sum(g1 * (x - m_new) ** 2 + g2 * (x + m_new) ** 2)) for g1, g2, x in keep)
        s_new = max((res / den_m) ** 0.5, 1e-3)
        hist.append(ll)
        done = it > 0 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2])
        q, m, s = q_new, m_new, s_new
        if done:
            break
    return dict(q=q, m=m, s=s, ll_hist=hist)


def em_decode(c, obs, blocks, params, alpha=1.0):
    """Viterbi with the learned parameters. Returns {half_index: path} aligned with blocks."""
    lam = float(np.log((1 - params["q"]) / params["q"]))
    out = []
    for a, b in blocks:
        x, o = c[a:b + 1], obs[a:b + 1]
        lp_h = np.where(o, -0.5 * alpha * ((x - params["m"]) / params["s"]) ** 2, 0.0)
        lp_a = np.where(o, -0.5 * alpha * ((x + params["m"]) / params["s"]) ** 2, 0.0)
        out.append(viterbi_2state(lp_h, lp_a, lam))
    return out, lam


def gap_dead_ball(theta=0.5):
    """004's gap classifier: list of (f0, f1, p_dead) for ball gaps >= 2 s in the graded halves, plus the table."""
    import pandas as pd
    from games import half_video_blocks
    from load_smart_tagging import load as load_tags
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    P, B, T = load_atd_arrays()
    FPS, L, W = 25, 105.3, 68.0
    bx, by = B[:, 0] * L, B[:, 1] * W
    vis = np.isfinite(bx)
    spd = P[..., 2]; fin = np.isfinite(spd); nf = fin.sum(1)
    with np.errstate(all="ignore"):
        med = np.where(nf > 0, np.nanmedian(np.where(fin, spd, np.nan), axis=1), np.nan)
        mov = np.where(nf > 0, (np.where(fin, spd, 0) > 1.5).sum(1) / np.maximum(nf, 1), np.nan)
    bd = lambda i: min(bx[i], L - bx[i], by[i], W - by[i])
    tags = pd.DataFrame([{**{k: v for k, v in t.items() if k != "tags"}, **t["tags"]} for t in load_tags()])
    sps = tags[tags.code == "SET PIECES"].start_frame.values
    rows = []
    for half, a, b in half_video_blocks("BAR-ATH"):
        s, e = runs(~vis[a:b + 1])
        for si, ei in zip(s, e):
            f0, f1 = a + si - 1, a + ei
            if f0 < a or f1 > b or (ei - si) / FPS < 2:
                continue
            dur = (ei - si) / FPS
            disp = float(np.hypot(bx[f1] - bx[f0], by[f1] - by[f0]))
            with np.errstate(all="ignore"):
                sp_, mv_ = np.nanmean(med[a + si:a + ei]), np.nanmean(mov[a + si:a + ei])
            rows.append(dict(half=half, f0=a + si, f1=a + ei, dur=dur, d_start=bd(f0), d_end=bd(f1), disp=disp, v_imp=disp / dur,
                             spd=sp_, mov=mv_, tag=bool(((sps >= f0 - 5 * FPS) & (sps <= f1 + 5 * FPS)).any())))
    G = pd.DataFrame(rows)
    X = pd.DataFrame({"a": np.log(G.dur), "b": G.d_start.clip(-5, 40), "c": G.d_end.clip(-5, 40), "d": np.log1p(G.disp),
                      "e": G.v_imp.clip(0, 30), "f": G.spd.fillna(G.spd.median()), "g": G.mov.fillna(G.mov.median())})
    y = G.tag.astype(int).values
    mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X, y)
    pb = np.clip(mdl.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    pr = y.mean()
    G["p_dead"] = 1 / (1 + np.exp(-(np.log(pb / (1 - pb)) + np.log(pr / (1 - pr)))))
    return G


def integrated_autocorr_time(resid, obs, max_lag=250):
    """Integrated autocorrelation time of a residual series over contiguous observed stretches (1 + 2*sum rho_k, cut at first <= 0)."""
    stretches = []
    st, en = runs(obs)
    for a, b in zip(st, en):
        if b - a > 4 * max_lag:
            stretches.append(resid[a:b])
    if not stretches:
        return 1.0
    num = np.zeros(max_lag + 1); den = 0.0
    for x in stretches:
        x = x - x.mean()
        den += float(np.sum(x * x))
        for k in range(max_lag + 1):
            num[k] += float(np.sum(x[: len(x) - k] * x[k:]))
    rho = num / den
    tau = 1.0
    for k in range(1, max_lag + 1):
        if rho[k] <= 0:
            break
        tau += 2 * rho[k]
    return tau


def external_possession_rate():
    """(rate per minute, sd, per-match DataFrame-like list) from the cached StatsBomb open-data events (see 006)."""
    import glob, json
    rates = []
    for f in sorted(glob.glob(os.path.join(CACHE, "ext", "ev_*.json"))):
        ev = [e for e in json.load(open(f)) if e.get("period") in (1, 2) and "possession" in e]
        def ts(e):
            h, m, s = e["timestamp"].split(":"); return int(h) * 3600 + int(m) * 60 + float(s)
        minutes = sum((max(ts(e) for e in ev if e["period"] == p) - min(ts(e) for e in ev if e["period"] == p)) / 60 for p in (1, 2))
        rates.append(len({(e["period"], e["possession"]) for e in ev}) / minutes)
    if not rates:
        raise FileNotFoundError("run experiments/006_external_calibration first (it downloads the reference events)")
    return float(np.mean(rates)), float(np.std(rates, ddof=1)), rates
