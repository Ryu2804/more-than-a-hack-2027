"""Dataset-agnostic features for the supervised possession / dead-ball model (013).

Same code is applied to other matches (Metrica sample games) and to this match's ATD, so a model trained on the former
transfers. Input arrays: P (N,K,2) normalised x,y (NaN = absent), B (N,2) ball, T (K,) 0 home / 1 away, half (N,) 1|2.
Player-configuration triggers follow the IFAB-law set-piece triggers of Vidal-Codina et al. (arXiv 2202.00804); see RESEARCH.md.
"""
import numpy as np
import pandas as pd

L, W, FPS = 105.3, 68.0, 25
PLAYER_FEATS = ["depth a", "depth b", "len a", "len b", "wid a", "wid b", "compact a", "compact b", "speed a", "speed b",
                "d depth", "d len", "d compact", "d speed",
                "min dist corner", "min dist centre", "n beyond touchline", "n own goal area a", "n own goal area b",
                "frac own half a", "frac own half b", "speed all", "frac moving all"]
BALL_FEATS = ["control c", "ball obs", "log ball speed", "ball inside dist", "log nearest dist"]
import os as _os
TOUCH_FEATS = _os.environ.get("TOUCH_FEATS", "0") == "1"      # experiment 019: last-touch memory (held team, time since touch) as extra ball channels
if TOUCH_FEATS:
    BALL_FEATS = BALL_FEATS + ["touch-held team", "log time since touch"]


def _roll(a, w=50, how="mean"):
    r = pd.Series(a).rolling(w, center=True, min_periods=max(w // 5, 1))
    return (r.mean() if how == "mean" else r.median()).values


def attack_dirs(X, T, half, pres, min_frames=1000):
    """(team, half) -> +1 attacks to +x, -1 to -x.

    Each team's goalkeeper is the slot whose mean x stays closest to a goal line. The teams must defend opposite ends within a
    half and swap at half time, so we pick the one of the two consistent hypotheses with the smaller summed goalkeeper distances
    (a per-team density peak alone is unreliable and broke on 2 of 3 training games)."""
    gl, gr = {}, {}                                       # distance of the most goal-line-hugging slot to the left / right goal line
    for h in (1, 2):
        m = half == h
        for t in (0, 1):
            cols = np.flatnonzero(T == t)
            sub, pr = X[m][:, cols], pres[m][:, cols]
            cnt = pr.sum(0)
            with np.errstate(all="ignore"):
                mean = np.where(cnt >= min_frames, np.nanmean(np.where(pr, sub, np.nan), 0), np.nan)
            gl[(t, h)] = np.nanmin(mean) if np.isfinite(mean).any() else L / 2
            gr[(t, h)] = (L - np.nanmax(mean)) if np.isfinite(mean).any() else L / 2
    # H0: home defends left in half 1 (away right), swapped in half 2;  H1: the reverse
    c0 = gl[(0, 1)] + gr[(1, 1)] + gr[(0, 2)] + gl[(1, 2)]
    c1 = gr[(0, 1)] + gl[(1, 1)] + gl[(0, 2)] + gr[(1, 2)]
    if c0 <= c1:
        return {(0, 1): 1, (1, 1): -1, (0, 2): -1, (1, 2): 1}      # defends left => attacks +x
    return {(0, 1): -1, (1, 1): 1, (0, 2): 1, (1, 2): -1}


def build(P, B, T, half, soft_sigma=3.0):
    """Returns dict(player (2,N,Fp), ball (2,N,Fb), pvalid (2,N), ok (N,) ) with a-view = team a attacking to +x.
    Ball channels are view-signed (control c flips sign for the second view)."""
    N, K, _ = P.shape
    X, Y = P[..., 0] * L, P[..., 1] * W
    pres = np.isfinite(X)
    bx, by = B[:, 0] * L, B[:, 1] * W
    vis = np.isfinite(bx)
    att = attack_dirs(X, T, half, pres)

    # per-player speed from positions (identical definition for every dataset)
    sp = np.full((N, K), np.nan, np.float32)
    sp[1:] = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)) * FPS
    sp[sp > 12] = np.nan
    with np.errstate(all="ignore"):
        def tstats(t):
            m = pres & (T == t)[None]
            n = m.sum(1)
            xm, ym, sm = np.where(m, X, np.nan), np.where(m, Y, np.nan), np.where(m & np.isfinite(sp), sp, np.nan)
            mx, my = np.nanmean(xm, 1), np.nanmean(ym, 1)
            o = {"mx": mx, "my": my, "sx": np.nanstd(xm, 1), "sy": np.nanstd(ym, 1),
                 "cmp": np.nanmean(np.hypot(xm - mx[:, None], ym - my[:, None]), 1), "spd": np.nanmean(sm, 1)}
            for k in o: o[k][n < 5] = np.nan
            return {k: _roll(v) for k, v in o.items()}
        S = [tstats(0), tstats(1)]

        # pooled player-configuration triggers (ball-free)
        d_corner = np.min([np.nanmin(np.where(pres, np.hypot(X - cx, Y - cy), np.inf), 1) for cx, cy in ((0, 0), (0, W), (L, 0), (L, W))], 0)
        d_centre = np.nanmin(np.where(pres, np.hypot(X - L / 2, Y - W / 2), np.inf), 1)
        n_out = (pres & ((Y < -0.3) | (Y > W + 0.3))).sum(1)
        spd_all = np.nanmean(np.where(np.isfinite(sp), sp, np.nan), 1)
        mov_all = np.nanmean(np.where(np.isfinite(sp), sp > 1.5, np.nan), 1)
        pool = {"dc": _roll(np.clip(d_corner, 0, 40)), "dm": _roll(np.clip(d_centre, 0, 40)), "no": _roll(n_out.astype(float)),
                "sa": _roll(spd_all), "ma": _roll(mov_all)}

        # soft control and nearest distance (ball channels)
        d = np.hypot(X - bx[:, None], Y - by[:, None])
        d = np.where(np.isfinite(d), d, np.inf)
        w = np.exp(-d / soft_sigma)
        SH, SA = (w * (T == 0)[None]).sum(1), (w * (T == 1)[None]).sum(1)
        tot = SH + SA
        c = (SH - SA) / (tot + 0.05)
        obs = vis & (tot >= 0.05)
        near = np.where(vis, d.min(1), np.nan)
        bsp = np.full(N, np.nan); bsp[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS; bsp[bsp > 40] = np.nan
        inside = np.minimum.reduce([bx, L - bx, by, W - by])
        dH = np.where((T == 0)[None], d, np.inf).min(1); dA = np.where((T == 1)[None], d, np.inf).min(1)
        touch = vis & (np.minimum(dH, dA) < 0.8)
        last = np.maximum.accumulate(np.where(touch, np.arange(N), -1))
        held = np.where(last >= 0, np.where(dH <= dA, 1.0, -1.0)[np.clip(last, 0, N - 1)], 0.0)
        tsince = np.where(last >= 0, np.minimum((np.arange(N) - last) / FPS, 30.0), 30.0)

    half_f = np.where(half > 0, half, 1)
    player, ball, pvalid = [], [], []
    for a_, b_ in ((0, 1), (1, 0)):
        da = np.array([att[(a_, h)] for h in (1, 2)])[half_f - 1]
        fr = lambda x, da=da: np.where(da == 1, x, L - x)            # a attacks +x
        own_half = lambda mx, da=da: np.where(da == 1, mx < L / 2, mx > L / 2)
        # fraction of a team's present players in its own half; own goal area counts
        def frac_own(t, dt):
            m = pres & (T == t)[None]
            xm = np.where(m, X, np.nan)
            inown = np.where(dt[:, None] == 1, xm < L / 2, xm > L / 2)
            with np.errstate(all="ignore"):
                return _roll(np.nanmean(np.where(m, inown, np.nan), 1))
        def n_goal_area(t, dt):
            m = pres & (T == t)[None]
            own_x = np.where(dt[:, None] == 1, X < 5.5, X > L - 5.5)
            return _roll((m & own_x & (np.abs(Y - W / 2) < 9.15)).sum(1).astype(float))
        db = np.array([att[(b_, h)] for h in (1, 2)])[half_f - 1]
        f = np.stack([fr(S[a_]["mx"]), fr(S[b_]["mx"]), S[a_]["sx"], S[b_]["sx"], S[a_]["sy"], S[b_]["sy"],
                      S[a_]["cmp"], S[b_]["cmp"], S[a_]["spd"], S[b_]["spd"],
                      fr(S[a_]["mx"]) - fr(S[b_]["mx"]), S[a_]["sx"] - S[b_]["sx"], S[a_]["cmp"] - S[b_]["cmp"], S[a_]["spd"] - S[b_]["spd"],
                      pool["dc"], pool["dm"], pool["no"], n_goal_area(a_, da), n_goal_area(b_, db),
                      frac_own(a_, da), frac_own(b_, db), pool["sa"], pool["ma"]], 1)
        sign = 1.0 if a_ == 0 else -1.0
        bch = np.stack([np.where(obs, c * sign, 0.0), obs.astype(float), np.where(obs, np.log1p(_roll(bsp, 25, "median")), 0.0),
                        np.where(vis, np.clip(inside, -5, 15) / 15, 0.0), np.where(vis, np.log1p(np.clip(near, 0, 30)), 0.0)] + ([held * sign, np.log1p(tsince)] if TOUCH_FEATS else []), 1)
        player.append(f); ball.append(bch); pvalid.append(np.isfinite(f).all(1))
    return dict(player=np.stack(player).astype(np.float32), ball=np.stack(ball).astype(np.float32), pvalid=np.stack(pvalid), obs=obs, vis=vis)


def fill_ball_gaps(B, max_gap):
    """Linear interpolation of ball positions across NaN gaps of at most max_gap frames (gaps at the edges are left alone). B (N,2+)."""
    out = B.copy()
    if max_gap <= 0:
        return out
    ok = np.isfinite(B[:, 0])
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return out
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b, :2] = B[a, :2] * (1 - w)[:, None] + B[b, :2] * w[:, None]
    return out


PP_K, PP_F = 48, 10
USE_PP = _os.environ.get("USE_PP", "0") == "1"      # experiment 020: per-player set encoder


def player_set(P, B, T, half, ix, K=PP_K):
    """Per-player, per-step features at frames ix for both views (view a: team a attacks +x), padded to K slots.
    Returns pp (2, len(ix), K, PP_F) float32 and mask (2, len(ix), K) bool. Dataset-agnostic: positions, velocities, ball distance and closing speed,
    nearest-opponent distance. Ball-dependent channels (6, 7, 8) are the ones zeroed when the ball is hidden."""
    N, Kp, _ = P.shape
    X, Y = P[..., 0] * L, P[..., 1] * W
    bx, by = B[:, 0] * L, B[:, 1] * W
    pres = np.isfinite(X)
    att = attack_dirs(X, T, half, pres)
    ix = np.asarray(ix); ip = np.clip(ix - STRIDE_PP, 0, None)
    x, y, xp, yp = X[ix], Y[ix], X[ip], Y[ip]
    dt = np.maximum((ix - ip) / FPS, 1e-3)[:, None]
    vx, vy = (x - xp) / dt, (y - yp) / dt
    ok_v = np.isfinite(vx) & np.isfinite(vy) & (np.hypot(vx, vy) < 12)
    bxi, byi = bx[ix][:, None], by[ix][:, None]; visb = np.isfinite(bxi)
    d = np.hypot(x - bxi, y - byi); dp = np.hypot(xp - bx[ip][:, None], yp - by[ip][:, None])
    closing = np.where(np.isfinite(d) & np.isfinite(dp), (dp - d) / dt, 0.0)
    h_ix = np.where(half[ix] > 0, half[ix], 1)
    out = np.zeros((2, len(ix), K, PP_F), np.float32); mask = np.zeros((2, len(ix), K), bool)
    for v, a in enumerate((0, 1)):
        da = np.array([att[(a, h)] for h in (1, 2)])[h_ix - 1][:, None]
        xa = np.where(da == 1, x, L - x); vxa = np.where(da == 1, vx, -vx)
        same = (T == a)[None].repeat(len(ix), 0)
        # nearest opponent distance and team centroid x
        opp = np.where(same, np.nan, 1.0)
        with np.errstate(all="ignore"):
            cx = np.nanmean(np.where(same & np.isfinite(xa), xa, np.nan), 1, keepdims=True)
            dopp = np.full(x.shape, 30.0)
            for j in range(Kp):
                dj = np.hypot(x[:, [j]] - x, y[:, [j]] - y); dj = np.where(~same & np.isfinite(dj), dj, np.inf)
                m_ = dj.min(1); dopp[:, j] = np.where(np.isfinite(m_), np.minimum(m_, 30.0), 30.0)
        feats = np.stack([same.astype(np.float32), xa / L, y / W, np.where(ok_v, vxa, 0) / 8, np.where(ok_v, vy, 0) / 8,
                          np.where(ok_v, np.hypot(vx, vy), 0) / 8, np.where(visb & np.isfinite(d), np.minimum(d, 30) / 30, 0),
                          (visb & np.isfinite(d)).astype(np.float32), np.clip(closing, -12, 12) / 8, (xa - np.nan_to_num(cx, nan=L / 2)) / L], -1)
        feats = np.nan_to_num(feats).astype(np.float32); feats[..., 10 - 1] = feats[..., 9]
        # last channel: nearest-opponent distance replaces the duplicate slot
        feats[..., 9] = np.minimum(dopp, 30) / 30
        k = min(Kp, K); out[v, :, :k] = feats[:, :k]; mask[v, :, :k] = pres[ix][:, :k]
    return out, mask


STRIDE_PP = 5
