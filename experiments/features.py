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
                        np.where(vis, np.clip(inside, -5, 15) / 15, 0.0), np.where(vis, np.log1p(np.clip(near, 0, 30)), 0.0)], 1)
        player.append(f); ball.append(bch); pvalid.append(np.isfinite(f).all(1))
    return dict(player=np.stack(player).astype(np.float32), ball=np.stack(ball).astype(np.float32), pvalid=np.stack(pvalid), obs=obs, vis=vis)
