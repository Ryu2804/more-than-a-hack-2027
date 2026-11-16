"""SkillCorner open data (https://github.com/SkillCorner/opendata, MIT licence; credit SkillCorner) as training/validation matches.

20 A-League 2024/25 matches of automatic broadcast tracking at 10 fps. Same output format as other_games.load:
P (N,K,2) normalised x,y at 25 fps (players detected or extrapolated by the provider, ball NaN when undetected), T (K,), B (N,2), half (N,),
y (N,) H=0 / A=1 / D=2 / -1. Labels come from the provider's phases-of-play file: phases exist only while the ball is in play, so
frames inside a phase are alive with team_in_possession, and every other frame of a period is dead ball.
"""
import csv
import json
import os
import urllib.request

import numpy as np

import common

D = os.path.join(common.CACHE, "ext_skillcorner")
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{m}/{m}_{f}"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{m}/{m}_{f}"
MATCH_IDS = [1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
             2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461]


def _get(m, name, url):
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, f"{m}_{name}")
    if not os.path.exists(p):
        urllib.request.urlretrieve(url.format(m=m, f=name), p)
    return p


def load(m, keep_files=True):
    match = json.load(open(_get(m, "match.json", RAW)))
    home, away = match["home_team"]["id"], match["away_team"]["id"]
    L, W = float(match["pitch_length"]), float(match["pitch_width"])
    team_of = {}
    for p in match["players"]:
        team_of[p["id"]] = 0 if p["team_id"] == home else 1
    rows = []
    with open(_get(m, "tracking_extrapolated.jsonl", MEDIA)) as f:
        for line in f:
            rows.append(json.loads(line))
    n10 = len(rows)
    pid = sorted({p["player_id"] for r in rows for p in r["player_data"] if p["player_id"] in team_of})
    col = {p: i for i, p in enumerate(pid)}
    K = len(pid)
    P10 = np.full((n10, K, 2), np.nan, np.float32); B10 = np.full((n10, 2), np.nan, np.float32); per10 = np.zeros(n10, np.int8)
    for i, r in enumerate(rows):
        per10[i] = r["period"] or 0
        b = r["ball_data"]
        if b.get("is_detected") and b.get("x") is not None:
            B10[i] = (b["x"], b["y"])
        for p in r["player_data"]:
            if p["player_id"] in col and p.get("x") is not None:       # detected OR extrapolated: keeps ~22 players per frame like a tactical-camera tracker
                P10[i, col[p["player_id"]]] = (p["x"], p["y"])
    del rows
    # phases of play -> labels at 10 fps
    y10 = np.full(n10, -1, np.int8)
    y10[per10 > 0] = 2                                              # every frame of a period is dead unless inside a phase
    with open(_get(m, "phases_of_play.csv", RAW)) as f:
        for r in csv.DictReader(f):
            a, b = int(r["frame_start"]), int(r["frame_end"])
            tm = 0 if int(r["team_in_possession_id"]) == home else 1
            y10[a:b] = tm
    # 10 -> 25 fps: linear interpolation across short gaps only (<= 0.35 s), labels by the containing 10 fps frame
    n25 = int((n10 - 1) * 2.5) + 1
    t10 = np.arange(n10) / 10.0; t25 = np.arange(n25) / 25.0
    def interp(series):
        ok = np.isfinite(series)
        if ok.sum() < 2:
            return np.full(n25, np.nan, np.float32)
        tv = t10[ok]
        out = np.interp(t25, tv, series[ok], left=np.nan, right=np.nan)
        j = np.searchsorted(tv, t25)
        d = np.minimum(np.abs(t25 - tv[np.clip(j, 0, len(tv) - 1)]), np.abs(t25 - tv[np.clip(j - 1, 0, len(tv) - 1)]))
        out[d > 0.35] = np.nan
        return out.astype(np.float32)
    P25 = np.stack([np.stack([interp(P10[:, k, 0]), interp(P10[:, k, 1])], 1) for k in range(K)], 1)
    B25 = np.stack([interp(B10[:, 0]), interp(B10[:, 1])], 1)
    idx10 = np.minimum((np.arange(n25) * 0.4).astype(int), n10 - 1)
    # normalise (centred metres -> 0..1); scale to the 105.3 x 68 frame used by features.py happens there
    P25[..., 0] = (P25[..., 0] + L / 2) / L; P25[..., 1] = (P25[..., 1] + W / 2) / W
    B25[:, 0] = (B25[:, 0] + L / 2) / L; B25[:, 1] = (B25[:, 1] + W / 2) / W
    T = np.array([team_of[p] for p in pid])
    out = dict(P=P25, T=T, B=B25, half=per10[idx10].astype(int), y=y10[idx10].astype(np.int8), name=f"skillcorner_{m}")
    if not keep_files:
        for nm in ("tracking_extrapolated.jsonl",):
            try: os.remove(os.path.join(D, f"{m}_{nm}"))
            except OSError: pass
    return out
