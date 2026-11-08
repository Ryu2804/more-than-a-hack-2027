"""IDSSE / DFL open data (Bassek et al. 2025, Scientific Data; Bundesliga 2022/23; TRACAB 25 fps; CC-BY 4.0; figshare 28196177, HF pysport/idsse-data).

Loaded inside the sandbox only. Same dict format as skillcorner_games.load: P (N,K,2) normalised, T (K,) 0 home / 1 away, B (N,2), half (N,),
y (N,) 0 home / 1 away / 2 dead, name. Labels: BallStatus == 0 -> dead (2); else BallPossession 1 -> home (0), 2 -> away (1).
"""
import re
import urllib.request
from pathlib import Path

import numpy as np

L, W = 105.0, 68.0
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
MATCHES = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002", "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)">(.*?)</FrameSet>', re.S)
FR = re.compile(r'<Frame N="(\d+)"[^>]* X="(-?[\d.]+)" Y="(-?[\d.]+)"(?:[^>]*BallPossession="(\d)" BallStatus="(\d)")?')


def _get(mid, kind, cache):
    com = MATCHES[mid]
    f = {"info": f"DFL_02_01_matchinformation_DFL-COM-0000{com[-2:]}_DFL-MAT-{mid}.xml", "pos": f"DFL_04_03_positions_raw_observed_DFL-COM-0000{com[-2:]}_DFL-MAT-{mid}.xml"}[kind]
    p = Path(cache) / f
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + f, p)
    return p


def load(mid, cache="/tmp/idsse", keep_files=True):
    info = _get(mid, "info", cache).read_text(encoding="utf8")
    teams = dict(re.findall(r'<Team TeamId="([^"]+)"[^>]* Role="(home|guest)"', info))
    txt = _get(mid, "pos", cache).read_text(encoding="utf8")
    secs = {}
    for sec, team, person, body in FS.findall(txt):
        secs.setdefault(sec, {})[(team, person)] = np.array([[int(m[0]), float(m[1]), float(m[2]), int(m[3]) if m[3] else -1, int(m[4]) if m[4] else -1] for m in FR.findall(body)])
    order = [s for s in ("firstHalf", "secondHalf") if s in secs]
    persons = sorted({k for s in order for k in secs[s] if teams.get(k[0]) in ("home", "guest")}, key=lambda k: (teams[k[0]] != "home", k[1]))
    T = np.array([0 if teams[k[0]] == "home" else 1 for k in persons], np.int8)
    GAP = 250
    parts = []
    for s in order:
        n0 = min(a[0, 0] for a in secs[s].values()); n1 = max(a[-1, 0] for a in secs[s].values())
        parts.append((s, int(n0), int(n1)))
    n_tot = sum(b - a + 1 + GAP for _, a, b in parts)
    P = np.full((n_tot, len(persons), 2), np.nan, np.float32); B = np.full((n_tot, 2), np.nan, np.float32)
    half = np.zeros(n_tot, int); y = np.full(n_tot, -1, np.int8)
    off = 0
    for h, (s, a, b) in enumerate(parts, 1):
        n = b - a + 1
        for j, k in enumerate(persons):
            arr = secs[s].get(k)
            if arr is not None:
                P[off + arr[:, 0].astype(int) - a, j] = arr[:, 1:3]
        for (team, _), arr in secs[s].items():
            if team == "BALL":
                ix = off + arr[:, 0].astype(int) - a
                B[ix] = arr[:, 1:3]
                pos, st = arr[:, 3], arr[:, 4]
                lab = np.where(st == 0, 2, np.where(pos == 1, 0, np.where(pos == 2, 1, -1)))
                y[ix] = lab
        half[off:off + n] = h
        off += n + GAP
    for A in (P, B):
        A[..., 0] = (A[..., 0] + L / 2) / L; A[..., 1] = (A[..., 1] + W / 2) / W
    return dict(P=P, T=T, B=B, half=half, y=y, name=f"idsse_{mid}")
