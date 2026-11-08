"""Other-match data for supervised training (allowed by RULES.md: other games only).

Source: Metrica Sports public sample games 1-3 (tracking + events), https://github.com/metrica-sports/sample-data
(acknowledge the source in the technical document). Cached in _cache/ext_metrica (gitignored).

Labels are DERIVED FROM EVENT DATA of those games: per frame H(0) / A(1) / D(2) / -1 unknown.
  possessor = team of the latest PASS / CARRY / SHOT / RECOVERY / SET PIECE event (loose ball keeps the last team,
              as CHALLENGE.md describes); BALL LOST and CHALLENGE do not change it.
  dead      = from BALL OUT / FAULT RECEIVED / CARD / goal until the next ball-in-play event (PASS/CARRY/SHOT/RECOVERY/SET PIECE).
"""
import csv
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

import common

D = os.path.join(common.CACHE, "ext_metrica")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FPS, L, W = 25, 105.3, 68.0


def _get(name, sub):
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, name)
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"{BASE}/{sub}/{name}", p)
    return p


def _csv_tracking(n):
    out = {}
    for side in ("Home", "Away"):
        p = _get(f"Sample_Game_{n}_RawTrackingData_{side}_Team.csv", f"Sample_Game_{n}")
        df = pd.read_csv(p, header=None, skiprows=3)
        out[side] = df
    h, a = out["Home"], out["Away"]
    period, frame = h[0].values.astype(int), h[1].values.astype(int)
    hx = h.iloc[:, 3:-2].values.astype(np.float32); ax = a.iloc[:, 3:-2].values.astype(np.float32)
    B = h.iloc[:, -2:].values.astype(np.float32)
    P = np.concatenate([hx.reshape(len(h), -1, 2), ax.reshape(len(a), -1, 2)], 1)
    T = np.array([0] * (hx.shape[1] // 2) + [1] * (ax.shape[1] // 2))
    return P, T, B, period, frame


def _csv_events(n):
    p = _get(f"Sample_Game_{n}_RawEventsData.csv", f"Sample_Game_{n}")
    rows = []
    for r in csv.DictReader(open(p)):
        rows.append(dict(team=0 if r["Team"] == "Home" else 1, type=r["Type"], sub=r["Subtype"], period=int(r["Period"]),
                         start=int(float(r["Start Frame"])), end=int(float(r["End Frame"]))))
    return pd.DataFrame(rows)


def _game3():
    tr = _get("Sample_Game_3_tracking.txt", "Sample_Game_3")
    meta = open(_get("Sample_Game_3_metadata.xml", "Sample_Game_3")).read()
    team_of = dict(re.findall(r'<Player id="(P\d+)" teamId="(\w+)"', meta))
    # tracking columns follow the data-format spec's PlayerChannelRef order, NOT the order of the PlayerChannel list
    chan_to_player = dict(re.findall(r'<PlayerChannel channelId="x" id="(player\d+_x)" playerId="(P\d+)"', meta))
    spec_x = re.findall(r'<PlayerChannelRef playerChannelId="(player\d+_x)"', meta)
    chan = [chan_to_player[c] for c in spec_x]
    rows = []
    with open(tr) as f:
        for line in f:
            p = line.split(":")
            rows.append((int(p[0]), p[1], p[2]))
    n = rows[-1][0]
    ncol = len(rows[0][1].split(";"))
    P = np.full((n, ncol, 2), np.nan, np.float32); B = np.full((n, 2), np.nan, np.float32)
    for fr, pl, ba in rows:
        P[fr - 1] = np.array(pl.replace(";", ",").split(","), dtype=np.float32).reshape(ncol, 2)
        B[fr - 1] = np.array(ba.split(","), dtype=np.float32)
    T = np.array([0 if team_of[c] == "FIFATMA" else 1 for c in chan[:ncol]])
    ev = json.load(open(_get("Sample_Game_3_events.json", "Sample_Game_3")))["data"]
    rows = []
    for e in ev:
        sub = e.get("subtypes"); sub = sub["name"] if isinstance(sub, dict) else (sub[0]["name"] if isinstance(sub, list) and sub else "")
        rows.append(dict(team=0 if e["team"]["id"] == "FIFATMA" else 1, type=e["type"]["name"], sub=sub, period=e["period"],
                         start=e["start"]["frame"], end=e["end"]["frame"]))
    ev_df = pd.DataFrame(rows)
    # one continuous tracking range holds both halves: derive the period of each frame from the event spans
    period = np.zeros(n, int)
    for p in (1, 2):
        e = ev_df[ev_df.period == p]
        period[int(e.start.min()) - 1:int(e.end.max())] = p
    return P, T, B, period, np.arange(1, n + 1), ev_df


def derive_labels(ev, n_frames, period, frame):
    """Per-frame H/A/D/-1 from events (see module docstring)."""
    ev = ev.sort_values(["start", "end"]).reset_index(drop=True)
    y = np.full(n_frames, -1, np.int8)
    f0 = {int(p): int(frame[period == p].min()) for p in np.unique(period) if p > 0}
    f1 = {int(p): int(frame[period == p].max()) for p in np.unique(period) if p > 0}
    for p in f0:
        evp = ev[ev.period == p]
        marks = []                                   # (frame, kind, team)
        for _, e in evp.iterrows():
            if e.type in ("PASS", "CARRY", "RECOVERY", "SET PIECE"):
                marks.append((e.start, "poss", e.team)); marks.append((e.start, "alive", -1))   # a ball-in-play event restarts play
            elif e.type == "SHOT":
                marks.append((e.start, "poss", e.team)); marks.append((e.start, "alive", -1))
                if "GOAL" in str(e["sub"]).upper(): marks.append((e.end, "dead", -1))
            elif e.type in ("BALL OUT", "FAULT RECEIVED", "CARD"):
                marks.append((e.start, "dead", -1))
        marks.sort(key=lambda m: (m[0], {"alive": 0, "poss": 1, "dead": 2}[m[1]]))
        poss, dead, last = None, False, None
        for fr, kind, tm in marks + [(f1[p] + 1, "end", -1)]:
            if last is not None and poss is not None:
                a, b = max(last, f0[p]), min(fr, f1[p] + 1)
                if b > a: y[a - 1:b - 1] = 2 if dead else poss
            if kind == "poss": poss = tm
            elif kind == "dead": dead = True
            elif kind == "alive": dead = False
            last = fr
    return y


def load(n):
    """dict(P (N,K,2) normalised, T (K,), B (N,2), half (N,), y (N,), name)"""
    if n in (1, 2):
        P, T, B, period, frame = _csv_tracking(n); ev = _csv_events(n)
    else:
        P, T, B, period, frame, ev = _game3()
    y = derive_labels(ev, len(B), period, frame)
    y[period == 0] = -1
    return dict(P=P, T=T, B=B, half=period.astype(int), y=y, ev=ev, name=f"metrica_{n}")
