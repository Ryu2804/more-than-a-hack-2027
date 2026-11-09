"""exp_3 label definitions on the IDSSE proxy: ``L_track(tau)`` and ``L_ctrl``.

Codes as in exp_0/exp_1: 0 H, 1 A, 2 D (dead ball), 3 unlabelled. Both relabelings only
ever move a *team* run to the previous team; D frames, unlabelled frames and half
boundaries are never touched, and nothing is merged across halves.

``L_ctrl`` (PREREG section 2) reads CHALLENGE.md's "controlled possession" with the
provider frame labels plus the open IDSSE event data (DFL ``events_raw`` XML, CC BY 4.0):
a team-k run is a controlled possession iff
  (a) it starts the half or follows a dead-ball run, or
  (b) it lasts at least ``TAU_HOLD_S`` seconds, or
  (c) team k has a controlled on-ball event within ``EVENT_MARGIN_S`` of the run;
otherwise, processed left to right, it takes the team of the previous (relabelled) run.

Events are synced to tracking frames from the half kick-off events only:
``frame = first_frame_of_half + round((event_time - kickoff_time) * 25)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from lxml import etree

ROOT = Path(__file__).resolve().parent
EVENT_DIR = ROOT / "external" / "idsse_events"
META_DIR = ROOT / "external" / "idsse_meta"
FPS = 25

TAU_HOLD_S = 2.0
EVENT_MARGIN_S = 0.5
TRACK_TAUS_S = (1.0, 2.0, 3.0)
# Controlled on-ball events (fixed from the schema before any fit; PREREG section 2).
CONTROLLED_PLAY_EVALUATIONS = ("successfullyCompleted", "successful")  # Play = Pass or Cross
CONTROLLED_CLAIM_TYPES = ("InterceptedBall", "BallClaimed", "BallHeld")  # BallClaiming@Type
CONTROLLED_TACKLE_RESULTS = ("ballClaimed", "ballControlRetained", "dribbledAround")
# Sync refinement (label-free; fixed after the kick-off-only residuals, before the snapped ones):
# kick-off-synced event times run late by -3 s ... +10 s vs the provider's CalculatedFrame on
# shots (5-95 %: -1.8 s ... +4.8 s), so each event is snapped to its player's closest approach to
# the ball within [-5 s, +2 s]; acceptance is judged on the shots' CalculatedFrame only.
SNAP_BEFORE = 125  # frames before the synced frame (event logged late)
SNAP_AFTER = 50
SNAP_MAX_DISTANCE_M = 2.0
SYNC_MEDIAN_MAX = 5  # frames: events used iff pooled median |shot residual| <= 0.2 s ...
SYNC_P90_MAX = 12  # ... and pooled p90 |shot residual| <= 0.48 s; else fall back to (a) + (b)
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
TEAM_CODES = {"home": 0, "guest": 1}  # match information Team@Role (same rule as the judge loader)


# ---------------------------------------------------------------- relabelings
def period_runs(labels: np.ndarray, half: np.ndarray, period: int) -> list[tuple[int, int, int]]:
    """Maximal [start, stop) runs of constant label inside one period, in array order."""
    index = np.flatnonzero(half == period)
    if len(index) == 0:
        return []
    if np.any(np.diff(index) != 1):
        raise ValueError("Periods must be contiguous blocks")
    values = labels[index]
    cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(values)]
    return [(int(index[0] + a), int(index[0] + b), int(values[a])) for a, b in zip(starts, stops)]


def _periods(half: np.ndarray) -> list[int]:
    return [int(p) for p in np.unique(half) if p > 0]


def l_track(labels: np.ndarray, half: np.ndarray, tau_s: float) -> np.ndarray:
    """Tracking-only relabel: a team run shorter than ``tau_s`` that follows the other team and
    is followed by that team, a dead ball or the end of the half joins the previous team."""
    tau = tau_s * FPS
    out = labels.copy()
    for period in _periods(half):
        runs = period_runs(labels, half, period)
        previous = None  # label of the previous team/D run in the output
        for i, (start, stop, code) in enumerate(runs):
            if code == 2:
                previous = 2
                continue
            if code not in (0, 1):
                continue  # unlabelled runs are transparent
            following = next((c for _, _, c in runs[i + 1:] if c in (0, 1, 2)), 2)
            if (previous in (0, 1) and previous != code and stop - start < tau
                    and following in (previous, 2)):
                out[start:stop] = previous
            previous = int(out[start])
    return out


def l_ctrl(labels: np.ndarray, half: np.ndarray, event_index: np.ndarray, event_team: np.ndarray,
           tau_hold_s: float = TAU_HOLD_S, margin_s: float = EVENT_MARGIN_S,
           use_events: bool = True) -> np.ndarray:
    """Controlled-possession labels (module docstring rules (a)-(c)); D never changes."""
    event_index = np.asarray(event_index, dtype="int64")
    event_team = np.asarray(event_team, dtype="int64")
    tau, margin = tau_hold_s * FPS, margin_s * FPS
    out = labels.copy()
    for period in _periods(half):
        in_period = half[event_index] == period if len(event_index) else np.zeros(0, bool)
        frames, teams = event_index[in_period], event_team[in_period]
        previous = None
        for start, stop, code in period_runs(labels, half, period):
            if code == 2:
                previous = 2
                continue
            if code not in (0, 1):
                continue
            controlled = (previous is None or previous == 2 or stop - start >= tau
                          or (use_events and bool(np.any(
                              (teams == code) & (frames >= start - margin) & (frames <= stop - 1 + margin)))))
            if not controlled:
                out[start:stop] = previous
            previous = int(out[start])
    return out


# ---------------------------------------------------------------- statistics
def possession_count(labels: np.ndarray, half: np.ndarray) -> int:
    return sum(1 for p in _periods(half) for _, _, c in period_runs(labels, half, p) if c in (0, 1))


def label_stats(labels: np.ndarray, half: np.ndarray, provider: np.ndarray) -> dict:
    """Possessions/min (per minute of half time), share of frames changed vs provider labels."""
    graded = half > 0
    minutes = graded.sum() / FPS / 60
    alive = graded & (labels < 2)
    count = possession_count(labels, half)
    return {"possessions": count, "possessions_per_min": float(count / minutes),
            "mean_possession_s": float(alive.sum() / max(1, count) / FPS),
            "frames_changed_share": float(np.mean(labels[graded] != provider[graded])),
            "dead_share": float(np.mean(labels[graded] == 2))}


def agreement(a: np.ndarray, b: np.ndarray, half: np.ndarray) -> float:
    graded = half > 0
    return float(np.mean(a[graded] == b[graded]))


# ---------------------------------------------------------------- events
@dataclass(frozen=True)
class MatchEvents:
    """Controlled on-ball events of one match, one entry per event (arrays aligned)."""
    index: np.ndarray  # match-array index (25 fps) of each event
    team: np.ndarray  # 0 home, 1 away
    kind: np.ndarray  # event family: pass_or_cross, shot, claim, tackle_won
    slot: np.ndarray  # clean-match player slot of the acting player, -1 if unknown
    calculated: np.ndarray  # provider CalculatedFrame (shots only), -1 otherwise
    dropped: int  # events outside both halves (kick-off sync) or never located (snap)

    def subset(self, keep: np.ndarray, index: np.ndarray | None = None, dropped: int = 0) -> "MatchEvents":
        return MatchEvents(self.index[keep] if index is None else index, self.team[keep], self.kind[keep],
                           self.slot[keep], self.calculated[keep], self.dropped + dropped)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _file(folder: Path, kind: str, name: str) -> Path:
    found = sorted(folder.glob(f"DFL_*_{kind}_*DFL-MAT-{name}.xml"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {kind} file for {name}")
    return found[0]


@lru_cache(maxsize=8)
def team_roles(name: str) -> dict:
    """DFL club id -> 0 home / 1 away, from the open match information file."""
    tree = etree.parse(str(_file(META_DIR, "matchinformation", name)))
    return {team.get("TeamId"): TEAM_CODES[team.get("Role")] for team in tree.iter("Team")
            if team.get("Role") in TEAM_CODES}


@lru_cache(maxsize=8)
def player_slots(name: str) -> dict:
    """DFL person id -> clean-match slot, in the judge loader's order (Teams/Team/Players/Player)."""
    tree = etree.parse(str(_file(META_DIR, "matchinformation", name)))
    ids = [p.get("PersonId") for team in tree.findall(".//Teams/Team") for p in team.findall("./Players/Player")]
    return {person: slot for slot, person in enumerate(ids)}


def controlled_team(element) -> str | None:
    """Club id of the team that controls the ball in this event element, else None."""
    tag = element.tag
    if tag == "Play" and element.get("Evaluation") in CONTROLLED_PLAY_EVALUATIONS:
        return element.get("Team")
    if tag == "ShotAtGoal":
        return element.get("Team")
    if tag == "BallClaiming" and element.get("Type") in CONTROLLED_CLAIM_TYPES:
        return element.get("Team")
    if tag == "TacklingGame" and element.get("WinnerResult") in CONTROLLED_TACKLE_RESULTS:
        return element.get("WinnerTeam")
    return None


def event_player(element) -> str | None:
    return element.get("Winner") if element.tag == "TacklingGame" else element.get("Player")


FAMILIES = {"Play": "pass_or_cross", "ShotAtGoal": "shot", "BallClaiming": "claim",
            "TacklingGame": "tackle_won"}


def raw_events(name: str) -> list[dict]:
    """Every controlled on-ball element of the event file with its outer EventTime."""
    root = etree.parse(str(_file(EVENT_DIR, "events_raw", name))).getroot()
    roles, slots = team_roles(name), player_slots(name)
    rows = []
    for event in root.iter("Event"):
        if not len(event):
            continue
        time = _time(event.get("EventTime"))
        calculated = event.get("CalculatedFrame")
        for element in event.iter():
            club = controlled_team(element) if element is not event else None
            if club is None:
                continue
            if club not in roles:
                raise ValueError("Event team not in match information")
            is_shot = calculated is not None and element.tag == "ShotAtGoal"
            rows.append({"time": time, "team": roles[club], "kind": FAMILIES[element.tag],
                         "slot": slots.get(event_player(element), -1),
                         "calculated": int(calculated) if is_shot else -1})
    return rows


def kickoff_times(name: str) -> dict:
    """Earliest KickOff EventTime per GameSection (the half kick-offs)."""
    root = etree.parse(str(_file(EVENT_DIR, "events_raw", name))).getroot()
    times: dict[int, datetime] = {}
    for event in root.iter("Event"):
        for kickoff in event.iter("KickOff"):
            period = SECTIONS.get(kickoff.get("GameSection"))
            if period is None:
                continue
            time = _time(event.get("EventTime"))
            times[period] = min(times.get(period, time), time)
    if set(times) != {1, 2}:
        raise ValueError("Both half kick-offs are required for syncing")
    return times


def event_frame(time: datetime, kickoff: datetime, first_frame: int) -> int:
    """Provider frame number of an event time, synced from the half kick-off."""
    return first_frame + int(round((time - kickoff).total_seconds() * FPS))


def sync_events(name: str, frame: np.ndarray, half: np.ndarray) -> MatchEvents:
    """Controlled events -> match-array indices via the half kick-off events only."""
    kickoffs = kickoff_times(name)
    positions = {p: np.flatnonzero(half == p) for p in (1, 2)}
    first = {p: int(frame[positions[p][0]]) for p in (1, 2)}
    rows, index, dropped = [], [], 0
    for row in raw_events(name):
        period = 2 if row["time"] >= kickoffs[2] else 1 if row["time"] >= kickoffs[1] else 0
        offset = (event_frame(row["time"], kickoffs[period], first[period]) - first[period]
                  if period else -1)
        if not 0 <= offset < (len(positions[period]) if period else 0):
            dropped += 1
            continue
        rows.append(row)
        index.append(int(positions[period][offset]))
    column = lambda key, dtype: np.array([r[key] for r in rows], dtype=dtype)  # noqa: E731
    return MatchEvents(np.array(index, dtype="int64"), column("team", "int64"), column("kind", str),
                       column("slot", "int64"), column("calculated", "int64"), dropped)


def snap_events(events: MatchEvents, players: np.ndarray, ball: np.ndarray,
                half: np.ndarray) -> MatchEvents:
    """Label-free refinement of the kick-off sync (clean tracking, no labels): move each event to
    the frame within [-SNAP_BEFORE, +SNAP_AFTER] frames of its synced frame (same half) where its
    player is closest to the ball; drop it if that distance never reaches SNAP_MAX_DISTANCE_M or
    the player is unknown."""
    keep, index = [], []
    for k, (frame, slot) in enumerate(zip(events.index, events.slot)):
        if slot < 0:
            continue
        window = np.arange(max(frame - SNAP_BEFORE, 0), min(frame + SNAP_AFTER + 1, len(half)))
        window = window[half[window] == half[frame]]
        distance = np.linalg.norm(players[window, slot, :2] - ball[window], axis=1)
        finite = np.isfinite(distance)
        if not finite.any() or distance[finite].min() > SNAP_MAX_DISTANCE_M:
            continue
        keep.append(k)
        index.append(int(window[finite][int(np.argmin(distance[finite]))]))
    keep_array = np.array(keep, dtype="int64")
    return events.subset(keep_array, np.array(index, dtype="int64"), len(events.index) - len(keep))


def shot_residuals(events: MatchEvents, frame: np.ndarray) -> np.ndarray:
    """Synced provider frame minus the provider's own CalculatedFrame, shots that carry one."""
    has = events.calculated >= 0
    return (frame[events.index[has]] - events.calculated[has]).astype("int64")


def sync_quality(residuals: np.ndarray) -> dict:
    absolute = np.abs(residuals)
    passed = bool(np.median(absolute) <= SYNC_MEDIAN_MAX and np.quantile(absolute, 0.9) <= SYNC_P90_MAX)
    return {"shots": int(len(residuals)), "median_abs_frames": float(np.median(absolute)),
            "p90_abs_frames": float(np.quantile(absolute, 0.9)), "median_frames": float(np.median(residuals)),
            "accepted": passed}


def possession_change_lag(labels: np.ndarray, events: MatchEvents, max_lag: int = 50) -> dict:
    """Agreement with provider possession: share of controlled events whose team owns the ball
    (provider label) at event frame + lag, for lag in [-max_lag, max_lag] frames."""
    lags = np.arange(-max_lag, max_lag + 1)
    shares = []
    for lag in lags:
        position = np.clip(events.index + lag, 0, len(labels) - 1)
        shares.append(float(np.mean(labels[position] == events.team)))
    best = int(lags[int(np.argmax(shares))])
    return {"agreement_at_lag0": shares[max_lag], "best_lag_frames": best,
            "agreement_at_best_lag": max(shares)}


def change_residuals(labels: np.ndarray, half: np.ndarray, events: MatchEvents,
                     kinds=("claim", "tackle_won"), reach: int = 125) -> np.ndarray:
    """Frames from each possession-winning event (claim, tackle won) to the nearest provider change
    *into* that team within +-reach frames in the same half (NaN if none)."""
    change = np.zeros(len(labels), dtype="int64") - 1
    same = (half[1:] == half[:-1]) & (half[1:] > 0)
    into = np.flatnonzero(same & (labels[1:] != labels[:-1]) & (labels[1:] < 2)) + 1
    change[into] = labels[into]
    out = []
    for frame, team, kind in zip(events.index, events.team, events.kind):
        if kind not in kinds:
            continue
        lo, hi = max(frame - reach, 0), min(frame + reach + 1, len(labels))
        hits = np.flatnonzero(change[lo:hi] == team) + lo
        out.append(float(hits[np.argmin(np.abs(hits - frame))] - frame) if len(hits) else np.nan)
    return np.array(out)


def match_events(name: str, clean) -> tuple[MatchEvents, dict]:
    """Kick-off sync, then snap; returns the events used by L_ctrl and the sync report."""
    synced = sync_events(name, clean.frame, clean.half)
    snapped = snap_events(synced, clean.players, clean.ball, clean.half)
    report = {"kickoff_sync": sync_quality(shot_residuals(synced, clean.frame)),
              "snapped_sync": sync_quality(shot_residuals(snapped, clean.frame)),
              "events_synced": int(len(synced.index)), "events_snapped": int(len(snapped.index)),
              "dropped_outside_half": int(synced.dropped), "dropped_unlocated": int(snapped.dropped - synced.dropped)}
    return snapped, report
