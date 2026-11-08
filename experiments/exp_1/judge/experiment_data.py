"""Notebook data contracts; no target labels or target data enter logs."""

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import io
import json
import warnings

from lxml import etree
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent / "more-than-a-hack-2027"
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
FPS = 25


@dataclass(frozen=True)
class Match:
    name: str
    players: np.ndarray  # frame, slot, (x metres, y metres, speed m/s)
    ball: np.ndarray  # frame, (x metres, y metres); never provider speed/state
    teams: np.ndarray  # team per anonymous slot
    half: np.ndarray
    frame: np.ndarray
    labels: np.ndarray | None
    pitch: tuple[float, float]


def labels_from_ball(status: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Sportec BallStatus=1 is alive; BallPossession=1/2 is home/away."""
    return np.where(
        status == 0,
        2,
        np.where(
            (status == 1) & (owner == 1),
            0,
            np.where((status == 1) & (owner == 2), 1, 3),
        ),
    ).astype("int8")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype("int8"))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def save_match(match: Match, path: Path) -> None:
    if match.labels is None:
        raise ValueError("Match cache is external-labelled-data only")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=match.name,
        players=match.players,
        ball=match.ball,
        teams=match.teams,
        half=match.half,
        frame=match.frame,
        labels=match.labels,
        pitch=match.pitch,
    )


def read_match(path: Path) -> Match:
    with np.load(path, allow_pickle=False) as z:
        return Match(
            str(z["name"]),
            z["players"],
            z["ball"],
            z["teams"],
            z["half"],
            z["frame"],
            z["labels"],
            tuple(z["pitch"]),
        )


def _external_objects(path: Path) -> dict:
    """Stream XML per object instead of constructing millions of kloppy objects."""
    objects = {}
    current, rows = None, []
    sections = {"firstHalf": 1, "secondHalf": 2}
    for event, element in etree.iterparse(
        str(path), events=("start", "end"), huge_tree=True
    ):
        if event == "start" and element.tag == "FrameSet":
            current = (
                sections.get(element.get("GameSection"), 0),
                "ball" if element.get("TeamId") == "BALL" else element.get("PersonId"),
            )
            rows = []
        elif event == "end" and element.tag == "Frame" and current:
            a = element.attrib
            rows.append(
                (
                    int(a["N"]),
                    float(a["X"]),
                    float(a["Y"]),
                    int(a.get("BallStatus", -1)),
                    int(a.get("BallPossession", 0)),
                )
            )
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
        elif event == "end" and element.tag == "FrameSet":
            if current and current[0] and rows:
                objects[current] = np.asarray(rows, dtype="float64")
            rows, current = [], None
            element.clear()
    return objects


def _derived_speed(xy: np.ndarray) -> np.ndarray:
    """Common displacement-based speed; avoid provider-dependent speed units."""
    before = np.concatenate([xy[:5], xy[:-5]])
    after = np.concatenate([xy[5:], xy[-5:]])
    speed = np.linalg.norm(after - before, axis=-1) / 0.4
    return np.clip(speed, 0, 12).astype("float32")


def load_idsse(name: str) -> Match:
    if name not in MATCH_IDS:
        raise ValueError("Unknown IDSSE match")
    cache = ROOT / "cache" / (name + "_clean.npz")
    if cache.exists():
        return read_match(cache)
    source = ROOT / "external" / "idsse"
    meta = next(source.glob("*matchinformation*" + name + ".xml"))
    raw = next(source.glob("*positions_raw*" + name + ".xml"))
    tree = etree.parse(str(meta))
    environment = tree.find(".//Environment")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    ids, teams = [], []
    for team in tree.findall(".//Teams/Team"):
        code = 0 if team.get("Role") == "home" else 1
        for player in team.findall("./Players/Player"):
            ids.append(player.get("PersonId"))
            teams.append(code)
    objects = _external_objects(raw)
    blocks = []
    for half in (1, 2):
        b = objects[(half, "ball")]
        first, last = int(b[:, 0].min()), int(b[:, 0].max())
        n = last - first + 1
        players = np.full((n, len(ids), 3), np.nan, dtype="float32")
        ball = np.full((n, 2), np.nan, dtype="float32")
        labels = np.full(n, 3, dtype="int8")
        for slot, identity in enumerate(ids):
            rows = objects.get((half, identity))
            if rows is None:
                continue
            index = rows[:, 0].astype("int64") - first
            keep = (index >= 0) & (index < n)
            players[index[keep], slot, :2] = rows[keep, 1:3] + np.array(pitch) / 2
        players[:, :, 2] = _derived_speed(players[:, :, :2])
        index = b[:, 0].astype("int64") - first
        ball[index] = b[:, 1:3] + np.array(pitch) / 2
        labels[index] = labels_from_ball(b[:, 3], b[:, 4])
        blocks.append(
            (
                players,
                ball,
                np.full(n, half, dtype="int8"),
                np.arange(first, last + 1),
                labels,
            )
        )
        if half == 1:
            # Official valid_runs uses X as a boundary. Never merge spells across halves.
            blocks.append(
                (
                    np.full((1, len(ids), 3), np.nan, dtype="float32"),
                    np.full((1, 2), np.nan, dtype="float32"),
                    np.zeros(1, dtype="int8"),
                    np.array([-1]),
                    np.full(1, 3, dtype="int8"),
                )
            )
    arrays = [np.concatenate([block[i] for block in blocks]) for i in range(5)]
    match = Match(
        name,
        arrays[0],
        arrays[1],
        np.array(teams, dtype="int8"),
        arrays[2],
        arrays[3],
        arrays[4],
        pitch,
    )
    save_match(match, cache)
    return match


def _corrupt_observations(
    players: np.ndarray,
    ball: np.ndarray,
    teams: np.ndarray,
    half: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Only observable inputs accepted here; possession/dead labels unavailable."""
    rng = np.random.default_rng(seed)
    p, b = players.copy(), ball.copy()
    for period in np.unique(half):
        if period == 0:
            continue
        index = np.flatnonzero(half == period)
        n = len(index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            collective = np.nanmean(players[index, :, 2], axis=1)
        slow = 1 / (1 + np.nan_to_num(collective, nan=2))
        weights = 0.5 + uniform_filter1d(slow, min(n, 251))
        missing = np.zeros(n, dtype=bool)
        target = int(0.41 * n)
        count = max(1, round(target / (17 * FPS)))
        extra = rng.exponential(1.0, count)
        lengths = 126 + np.floor(extra / extra.sum() * (target - 126 * count)).astype(
            int
        )
        lengths[0] += target - int(lengths.sum())
        for length in sorted(lengths, reverse=True):
            positions = np.arange(max(0, n - length))
            cumulative = np.r_[0, np.cumsum(missing)]
            free = (
                cumulative[np.minimum(n, positions + length + 125)]
                - cumulative[np.maximum(0, positions - 125)]
            ) == 0
            positions = positions[free]
            if not len(positions):
                raise ValueError(
                    "Cannot place label-free ball gaps with frozen targets"
                )
            probability = weights[positions] / weights[positions].sum()
            start = int(rng.choice(positions, p=probability))
            missing[start : start + length] = True
        b[index[missing]] = np.nan
        for slot in range(p.shape[1]):
            absent = np.zeros(n, dtype=bool)
            position = int(rng.integers(0, 450))
            while position < n:
                length = int(rng.integers(4, 35))
                absent[position : position + length] = True
                position += int(25 * rng.exponential(13.6 / np.log(2))) + length
            absent |= rng.random(n) < 0.055
            p[index[absent], slot] = np.nan
        # Team-preserving tracklet permutation: identity never becomes a feature.
        for team in (0, 1):
            slots = np.flatnonzero(teams == team)
            start = 0
            while start < n:
                stop = min(
                    n, start + max(1, int(rng.exponential(13.6 / np.log(2)) * FPS))
                )
                section = index[start:stop]
                p[np.ix_(section, slots)] = p[np.ix_(section, rng.permutation(slots))]
                start = stop
    # Additional duplicate slots model false-positive track detections.
    duplicates = np.full((len(p), 2, 3), np.nan, dtype="float32")
    for team in (0, 1):
        slot = rng.choice(np.flatnonzero(teams == team))
        visible = uniform_filter1d(rng.random(len(p)), 125) > 0.505
        duplicates[visible, team] = p[visible, slot]
    return np.concatenate([p, duplicates], axis=1), b


def degrade(match: Match, seed: int = 20261108) -> Match:
    p, b = _corrupt_observations(
        match.players, match.ball, match.teams, match.half, seed
    )
    return replace(
        match, players=p, ball=b, teams=np.r_[match.teams, [0, 1]].astype("int8")
    )


def smooth_baseline(near: np.ndarray, half: np.ndarray, window: int = 50) -> np.ndarray:
    output = np.full(len(near), 2, dtype="int8")
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        values = near[index]
        position = np.arange(len(values))
        lo, hi = np.maximum(0, position - window), np.minimum(
            len(values), position + window + 1
        )
        home = np.r_[0, np.cumsum(values == 0)]
        away = np.r_[0, np.cumsum(values == 1)]
        h, a = home[hi] - home[lo], away[hi] - away[lo]
        output[index] = np.where(h + a > 0, np.where(h >= a, 0, 1), 2)
    return output


def degradation_summary(match: Match) -> dict:
    visible = np.isfinite(match.ball).all(axis=1)
    lengths = [
        stop - start
        for period in (1, 2)
        for start, stop in runs(~visible[match.half == period])
    ]
    counts = np.isfinite(match.players[:, :, 0]).sum(axis=1)
    return {
        "match": match.name,
        "frames": len(visible),
        "ball_visible": float(visible.mean()),
        "gap_mean_seconds": float(np.mean(lengths) / FPS) if lengths else 0.0,
        "missing_in_long_gaps": (
            float(sum(x for x in lengths if x > 125) / sum(lengths)) if lengths else 0.0
        ),
        "mean_objects": float(counts.mean()),
        **(
            {
                "label_" + label: float(np.mean(match.labels == code))
                for code, label in enumerate("HADX")
            }
            if match.labels is not None
            else {}
        ),
    }


def load_target(source: str = "tactical") -> Match:
    """Read raw target inputs locally; anonymous team mapping only from metadata."""
    if source not in ("tactical", "broadcast"):
        raise ValueError("Unknown recording")
    meta = REPO / "data" / source / ("BAR-ATH_" + source + "_atd.xml")
    tree = etree.parse(str(meta))
    home = next(tree.iter("Team")).get("id")
    teams = np.array(
        [0 if p.get("teamId") == home else 1 for p in tree.iter("Player")], dtype="int8"
    )
    blocks = []
    raw = REPO / "data" / source / ("BAR-ATH_" + source + "_atd.txt")
    with raw.open() as handle:
        while lines := handle.readlines(32_000_000):
            text = "".join(lines).replace(":", ",").replace(";", ",")
            block = (
                __import__("pandas")
                .read_csv(io.StringIO(text), header=None, dtype="float32")
                .to_numpy()
            )
            if block.shape[1] != 1 + 3 * len(teams) + 3:
                raise ValueError("Target raw tracking schema differs from metadata")
            blocks.append(block)
    values = np.concatenate(blocks)
    frame = values[:, 0].astype("int64") - 1
    players = values[:, 1 : 1 + 3 * len(teams)].reshape(-1, len(teams), 3).copy()
    ball = values[:, -3:-1].copy()
    if not np.array_equal(frame, np.arange(len(frame))):
        raise ValueError("Target tracking frame alignment failed")
    pitch = (105.3, 68.0)
    players[:, :, :2] *= np.array(pitch)
    ball *= np.array(pitch)
    half = np.zeros(len(frame), dtype="int8")
    if source == "tactical":
        half[11185:90652], half[110808:190198] = 1, 2
        for period in (1, 2):
            index = np.flatnonzero(half == period)
            players[index, :, 2] = _derived_speed(players[index, :, :2])
    return Match("BAR-ATH-" + source, players, ball, teams, half, frame, None, pitch)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
