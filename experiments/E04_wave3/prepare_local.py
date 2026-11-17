from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

WAVE3 = Path(__file__).resolve().parent
REPO = WAVE3.parents[1]
DATA_DIR = WAVE3 / "cache" / "data"
AUDIT_PATH = WAVE3 / "cache" / "local_audit.json"
IDSSE_ROOT = WAVE3 / "cache" / "idsse_parser"
FPS = 5
STRIDE = 5
PITCH = np.asarray((105.3, 68.0), dtype=np.float32)
TARGET_FRAMES = 199319
TARGET_PLAYER_SLOTS = 46
TARGET_SLOT_WIDTH = 141
TARGET_UNKNOWN_SLOTS = (21, 22, 43)
MATCHES = {1: "metrica1", 2: "metrica2", 3: "metrica3"}
SEED_BASE = 20262108


def _load_module(path: Path, role: str):
    name = f"_farras_wave3_local_{role}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


feature_data = _load_module(WAVE3.parent / "E02_wave1" / "data.py", "wave1_data")
reference = _load_module(WAVE3 / "reference" / "jeremy_data.py", "jeremy_data")
reference.ROOT = IDSSE_ROOT


def _half_runs(half: np.ndarray) -> list[tuple[int, int]]:
    blocks = []
    for period in np.unique(half):
        if period <= 0:
            continue
        edges = np.diff(np.r_[False, half == period, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        blocks.extend((int(start), int(stop)) for start, stop in zip(starts, stops))
    blocks.sort()
    return blocks


def _sample_indices(half: np.ndarray):
    sampled_parts = []
    blocks = []
    blocks25 = []
    offset = 0
    for start, stop in _half_runs(half):
        indices = np.arange(start, stop, STRIDE, dtype=np.int64)
        if not len(indices):
            continue
        sampled_parts.append(indices)
        blocks.append((offset, offset + len(indices)))
        blocks25.append((start, stop))
        offset += len(indices)
    if not sampled_parts:
        raise ValueError("Match contains no positive-half samples")
    return (
        np.concatenate(sampled_parts),
        np.asarray(blocks, dtype=np.int64).reshape(-1, 2),
        np.asarray(blocks25, dtype=np.int64).reshape(-1, 2),
    )


def _derived_players(xy_m: np.ndarray, half: np.ndarray) -> np.ndarray:
    players = np.full((*xy_m.shape[:2], 3), np.nan, dtype=np.float32)
    players[:, :, :2] = xy_m
    for start, stop in _half_runs(half):
        players[start:stop, :, 2] = reference._derived_speed(players[start:stop, :, :2])
    return players


def _labels_for_reference(labels: np.ndarray) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int8).copy()
    result[result < 0] = 3
    if np.any((result < 0) | (result > 3)):
        raise ValueError("Unexpected event-derived label code")
    return result


def _load_metrica_csv(game: int):
    source = feature_data.RYU / "_cache" / "ext_metrica"
    paths = {
        "events": source / f"Sample_Game_{game}_RawEventsData.csv",
        "home": source / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv",
        "away": source / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv",
    }
    home, away = [
        pd.read_csv(paths[side], header=None, skiprows=3)
        for side in ("home", "away")
    ]
    if not np.array_equal(home.iloc[:, :2], away.iloc[:, :2]):
        raise ValueError(f"Metrica Game {game} Home/Away timelines differ")
    if (home.shape[1] < 5 or away.shape[1] < 5
            or (home.shape[1] - 5) % 2 or (away.shape[1] - 5) % 2):
        raise ValueError(f"Metrica Game {game} tracking columns do not contain x/y pairs")
    half = home.iloc[:, 0].to_numpy(dtype=np.int8)
    frame = home.iloc[:, 1].to_numpy(dtype=np.int64)
    home_xy = home.iloc[:, 3:-2].to_numpy(dtype=np.float32).reshape(len(home), -1, 2)
    away_xy = away.iloc[:, 3:-2].to_numpy(dtype=np.float32).reshape(len(away), -1, 2)
    ball_xy = home.iloc[:, -2:].to_numpy(dtype=np.float32)
    events = pd.read_csv(paths["events"])
    proxy = feature_data.event_labels(events, half, frame)
    teams = np.r_[np.zeros(home_xy.shape[1], dtype=np.int8),
                  np.ones(away_xy.shape[1], dtype=np.int8)]
    xy_m = np.concatenate([home_xy, away_xy], axis=1) * PITCH
    ball_m = ball_xy * PITCH
    match = reference.Match(
        MATCHES[game],
        _derived_players(xy_m, half),
        ball_m.astype(np.float32, copy=False),
        teams,
        half,
        frame,
        _labels_for_reference(proxy),
        tuple(float(value) for value in PITCH),
    )
    return match, paths


def _load_metrica_game3():
    source = feature_data.RYU / "_cache" / "ext_metrica"
    paths = {
        "tracking": source / "Sample_Game_3_tracking.txt",
        "metadata": source / "Sample_Game_3_metadata.xml",
        "events": source / "Sample_Game_3_events.json",
    }
    meta = ET.parse(paths["metadata"]).getroot()
    events = json.loads(paths["events"].read_text(encoding="utf-8"))["data"]
    team_of = {
        player.attrib["id"]: 0 if player.attrib["teamId"] == "FIFATMA" else 1
        for player in meta.iter("Player")
    }
    ids = list(team_of)
    index_of = {player: index for index, player in enumerate(ids)}
    channels = {channel.attrib["id"]: channel.attrib for channel in meta.iter("PlayerChannel")}
    formats = sorted(
        meta.iter("DataFormatSpecification"),
        key=lambda item: int(item.attrib["startFrame"]),
    )
    raw_rows, ball_rows, frame_rows = [], [], []
    with paths["tracking"].open(encoding="utf-8") as stream:
        for line in stream:
            frame_value, player_values, ball_values = line.strip().split(":")
            frame_rows.append(int(frame_value))
            raw_rows.append(np.fromstring(player_values.replace(";", ","), sep=",", dtype=np.float32))
            ball_rows.append(np.fromstring(ball_values, sep=",", dtype=np.float32))
    frames = np.asarray(frame_rows, dtype=np.int64)
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("Game 3 tracking is not contiguous one-based frames")
    raw = np.stack(raw_rows).reshape(len(frames), -1, 2)
    ball_xy = np.stack(ball_rows)
    players_xy = np.full((len(frames), len(ids), 2), np.nan, dtype=np.float32)
    coverage = np.zeros(len(frames), dtype=np.uint8)
    for fmt in formats:
        start, stop = int(fmt.attrib["startFrame"]) - 1, int(fmt.attrib["endFrame"])
        refs = [channels[ref.attrib["playerChannelId"]] for ref in fmt.iter("PlayerChannelRef")]
        if len(refs) != 2 * raw.shape[1]:
            raise ValueError("Player-slot width differs from active metadata format")
        selected = []
        for x, y in zip(refs[::2], refs[1::2]):
            if (x["channelId"] != "x" or y["channelId"] != "y"
                    or x["playerId"] != y["playerId"]):
                raise ValueError("Metadata coordinates are not paired x/y for same player")
            selected.append(index_of[x["playerId"]])
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate active player identity")
        players_xy[start:stop, selected, :] = raw[start:stop]
        coverage[start:stop] += 1
    if not np.all(coverage == 1):
        raise ValueError("Metadata formats must partition every tracking frame once")

    half = np.zeros(len(frames), dtype=np.int8)
    event_rows = []
    for period in sorted({event["period"] for event in events}):
        group = [event for event in events if event["period"] == period]
        start = min(event["start"]["frame"] for event in group)
        stop = max(event["end"]["frame"] for event in group)
        if start < 1 or stop > len(frames) or np.any(half[start - 1:stop]):
            raise ValueError("Event period bounds overlap or exceed tracking")
        half[start - 1:stop] = period
    for event in events:
        subtype = event["subtypes"]
        subtypes = [subtype] if isinstance(subtype, dict) else (subtype or [])
        event_rows.append({
            "Period": event["period"],
            "Team": "Home" if event["team"]["id"] == "FIFATMA" else "Away",
            "Type": event["type"]["name"],
            "Subtype": "|".join(item["name"] for item in subtypes),
            "Start Frame": event["start"]["frame"],
            "End Frame": event["end"]["frame"],
        })
    proxy = feature_data.event_labels(pd.DataFrame(event_rows), half, frames)
    teams = np.asarray([team_of[player] for player in ids], dtype=np.int8)
    match = reference.Match(
        MATCHES[3],
        _derived_players(players_xy * PITCH, half),
        (ball_xy * PITCH).astype(np.float32, copy=False),
        teams,
        half,
        frames,
        _labels_for_reference(proxy),
        tuple(float(value) for value in PITCH),
    )
    return match, paths


def _nearest_distances(players_m: np.ndarray, ball_m: np.ndarray, teams: np.ndarray) -> np.ndarray:
    distances = np.full((len(ball_m), 2), np.nan, dtype=np.float32)
    visible_ball = np.isfinite(ball_m).all(axis=1)
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        if not len(slots):
            raise ValueError(f"No player slots assigned to team {team}")
        for start in range(0, len(ball_m), 8192):
            stop = min(len(ball_m), start + 8192)
            player_xy = players_m[start:stop, slots, :2]
            ball_xy = ball_m[start:stop]
            dx = player_xy[:, :, 0] - ball_xy[:, None, 0]
            dy = player_xy[:, :, 1] - ball_xy[:, None, 1]
            np.hypot(dx, dy, out=dx)
            valid = np.isfinite(player_xy).all(axis=2) & visible_ball[start:stop, None]
            dx[~valid] = np.inf
            nearest = dx.min(axis=1)
            nearest[~np.isfinite(nearest)] = np.nan
            distances[start:stop, team] = nearest
    return distances


def _team_votes(players_m: np.ndarray, ball_m: np.ndarray, nearest_m: np.ndarray,
                sampled: np.ndarray, slots: int) -> tuple[np.ndarray, list[dict]]:
    home_votes = np.zeros(slots, dtype=np.int64)
    away_votes = np.zeros(slots, dtype=np.int64)
    present = np.zeros(slots, dtype=bool)
    for offset in range(0, len(sampled), 8192):
        indices = sampled[offset:offset + 8192]
        player_xy = players_m[indices, :slots, :2]
        ball_xy = ball_m[indices]
        nearest = nearest_m[indices]
        visible = np.isfinite(ball_xy).all(axis=1) & np.isfinite(nearest).all(axis=1)
        near_equal = np.isclose(nearest[:, 0], nearest[:, 1], rtol=1e-4, atol=1e-3)
        eligible = visible & ~near_equal
        dx = player_xy[:, :, 0] - ball_xy[:, None, 0]
        dy = player_xy[:, :, 1] - ball_xy[:, None, 1]
        np.hypot(dx, dy, out=dx)
        present |= np.isfinite(player_xy).all(axis=2).any(axis=0)
        home_match = np.isclose(dx, nearest[:, 0, None], rtol=1e-4, atol=1e-3)
        away_match = np.isclose(dx, nearest[:, 1, None], rtol=1e-4, atol=1e-3)
        home_votes += np.count_nonzero(home_match & ~away_match & eligible[:, None], axis=0)
        away_votes += np.count_nonzero(away_match & ~home_match & eligible[:, None], axis=0)
    teams = np.full(slots, -1, dtype=np.int8)
    evidence = []
    for slot in range(slots):
        home_count = int(home_votes[slot])
        away_count = int(away_votes[slot])
        matches = home_count + away_count
        purity = (max(home_count, away_count) / matches) if matches else 0.0
        if present[slot]:
            if matches < 20 or purity < 0.99:
                raise ValueError(
                    f"Target slot {slot} team evidence insufficient: "
                    f"matches={matches}, purity={purity:.6f}"
                )
            teams[slot] = 0 if home_count > away_count else 1
        evidence.append({
            "slot": slot,
            "present_on_sampled_grid": bool(present[slot]),
            "home_votes": home_count,
            "away_votes": away_count,
            "matches": matches,
            "purity": float(purity),
            "assigned_team": int(teams[slot]),
        })
    return teams, evidence


def _check_target_teams(teams: np.ndarray, players_m: np.ndarray, sampled: np.ndarray) -> None:
    if teams.shape != (TARGET_PLAYER_SLOTS,):
        raise ValueError("Target team vector has unexpected width")
    counts = [int(np.count_nonzero(teams == team)) for team in (0, 1)]
    unknown = np.flatnonzero(teams == -1).tolist()
    if counts != [22, 21] or unknown != list(TARGET_UNKNOWN_SLOTS):
        raise ValueError(
            f"Target team evidence disagrees with expected active slots: "
            f"home={counts[0]}, away={counts[1]}, unknown={unknown}"
        )
    if np.isfinite(players_m[sampled][:, teams == -1, :2]).any():
        raise ValueError("Target unknown-team slots are present on sampled grid")


def _validate_target_distances(players_m: np.ndarray, ball_m: np.ndarray, teams: np.ndarray,
                               cached: np.ndarray) -> dict:
    reconstructed = _nearest_distances(players_m, ball_m, teams)
    visible = np.isfinite(ball_m).all(axis=1)
    available = np.isfinite(reconstructed)
    cached_available = np.isfinite(cached)
    if not np.array_equal(available[visible], cached_available[visible]):
        raise ValueError("Reconstructed and cached team-distance availability disagree")
    comparable = visible[:, None] & available & cached_available
    if not np.allclose(reconstructed[comparable], cached[comparable], rtol=1e-4, atol=1e-3):
        delta = np.abs(reconstructed[comparable] - cached[comparable])
        maximum = float(np.nanmax(delta)) if delta.size else 0.0
        raise ValueError(f"Reconstructed target nearest distances disagree; max={maximum:.9g}m")
    delta = np.abs(reconstructed[comparable] - cached[comparable])
    return {
        "visible_frames_checked": int(visible.sum()),
        "finite_team_distances_checked": int(comparable.sum()),
        "unavailable_team_distances_on_visible_frames": int((~available[visible]).sum()),
        "max_abs_discrepancy_m": float(np.max(delta)) if delta.size else 0.0,
        "rtol": 1e-4,
        "atol_m": 1e-3,
    }


def _load_target():
    source = feature_data.RYU / "_cache"
    paths = {
        "tracking": source / "atd_full.npy",
        "nearest_distances": source / "team_dist.npy",
    }
    flat = np.load(paths["tracking"], mmap_mode="r", allow_pickle=False)
    cached_distances = np.load(paths["nearest_distances"], mmap_mode="r", allow_pickle=False)
    if flat.shape != (TARGET_FRAMES, TARGET_SLOT_WIDTH):
        raise ValueError(f"Target ATD tracking shape differs: {flat.shape}")
    if cached_distances.shape != (TARGET_FRAMES, 2):
        raise ValueError(f"Target nearest-distance shape differs: {cached_distances.shape}")

    games = _load_module(REPO / "games.py", "games")
    blocks25 = [(int(start), int(stop) + 1) for _, start, stop in games.half_video_blocks("BAR-ATH")]
    half = np.zeros(TARGET_FRAMES, dtype=np.int8)
    for period, (start, stop) in enumerate(blocks25, start=1):
        if start < 0 or stop > len(half) or np.any(half[start:stop]):
            raise ValueError("Target half boundaries overlap or exceed tracking")
        half[start:stop] = period
    sampled, blocks, blocks25_array = _sample_indices(half)

    player_xy = flat[:, :-3].reshape(TARGET_FRAMES, TARGET_PLAYER_SLOTS, 3)[:, :, :2]
    players_m = np.asarray(player_xy, dtype=np.float32).copy()
    players_m *= PITCH
    ball_m = np.asarray(flat[:, -3:-1], dtype=np.float32).copy()
    ball_m *= PITCH
    cached = np.asarray(cached_distances, dtype=np.float32)
    teams, evidence = _team_votes(players_m, ball_m, cached, sampled, TARGET_PLAYER_SLOTS)
    _check_target_teams(teams, players_m, sampled)
    distance_audit = _validate_target_distances(
        players_m[sampled], ball_m[sampled], teams, cached[sampled])
    distance_audit["scope"] = "sampled graded frames at5fps; no team claim for omitted frames"
    match = reference.Match(
        "target",
        _derived_players(players_m, half),
        ball_m,
        teams,
        half,
        np.arange(TARGET_FRAMES, dtype=np.int64),
        None,
        tuple(float(value) for value in PITCH),
    )
    return match, paths, evidence, distance_audit


def _label_counts(labels: np.ndarray) -> dict:
    return {
        "home": int(np.count_nonzero(labels == 0)),
        "away": int(np.count_nonzero(labels == 1)),
        "dead": int(np.count_nonzero(labels == 2)),
        "unknown": int(np.count_nonzero(labels == -1)),
    }


def _write_pack(name: str, regime: str, match, source: str) -> dict:
    sampled, blocks, blocks25 = _sample_indices(match.half)
    teams = np.asarray(match.teams, dtype=np.int8)
    if match.players.shape[1] != len(teams):
        raise ValueError(f"Player/team slot mismatch for {name}_{regime}")
    if np.any((teams < -1) | (teams > 1)):
        raise ValueError(f"Invalid team code for {name}_{regime}")
    players_m = np.asarray(match.players[:, :, :2], dtype=np.float32)
    ball_m = np.asarray(match.ball, dtype=np.float32)
    players = players_m[sampled] / PITCH
    ball = ball_m[sampled] / PITCH
    if np.any(np.isfinite(players[:, teams == -1, :])):
        raise ValueError(f"Unknown-team slots present in {name}_{regime}")
    distances = _nearest_distances(players_m[sampled], ball_m[sampled], teams)
    X = feature_data.features(players, ball, distances, blocks.tolist())
    if not np.isfinite(X).all():
        raise ValueError(f"Nonfinite features for {name}_{regime}")
    if match.labels is None:
        y25 = np.full(len(match.half), -1, dtype=np.int8)
    else:
        y25 = np.asarray(match.labels, dtype=np.int8).copy()
        y25[y25 == 3] = -1
    if np.any((y25 < -1) | (y25 > 2)):
        raise ValueError(f"Unexpected prepared label code for {name}_{regime}")
    y = y25[sampled]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / f"{name}_{regime}.npz"
    part = output.with_name(output.name + ".part")
    try:
        with part.open("wb") as stream:
            np.savez_compressed(
                stream,
                X=X.astype(np.float32, copy=False),
                players=players.astype(np.float32, copy=False),
                teams=teams,
                ball=ball.astype(np.float32, copy=False),
                blocks=blocks,
                sampled=sampled,
                y=y.astype(np.int8, copy=False),
                y25=y25,
                blocks25=blocks25,
                n_frames=np.asarray(len(y25), dtype=np.int64),
                fps=np.asarray(FPS, dtype=np.int64),
            )
        os.replace(part, output)
    finally:
        if part.exists():
            part.unlink()

    present_players = int(np.isfinite(players[:, :, :2]).all(axis=2).sum())
    total_players = int(players.shape[0] * players.shape[1])
    result = {
        "path": output.relative_to(WAVE3).as_posix(),
        "sha256": feature_data.digest(output),
        "source": source,
        "frames": int(len(y25)),
        "samples": int(len(sampled)),
        "blocks": int(len(blocks)),
        "feature_shape": list(X.shape),
        "player_slots": int(players.shape[1]),
        "team_slot_counts": {
            "home": int(np.count_nonzero(teams == 0)),
            "away": int(np.count_nonzero(teams == 1)),
            "unknown": int(np.count_nonzero(teams == -1)),
        },
        "visible_ball_samples": int(np.isfinite(ball).all(axis=1).sum()),
        "visible_player_slots": present_players,
        "total_player_slots": total_players,
        "player_visible_fraction": float(present_players / total_players) if total_players else 0.0,
        "proxy_label_counts": _label_counts(y25),
    }
    return result


def _source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    result = {}
    for role, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"Required local source missing: {path}")
        try:
            key = path.relative_to(feature_data.RYU).as_posix()
        except ValueError:
            key = path.as_posix()
        result[f"{role}:{key}"] = feature_data.digest(path)
    return result


def prepare() -> dict:
    source_files: dict[str, Path] = {}
    pack_audit = {}
    metrica_clean = {}
    for game in (1, 2):
        match, paths = _load_metrica_csv(game)
        metrica_clean[game] = match
        source_files.update({f"game{game}_{role}": path for role, path in paths.items()})
    metrica_clean[3], game3_paths = _load_metrica_game3()
    source_files.update({f"game3_{role}": path for role, path in game3_paths.items()})

    target, target_paths, team_evidence, distance_audit = _load_target()
    source_files.update({f"target_{role}": path for role, path in target_paths.items()})
    source_hashes = _source_hashes(source_files)

    for game, clean in metrica_clean.items():
        name = MATCHES[game]
        pack_audit[f"{name}_clean"] = _write_pack(name, "clean", clean, "native Metrica tracking and event-derived proxy labels")
        for round_index in range(3):
            seed = SEED_BASE + 1000 * round_index + game
            degraded = reference.degrade(clean, seed=seed)
            record = _write_pack(name, f"r{round_index}", degraded, "Jeremy reference label-free degradation")
            record["seed"] = seed
            pack_audit[f"{name}_r{round_index}"] = record
    pack_audit["target_clean"] = _write_pack(
        "target", "clean", target,
        "native cached ATD tracking; team slots inferred only from cached nearest distances",
    )

    audit = {
        "format": "farras-wave3-local-audit-v1",
        "source_root": feature_data.RYU.as_posix(),
        "source_sha256": source_hashes,
        "feature_code_sha256": feature_data.digest(Path(feature_data.__file__)),
        "reference_code_sha256": feature_data.digest(WAVE3 / "reference" / "jeremy_data.py"),
        "game3_parser_sha256": feature_data.digest(Path(__file__)),
        "packs": pack_audit,
        "target_team_assignment": {
            "method": "per-slot geometric distance matches to cached home/away nearest distances",
            "rtol": 1e-4,
            "atol_m": 1e-3,
            "minimum_matches": 20,
            "minimum_purity": 0.99,
            "unknown_slots": list(TARGET_UNKNOWN_SLOTS),
            "slot_votes": team_evidence,
            "nearest_distance_validation": distance_audit,
            "provenance": "No target team metadata or labels read or inferred from XML; unknown labels remain -1.",
        },
        "label_provenance": "Metrica labels are wave1 event-derived proxies, not corrected possession ground truth; target labels are unknown.",
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_PATH.with_name(AUDIT_PATH.name + ".part")
    try:
        temporary.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, AUDIT_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
    return audit


def main() -> None:
    argparse.ArgumentParser(description="Prepare local Metrica and target packs for wave3.").parse_args()
    audit = prepare()
    print(json.dumps({"packs": len(audit["packs"]), "audit": AUDIT_PATH.as_posix()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
