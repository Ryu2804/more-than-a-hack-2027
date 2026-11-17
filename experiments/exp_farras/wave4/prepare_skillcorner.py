"""Prepare SkillCorner Open Data matches for Wave 4.

Source: SkillCorner Open Data (MIT License), https://github.com/SkillCorner/opendata.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


WAVE4 = Path(__file__).resolve().parent
RAW_ROOT = WAVE4 / "cache" / "raw"
RAW_DIR = RAW_ROOT / "ext_skillcorner"
DATA_DIR = WAVE4 / "cache" / "data"
AUDIT_PATH = WAVE4 / "cache" / "skillcorner_audit.json"
WAVE1_FEATURES = WAVE4.parent / "wave1" / "data.py"
CHUNK_SIZE = 1024 * 1024
DISTANCE_CHUNK = 8192
FPS = 5
STRIDE = 5
INTERPOLATION_MAX_GAP_SECONDS = 0.35


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _git_blob_oid(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _load_reference() -> tuple[list[int], str, str]:
    source_path = WAVE4 / "reference" / "skillcorner_games.py"
    module_name = f"_wave4_skillcorner_source_{uuid.uuid4().hex}"
    common_stub = types.ModuleType("common")
    common_stub.CACHE = str(RAW_ROOT)
    had_common = "common" in sys.modules
    previous_common = sys.modules.get("common")
    sys.modules["common"] = common_stub
    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned SkillCorner source module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if had_common:
            sys.modules["common"] = previous_common
        else:
            sys.modules.pop("common", None)
    match_ids = [int(match_id) for match_id in module.MATCH_IDS]
    if len(match_ids) != 20 or len(set(match_ids)) != 20:
        raise ValueError("Pinned SkillCorner source must define 20 unique matches")
    return match_ids, module.RAW, module.MEDIA


def _https_url(url: str) -> str:
    if not url.startswith("https://"):
        raise ValueError("SkillCorner input URL must use HTTPS")
    return url


def _fetch_lfs_pointer(url: str) -> tuple[bytes, str, int]:
    request = Request(_https_url(url), headers={"User-Agent": "wave4-skillcorner-preparer"})
    with urlopen(request, timeout=60) as response:
        pointer = response.read(4096)
    if len(pointer) >= 4096:
        raise ValueError("SkillCorner tracking LFS pointer exceeds expected size")
    lines = pointer.decode("ascii").strip().splitlines()
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        raise ValueError("SkillCorner tracking RAW source is not a Git LFS pointer")
    oid = next((line.removeprefix("oid sha256:") for line in lines if line.startswith("oid sha256:")), "")
    size_text = next((line.removeprefix("size ") for line in lines if line.startswith("size ")), "")
    if not re.fullmatch(r"[0-9a-f]{64}", oid) or not size_text.isdecimal():
        raise ValueError("SkillCorner tracking LFS pointer has invalid oid or size")
    expected_size = int(size_text)
    if expected_size <= 0:
        raise ValueError("SkillCorner tracking LFS pointer declares invalid size")
    return pointer, oid, expected_size


def _download(url: str, destination: Path, expected_sha256: str | None = None,
              expected_size: int | None = None) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(_https_url(url), headers={"User-Agent": "wave4-skillcorner-preparer"})
            with urlopen(request, timeout=120) as response:
                for block in iter(lambda: response.read(CHUNK_SIZE), b""):
                    size += len(block)
                    if expected_size is not None and size > expected_size:
                        raise ValueError("SkillCorner download exceeds declared LFS size")
                    output.write(block)
                    digest.update(block)
        sha256 = digest.hexdigest()
        if size == 0:
            raise ValueError("SkillCorner download is empty")
        if expected_size is not None and size != expected_size:
            raise ValueError("SkillCorner download size differs from declared LFS size")
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ValueError("SkillCorner download SHA-256 differs from declared LFS oid")
        os.replace(temporary_name, destination)
        return sha256, size
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _ensure_tracking_input(path: Path, raw_url: str, media_url: str) -> dict:
    pointer, expected_sha256, expected_size = _fetch_lfs_pointer(raw_url)
    pointer_sha256 = hashlib.sha256(pointer).hexdigest()
    if path.exists():
        size = path.stat().st_size
        if size != expected_size:
            raise ValueError(f"SkillCorner tracking LFS size verification failed: {path.name}")
        sha256 = expected_sha256
    else:
        sha256, size = _download(media_url, path, expected_sha256, expected_size)
    return {
        "path": path.relative_to(WAVE4).as_posix(),
        "url": _https_url(media_url),
        "sha256": sha256,
        "size_bytes": size,
        "lfs_oid_sha256": expected_sha256,
        "lfs_size_bytes": expected_size,
        "lfs_pointer_sha256": pointer_sha256,
        "lfs_pointer_size_bytes": len(pointer),
        "lfs_pointer_git_blob_oid": _git_blob_oid(pointer),
        "verification": "LFS_pointer_oid_and_size",
    }


def _ensure_regular_input(path: Path, url: str) -> dict:
    if not path.exists():
        _download(url, path)
    sha256, size = _sha256_file(path)
    if size <= 0:
        raise ValueError(f"SkillCorner input is empty: {path.name}")
    return {
        "path": path.relative_to(WAVE4).as_posix(),
        "url": _https_url(url),
        "sha256": sha256,
        "size_bytes": size,
        "verification": "downloaded_or_local_sha256_and_size",
    }


def _read_metadata(path: Path) -> tuple[dict, dict[int, int], float, float]:
    with path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("SkillCorner match metadata must be a JSON object")
    home = metadata.get("home_team", {}).get("id")
    away = metadata.get("away_team", {}).get("id")
    if home is None or away is None or home == away:
        raise ValueError("SkillCorner match metadata has invalid home/away team IDs")
    try:
        length = float(metadata["pitch_length"])
        width = float(metadata["pitch_width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("SkillCorner match metadata has invalid pitch dimensions") from error
    if not np.isfinite([length, width]).all() or length <= 0 or width <= 0:
        raise ValueError("SkillCorner match metadata has invalid pitch dimensions")
    roster = metadata.get("players")
    if not isinstance(roster, list) or not roster:
        raise ValueError("SkillCorner match metadata has no player roster")
    team_of: dict[int, int] = {}
    for player in roster:
        try:
            player_id = player["id"]
            team_id = player["team_id"]
        except (KeyError, TypeError) as error:
            raise ValueError("SkillCorner roster entry lacks player/team ID") from error
        if team_id == home:
            side = 0
        elif team_id == away:
            side = 1
        else:
            raise ValueError("SkillCorner roster contains an unknown team ID")
        if player_id in team_of and team_of[player_id] != side:
            raise ValueError("SkillCorner player ID is assigned to both teams")
        team_of[player_id] = side
    if not team_of or set(team_of.values()) != {0, 1}:
        raise ValueError("SkillCorner roster must contain players from both teams")
    return metadata, team_of, length, width


def _tracking_record(line: bytes, line_number: int) -> dict:
    try:
        row = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid SkillCorner tracking JSON at line {line_number}") from error
    if not isinstance(row, dict):
        raise ValueError(f"SkillCorner tracking row is not an object at line {line_number}")
    if not isinstance(row.get("player_data"), list) or not isinstance(row.get("ball_data"), dict):
        raise ValueError(f"SkillCorner tracking row lacks player/ball data at line {line_number}")
    return row


def _scan_tracking(path: Path, team_of: dict[int, int]) -> tuple[list[int], str, int, int]:
    player_ids: set[int] = set()
    digest = hashlib.sha256()
    size = 0
    frame_count = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            size += len(line)
            digest.update(line)
            if not line.strip():
                continue
            row = _tracking_record(line, line_number)
            for player in row["player_data"]:
                if not isinstance(player, dict):
                    raise ValueError(f"SkillCorner player row is not an object at line {line_number}")
                player_id = player.get("player_id")
                if player_id in team_of:
                    player_ids.add(player_id)
            frame_count += 1
    if frame_count == 0 or not player_ids:
        raise ValueError("SkillCorner tracking contains no frames or rostered players")
    return sorted(player_ids), digest.hexdigest(), size, frame_count


def _integer_period(value, line_number: int) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"Invalid SkillCorner period at line {line_number}")
    try:
        period = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid SkillCorner period at line {line_number}") from error
    if str(value).strip() not in (str(period), f"{period}.0") or not 0 <= period <= 127:
        raise ValueError(f"Invalid SkillCorner period at line {line_number}")
    return period


def _parse_tracking(path: Path, team_of: dict[int, int], player_ids: list[int],
                    expected_digest: str, expected_size: int, expected_frames: int):
    player_columns = {player_id: column for column, player_id in enumerate(player_ids)}
    players = np.full((expected_frames, len(player_ids), 2), np.nan, dtype=np.float32)
    ball = np.full((expected_frames, 2), np.nan, dtype=np.float32)
    period = np.zeros(expected_frames, dtype=np.int8)
    digest = hashlib.sha256()
    size = 0
    frame_index = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            size += len(line)
            digest.update(line)
            if not line.strip():
                continue
            row = _tracking_record(line, line_number)
            if frame_index >= expected_frames:
                raise ValueError("SkillCorner tracking changed between passes")
            period[frame_index] = _integer_period(row.get("period"), line_number)
            ball_row = row["ball_data"]
            if ball_row.get("is_detected") and ball_row.get("x") is not None:
                ball[frame_index] = (float(ball_row["x"]), float(ball_row["y"]))
            for player in row["player_data"]:
                player_id = player.get("player_id")
                column = player_columns.get(player_id)
                if column is not None and player.get("x") is not None:
                    players[frame_index, column] = (float(player["x"]), float(player["y"]))
            frame_index += 1
    if (frame_index != expected_frames or size != expected_size or
            digest.hexdigest() != expected_digest):
        raise ValueError("SkillCorner tracking changed between passes")
    teams = np.asarray([team_of[player_id] for player_id in player_ids], dtype=np.int8)
    if np.any((teams != 0) & (teams != 1)):
        raise ValueError("SkillCorner tracking contains unknown team assignments")
    return players, ball, period, teams


def _read_phases(path: Path, n10: int, home_id, away_id) -> np.ndarray:
    labels = np.full(n10, -1, dtype=np.int8)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_start", "frame_end", "team_in_possession_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("SkillCorner phases file lacks required columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                start = int(row["frame_start"])
                stop = int(row["frame_end"])
                team_id = int(row["team_in_possession_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid SkillCorner phase row {row_number}") from error
            if start < 0 or stop < start or stop > n10:
                raise ValueError(f"SkillCorner phase range is outside tracking at row {row_number}")
            if team_id == home_id:
                label = 0
            elif team_id == away_id:
                label = 1
            else:
                raise ValueError(f"SkillCorner phase has unknown team ID at row {row_number}")
            labels[start:stop] = label
    return labels


def _interpolate_samples(series: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    finite = np.isfinite(series)
    source_indices = np.flatnonzero(finite)
    if len(source_indices) < 2:
        return np.full(len(sampled), np.nan, dtype=np.float32)
    source_times = source_indices.astype(np.float64) / 10.0
    query_times = sampled.astype(np.float64) / 25.0
    values = np.interp(query_times, source_times, series[finite], left=np.nan, right=np.nan)
    right = np.searchsorted(source_times, query_times)
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    nearest = np.minimum(np.abs(query_times - source_times[right]),
                         np.abs(query_times - source_times[left]))
    values[nearest > INTERPOLATION_MAX_GAP_SECONDS] = np.nan
    return values.astype(np.float32)


def _period_blocks(period25: np.ndarray) -> list[tuple[int, int]]:
    blocks = []
    for value in np.unique(period25):
        if value <= 0:
            continue
        edges = np.diff(np.r_[False, period25 == value, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        blocks.extend((int(start), int(stop)) for start, stop in zip(starts, stops))
    blocks.sort()
    return blocks


def _sample_indices(period25: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_parts = []
    blocks = []
    blocks25 = []
    offset = 0
    for start, stop in _period_blocks(period25):
        indices = np.arange(start, stop, STRIDE, dtype=np.int64)
        if not len(indices):
            continue
        sampled_parts.append(indices)
        blocks.append((offset, offset + len(indices)))
        blocks25.append((start, stop))
        offset += len(indices)
    if not sampled_parts:
        raise ValueError("SkillCorner match contains no positive-period samples")
    return (
        np.concatenate(sampled_parts),
        np.asarray(blocks, dtype=np.int64).reshape(-1, 2),
        np.asarray(blocks25, dtype=np.int64).reshape(-1, 2),
    )


def _nearest_distances(players: np.ndarray, ball: np.ndarray, teams: np.ndarray) -> np.ndarray:
    distances = np.full((len(ball), 2), np.nan, dtype=np.float32)
    for start in range(0, len(ball), DISTANCE_CHUNK):
        stop = min(start + DISTANCE_CHUNK, len(ball))
        player_chunk = players[start:stop]
        ball_chunk = ball[start:stop]
        ball_visible = np.isfinite(ball_chunk).all(axis=1)
        for team in (0, 1):
            slots = np.flatnonzero(teams == team)
            if not len(slots):
                raise ValueError("SkillCorner team has no rostered tracking players")
            xy = player_chunk[:, slots]
            valid = np.isfinite(xy).all(axis=2) & ball_visible[:, None]
            dx = (xy[:, :, 0] - ball_chunk[:, None, 0]) * 105.3
            dy = (xy[:, :, 1] - ball_chunk[:, None, 1]) * 68.0
            np.hypot(dx, dy, out=dx)
            dx[~valid] = np.inf
            nearest = dx.min(axis=1)
            nearest[~np.isfinite(nearest)] = np.nan
            distances[start:stop, team] = nearest
    return distances


def _load_feature_builder():
    module_name = f"_wave4_wave1_data_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, WAVE1_FEATURES)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load Wave 1 feature builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, module_name


def _save_pack(path: Path, data: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    sha256, size = _sha256_file(path)
    return {"path": path.relative_to(WAVE4).as_posix(), "sha256": sha256, "size_bytes": size}


def _label_counts(labels: np.ndarray) -> dict:
    return {
        "home": int(np.count_nonzero(labels == 0)),
        "away": int(np.count_nonzero(labels == 1)),
        "dead": int(np.count_nonzero(labels == 2)),
        "unknown": int(np.count_nonzero(labels == -1)),
    }


def _prepare_match(match_id: int, index: int, raw_template: str, media_template: str,
                   feature_data) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW_DIR / f"{match_id}_match.json"
    phases_path = RAW_DIR / f"{match_id}_phases_of_play.csv"
    tracking_path = RAW_DIR / f"{match_id}_tracking_extrapolated.jsonl"
    metadata_url = raw_template.format(m=match_id, f="match.json")
    phases_url = raw_template.format(m=match_id, f="phases_of_play.csv")
    tracking_raw_url = raw_template.format(m=match_id, f="tracking_extrapolated.jsonl")
    tracking_media_url = media_template.format(m=match_id, f="tracking_extrapolated.jsonl")
    inputs = {
        "metadata": _ensure_regular_input(metadata_path, metadata_url),
        "phases": _ensure_regular_input(phases_path, phases_url),
        "tracking": _ensure_tracking_input(tracking_path, tracking_raw_url, tracking_media_url),
    }
    metadata, team_of, length, width = _read_metadata(metadata_path)
    home_id = metadata["home_team"]["id"]
    away_id = metadata["away_team"]["id"]
    player_ids, tracking_sha256, tracking_size, frame_count10 = _scan_tracking(tracking_path, team_of)
    if tracking_sha256 != inputs["tracking"]["sha256"] or tracking_size != inputs["tracking"]["size_bytes"]:
        raise ValueError(f"SkillCorner tracking audit mismatch for {match_id}")
    players10, ball10, period10, teams = _parse_tracking(
        tracking_path, team_of, player_ids, tracking_sha256, tracking_size, frame_count10
    )
    phase_labels10 = _read_phases(phases_path, frame_count10, home_id, away_id)
    labels10 = np.full(frame_count10, -1, dtype=np.int8)
    active10 = period10 > 0
    labels10[active10] = 2
    phase_active = active10 & (phase_labels10 >= 0)
    labels10[phase_active] = phase_labels10[phase_active]

    frame_count25 = int((frame_count10 - 1) * 2.5) + 1
    containing10 = np.minimum((np.arange(frame_count25) * 0.4).astype(int), frame_count10 - 1)
    period25 = period10[containing10]
    labels25 = labels10[containing10].astype(np.int8, copy=True)
    labels25[period25 == 0] = -1
    sampled, blocks, blocks25 = _sample_indices(period25)

    players = np.full((len(sampled), len(player_ids), 2), np.nan, dtype=np.float32)
    for column in range(len(player_ids)):
        for axis in range(2):
            players[:, column, axis] = _interpolate_samples(players10[:, column, axis], sampled)
    ball = np.column_stack([
        _interpolate_samples(ball10[:, axis], sampled) for axis in range(2)
    ]).astype(np.float32, copy=False)
    players[..., 0] = (players[..., 0] + length / 2.0) / length
    players[..., 1] = (players[..., 1] + width / 2.0) / width
    ball[:, 0] = (ball[:, 0] + length / 2.0) / length
    ball[:, 1] = (ball[:, 1] + width / 2.0) / width

    distances = _nearest_distances(players, ball, teams)
    X = feature_data.features(players, ball, distances, blocks.tolist())
    if X.shape != (len(sampled), 16) or not np.isfinite(X).all():
        raise ValueError(f"Wave 1 features are invalid for SkillCorner match {match_id}")
    y = labels25[sampled]
    output_path = DATA_DIR / f"skillcorner_{match_id}_clean.npz"
    pack = _save_pack(output_path, {
        "X": X.astype(np.float32, copy=False),
        "players": players.astype(np.float32, copy=False),
        "teams": teams,
        "ball": ball.astype(np.float32, copy=False),
        "blocks": blocks,
        "sampled": sampled,
        "y": y.astype(np.int8, copy=False),
        "y25": labels25,
        "blocks25": blocks25,
        "n_frames": np.asarray(frame_count25, dtype=np.int64),
        "fps": np.asarray(FPS, dtype=np.int64),
    })
    sample_counts = _label_counts(y)
    result = {
        "match_id": match_id,
        "partition": "development" if index < 14 else "locked_final",
        "inputs": inputs,
        "prepared": pack,
        "frame_count_10fps": int(frame_count10),
        "frame_count_25fps": int(frame_count25),
        "sample_count_5fps": int(len(sampled)),
        "period_block_count": int(len(blocks)),
        "player_slot_count": int(len(teams)),
        "team_slot_counts": {
            "home": int(np.count_nonzero(teams == 0)),
            "away": int(np.count_nonzero(teams == 1)),
        },
        "pitch_length_m": length,
        "pitch_width_m": width,
        "label_counts_25fps": _label_counts(labels25),
        "sample_label_counts": sample_counts,
    }
    del players10, ball10, period10, phase_labels10, labels10, period25, labels25
    del players, ball, distances, X, y, teams, sampled, blocks, blocks25
    gc.collect()
    return result


def prepare() -> dict:
    match_ids, raw_template, media_template = _load_reference()
    feature_data, feature_module_name = _load_feature_builder()
    try:
        matches = {}
        for index, match_id in enumerate(match_ids):
            matches[str(match_id)] = _prepare_match(
                match_id, index, raw_template, media_template, feature_data
            )
        audit = {
            "schema_version": 1,
            "audit_path": AUDIT_PATH.relative_to(WAVE4).as_posix(),
            "source": {
                "name": "SkillCorner Open Data",
                "credit": "SkillCorner",
                "license": "MIT License",
                "repository": "https://github.com/SkillCorner/opendata",
            },
            "reference_parser": {
                "path": "reference/skillcorner_games.py",
                "sha256": _sha256_file(WAVE4 / "reference" / "skillcorner_games.py")[0],
            },
            "preparer": {
                "path": "prepare_skillcorner.py",
                "sha256": _sha256_file(Path(__file__))[0],
            },
            "sampling": {
                "native_fps": 10,
                "prepared_fps": FPS,
                "stride_25fps_frames": STRIDE,
                "interpolation_max_gap_seconds": INTERPOLATION_MAX_GAP_SECONDS,
                "label_alignment": "containing 10fps frame",
                "phase_intervals": "frame_start inclusive, frame_end exclusive",
            },
            "matches": matches,
        }
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = AUDIT_PATH.with_name(AUDIT_PATH.name + ".part")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(audit, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, AUDIT_PATH)
        finally:
            if temporary.exists():
                temporary.unlink()
        return audit
    finally:
        sys.modules.pop(feature_module_name, None)


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
