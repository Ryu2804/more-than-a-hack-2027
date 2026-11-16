from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests


WAVE3 = Path(__file__).resolve().parent
IDSSE_ROOT = (WAVE3 / "cache" / "idsse_parser").resolve()
IDSSE_DATA = IDSSE_ROOT / "external" / "idsse"
DATA_DIR = WAVE3 / "cache" / "data"
AUDIT_PATH = WAVE3 / "cache" / "idsse_audit.json"
INVENTORY_URL = "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main"
DATASET_URL = "https://huggingface.co/datasets/pysport/idsse-data"
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
REGIMES = ("r0", "r1", "r2")
FPS = 5
STRIDE = 5
CHUNK_SIZE = 8 * 1024 * 1024


def _load_unique_module(path: Path, role: str):
    name = f"_farras_wave3_{role}_{uuid.uuid4().hex}"
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


def _source_entries(inventory: list[dict]) -> list[dict]:
    selected = []
    for match in MATCH_IDS:
        for label, marker in (("metadata", "matchinformation"), ("tracking", "positions_raw")):
            candidates = [
                entry
                for entry in inventory
                if entry.get("type") == "file"
                and marker in str(entry.get("path", ""))
                and str(entry.get("path", "")).endswith(f"{match}.xml")
            ]
            if len(candidates) != 1:
                raise ValueError(f"Expected one {label} source for {match}; found {len(candidates)}")
            entry = candidates[0]
            source_path = str(entry["path"])
            pure_path = PurePosixPath(source_path)
            if pure_path.is_absolute() or len(pure_path.parts) != 1 or ".." in pure_path.parts:
                raise ValueError(f"Unexpected IDSSE inventory path: {source_path}")
            size = int(entry["size"])
            if size <= 0:
                raise ValueError(f"Invalid IDSSE source size for {source_path}")
            lfs = entry.get("lfs")
            expected_sha256 = None
            if lfs is not None:
                expected_sha256 = str(lfs.get("oid", "")).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                    raise ValueError(f"Invalid LFS SHA-256 for {source_path}")
                if int(lfs.get("size", -1)) != size:
                    raise ValueError(f"Inventory size mismatch for {source_path}")
            expected_git_oid = None
            if lfs is None:
                expected_git_oid = str(entry.get("oid", "")).lower()
                if not re.fullmatch(r"[0-9a-f]{40}", expected_git_oid):
                    raise ValueError(f"Invalid inventory object ID for {source_path}")
            source_url = (
                "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
                + quote(source_path, safe="/")
            )
            selected.append(
                {
                    "match": match,
                    "kind": label,
                    "source_path": source_path,
                    "local_name": pure_path.name,
                    "source_url": source_url,
                    "inventory_oid": str(entry.get("oid", "")),
                    "expected_git_oid": expected_git_oid,
                    "expected_sha256": expected_sha256,
                    "size_bytes": size,
                }
            )
    if len({entry["local_name"] for entry in selected}) != len(selected):
        raise ValueError("IDSSE inventory contains colliding source filenames")
    return selected


def _measure_file(path: Path) -> dict:
    expected_size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            size += len(block)
            sha256.update(block)
            git_blob.update(block)
    return {"size_bytes": size, "sha256": sha256.hexdigest(), "git_blob_oid": git_blob.hexdigest()}


def _validate_file(record: dict, observed: dict) -> None:
    if observed["size_bytes"] != record["size_bytes"]:
        raise ValueError(f"IDSSE size verification failed for {record['source_path']}")
    expected_sha256 = record["expected_sha256"]
    if expected_sha256 is not None and observed["sha256"] != expected_sha256:
        raise ValueError(f"IDSSE LFS SHA-256 verification failed for {record['source_path']}")
    expected_git_oid = record["expected_git_oid"]
    if expected_git_oid is not None and observed["git_blob_oid"] != expected_git_oid:
        raise ValueError(f"IDSSE inventory object verification failed for {record['source_path']}")


def _download_file(session: requests.Session, record: dict, destination: Path) -> dict:
    part = destination.with_name(destination.name + ".part")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {record['size_bytes']}\0".encode("ascii"))
    size = 0
    try:
        with session.get(
            record["source_url"],
            stream=True,
            timeout=(30, 120),
            headers={"Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            with part.open("wb") as output:
                for block in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not block:
                        continue
                    size += len(block)
                    if size > record["size_bytes"]:
                        raise ValueError(f"IDSSE download exceeds inventory size for {record['source_path']}")
                    output.write(block)
                    sha256.update(block)
                    git_blob.update(block)
        observed = {
            "size_bytes": size,
            "sha256": sha256.hexdigest(),
            "git_blob_oid": git_blob.hexdigest(),
        }
        _validate_file(record, observed)
        os.replace(part, destination)
        return observed
    finally:
        if part.exists():
            part.unlink()


def _manifest_record(record: dict, observed: dict) -> dict:
    return {
        "match": record["match"],
        "kind": record["kind"],
        "source_path": record["source_path"],
        "source_url": record["source_url"],
        "inventory_oid": record["inventory_oid"],
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with part.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


def download_sources() -> dict:
    IDSSE_DATA.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        response = session.get(INVENTORY_URL, timeout=(30, 60))
        response.raise_for_status()
        inventory = response.json()
        if not isinstance(inventory, list):
            raise ValueError("Unexpected Hugging Face IDSSE inventory response")
        records = _source_entries(inventory)
        files = []
        for record in records:
            destination = IDSSE_DATA / record["local_name"]
            observed = None
            if destination.is_file():
                cached = _measure_file(destination)
                try:
                    _validate_file(record, cached)
                    observed = cached
                except ValueError:
                    observed = None
            if observed is None:
                observed = _download_file(session, record, destination)
            files.append(_manifest_record(record, observed))
    manifest = {
        "dataset_url": DATASET_URL,
        "inventory_url": INVENTORY_URL,
        "files": files,
    }
    _atomic_json(IDSSE_DATA / "manifest.json", manifest)
    return manifest


def _period_blocks(half, np) -> list[tuple[int, int]]:
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


def _sample_indices(half, np) -> tuple[object, object, object]:
    sampled_parts = []
    blocks = []
    blocks25 = []
    offset = 0
    for start, stop in _period_blocks(half, np):
        indices = np.arange(start, stop, STRIDE, dtype=np.int64)
        if len(indices) == 0:
            continue
        sampled_parts.append(indices)
        blocks.append((offset, offset + len(indices)))
        blocks25.append((start, stop))
        offset += len(indices)
    if not sampled_parts:
        raise ValueError("IDSSE match contains no half samples")
    return (
        np.concatenate(sampled_parts),
        np.asarray(blocks, dtype=np.int64).reshape(-1, 2),
        np.asarray(blocks25, dtype=np.int64).reshape(-1, 2),
    )


def _nearest_distances(players, ball, teams, pitch, np):
    distances = np.full((len(ball), 2), np.nan, dtype=np.float32)
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        if len(slots) == 0:
            continue
        for start in range(0, len(ball), 8192):
            stop = min(len(ball), start + 8192)
            player_xy = players[start:stop, slots]
            ball_xy = ball[start:stop]
            valid = np.isfinite(player_xy).all(axis=2) & np.isfinite(ball_xy).all(axis=1)[:, None]
            dx = (player_xy[:, :, 0] - ball_xy[:, None, 0]) * pitch[0]
            dy = (player_xy[:, :, 1] - ball_xy[:, None, 1]) * pitch[1]
            np.hypot(dx, dy, out=dx)
            dx[~valid] = np.inf
            nearest = dx.min(axis=1)
            nearest[~np.isfinite(nearest)] = np.nan
            distances[start:stop, team] = nearest
    return distances


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_counts(labels, np) -> dict:
    return {
        "home": int(np.count_nonzero(labels == 0)),
        "away": int(np.count_nonzero(labels == 1)),
        "dead": int(np.count_nonzero(labels == 2)),
        "unknown": int(np.count_nonzero(labels == -1)),
    }


def _save_regime(name: str, round_index: int, clean, reference, feature_data, np) -> dict:
    seed = 20261108 + 1000 * round_index + MATCH_IDS.index(name)
    degraded = reference.degrade(clean, seed=seed)
    sampled, blocks, blocks25 = _sample_indices(degraded.half, np)
    pitch = np.asarray(degraded.pitch, dtype=np.float32)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError(f"Invalid pitch dimensions for {name}")
    players = degraded.players[sampled, :, :2]
    players /= pitch
    ball = degraded.ball[sampled]
    ball /= pitch
    teams = np.asarray(degraded.teams, dtype=np.int8)
    if np.any((teams != 0) & (teams != 1)):
        raise ValueError(f"IDSSE match {name} contains unknown team slots")
    if players.shape[1] != len(teams):
        raise ValueError(f"IDSSE player/team slot mismatch for {name}")
    distances = _nearest_distances(players, ball, teams, pitch, np)
    X = feature_data.features(players, ball, distances, blocks.tolist())
    if degraded.labels is None:
        raise ValueError(f"IDSSE labels unavailable for {name}")
    y25 = np.asarray(degraded.labels, dtype=np.int8).copy()
    y25[y25 == 3] = -1
    if np.any((y25 < -1) | (y25 > 2)):
        raise ValueError(f"Unexpected IDSSE label code for {name}")
    y = y25[sampled]
    output_path = DATA_DIR / f"{name}_r{round_index}.npz"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = output_path.with_name(output_path.name + ".part")
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
        os.replace(part, output_path)
    finally:
        if part.exists():
            part.unlink()
    visible_ball = int(np.isfinite(ball).all(axis=1).sum())
    visible_player_slots = int(np.isfinite(players).all(axis=2).sum())
    total_player_slots = int(players.shape[0] * players.shape[1])
    result = {
        "seed": seed,
        "path": output_path.relative_to(WAVE3).as_posix(),
        "sha256": _sha256_file(output_path),
        "sample_count": int(len(sampled)),
        "block_count": int(len(blocks)),
        "ball_visible_samples": visible_ball,
        "ball_visible_fraction": float(visible_ball / len(ball)) if len(ball) else 0.0,
        "visible_player_slots": visible_player_slots,
        "total_player_slots": total_player_slots,
        "player_visible_fraction": float(visible_player_slots / total_player_slots) if total_player_slots else 0.0,
    }
    return result


def _audit_summary(audit: dict) -> dict:
    matches = audit["matches"]
    labels = {key: 0 for key in ("home", "away", "dead", "unknown")}
    total_frames = 0
    total_samples = 0
    for match in matches.values():
        total_frames += match["n_frames"]
        for key, value in match["label_counts"].items():
            labels[key] += value
        total_samples += sum(regime["sample_count"] for regime in match["regimes"].values())
    return {
        "completed_matches": len(matches),
        "prepared_files": sum(len(match["regimes"]) for match in matches.values()),
        "total_frames": total_frames,
        "total_sample_rows": total_samples,
        "label_counts": labels,
    }


def prepare(download_only: bool = False) -> dict:
    manifest = download_sources()
    if download_only:
        return manifest
    import numpy as np

    reference_path = (WAVE3 / "reference" / "jeremy_data.py").resolve(strict=True)
    features_path = (WAVE3.parent / "wave1" / "data.py").resolve(strict=True)
    reference = _load_unique_module(reference_path, "jeremy_data")
    feature_data = _load_unique_module(features_path, "wave1_data")
    reference.ROOT = IDSSE_ROOT
    audit = {
        "dataset_url": DATASET_URL,
        "inventory_url": INVENTORY_URL,
        "source_manifest": (IDSSE_DATA / "manifest.json").relative_to(WAVE3).as_posix(),
        "source_hashes": {entry["source_path"]: entry["sha256"] for entry in manifest["files"]},
        "matches": {},
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(AUDIT_PATH, {**audit, "aggregate": _audit_summary(audit)})
    for name in MATCH_IDS:
        clean = reference.load_idsse(name)
        labels = np.asarray(clean.labels, dtype=np.int8).copy()
        labels[labels == 3] = -1
        match_audit = {
            "n_frames": int(len(clean.half)),
            "block_count": int(len(_period_blocks(clean.half, np))),
            "label_counts": _label_counts(labels, np),
            "regimes": {},
        }
        del labels
        for round_index, regime in enumerate(REGIMES):
            match_audit["regimes"][regime] = _save_regime(
                name, round_index, clean, reference, feature_data, np
            )
        del clean
        gc.collect()
        audit["matches"][name] = match_audit
        _atomic_json(AUDIT_PATH, {**audit, "aggregate": _audit_summary(audit)})
    return {**audit, "aggregate": _audit_summary(audit)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare IDSSE matches for wave3.")
    parser.add_argument("--download-only", action="store_true", help="Download and verify IDSSE inputs without preparing packs.")
    args = parser.parse_args()
    prepare(download_only=args.download_only)


if __name__ == "__main__":
    main()
