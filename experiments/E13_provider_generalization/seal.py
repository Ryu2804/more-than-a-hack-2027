"""Write-once, manifest-bound artifact inventory; verification rehashes every file.

The manifest digest and expected IDs must come from the frozen protocol, not the
seal being checked. This is an integrity gate, not an OS security boundary.
"""

import hashlib
import json
import os
from pathlib import Path


_PAYLOAD_KEYS = {"schema_version", "manifest_sha256", "artifacts", "artifact_count"}
_ARTIFACT_KEYS = {"id", "path", "sha256"}


def _digest(value, field):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    return value.lower()


def _expected_ids(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("expected_ids must be a nonempty collection of IDs")
    try:
        ids = list(values)
    except TypeError as error:
        raise ValueError("expected_ids must be a nonempty collection of IDs") from error
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("expected_ids must contain nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("expected_ids contains duplicate IDs")
    return set(ids)


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts(values, expected):
    if not isinstance(values, list):
        raise ValueError("artifacts must be a list")
    rows = []
    ids = set()
    paths = set()
    physical_files = set()
    for value in values:
        if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
            raise ValueError("artifact rows must contain only id, path and sha256")
        identifier = value["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("artifact IDs must be nonempty strings")
        if identifier in ids:
            raise ValueError(f"duplicate artifact ID: {identifier}")
        ids.add(identifier)
        raw_path = value["path"]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError(f"artifact path must be an absolute string: {identifier}")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"artifact is not a regular file: {identifier}")
        path_key = os.path.normcase(str(path))
        stat = path.stat()
        physical_key = (stat.st_dev, stat.st_ino)
        if path_key in paths or (stat.st_ino and physical_key in physical_files):
            raise ValueError(f"duplicate physical artifact path: {identifier}")
        paths.add(path_key)
        if stat.st_ino:
            physical_files.add(physical_key)
        digest = _digest(value["sha256"], f"artifact {identifier} sha256")
        if _hash_file(path) != digest:
            raise ValueError(f"artifact SHA-256 mismatch: {identifier}")
        rows.append({"id": identifier, "path": str(path), "sha256": digest})
    if ids != expected:
        missing = sorted(expected - ids)
        added = sorted(ids - expected)
        raise ValueError(f"artifact coverage mismatch: missing={missing}, added={added}")
    return sorted(rows, key=lambda row: row["id"])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def create(path, manifest_sha256, artifacts, expected_ids):
    """Validate all artifacts before exclusively creating a new seal.

    Failed writes remove only the file created by this call. Existing seals are
    never opened for writing, including when their contents would be identical.
    """
    manifest = _digest(manifest_sha256, "manifest_sha256")
    expected = _expected_ids(expected_ids)
    rows = _artifacts(artifacts, expected)
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest,
        "artifacts": rows,
        "artifact_count": len(rows),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    target = Path(path)
    created = False
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            target.unlink()
        raise
    return payload


def verify(path, manifest_sha256, expected_ids):
    """Reject malformed, incomplete or changed artifacts before any scoring."""
    manifest = _digest(manifest_sha256, "manifest_sha256")
    expected = _expected_ids(expected_ids)
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("invalid seal schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported seal schema version")
    if payload["manifest_sha256"] != manifest:
        raise ValueError("seal manifest SHA-256 mismatch")
    count = payload["artifact_count"]
    if type(count) is not int or count != len(expected):
        raise ValueError("seal artifact_count mismatch")
    rows = _artifacts(payload["artifacts"], expected)
    if payload["artifacts"] != rows:
        raise ValueError("seal artifacts are not canonical")
    return payload
