import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import seal


class SealTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "prediction_seal.json"
        self.manifest = hashlib.sha256(b"frozen protocol").hexdigest()
        self.rows = [
            self.artifact("prediction-b", b"probabilities and decoded codes b"),
            self.artifact("fit-metadata", b"training metadata"),
            self.artifact("prediction-a", b"probabilities and decoded codes a"),
        ]
        self.expected = [row["id"] for row in self.rows]

    def artifact(self, identifier, contents):
        path = self.root / (identifier + ".bin")
        path.write_bytes(contents)
        return {
            "id": identifier,
            "path": str(path),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }

    def create(self):
        return seal.create(self.path, self.manifest, self.rows, self.expected)

    def replace_payload(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_roundtrip_is_sorted_and_does_not_mutate_callers(self):
        original = copy.deepcopy(self.rows)
        expected = self.expected.copy()
        payload = self.create()
        self.assertEqual(self.rows, original)
        self.assertEqual(self.expected, expected)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["manifest_sha256"], self.manifest)
        self.assertEqual(payload["artifact_count"], 3)
        self.assertEqual([row["id"] for row in payload["artifacts"]], sorted(expected))
        self.assertEqual(seal.verify(self.path, self.manifest, expected), payload)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), payload)
        payload["artifacts"][0]["sha256"] = "0" * 64
        self.assertNotEqual(seal.verify(self.path, self.manifest, expected), payload)

    def test_complete_protocol_requires_all_300_artifacts(self):
        rows = []
        for fit in range(10):
            for kind in ("metadata", "checkpoint", "replay-proof"):
                identifier = f"fit-{fit}-{kind}"
                rows.append(self.artifact(identifier, identifier.encode()))
        for prediction in range(270):
            identifier = f"joint-prob-codes-{prediction}"
            rows.append(self.artifact(identifier, identifier.encode()))
        expected = [row["id"] for row in rows]
        self.assertEqual(len(expected), 300)
        missing_path = self.root / "incomplete.json"
        with self.assertRaises(ValueError):
            seal.create(missing_path, self.manifest, rows[:-1], expected)
        self.assertFalse(missing_path.exists())
        payload = seal.create(self.path, self.manifest, rows, expected)
        self.assertEqual(payload["artifact_count"], 300)
        self.assertEqual(seal.verify(self.path, self.manifest, expected), payload)

    def test_existing_seal_cannot_be_overwritten_even_identically(self):
        self.create()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(self.path.read_bytes(), before)
        self.path.write_bytes(b"previous incomplete or invalid seal")
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_later_artifact_validation_never_creates_seal(self):
        for failure in ("digest", "missing", "directory"):
            with self.subTest(failure=failure):
                rows = copy.deepcopy(self.rows)
                if failure == "digest":
                    rows[-1]["sha256"] = "0" * 64
                elif failure == "missing":
                    rows[-1]["path"] = str(self.root / "not-created.bin")
                else:
                    rows[-1]["path"] = str(self.root)
                with self.assertRaises((OSError, ValueError)):
                    seal.create(self.path, self.manifest, rows, self.expected)
                self.assertFalse(self.path.exists())

    def test_create_rejects_missing_added_and_duplicate_ids(self):
        extra = self.artifact("unregistered", b"not in protocol")
        cases = [
            self.rows[:-1],
            self.rows + [extra],
            self.rows + [self.rows[0]],
            [],
        ]
        for rows in cases:
            with self.subTest(ids=[row["id"] for row in rows]):
                with self.assertRaises(ValueError):
                    seal.create(self.path, self.manifest, rows, self.expected)
                self.assertFalse(self.path.exists())

    def test_expected_ids_must_be_nonempty_unique_strings(self):
        self.create()
        for expected in ([], ["x", "x"], [None], [1], [""], "prediction-a", None):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    seal.create(self.root / "invalid.json", self.manifest, self.rows, expected)
                with self.assertRaises(ValueError):
                    seal.verify(self.path, self.manifest, expected)
                self.assertFalse((self.root / "invalid.json").exists())

    def test_create_rejects_duplicate_resolved_and_hardlink_paths(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["path"] = str(self.root / ".." / self.root.name / Path(rows[0]["path"]).name)
        rows[-1]["sha256"] = rows[0]["sha256"]
        with self.assertRaises(ValueError):
            seal.create(self.path, self.manifest, rows, self.expected)
        self.assertFalse(self.path.exists())
        alias = self.root / "hardlink.bin"
        try:
            os.link(self.rows[0]["path"], alias)
        except OSError as error:
            self.skipTest(f"filesystem cannot create a hard link: {error}")
        rows[-1]["path"] = str(alias)
        with self.assertRaises(ValueError):
            seal.create(self.path, self.manifest, rows, self.expected)
        self.assertFalse(self.path.exists())

    def test_create_rejects_malformed_rows_and_digests(self):
        malformed = [None, {}, dict(self.rows[0], extra=True)]
        malformed += [dict(self.rows[0], id=value) for value in ("", None, 1)]
        malformed += [dict(self.rows[0], path=value) for value in (None, "relative.bin", 1)]
        malformed += [dict(self.rows[0], sha256=value) for value in (None, "f" * 63, "g" * 64)]
        for row in malformed:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    seal.create(self.path, self.manifest, [row, *self.rows[1:]], self.expected)
                self.assertFalse(self.path.exists())
        for manifest in (None, "short", "z" * 64):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                seal.create(self.path, manifest, self.rows, self.expected)

    def test_consumer_detects_one_byte_tampering_in_each_artifact(self):
        self.create()
        for row in self.rows:
            with self.subTest(artifact=row["id"]):
                path = Path(row["path"])
                before = path.read_bytes()
                path.write_bytes(bytes([before[0] ^ 1]) + before[1:])
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    seal.verify(self.path, self.manifest, self.expected)
                path.write_bytes(before)

    def test_consumer_rejects_deleted_artifact(self):
        self.create()
        Path(self.rows[-1]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            seal.verify(self.path, self.manifest, self.expected)

    def test_consumer_rejects_wrong_protocol_and_expected_inventory(self):
        self.create()
        wrong_manifest = hashlib.sha256(b"changed protocol").hexdigest()
        for manifest, expected in (
            (wrong_manifest, self.expected),
            (self.manifest, self.expected[:-1]),
            (self.manifest, [*self.expected[:-1], "substituted-prediction"]),
            (self.manifest, [*self.expected, "additional-prediction"]),
        ):
            with self.subTest(manifest=manifest, expected=expected):
                with self.assertRaises(ValueError):
                    seal.verify(self.path, manifest, expected)

    def test_consumer_rejects_modified_seal_inventory(self):
        original = self.create()
        extra = self.artifact("unregistered", b"new prediction")
        cases = []
        missing = copy.deepcopy(original)
        missing["artifacts"].pop()
        cases.append(missing)
        added = copy.deepcopy(original)
        added["artifacts"].append(extra)
        cases.append(added)
        duplicate_id = copy.deepcopy(original)
        duplicate_id["artifacts"][-1] = copy.deepcopy(duplicate_id["artifacts"][0])
        cases.append(duplicate_id)
        duplicate_path = copy.deepcopy(original)
        duplicate_path["artifacts"][-1]["path"] = duplicate_path["artifacts"][0]["path"]
        duplicate_path["artifacts"][-1]["sha256"] = duplicate_path["artifacts"][0]["sha256"]
        cases.append(duplicate_path)
        bad_digest = copy.deepcopy(original)
        bad_digest["artifacts"][0]["sha256"] = "0" * 64
        cases.append(bad_digest)
        reversed_rows = copy.deepcopy(original)
        reversed_rows["artifacts"].reverse()
        cases.append(reversed_rows)
        for payload in cases:
            with self.subTest(payload=payload):
                self.replace_payload(payload)
                with self.assertRaises(ValueError):
                    seal.verify(self.path, self.manifest, self.expected)

    def test_consumer_rejects_schema_changes_and_duplicate_json_keys(self):
        original = self.create()
        changes = [
            {"schema_version": 2},
            {"schema_version": True},
            {"schema_version": 1.0},
            {"artifact_count": 2},
            {"artifact_count": 3.0},
            {"artifact_count": True},
            {"artifacts": {}},
            {"manifest_sha256": "not-a-hash"},
            {"unexpected": "field"},
        ]
        cases = [dict(original, **change) for change in changes]
        cases += [None, [], {key: value for key, value in original.items() if key != "artifact_count"}]
        for payload in cases:
            with self.subTest(payload=payload):
                self.replace_payload(payload)
                with self.assertRaises(ValueError):
                    seal.verify(self.path, self.manifest, self.expected)
        serialized = json.dumps(original)
        self.path.write_text('{"schema_version":1,' + serialized[1:], encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            seal.verify(self.path, self.manifest, self.expected)
        duplicate_row_key = serialized.replace('"id":', '"id":"duplicate", "id":', 1)
        self.path.write_text(duplicate_row_key, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            seal.verify(self.path, self.manifest, self.expected)
        self.path.write_text(serialized[:len(serialized) // 2], encoding="utf-8")
        with self.assertRaises(ValueError):
            seal.verify(self.path, self.manifest, self.expected)


if __name__ == "__main__":
    unittest.main()
