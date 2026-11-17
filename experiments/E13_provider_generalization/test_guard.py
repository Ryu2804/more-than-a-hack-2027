import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import guard


_CHILD_SETUP = """
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
from guard import ReadGuard


def denied_open(operation, current_guard):
    before = current_guard.report()["blocked_attempts"]
    try:
        result = operation()
    except PermissionError:
        assert current_guard.report()["blocked_attempts"] == before + 1
    else:
        if hasattr(result, "close"):
            result.close()
        elif isinstance(result, int):
            os.close(result)
        raise AssertionError("Forbidden open succeeded")

"""


class ReadGuardChecks(unittest.TestCase):
    def test_root_boundaries_relative_paths_bytes_and_extensions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            base = Path(directory)
            root = base / "labels"
            current = guard.ReadGuard(denied_roots=[root])
            for path in (root, root / "match" / "gt.bin", os.fsencode(root / "gt.bin")):
                with self.subTest(path=path), self.assertRaises(PermissionError):
                    current.check(path)
            current.check(base / "labels-copy" / "gt.bin")
            current.check(base / "allowed.bin")
            relative = os.path.relpath(root / "gt.bin")
            with self.assertRaises(PermissionError):
                current.check(relative)
            current.add(files=[base / "old-result.json"], roots=[base / "old-output"])
            for path in (base / "old-result.json", base / "old-output" / "result.bin"):
                with self.assertRaises(PermissionError):
                    current.check(path)
            current.check(base / "old-result.json.bak")

    def test_modes_flags_and_file_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "denied.bin"
            current = guard.ReadGuard(denied_files=[target])
            for mode in ("r", "rb", "r+", "w+", "a+b", "x+", None,
                         os.O_RDONLY, os.O_RDWR, os.O_RDWR | os.O_CREAT):
                with self.subTest(mode=mode), self.assertRaises(PermissionError):
                    current.check(target, mode)
            for mode in ("w", "wb", "a", "ab", "x", "xb", os.O_WRONLY,
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND):
                with self.subTest(mode=mode):
                    current.check(target, mode)
            for descriptor in (0, 1, 17):
                current.check(descriptor, "r+")
            current.check(target.parent / "another.bin")

    @unittest.skipUnless(os.name == "nt", "Windows path semantics")
    def test_windows_case_separators_and_extended_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "MixedCase"
            root.mkdir()
            target = root / "Labels.bin"
            target.write_bytes(b"synthetic")
            current = guard.ReadGuard(denied_roots=[root])
            variants = (
                str(target).upper(),
                str(target).replace("\\", "/"),
                str(root / "unused" / ".." / "Labels.bin"),
                "\\\\?\\" + str(target.resolve()),
            )
            for path in variants:
                with self.subTest(path=path), self.assertRaises(PermissionError):
                    current.check(path)
            self.assertEqual(len({row["path"] for row in current.report()["blocked"]}), 1)

    def test_probe_requires_installed_guard_and_denied_target(self):
        current = guard.ReadGuard(denied_files=["synthetic-denied.bin"])
        with self.assertRaises(RuntimeError):
            current.probe("synthetic-denied.bin")
        self.assertEqual(current.report()["blocked_attempts"], 0)

    def test_report_is_detached_and_states_scope(self):
        current = guard.ReadGuard(denied_files=["synthetic-denied.bin"])
        with self.assertRaises(PermissionError):
            current.check("synthetic-denied.bin")
        report = current.report()
        self.assertEqual(report["successful_forbidden_reads"], 0)
        self.assertEqual(report["nonprobe_blocked_attempts"], 1)
        self.assertIn("not independent security verification", report["limitation"])
        self.assertIn("not an OS sandbox", report["limitation"])
        report["blocked"][0]["probe"] = True
        report["denied_files"].clear()
        self.assertEqual(current.report()["nonprobe_blocked_attempts"], 1)
        self.assertEqual(len(current.report()["denied_files"]), 1)


class IsolatedReadGuard(unittest.TestCase):
    def child(self, body):
        setup = """
base = Path(directory)
blocked = base / "blocked"
blocked.mkdir()
target = blocked / "secret.bin"
target.write_bytes(b"synthetic secret")
allowed = base / "allowed.bin"
allowed.write_bytes(b"synthetic allowed")
"""
        script = (_CHILD_SETUP + "with tempfile.TemporaryDirectory() as directory:\n"
                  + textwrap.indent(textwrap.dedent(setup + body), "    "))
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(Path(guard.__file__).resolve().parent)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_actual_builtin_io_path_and_os_opens_are_denied(self):
        report = self.child("""
current = ReadGuard(denied_roots=[blocked]).install()
current.install()
operations = [
    lambda: open(target, "rb"),
    lambda: io.open(target, "r"),
    lambda: target.open("rb"),
    lambda: os.open(target, os.O_RDONLY),
    lambda: os.open(target, os.O_RDWR),
    lambda: open(target, "r+"),
    lambda: open(target, "w+"),
    lambda: open(target, "a+"),
    lambda: open(blocked / "new.bin", "x+"),
    lambda: open(os.fsencode(target), "rb"),
]
for operation in operations:
    denied_open(operation, current)
assert target.stat().st_size == len(b"synthetic secret")
assert allowed.read_bytes() == b"synthetic allowed"
print(json.dumps(current.report()))
""")
        self.assertEqual(report["blocked_attempts"], 10)
        self.assertEqual(report["nonprobe_blocked_attempts"], 10)
        self.assertEqual(report["probe_count"], 0)

    def test_actual_write_only_opens_and_later_additions(self):
        report = self.child("""
current = ReadGuard(denied_roots=[blocked]).install()
with open(target, "wb") as handle:
    handle.write(b"new")
with io.open(target, "ab") as handle:
    handle.write(b" appended")
created = blocked / "created.bin"
with created.open("xb") as handle:
    handle.write(b"created")
fd = os.open(target, os.O_WRONLY | os.O_APPEND)
try:
    os.write(fd, b"!")
finally:
    os.close(fd)
assert target.stat().st_size == len(b"new appended!")
assert created.stat().st_size == len(b"created")
assert current.report()["blocked_attempts"] == 0
current.add(files=[allowed])
denied_open(lambda: allowed.open("rb"), current)
print(json.dumps(current.report()))
""")
        self.assertEqual(report["blocked_attempts"], 1)

    def test_exact_file_deny_keeps_siblings_and_source_imports_readable(self):
        report = self.child("""
module = base / "readguard_fixture.py"
module.write_text("VALUE = 37\\n", encoding="utf-8")
sys.path.insert(0, str(base))
current = ReadGuard(denied_files=[target]).install()
denied_open(lambda: open(target, "rb"), current)
sibling = blocked / "sibling.bin"
sibling.write_bytes(b"sibling")
assert sibling.read_bytes() == b"sibling"
assert importlib.import_module("readguard_fixture").VALUE == 37
assert "torch" not in sys.modules
assert "bridge" not in sys.modules
print(json.dumps(current.report()))
""")
        self.assertEqual(report["blocked_attempts"], 1)

    def test_probes_open_real_and_missing_paths_and_are_accounted_separately(self):
        report = self.child("""
current = ReadGuard(denied_roots=[blocked]).install()
first = current.probe(target)
second = current.probe(blocked / "does-not-exist.bin")
assert first["probe"] is True and second["probe"] is True
assert current.report()["blocked_attempts"] == 2
try:
    current.probe(allowed)
except ValueError:
    pass
else:
    raise AssertionError("Allowed probe was accepted")
denied_open(lambda: open(target, "rb"), current)
report = current.report()
assert [row["probe"] for row in report["blocked"]] == [True, True, False]
print(json.dumps(report))
""")
        self.assertEqual(report["probe_count"], 2)
        self.assertEqual(report["nonprobe_blocked_attempts"], 1)
        self.assertEqual(report["successful_forbidden_reads"], 0)

    def test_probe_marker_is_thread_local(self):
        report = self.child("""
import threading
current = ReadGuard(denied_roots=[blocked]).install()
original_open = open
other_thread = []
def interleaved_open(path, mode):
    thread = threading.Thread(target=lambda: other_thread.append(
        denied_open(lambda: original_open(target, "rb"), current)))
    thread.start()
    thread.join()
    return original_open(path, mode)
import builtins
builtins.open = interleaved_open
try:
    current.probe(target)
finally:
    builtins.open = original_open
assert len(other_thread) == 1
print(json.dumps(current.report()))
""")
        self.assertEqual(report["probe_count"], 1)
        self.assertEqual(report["nonprobe_blocked_attempts"], 1)

    def test_symlink_aliases_resolve_for_configuration_and_access(self):
        report = self.child("""
file_alias = base / "file-link.bin"
dir_alias = base / "directory-link"
try:
    file_alias.symlink_to(target)
    dir_alias.symlink_to(blocked, target_is_directory=True)
except (OSError, NotImplementedError) as error:
    print(json.dumps({"symlinks": False, "reason": str(error)}))
else:
    current = ReadGuard(denied_roots=[dir_alias]).install()
    denied_open(lambda: file_alias.open("rb"), current)
    denied_open(lambda: (dir_alias / "secret.bin").open("rb"), current)
    denied_open(lambda: target.open("rb"), current)
    assert len({row["path"] for row in current.report()["blocked"]}) == 1
    extra = base / "extra.bin"
    extra.write_bytes(b"extra")
    extra_alias = base / "extra-link.bin"
    extra_alias.symlink_to(extra)
    current.add(files=[extra_alias])
    denied_open(lambda: extra.open("rb"), current)
    print(json.dumps(current.report()))
""")
        if report.get("symlinks") is False:
            self.skipTest(report["reason"])
        self.assertEqual(report["blocked_attempts"], 4)

    def test_numpy_load_obeys_the_open_hook_when_available(self):
        report = self.child("""
try:
    import numpy as np
except ModuleNotFoundError:
    print(json.dumps({"numpy": False}))
else:
    array_path = blocked / "array.npy"
    np.save(array_path, np.array([2, 3, 5]))
    current = ReadGuard(denied_files=[array_path]).install()
    denied_open(lambda: np.load(array_path, allow_pickle=False), current)
    print(json.dumps(current.report()))
""")
        if report.get("numpy") is False:
            self.skipTest("NumPy is not installed in the child interpreter")
        self.assertEqual(report["blocked_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
