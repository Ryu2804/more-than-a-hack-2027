"""Read isolation for a trusted Python pipeline, not an OS security sandbox.

Only Python ``open`` audit events are intercepted. Existing descriptors, native
code, subprocesses, and filesystem races are outside this guard's guarantees.
"""

import os
from pathlib import Path
import sys
import threading


class ReadGuard:
    def __init__(self, denied_roots=(), denied_files=()):
        self._roots = set()
        self._files = set()
        self._blocked = []
        self._installed = False
        self._local = threading.local()
        self.add(roots=denied_roots, files=denied_files)

    @staticmethod
    def _normalize(path):
        normalized = os.path.normcase(str(Path(os.fsdecode(os.fspath(path))).resolve()))
        if os.name == "nt":
            if normalized.startswith("\\\\?\\unc\\"):
                normalized = "\\\\" + normalized[8:]
            elif normalized.startswith("\\\\?\\"):
                normalized = normalized[4:]
        return normalized

    @staticmethod
    def _reads(mode):
        if isinstance(mode, int):
            access_mask = getattr(os, "O_ACCMODE", os.O_WRONLY | os.O_RDWR)
            return mode & access_mask != os.O_WRONLY
        if isinstance(mode, str):
            return "r" in mode or "+" in mode or not any(flag in mode for flag in "wax")
        return True

    def add(self, roots=(), files=()):
        self._roots.update(self._normalize(path) for path in roots)
        self._files.update(self._normalize(path) for path in files)

    def _denied(self, path):
        return path in self._files or any(
            path == root or path.startswith(root.rstrip(os.sep) + os.sep)
            for root in self._roots
        )

    def check(self, path, mode="r"):
        if isinstance(path, int) or not self._reads(mode):
            return
        normalized = self._normalize(path)
        if self._denied(normalized):
            self._blocked.append({
                "path": normalized,
                "probe": normalized == getattr(self._local, "probe_path", None),
            })
            raise PermissionError(f"ReadGuard denied read: {normalized}")

    def _audit(self, event, args):
        if event == "open":
            path, mode, flags = args
            self.check(path, flags if isinstance(flags, int) else mode)

    def install(self):
        if not self._installed:
            sys.addaudithook(self._audit)
            self._installed = True
        return self

    def probe(self, path):
        """Attempt a forbidden open without reading bytes; require our hook to deny it."""
        if not self._installed:
            raise RuntimeError("Install ReadGuard before probing")
        normalized = self._normalize(path)
        if not self._denied(normalized):
            raise ValueError("A probe must target a denied path")
        previous = getattr(self._local, "probe_path", None)
        self._local.probe_path = normalized
        before = len(self._blocked)
        try:
            try:
                with open(path, "rb"):
                    pass
            except PermissionError:
                records = self._blocked[before:]
                for record in records:
                    if record["path"] == normalized and record["probe"]:
                        return dict(record)
                raise RuntimeError("Probe was not denied by this ReadGuard") from None
            raise RuntimeError("ReadGuard failed to intercept the forbidden probe")
        finally:
            self._local.probe_path = previous

    def report(self):
        blocked = [dict(record) for record in self._blocked]
        probes = [dict(record) for record in blocked if record["probe"]]
        return {
            "installed": self._installed,
            "denied_roots": sorted(self._roots),
            "denied_files": sorted(self._files),
            "blocked_attempts": len(blocked),
            "nonprobe_blocked_attempts": len(blocked) - len(probes),
            "probe_count": len(probes),
            "blocked": blocked,
            "probes": probes,
            "successful_forbidden_reads": 0,
            "limitation": (
                "Zero successful forbidden reads describes intercepted open events "
                "by construction, not independent security verification. This trusted "
                "Python guard is not an OS sandbox and cannot establish no overfit."
            ),
        }
