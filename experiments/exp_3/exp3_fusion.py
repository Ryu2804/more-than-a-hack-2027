"""Broadcast ball fusion for BAR-ATH: pinned exp_0 acceptance rule + a replay guard.

Label-free. ``fusion_candidate`` (verbatim exp_0 snapshot, hash-checked) accepts the
broadcast ball per half only on geometric agreement with the tactical ball. The replay guard
then keeps a recovered frame only if the broadcast ball is physically reachable from the
nearest tactical ball observations on both sides (``REPLAY_SPEED`` x elapsed time).
Hygiene: returns aggregate counts only; arrays stay in the calling process.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json

import numpy as np

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
from exp1_common import FPS, TARGET_REPO
from exp3_common import FUSION_PIN, FUSION_SNAPSHOT

REPLAY_SPEED = 40.0  # m/s


def verify_pin() -> dict:
    """Snapshot file and its verbatim function text must match the pin record."""
    pin = json.loads(FUSION_PIN.read_text())
    text = FUSION_SNAPSHOT.read_text()
    function = text[text.index("def fusion_candidate("):]
    observed = {"snapshot_file_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "function_text_sha256": hashlib.sha256(function.encode()).hexdigest()}
    if any(observed[k] != pin[k] for k in observed):
        raise RuntimeError("Pinned fusion snapshot changed; diagnose before rerunning")
    return pin


def _snapshot():
    verify_pin()
    spec = importlib.util.spec_from_file_location("exp0_fusion_candidate", FUSION_SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = TARGET_REPO
    return module


def _neighbours(visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the previous / next visible frame (-1 / len if none) for each position."""
    n = len(visible)
    position = np.arange(n)
    previous = np.maximum.accumulate(np.where(visible, position, -1))
    following = np.minimum.accumulate(np.where(visible, position, n)[::-1])[::-1]
    return previous, following


def replay_guard(tactical_ball: np.ndarray, candidate_ball: np.ndarray, recovered: np.ndarray,
                 half: np.ndarray, speed: float = REPLAY_SPEED) -> np.ndarray:
    """Boolean mask of recovered frames that are reachable from both neighbouring tactical
    observations in the same half; frames without a neighbour on either side are rejected."""
    keep = np.zeros(len(half), dtype=bool)
    for period in (p for p in np.unique(half) if p > 0):
        index = np.flatnonzero(half == period)
        visible = np.isfinite(tactical_ball[index]).all(axis=1)
        previous, following = _neighbours(visible)
        local = np.flatnonzero(recovered[index])
        if len(local) == 0:
            continue
        before, after = previous[local], following[local]
        valid = (before >= 0) & (after < len(index))
        b = candidate_ball[index[local]]
        p = tactical_ball[index[np.clip(before, 0, len(index) - 1)]]
        q = tactical_ball[index[np.clip(after, 0, len(index) - 1)]]
        reach_before = speed * (local - before) / FPS
        reach_after = speed * (after - local) / FPS
        with np.errstate(invalid="ignore"):
            ok = (valid & (np.linalg.norm(b - p, axis=1) <= reach_before)
                  & (np.linalg.norm(b - q, axis=1) <= reach_after))
        keep[index[local[ok]]] = True
    return keep


def fuse(tactical, broadcast) -> tuple[object, dict]:
    """Fused match iff the pinned rule accepts both halves (after the replay guard), with an
    aggregate report; otherwise the tactical match unchanged."""
    candidate, reports = _snapshot().fusion_candidate(tactical, broadcast)
    tactical_visible = np.isfinite(tactical.ball).all(axis=1)
    recovered = ~tactical_visible & np.isfinite(candidate.ball).all(axis=1)
    keep = replay_guard(tactical.ball, candidate.ball, recovered, tactical.half)
    accepted_both = len(reports) == 2 and all(r["accepted"] for r in reports)
    ball = tactical.ball.copy()
    ball[keep] = candidate.ball[keep]
    graded = tactical.half > 0
    missing = int((~tactical_visible & graded).sum())
    report = {
        "halves": reports,
        "accepted_both_halves": bool(accepted_both),
        "use_fused_input": bool(accepted_both),
        "recovered_before_guard": int(recovered.sum()),
        "kept_after_guard": int(keep.sum()),
        "removed_by_replay_guard": int(recovered.sum() - keep.sum()),
        "missing_ball_frames_graded": missing,
        "recovered_share_of_missing_graded": float((keep & graded).sum() / max(1, missing)),
        "ball_coverage_graded_tactical": float(tactical_visible[graded].mean()),
        "ball_coverage_graded_fused": float(np.isfinite(ball[graded]).all(axis=1).mean()),
    }
    fused = replace(tactical, ball=ball) if accepted_both else tactical
    return fused, report
