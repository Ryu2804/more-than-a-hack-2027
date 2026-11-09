"""Verbatim snapshot of exp_0 ``experiment_target.fusion_candidate`` (pinned for exp_3).

exp_0 is still being edited, so exp_3 never imports it. The function body below is copied
byte-for-byte; its SHA-256 (of the function text alone) and the source file's SHA-256 at pin
time are in ``fusion_pin.json`` and are re-checked by ``exp3_fusion.verify_pin()``.
``REPO`` is set by the caller to the competition repository holding ``data/broadcast``.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

REPO = None  # set by exp3_fusion before use


def fusion_candidate(tactical, broadcast) -> tuple:
    """Reject misaligned/replay observations using geometry, never target labels."""
    homography = pd.read_csv(
        REPO / "data/broadcast/BAR-ATH_broadcast_homography.csv",
        usecols=["frame", "valid", "reprojection_error"],
    )
    valid_h = homography.valid.to_numpy().astype(bool) & (
        homography.reprojection_error.to_numpy() < 5.0
    )
    fused = tactical.ball.copy()
    reports = []
    for half, offset in ((1, 4014), (2, 26920)):
        target_index = np.flatnonzero(tactical.half == half)
        source_index = target_index - offset
        keep = (source_index >= 0) & (source_index < len(broadcast.ball))
        ti, bi = target_index[keep], source_index[keep]
        tv = np.isfinite(tactical.ball[ti]).all(axis=1)
        bv = np.isfinite(broadcast.ball[bi]).all(axis=1) & valid_h[bi]
        both = tv & bv
        error = np.linalg.norm(
            tactical.ball[ti[both]] - broadcast.ball[bi[both]], axis=1
        )
        accepted = (
            len(error) >= 1000
            and float(np.median(error)) <= 1.5
            and float(np.quantile(error, 0.9)) <= 5.0
        )
        recover = ~tv & bv
        if accepted:
            fused[ti[recover]] = broadcast.ball[bi[recover]]
        reports.append(
            {
                "half": half,
                "hypothesized_offset_frames": offset,
                "overlap_frames": len(error),
                "median_distance_m": float(np.median(error)) if len(error) else None,
                "p90_distance_m": (
                    float(np.quantile(error, 0.9)) if len(error) else None
                ),
                "accepted": bool(accepted),
                "recovered_frames": int(recover.sum()) if accepted else 0,
            }
        )
    return replace(tactical, ball=fused), reports
