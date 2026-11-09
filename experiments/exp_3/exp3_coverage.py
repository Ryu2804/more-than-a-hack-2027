"""S2 coverage view: re-reveal clean ball positions inside the ``target_like`` ball gaps.

Proxy for broadcast fusion (which supplies the ball on about 12 % of BAR-ATH's missing-ball
frames). Label-free and deterministic: in every maximal hidden-ball run of length L (graded
frames, same half) one contiguous block of ``round(REVEAL_SHARE * L)`` frames at a seeded
uniform offset gets the clean ball back. Players, teams, halves and labels are unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
import exp2_views as V
from exp1_common import BASE_SEED, MATCH_IDS, Match

REVEAL_SHARE = 0.12
SEED_OFFSET = 8_000_000


def reveal_seed(name: str) -> int:
    return BASE_SEED + MATCH_IDS.index(name) + SEED_OFFSET


def reveal_mask(view: Match, rng: np.random.Generator, share: float = REVEAL_SHARE) -> np.ndarray:
    """Frames to reveal: one contiguous block per hidden-ball run, chosen without labels."""
    hidden = ~V.ball_visible(view) & (view.half > 0)
    mask = np.zeros(len(view.half), dtype=bool)
    for start, stop in V.graded_runs(hidden, view.half):
        length = stop - start
        size = int(round(share * length))
        if size == 0:
            continue
        offset = int(rng.integers(0, length - size + 1))
        mask[start + offset:start + offset + size] = True
    return mask


def coverage_view(base: Match, clean_ball: np.ndarray, rng: np.random.Generator) -> tuple[Match, dict]:
    """``target_like`` of ``base`` with the clean ball revealed on the chosen blocks."""
    if clean_ball.shape != base.ball.shape:
        raise ValueError("Clean ball must align with the degraded match")
    view = V.target_like(base)
    chosen = reveal_mask(view, rng)
    reveal = chosen & np.isfinite(clean_ball).all(axis=1)
    ball = view.ball.copy()
    ball[reveal] = clean_ball[reveal]
    graded = view.half > 0
    hidden = graded & ~V.ball_visible(view)
    stats = {"hidden_frames": int(hidden.sum()), "revealed_frames": int(reveal.sum()),
             "revealed_share_of_hidden": float(reveal.sum() / max(1, hidden.sum())),
             "ball_visible_before": float(V.ball_visible(view)[graded].mean()),
             "ball_visible_after": float(np.isfinite(ball[graded]).all(axis=1).mean())}
    return replace(view, ball=ball), stats
