"""Aux-head targets from the *clean* IDSSE match (training matches only).

Per 5 fps step of a match's shared grid:

* ``aux_xy``  - clean ball (x, y), normalised exactly as exp_1's ``ball_x`` / ``ball_y`` globals
  (centred, divided by the half-pitch, times the label-free per-half flip sign);
* ``aux_ok``  - clean ball present;
* ``aux_team`` - team of the clean player nearest to the clean ball (0 home, 1 away, IGNORE).

The positions come from the clean match; the flip sign comes from the *observed* match of the
same corruption round (exp_1's tokens compute it there, and degradation can move a half's mean
home x across halfway), so the target lives in exactly the frame of that array's inputs. Targets
are attached to *training* arrays only; held-out matches never get them (``exp4_protocol``).
IDSSE only; BAR-ATH has no clean data and is never read here.
"""

from __future__ import annotations

import os

import numpy as np

import exp4_common  # noqa: F401,I001  (first: puts ../exp_3, ../exp_2, ../exp_1 on sys.path)
import exp1_tokens as T
from exp1_common import Match, load_clean, load_degraded, sample_indices
from exp4_common import AUX_DIR
from exp4_net import IGNORE

AUX_KEYS = ("aux_xy", "aux_ok", "aux_team")


def aux_targets(match: Match, indices: np.ndarray, sign: np.ndarray | None = None) -> dict:
    """Aux targets of one (clean) match on a 5 fps grid; observation geometry only, no labels.
    ``sign`` is the per-frame flip sign of the observed match (default: the match's own)."""
    length, width = match.pitch
    sign = (T._flip_signs(match) if sign is None else sign)[indices]
    ball = match.ball[indices]
    ok = np.isfinite(ball).all(axis=1)
    x = np.where(ok, (np.nan_to_num(ball[:, 0]) - length / 2) / (length / 2) * sign, 0.0)
    y = np.where(ok, (np.nan_to_num(ball[:, 1]) - width / 2) / (width / 2) * sign, 0.0)
    players = match.players[indices][:, :, :2]
    on_team = np.isin(match.teams, (0, 1))
    with np.errstate(invalid="ignore"):
        distance = np.linalg.norm(players - ball[:, None, :], axis=-1)
    distance = np.where(np.isfinite(distance) & on_team[None, :], distance, np.inf)
    nearest = distance.argmin(axis=1)
    has_player = np.isfinite(distance.min(axis=1))
    team = np.where(ok & has_player, match.teams[nearest], IGNORE).astype("int8")
    return {"aux_xy": np.column_stack([x, y]).astype("float32"), "aux_ok": ok, "aux_team": team}


def mirror_targets(targets: dict) -> dict:
    """Expected aux targets of the home/away mirrored match (numpy twin of exp4_net.mirror_aux)."""
    team = targets["aux_team"]
    swapped = np.where(team == 0, 1, np.where(team == 1, 0, team)).astype(team.dtype)
    return {"aux_xy": -targets["aux_xy"], "aux_ok": targets["aux_ok"].copy(), "aux_team": swapped}


def aux_path(name: str, round_index: int):
    return AUX_DIR / f"{name}_r{round_index}.npz"


def build_aux(name: str, round_index: int) -> dict:
    """Cached aux targets of one proxy match and corruption round on its 5 fps grid.
    ``target_aug`` arrays of a round share its players, hence its flip sign."""
    path = aux_path(name, round_index)
    if not path.exists():
        clean = load_clean(name)
        observed = load_degraded(name, round_index)
        if not np.array_equal(observed.half, clean.half):
            raise RuntimeError(f"Halves of {name} r{round_index} differ from the clean match")
        sign = T._flip_signs(observed)
        indices = sample_indices(clean.half)
        targets = aux_targets(clean, indices, sign)
        flipped = int((sign[indices] != T._flip_signs(clean)[indices]).sum())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, indices=indices, steps_sign_differs_from_clean=np.array(flipped), **targets)
        os.replace(tmp, path)
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def with_aux(tokens: dict, name: str, round_index: int) -> dict:
    """Training array + aux targets (new dict; the grids must match exactly)."""
    targets = build_aux(name, round_index)
    if not np.array_equal(targets["indices"], tokens["indices"]):
        raise RuntimeError(f"Aux grid of {name} is not the token grid")
    return {**tokens, **{key: targets[key] for key in AUX_KEYS}}


def consistency(tokens: dict, name: str, round_index: int) -> dict:
    """Aux ball target vs the observed ``ball_x`` / ``ball_y`` globals where the ball is visible.

    The degrader only hides the ball, so on visible steps they must agree up to float16
    rounding; this checks the normalisation and the flip sign of the aux target."""
    targets = build_aux(name, round_index)
    visible = tokens["glob"][:, 0].astype(bool)
    observed = tokens["glob"][:, 1:3].astype("float32")
    diff = np.abs(observed[visible] - targets["aux_xy"][visible])
    hidden = ~visible
    return {"visible_steps": int(visible.sum()),
            "visible_without_clean_ball": int((visible & ~targets["aux_ok"]).sum()),
            "max_abs_diff_visible": float(diff.max()) if len(diff) else 0.0,
            "hidden_steps": int(hidden.sum()),
            "hidden_with_ball_target": float(targets["aux_ok"][hidden].mean()) if hidden.any() else 0.0,
            "team_target_defined": float((targets["aux_team"] != IGNORE).mean()),
            "steps_sign_differs_from_clean": int(targets["steps_sign_differs_from_clean"])}


def team_agreement(name: str, state: np.ndarray) -> float:
    """Descriptive: share of alive steps where the clean nearest team equals the possession team."""
    team = build_aux(name, 0)["aux_team"]
    alive = (state < 2) & (team != IGNORE)
    return float((team[alive] == state[alive]).mean()) if alive.any() else float("nan")
