"""S3 (stretch, report only): sub-step ball-contact ("touch") features for SetSeq.

Four extra global features per 5 fps step at frame i, from the 25 fps frames [i-4, i] of the
same half (PREREG 5.1): per-team minimum player-ball distance, the peak change in ball
velocity, and the team of the nearest player at that peak. exp_1's modules are not edited:
``extended_globals()`` temporarily extends ``exp1_net``'s global-feature constants (names,
mirror negate / swap lists) for S3's own fits and predictions only.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import time
import warnings

import numpy as np
import torch

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
import exp1_net as N
import exp1_protocol as P1
import exp1_tokens as T
import exp2_protocol as P2
import exp2_views as V
import exp3_protocol as P
from exp1_common import BASE_SEED, FPS, MATCH_IDS, load_degraded, sample_indices, sha256
from exp3_common import EXP1, EXP2, PRED_DIR, ROOT, TOKEN_DIR, VIEWS

TOUCH_NAMES = ("home_min_dist", "away_min_dist", "peak_dv", "peak_team")
WINDOW = 5  # frames [i - 4, i]
DIST_SCALE, DIST_CAP = 10.0, 3.0
DV_SCALE, DV_CAP = 20.0, 3.0
BASE = len(T.GLOBAL_NAMES)
TOUCH_NEGATE = (BASE + 3,)
TOUCH_SWAP = ((BASE, BASE + 1),)
ARM = "exp3_touch"
CONFIG = N.TrainConfig(seed=BASE_SEED)


# ---------------------------------------------------------------- features
def _dv(ball: np.ndarray, half: np.ndarray) -> np.ndarray:
    """|v_g - v_{g-1}| (m/s) per frame with v_g = (b_g - b_{g-1}) * 25; NaN unless frames g-2..g
    are visible and in the same half."""
    out = np.full(len(half), np.nan)
    change = ball[2:] - 2 * ball[1:-1] + ball[:-2]
    same = (half[2:] == half[1:-1]) & (half[1:-1] == half[:-2]) & (half[2:] > 0)
    value = np.linalg.norm(change, axis=1) * FPS
    out[2:] = np.where(same & np.isfinite(value), value, np.nan)
    return out


def touch_features(match) -> np.ndarray:
    """(steps, 4) features on the shared 5 fps grid; observations only, no labels."""
    steps = sample_indices(match.half)
    ball, half = match.ball, match.half
    distance = np.linalg.norm(match.players[:, :, :2] - ball[:, None, :], axis=-1)  # NaN if absent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_min = np.nanmin(np.where((match.teams == 0)[None, :], distance, np.nan), axis=1)
        away_min = np.nanmin(np.where((match.teams == 1)[None, :], distance, np.nan), axis=1)
    finite_distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest_slot = finite_distance.argmin(axis=1)
    nearest_ok = np.isfinite(finite_distance.min(axis=1))
    nearest_team = np.where(nearest_ok, np.where(match.teams[nearest_slot] == 0, 1.0, -1.0), 0.0)
    dv = _dv(ball, half)
    out = np.zeros((len(steps), 4), dtype="float32")
    for k, frame in enumerate(steps):
        window = np.arange(max(frame - WINDOW + 1, 0), frame + 1)
        window = window[half[window] == half[frame]]
        for column, values in ((0, home_min), (1, away_min)):
            finite = values[window][np.isfinite(values[window])]
            out[k, column] = min(finite.min() / DIST_SCALE, DIST_CAP) if len(finite) else DIST_CAP
        valid = window[np.isfinite(dv[window])]
        if len(valid):
            peak = valid[int(np.argmax(dv[valid]))]
            out[k, 2] = min(dv[peak] / DV_SCALE, DV_CAP)
            out[k, 3] = nearest_team[peak]
    return out


def mirror_features(features: np.ndarray) -> np.ndarray:
    """Expected features of the home/away mirrored match (swap distances, negate peak team)."""
    out = features.copy()
    out[:, [0, 1]] = features[:, [1, 0]]
    out[:, 3] = -features[:, 3]
    return out


@contextmanager
def extended_globals():
    """Temporarily give exp1_net the 4 extra global columns and their mirror rules."""
    saved = (N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP)
    N.GLOBAL_NAMES = tuple(T.GLOBAL_NAMES) + TOUCH_NAMES
    N.GLOBAL_NEGATE = tuple(T.GLOBAL_NEGATE) + TOUCH_NEGATE
    N.GLOBAL_SWAP = tuple(T.GLOBAL_SWAP) + TOUCH_SWAP
    try:
        yield
    finally:
        N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP = saved


# ---------------------------------------------------------------- feature caches
def _observed(name: str, kind: str, round_index: int):
    if kind == "original":
        return load_degraded(name, round_index)
    if kind == "target_like":
        return V.target_like(load_degraded(name, 0))
    if kind == "target_aug":
        return V.target_aug(load_degraded(name, round_index),
                            np.random.default_rng(V.aug_seed(name, round_index)))
    raise ValueError(kind)


def feature_path(name: str, kind: str, round_index: int):
    return TOKEN_DIR / f"{name}_touch_{kind}_r{round_index}.npy"


def build_feature_cache() -> list[dict]:
    """Touch features for every training round and both held-out views; the regenerated
    observation must match the cached tokens' ball visibility (global column 0)."""
    rows = []
    jobs = [(n, kind, r) for n in MATCH_IDS for r, kind in P.RECIPE] + [(n, "target_like", 0) for n in MATCH_IDS]
    for name, kind, round_index in jobs:
        match = _observed(name, kind, round_index)
        cached = P2.load_tokens(name, kind, round_index)
        visible = np.isfinite(match.ball[cached["indices"]]).all(axis=1).astype("float16")
        if not np.array_equal(visible, cached["glob"][:, 0]):
            raise RuntimeError(f"Regenerated {kind} r{round_index} of {name} differs from the token cache")
        path = feature_path(name, kind, round_index)
        if not path.exists():
            np.save(path, touch_features(match))
        features = np.load(path)
        rows.append({"match": name, "kind": kind, "round": round_index,
                     "mean_home_min_dist": float(features[:, 0].mean()), "mean_peak_dv": float(features[:, 2].mean()),
                     "peak_team_defined": float((features[:, 3] != 0).mean())})
    return rows


def with_touch(tokens: dict, name: str, kind: str, round_index: int) -> dict:
    features = np.load(feature_path(name, kind, round_index))
    return {**tokens, "glob": np.concatenate([tokens["glob"], features.astype("float16")], axis=1)}


def view_tokens(name: str, view: str) -> dict:
    kind = "original" if view == "original" else "target_like"
    return with_touch(P.view_tokens(name, view), name, kind, 0)


# ---------------------------------------------------------------- fits
def fingerprint() -> str:
    code = {f: sha256(EXP1 / f) for f in P.FIT_CODE}
    code["exp2_views.py"] = sha256(EXP2 / "exp2_views.py")
    code["exp3_labels.py"] = sha256(ROOT / "exp3_labels.py")
    code["exp3_touch.py"] = sha256(ROOT / "exp3_touch.py")
    payload = {"arm": ARM, "recipe": P.RECIPE, "config": CONFIG.as_dict(), "train_label": P.TRAIN_LABEL,
               "events_used": P.EVENTS_USED, "touch": TOUCH_NAMES, "code": code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def train_arrays(names: list[str]) -> list[dict]:
    arrays = []
    for name in names:
        for round_index, kind in P.RECIPE:
            tokens = with_touch(P2.load_tokens(name, kind, round_index), name, kind, round_index)
            state, boundary = P.step_labels(name, P.TRAIN_LABEL, tokens)
            arrays.append({**tokens, "state": state, "boundary": boundary})
    return arrays


def _fit_job(train_names: list[str], predict_names: list[str]) -> tuple[dict, dict]:
    with extended_globals():
        model = N.train(train_arrays(train_names), CONFIG)
        arrays = {}
        for name in predict_names:
            for view in VIEWS:
                probability, boundary = N.predict(model, view_tokens(name, view), CONFIG.window)
                arrays[f"{view}__{name}_probability"] = probability
                arrays[f"{view}__{name}_boundary"] = boundary
    return arrays, {k: v.detach().cpu() for k, v in model.state_dict().items()}


def ensure_predictions(log=print) -> dict:
    """28 nested fits of the touch arm (cached by fingerprint), both views predicted."""
    folder = PRED_DIR / f"{ARM}_{fingerprint()}"
    folder.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for key, train_names, predict_names in P1.job_keys():
        path = folder / f"{key}.npz"
        if not path.exists():
            start = time.time()
            arrays, weights = _fit_job(train_names, predict_names)
            torch.save(weights, folder / f"{key}.pt")
            tmp = folder / f"{key}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            log(f"FIT {ARM} {key} {time.time() - start:.1f}s")
        predictions[key] = P._load_job(path, predict_names)
    return predictions
