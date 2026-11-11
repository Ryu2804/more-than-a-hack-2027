"""exp_7 views and the ``ball_recovered`` flag (PREREG 3.1-3.3) and their token caches.

* Training (PREREG 3.2): round 0 is exp_1's cached original view (flag 0); rounds 1-2 are exp_2's ``target_aug``
  view, then ``exp6_sim.simulate`` with (c, sigma, f) drawn once per (match, round) from a seeded generator.
* Evaluation (PREREG 3.3): exp_6's views exactly (``exp6_sim.make_view``, exp_6's seeds and run lengths), plus the
  AR(1) transfer view, which differs from the measured view only in the noise's temporal correlation.
* Flag: 1 iff the frame's ball comes from a recovery, i.e. the view has a ball where its base has none.

Fit code: part of every exp_7 fingerprint (the fits read these caches and predict these views).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import os

import numpy as np

import exp7_common as C  # noqa: I001  (first: environment, sys.path, safe exp_6 imports)
import exp1_net as N
import exp1_protocol as P1
import exp1_tokens as T
import exp2_protocol as P2
import exp2_views as V
from exp1_common import BASE_SEED, MATCH_IDS, JD, load_clean, load_degraded

S = C.S6
FLAG_NAME = "ball_recovered"
RECIPE = P2.RECIPES["target_aug"]  # round 0 original, rounds 1-2 target_aug (+ simulated detector errors)
SIM_ROUNDS = tuple(r for r, kind in RECIPE if kind == "target_aug")
TRAIN_SEED_OFFSET = 9000
TRAIN_RANGES = {"coverage": (0.0, 0.6), "noise": (0.0, 1.0), "fp": (0.0, 0.5)}  # U(low, high), drawn in this order
AR1_RHO = 0.9
EVAL_VIEWS = ("target_like", "measured", "stress", "clean25", "clean50", "ar1")
CACHED_VIEWS = EVAL_VIEWS[1:]  # target_like comes from exp_2's token cache (+ flag 0)
TOKEN_KEYS = ("tokens", "mask", "glob", "half", "indices")


# ---------------------------------------------------------------- flag
def recovered_mask(base: JD.Match, view: JD.Match) -> np.ndarray:
    """Graded frames where ``view`` has a ball and ``base`` has none (a simulated or detected recovery)."""
    return (base.half > 0) & ~V.ball_visible(base) & V.ball_visible(view)


def with_flag(item: dict, flag: np.ndarray) -> dict:
    if len(flag) != len(item["glob"]):
        raise ValueError("Flag is not aligned with the token grid")
    return {**item, "glob": np.concatenate([item["glob"], flag.astype("float16")[:, None]], axis=1)}


def without_flag(item: dict) -> dict:
    """The exp_1 input of a flagged item (for exp_2's saved models)."""
    return {**item, "glob": item["glob"][:, :len(T.GLOBAL_NAMES)]}


@contextmanager
def flag_globals():
    """Temporarily give exp1_net one extra global column; the home/away mirror leaves it unchanged."""
    saved = (N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP)
    N.GLOBAL_NAMES = tuple(T.GLOBAL_NAMES) + (FLAG_NAME,)
    N.GLOBAL_NEGATE, N.GLOBAL_SWAP = tuple(T.GLOBAL_NEGATE), tuple(T.GLOBAL_SWAP)
    try:
        yield
    finally:
        N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP = saved


def token_item(view: JD.Match, mask: np.ndarray, labels: bool) -> dict:
    """exp_1 tokens + flag column; with ``labels`` the provider state / boundary targets (exp_2's way)."""
    token_set = T.build_tokens(view)
    item = with_flag({key: getattr(token_set, key) for key in TOKEN_KEYS}, mask[token_set.indices])
    if labels:
        item["state"], item["boundary"] = T.step_targets(view.labels, token_set)
    return item


# ---------------------------------------------------------------- settings and seeds
def run_lengths() -> np.ndarray:
    """exp_6's BAR-ATH accepted run lengths (read in place; aggregate distribution only)."""
    lengths = np.load(C.RUN_LENGTHS)
    if not len(lengths):
        raise RuntimeError("exp_6 run-length distribution is empty")
    return lengths


def eval_settings() -> dict:
    q = C.measured_inputs()
    measured = S.measured(q)
    return {"target_like": S.BASE, "measured": measured, "stress": S.stress(q),
            "clean25": S.Setting("clean25", 0.25, 0.0, 0.0), "clean50": S.Setting("clean50", 0.5, 0.0, 0.0),
            "ar1": replace(measured, name="ar1")}


def train_seed(name: str, round_index: int) -> int:
    return BASE_SEED + TRAIN_SEED_OFFSET + 1000 * round_index + MATCH_IDS.index(name)


def draw_setting(rng: np.random.Generator) -> S.Setting:
    c, sigma, f = (float(rng.uniform(*TRAIN_RANGES[key])) for key in ("coverage", "noise", "fp"))
    return S.Setting("train", c, sigma, f)


def augment(base: JD.Match, clean: JD.Match, seed: int, lengths: np.ndarray) -> tuple[JD.Match, dict]:
    """One generator per (match, round): parameters first, then exp_6's simulator."""
    rng = np.random.default_rng(seed)
    setting = draw_setting(rng)
    view, stats = S.simulate(base, clean, setting, lengths, rng)
    return view, {**setting.as_dict(), **stats}


def train_view(name: str, round_index: int, lengths: np.ndarray, clean: JD.Match | None = None):
    """(target_aug base, simulated view, draw + simulator stats) of one training round."""
    base = V.target_aug(load_degraded(name, round_index), np.random.default_rng(V.aug_seed(name, round_index)))
    view, info = augment(base, clean if clean is not None else load_clean(name),
                         train_seed(name, round_index), lengths)
    return base, view, info


# ---------------------------------------------------------------- evaluation views
def ar1_noise(z: np.ndarray, sigma: float, rho: float) -> np.ndarray:
    """Stationary AR(1) per axis from standard normals z: e_0 = sigma z_0, e_t = rho e_{t-1} + sigma sqrt(1-rho^2) z_t."""
    scale = sigma * np.sqrt(1.0 - rho * rho)
    out = np.empty_like(z)
    out[0] = sigma * z[0]
    for t in range(1, len(z)):
        out[t] = rho * out[t - 1] + scale * z[t]
    return out


def simulate_ar1(base: JD.Match, clean: JD.Match, setting: S.Setting, lengths: np.ndarray,
                 rng: np.random.Generator, rho: float) -> tuple[JD.Match, dict]:
    """``exp6_sim.simulate`` with AR(1) noise on true-ball sub-runs. Same generator consumption, so the
    sub-runs and false-positive runs equal the measured view's; rho = 0 reproduces ``simulate`` exactly."""
    hidden = S.hidden_frames(base, clean.ball)
    target = int(round(setting.coverage * hidden.sum()))
    ball = base.ball.astype("float64")
    subruns = S.place_subruns(hidden, target, lengths, rng) if target else []
    fp_frames = 0
    for start, length in subruns:
        frames = slice(start, start + length)
        if rng.random() < setting.fp:
            track = S._player_track(clean.players[frames, :, :2].astype("float64"), rng)
            ball[frames] = track + rng.normal(0.0, S.FP_NOISE_M, (length, 2))
            fp_frames += length
        else:
            ball[frames] = clean.ball[frames] + ar1_noise(rng.normal(0.0, 1.0, (length, 2)), setting.noise, rho)
    view = replace(base, ball=ball.astype(base.ball.dtype))
    return view, {"hidden_frames": int(hidden.sum()), "target_frames": target, "subruns": len(subruns),
                  "revealed_frames": int(sum(length for _, length in subruns)), "fp_frames": fp_frames}


def eval_view(name: str, view_name: str, lengths: np.ndarray, base: JD.Match, clean: JD.Match):
    """One evaluation view of a held-out match from its ``target_like`` base (exp_6's seed)."""
    setting = eval_settings()[view_name]
    if view_name == "ar1":
        return simulate_ar1(base, clean, setting, lengths, np.random.default_rng(S.sim_seed(name)), AR1_RHO)
    return S.make_view(setting, base, clean, lengths, name)


# ---------------------------------------------------------------- token caches
def train_path(name: str, round_index: int) -> Path:
    return C.TOKEN_DIR / f"{name}_train_r{round_index}.npz"


def view_path(name: str, view_name: str) -> Path:
    return C.TOKEN_DIR / f"{name}_{view_name}.npz"


def _save(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **item)
    os.replace(tmp, path)


def _load(path: Path) -> dict:
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def _stats(base: JD.Match, view: JD.Match, mask: np.ndarray) -> dict:
    graded = base.half > 0
    return {"flagged_frames": int(mask.sum()), "ball_visible_before": float(V.ball_visible(base)[graded].mean()),
            "ball_visible_after": float(V.ball_visible(view)[graded].mean())}


def build_token_caches(log=print) -> list[dict]:
    """Training rounds 1-2 and the five simulated evaluation views of every match (aggregates returned)."""
    lengths = run_lengths()
    grid_of = {name: P2.load_tokens(name)["indices"] for name in MATCH_IDS}
    rows = []
    for name in MATCH_IDS:
        clean = load_clean(name)
        jobs = [("train", r) for r in SIM_ROUNDS] + [("view", v) for v in CACHED_VIEWS]
        target_like = V.target_like(load_degraded(name, 0))
        for kind, which in jobs:
            if kind == "train":
                base, view, info = train_view(name, which, lengths, clean)
                path = train_path(name, which)
            else:
                base = target_like
                view, info = eval_view(name, which, lengths, base, clean)
                path = view_path(name, which)
            mask = recovered_mask(base, view)
            if not path.exists():
                item = token_item(view, mask, labels=kind == "train")
                if not np.array_equal(item["indices"], grid_of[name]):
                    raise RuntimeError("View tokens are not aligned with the original grid")
                _save(path, item)
            rows.append({"match": name, "kind": kind, "which": which, **info, **_stats(base, view, mask)})
            log(f"TOKENS {name} {kind} {which}")
    return rows


def train_item(name: str, round_index: int) -> dict:
    """One training array: exp_1's cached round 0 with flag 0, or a cached simulated round."""
    if round_index == 0:
        tokens = P1.load_tokens(name, 0)
        return with_flag(tokens, np.zeros(len(tokens["glob"]), dtype=bool))
    return _load(train_path(name, round_index))


def view_item(name: str, view_name: str) -> dict:
    """One evaluation input: exp_2's cached target_like tokens with flag 0, or a cached simulated view."""
    if view_name == "target_like":
        tokens = {key: P2.load_tokens(name, "target_like", 0)[key] for key in TOKEN_KEYS}
        return with_flag(tokens, np.zeros(len(tokens["glob"]), dtype=bool))
    if view_name not in CACHED_VIEWS:
        raise ValueError(f"Unknown view {view_name}")
    return _load(view_path(name, view_name))


def unsimulated_equals_exp2(name: str, round_index: int) -> bool:
    """The target_aug view before simulation reproduces exp_2's cached tokens and targets exactly."""
    base = V.target_aug(load_degraded(name, round_index), np.random.default_rng(V.aug_seed(name, round_index)))
    mine = without_flag(token_item(base, np.zeros(len(base.half), dtype=bool), labels=True))
    cached = P2.load_tokens(name, "target_aug", round_index)
    return set(mine) == set(cached) and all(np.array_equal(mine[key], cached[key]) for key in cached)
