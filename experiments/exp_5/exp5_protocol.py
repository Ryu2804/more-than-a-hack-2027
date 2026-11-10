"""exp_5 fits: exp_4's touch-only SetSeq (``exp4_touch``) with fresh seeds.

Only pure exp_4 pieces are imported: ``exp4_protocol.train_tokens`` / ``view_tokens`` /
``globals_context`` / ``RECIPE`` / ``_claim`` and ``exp4_net.train`` / ``AUX_OFF`` / ``state_nll``.
exp_4's arm table, fingerprints, folders and decision are never used: exp_5 has its own arms,
fingerprints (hashing every fit module, exp_4's included) and ``exp_5/cache/predictions``.

This file is part of every exp_5 fingerprint, so it holds fit code only; evaluation lives in
``exp5_eval``. exp_4's touch predictions are read in place for the reproduction check only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import time

import numpy as np
import torch

import exp5_common as C  # noqa: I001  (first: env, sys.path, exp_1-exp_4 imports)
import exp1_net as N
import exp1_protocol as P1
import exp3_protocol as P3
import exp3_touch as S
import exp4_net as X
import exp4_protocol as P4
from exp1_common import sha256

RECIPE = P4.RECIPE  # round 0 original + rounds 1-2 target_aug (exp2_aug recipe, as exp4_touch)
SEEDS = (20261111, 20261112, 20261113)


@dataclass(frozen=True)
class ArmSpec:
    """exp4_touch's arm with only the seed free: touch columns on, aux head off, TrainConfig()."""

    seed: int
    touch: bool = True
    aux: X.AuxConfig = field(default=X.AUX_OFF)

    @property
    def config(self) -> N.TrainConfig:
        return replace(N.TrainConfig(), seed=self.seed)

    def as_dict(self) -> dict:  # same layout as exp4_protocol.ArmSpec.as_dict
        return {"touch": self.touch, "aux": self.aux.as_dict(), "seed": self.seed,
                "config": self.config.as_dict()}


ARMS = {f"exp5_touch{suffix}": ArmSpec(seed) for suffix, seed in zip(("", "_s1", "_s2"), SEEDS)}
FIT_ORDER = tuple(ARMS)  # seeds 20261111 -> 12 -> 13
REPRO = {"arm": "exp4_touch", "held_out": "J03WMX", "seed": 20261108, "max_abs_probability_diff": 1e-3}
FIT_CODE = {  # every module a fit executes or configures (exp_4's included); file -> folder
    "exp1_net.py": C.EXP1, "exp1_tokens.py": C.EXP1, "exp2_views.py": C.EXP2,
    "exp3_touch.py": C.EXP3, "exp4_net.py": C.EXP4, "exp4_aux.py": C.EXP4,
    "exp4_protocol.py": C.EXP4, "exp5_protocol.py": C.ROOT,
}


# ---------------------------------------------------------------- identity
def fingerprint(arm: str) -> str:
    code = {name: sha256(folder / name) for name, folder in FIT_CODE.items()}
    payload = {"arm": arm, "spec": ARMS[arm].as_dict(), "recipe": RECIPE, "matches": C.MATCH_IDS,
               "views": C.VIEWS, "train_label": C.TRAIN_LABEL, "touch_names": S.TOUCH_NAMES, "code": code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def prediction_folder(arm: str):
    return C.PRED_DIR / f"{arm}_{fingerprint(arm)}"


# ---------------------------------------------------------------- one fit
def train_arrays(spec: ArmSpec, names: list[str]) -> list[dict]:
    """exp2_aug recipe on the training matches, provider targets, touch columns (as exp4_touch)."""
    return [P4.train_tokens(name, kind, round_index, spec.touch)
            for name in names for round_index, kind in RECIPE]


def fit_spec(spec: ArmSpec, names: list[str]) -> tuple[X.SetSeqAux, list[dict]]:
    arrays = train_arrays(spec, names)
    with P4.globals_context(spec.touch):
        return X.train(arrays, spec.config, spec.aux), arrays


def new_model(spec: ArmSpec) -> X.SetSeqAux:
    """Untrained model of the arm's shape (for loading saved weights)."""
    with P4.globals_context(spec.touch):
        model = X.SetSeqAux(spec.config.width, spec.config.hidden, spec.config.seed + spec.aux.head_seed_offset)
    return model.to(N.DEVICE)


def predict_views(model, spec: ArmSpec, names: list[str]) -> dict:
    """Both held-out views of each match (held-out inputs carry no training targets)."""
    out = {}
    for name in names:
        for view in C.VIEWS:
            probability, boundary = N.predict(model, P4.view_tokens(name, view, spec.touch), spec.config.window)
            out[f"{view}__{name}_probability"] = probability
            out[f"{view}__{name}_boundary"] = boundary
    return out


def _fit_job(arm: str, train_names: list[str], predict_names: list[str]) -> tuple[dict, dict]:
    spec = ARMS[arm]
    model, arrays = fit_spec(spec, train_names)
    train_nll, train_steps = X.state_nll(model, arrays, spec.config.window)
    out = {"train_nll": np.array(train_nll), "train_steps": np.array(train_steps),
           **predict_views(model, spec, predict_names)}
    return out, {k: v.detach().cpu() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- nested fits of one arm
def missing_jobs(arm: str) -> list[str]:
    folder = prediction_folder(arm)
    return [key for key, *_ in P1.job_keys() if not (folder / f"{key}.npz").exists()]


def ensure_predictions(arm: str, log=print, fit: bool = True, poll_s: float = 20.0) -> dict:
    """Run (or reuse) the 28 pair/outer fits of one arm -> {job: {view: {match: (p, b)}}}.
    With ``fit=False`` a missing job is an error and nothing is created (evaluation never fits)."""
    if arm not in ARMS:
        raise ValueError(f"Unknown exp_5 arm {arm}")
    folder = prediction_folder(arm)
    if not fit:
        missing = missing_jobs(arm)
        if missing:
            raise RuntimeError(f"{arm}: {len(missing)} jobs not fitted yet; run 01_fits first")
    folder.mkdir(parents=True, exist_ok=True)
    pending = [job for job in P1.job_keys() if not (folder / f"{job[0]}.npz").exists()]
    while pending:
        waiting = []
        for key, train_names, predict_names in pending:
            path, lock = folder / f"{key}.npz", folder / f"{key}.lock"
            if path.exists():
                continue
            if not P4._claim(lock):
                waiting.append((key, train_names, predict_names))
                continue
            try:
                if not path.exists():
                    start = time.time()
                    arrays, weights = _fit_job(arm, train_names, predict_names)
                    torch.save(weights, folder / f"{key}.pt")
                    tmp = folder / f"{key}.tmp.npz"
                    np.savez(tmp, **arrays)
                    os.replace(tmp, path)
                    log(f"FIT {arm} {key} {time.time() - start:.1f}s train_nll {float(arrays['train_nll']):.4f}")
            finally:
                lock.unlink(missing_ok=True)
        pending = waiting
        if pending:
            time.sleep(poll_s)
    return {key: P3._load_job(folder / f"{key}.npz", names) for key, _, names in P1.job_keys()}


def proxy_fit_count() -> int:
    return sum(1 for _ in C.PRED_DIR.glob("*/*.pt")) if C.PRED_DIR.exists() else 0


def train_nll_rows(arm: str) -> list[dict]:
    folder = prediction_folder(arm)
    rows = []
    for key, _, _ in P1.job_keys():
        with np.load(folder / f"{key}.npz") as z:
            rows.append({"arm": arm, "job": key, "train_nll": float(z["train_nll"]),
                         "train_steps": int(z["train_steps"])})
    return rows


# ---------------------------------------------------------------- reproduction check
def reproduction_check() -> dict:
    """One fit through exp_5's code path with exp4_touch's seed, outer fold of J03WMX, against
    exp_4's cached prediction for that fold (read in place). Bit-exact expected."""
    spec = ArmSpec(seed=REPRO["seed"])
    held = REPRO["held_out"]
    model, _ = fit_spec(spec, [n for n in C.MATCH_IDS if n != held])
    fresh = predict_views(model, spec, [held])
    folder = C.exp4_touch_folder(REPRO["arm"])
    out = {"arm": REPRO["arm"], "seed": spec.seed, "held_out": held, "fits": 1,
           "exp4_cache": f"exp_4/cache/predictions/{folder.name}/outer_{held}.npz"}
    with np.load(folder / f"outer_{held}.npz") as cached:
        for view in C.VIEWS:
            p, b = fresh[f"{view}__{held}_probability"], fresh[f"{view}__{held}_boundary"]
            ref_p, ref_b = cached[f"{view}__{held}_probability"], cached[f"{view}__{held}_boundary"]
            out[view] = {"max_abs_probability_diff": float(np.abs(p - ref_p).max()),
                         "max_abs_boundary_diff": float(np.abs(b - ref_b).max()),
                         "bit_exact": bool(np.array_equal(p, ref_p) and np.array_equal(b, ref_b))}
    out["bit_exact"] = all(out[v]["bit_exact"] for v in C.VIEWS)
    out["passed"] = all(out[v]["max_abs_probability_diff"] <= REPRO["max_abs_probability_diff"] for v in C.VIEWS)
    return out
