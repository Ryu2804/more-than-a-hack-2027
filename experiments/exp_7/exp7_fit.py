"""exp_7 fits: exp_1 SetSeq with the ``ball_recovered`` flag, trained on exp_2's target_aug recipe with simulated
detector errors on rounds 1-2 (PREREG 2, 3.1-3.2). Provider labels; ``TrainConfig()`` except the seed.

Fit code only (part of every fingerprint). Each of the 28 nested jobs (exp_1's ``job_keys``) writes one resumable
``.npz`` (every evaluation view of its held-out matches, same weights) and one ``.pt``. Evaluation never fits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
import time

import numpy as np
import torch

import exp7_common as C  # noqa: I001  (first: environment, sys.path, safe exp_6 imports)
import exp1_net as N
import exp1_protocol as P1
import exp7_aug as A
from exp1_common import sha256

SEEDS = (20261117, 20261118, 20261119)


@dataclass(frozen=True)
class ArmSpec:
    seed: int

    @property
    def config(self) -> N.TrainConfig:
        return replace(N.TrainConfig(), seed=self.seed)

    def as_dict(self) -> dict:
        return {"seed": self.seed, "flag": A.FLAG_NAME, "config": self.config.as_dict()}


ARMS = {f"exp7_robust{suffix}": ArmSpec(seed) for suffix, seed in zip(("", "_s1", "_s2"), SEEDS)}
FIT_CODE = {  # every module a fit executes or configures: file -> folder
    "exp1_net.py": C.EXP1, "exp1_tokens.py": C.EXP1, "exp2_views.py": C.EXP2, "exp6_sim.py": C.EXP6,
    "exp7_aug.py": C.ROOT, "exp7_fit.py": C.ROOT,
}


# ---------------------------------------------------------------- identity
def fingerprint(arm: str) -> str:
    code = {name: sha256(folder / name) for name, folder in FIT_CODE.items()}
    payload = {"arm": arm, "spec": ARMS[arm].as_dict(), "recipe": A.RECIPE, "matches": C.MATCH_IDS,
               "views": A.EVAL_VIEWS, "train_label": C.TRAIN_LABEL, "train_ranges": A.TRAIN_RANGES,
               "train_seed_offset": A.TRAIN_SEED_OFFSET, "run_lengths": sha256(C.RUN_LENGTHS), "code": code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def prediction_folder(arm: str):
    return C.PRED_DIR / f"{arm}_{fingerprint(arm)}"


# ---------------------------------------------------------------- one fit
def train_arrays(names: list[str]) -> list[dict]:
    return [A.train_item(name, round_index) for name in names for round_index, _ in A.RECIPE]


def fit_model(spec: ArmSpec, names: list[str]) -> N.SetSeq:
    with A.flag_globals():
        return N.train(train_arrays(names), spec.config)


def new_model(spec: ArmSpec) -> N.SetSeq:
    """Untrained model of the arm's shape (for loading saved weights)."""
    with A.flag_globals():
        model = N.SetSeq(spec.config.width, spec.config.hidden)
    return model.to(N.DEVICE)


def predict_views(model: N.SetSeq, spec: ArmSpec, names: list[str]) -> dict:
    out = {}
    for name in names:
        for view in A.EVAL_VIEWS:
            probability, boundary = N.predict(model, A.view_item(name, view), spec.config.window)
            out[f"{view}__{name}_probability"] = probability
            out[f"{view}__{name}_boundary"] = boundary
    return out


def _fit_job(arm: str, train_names: list[str], predict_names: list[str]) -> tuple[dict, dict]:
    spec = ARMS[arm]
    model = fit_model(spec, train_names)
    return predict_views(model, spec, predict_names), {k: v.detach().cpu() for k, v in model.state_dict().items()}


def load_job(path, names: list[str]) -> dict:
    with np.load(path) as z:
        return {view: {n: (z[f"{view}__{n}_probability"], z[f"{view}__{n}_boundary"]) for n in names}
                for view in A.EVAL_VIEWS}


# ---------------------------------------------------------------- nested fits of one arm
def missing_jobs(arm: str) -> list[str]:
    folder = prediction_folder(arm)
    return [key for key, *_ in P1.job_keys() if not (folder / f"{key}.npz").exists()]


def ensure_predictions(arm: str, log=print, fit: bool = True) -> dict:
    """Run (or reuse) the 28 pair/outer fits of one arm -> {job: {view: {match: (p, b)}}}.
    With ``fit=False`` a missing job is an error and nothing is created (evaluation never fits)."""
    if arm not in ARMS:
        raise ValueError(f"Unknown exp_7 arm {arm}")
    folder = prediction_folder(arm)
    if not fit:
        missing = missing_jobs(arm)
        if missing:
            raise RuntimeError(f"{arm}: {len(missing)} jobs not fitted yet; run 01_fits first")
    else:
        folder.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for key, train_names, predict_names in P1.job_keys():
        path = folder / f"{key}.npz"
        if not path.exists():
            start = time.time()
            arrays, weights = _fit_job(arm, train_names, predict_names)
            torch.save(weights, folder / f"{key}.pt")
            tmp = folder / f"{key}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            log(f"FIT {arm} {key} {time.time() - start:.1f}s")
        predictions[key] = load_job(path, predict_names)
    return predictions


def proxy_fit_count() -> int:
    return sum(1 for _ in C.PRED_DIR.glob("*/*.pt")) if C.PRED_DIR.exists() else 0
