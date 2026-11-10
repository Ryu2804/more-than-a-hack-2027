"""exp_4 nested protocol: arms, fits, cached predictions and evaluation.

Nesting is exp_1's (``exp1_protocol.job_keys``): 21 pair models give inner validation
predictions, 7 outer models the held-out ones. Every fit predicts both views of its held-out
matches and saves its weights and its final training NLL. Evaluation (``exp3_protocol``)
re-selects temperature and decoder on the six inner validation matches against the evaluation
label set and scores the held-out match once. Evaluation never fits (``fit=False``).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import time

import numpy as np
import torch

import exp4_common  # noqa: F401,I001  (first: puts ../exp_3, ../exp_2, ../exp_1 on sys.path)
import exp1_net as N
import exp1_protocol as P1
import exp2_protocol as P2
import exp3_protocol as P3
import exp3_touch as S
import exp4_aux as A
import exp4_net as X
from exp1_common import read_json, sha256
from exp4_common import BASE_SEED, EXP1, EXP2, EXP3, MATCH_IDS, PRED_DIR, ROOT, TRAIN_LABEL, VIEWS

RECIPE = P2.RECIPES["target_aug"]  # round 0 original + rounds 1-2 target_aug (exp2_aug recipe)


@dataclass(frozen=True)
class ArmSpec:
    touch: bool
    aux: X.AuxConfig
    seed: int

    @property
    def config(self) -> N.TrainConfig:
        return replace(N.TrainConfig(), seed=self.seed)

    def as_dict(self) -> dict:
        return {"touch": self.touch, "aux": self.aux.as_dict(), "seed": self.seed,
                "config": self.config.as_dict()}


SEEDS = (BASE_SEED, BASE_SEED + 1, BASE_SEED + 2)  # 20261108 / 09 / 10
FAMILIES = {"exp4_full": (True, X.AUX_ON), "exp4_touch": (True, X.AUX_OFF), "exp4_aux": (False, X.AUX_ON)}
ARMS = {f"{family}{suffix}": ArmSpec(touch, aux, seed)
        for family, (touch, aux) in FAMILIES.items()
        for suffix, seed in zip(("", "_s1", "_s2"), SEEDS)}
FIT_ORDER = tuple(ARMS)  # exp4_full seeds first, then touch, then aux
EXP2_ARMS = ("exp2_aug", "exp2_aug_s1", "exp2_aug_s2")
EXP2_FINGERPRINTS = {arm: P2.fingerprint(arm) for arm in EXP2_ARMS}
ENSEMBLES = {f"{family}_ens3": tuple(f"{family}{s}" for s in ("", "_s1", "_s2")) for family in FAMILIES}
ENSEMBLES["ref_ens3"] = EXP2_ARMS
VARIANTS = {  # variant -> prediction source (decoder: exp_1 main, D.GAMMAS, for all)
    "exp4_full_ens3": ("exp4_full_ens3",),
    "exp4_touch_ens3": ("exp4_touch_ens3",),
    "exp4_aux_ens3": ("exp4_aux_ens3",),
    **{arm: (arm,) for arm in ARMS},
}
ENSEMBLE_VARIANTS = ("exp4_full_ens3", "exp4_touch_ens3", "exp4_aux_ens3")
REUSED_VARIANTS = ("exp3_ref_ens3", "exp2_aug")  # exp_3 result files, read in place
FIT_CODE = {"exp1_net.py": EXP1, "exp1_tokens.py": EXP1, "exp2_views.py": EXP2,
            "exp3_touch.py": EXP3, "exp4_net.py": ROOT, "exp4_aux.py": ROOT}


# ---------------------------------------------------------------- inputs
def fingerprint(arm: str) -> str:
    code = {name: sha256(folder / name) for name, folder in FIT_CODE.items()}
    payload = {"arm": arm, "spec": ARMS[arm].as_dict(), "recipe": RECIPE, "matches": MATCH_IDS,
               "views": VIEWS, "train_label": TRAIN_LABEL, "touch_names": S.TOUCH_NAMES, "code": code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def prediction_folder(arm: str):
    return PRED_DIR / f"{arm}_{fingerprint(arm)}"


def train_tokens(name: str, kind: str, round_index: int, touch: bool) -> dict:
    """Cached exp_1 / exp_2 tokens (provider state / boundary targets), touch columns optional."""
    tokens = P2.load_tokens(name, kind, round_index)
    return S.with_touch(tokens, name, kind, round_index) if touch else tokens


def view_tokens(name: str, view: str, touch: bool) -> dict:
    """Held-out inputs: never carry aux targets."""
    return S.view_tokens(name, view) if touch else P2.view_tokens(name, view)


def train_arrays(arm: str, names: list[str]) -> list[dict]:
    """exp2_aug recipe on the *training* matches; aux targets (clean data) only for them."""
    spec = ARMS[arm]
    arrays = []
    for name in names:
        for round_index, kind in RECIPE:
            tokens = train_tokens(name, kind, round_index, spec.touch)
            arrays.append(A.with_aux(tokens, name, round_index) if spec.aux.enabled else tokens)
    return arrays


def globals_context(touch: bool):
    return S.extended_globals() if touch else nullcontext()


def fit_model(arm: str, names: list[str]) -> tuple[X.SetSeqAux, list[dict]]:
    spec = ARMS[arm]
    arrays = train_arrays(arm, names)
    with globals_context(spec.touch):
        return X.train(arrays, spec.config, spec.aux), arrays


def new_model(arm: str) -> X.SetSeqAux:
    """Untrained model of an arm's shape (for loading saved weights)."""
    spec = ARMS[arm]
    with globals_context(spec.touch):
        model = X.SetSeqAux(spec.config.width, spec.config.hidden, spec.config.seed + spec.aux.head_seed_offset)
    return model.to(N.DEVICE)


# ---------------------------------------------------------------- fits
def _fit_job(arm: str, train_names: list[str], predict_names: list[str]) -> tuple[dict, dict]:
    spec = ARMS[arm]
    model, arrays = fit_model(arm, train_names)
    window = spec.config.window
    train_nll, train_steps = X.state_nll(model, arrays, window)
    out = {"train_nll": np.array(train_nll), "train_steps": np.array(train_steps)}
    for name in predict_names:
        for view in VIEWS:
            probability, boundary = N.predict(model, view_tokens(name, view, spec.touch), window)
            out[f"{view}__{name}_probability"] = probability
            out[f"{view}__{name}_boundary"] = boundary
    return out, {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _claim(lock) -> bool:
    """Per-job lock so two fit processes never fit the same job (stale locks are taken over)."""
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text().strip() or "0"), 0)
            return False
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True)
            return _claim(lock)
    with os.fdopen(handle, "w") as stream:
        stream.write(str(os.getpid()))
    return True


def ensure_predictions(arm: str, log=print, fit: bool = True, poll_s: float = 20.0) -> dict:
    """Run (or reuse) the 28 pair/outer fits of one arm -> {job: {view: {match: (p, b)}}}.
    With ``fit=False`` a missing job is an error (evaluation never starts a fit)."""
    folder = prediction_folder(arm)
    jobs = P1.job_keys()
    if not fit:
        missing = [key for key, *_ in jobs if not (folder / f"{key}.npz").exists()]
        if missing:
            raise RuntimeError(f"{arm}: {len(missing)} jobs not fitted yet; run 01_fits first")
    folder.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not (folder / f"{job[0]}.npz").exists()]
    while pending:
        waiting = []
        for key, train_names, predict_names in pending:
            path, lock = folder / f"{key}.npz", folder / f"{key}.lock"
            if path.exists():
                continue
            if not _claim(lock):
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
    return {key: P3._load_job(folder / f"{key}.npz", names) for key, _, names in jobs}


def fit_count() -> int:
    return sum(1 for _ in PRED_DIR.glob("*/*.pt")) if PRED_DIR.exists() else 0


def train_nll_rows(arm: str) -> list[dict]:
    folder = prediction_folder(arm)
    rows = []
    for key, _, _ in P1.job_keys():
        with np.load(folder / f"{key}.npz") as z:
            rows.append({"arm": arm, "job": key, "train_nll": float(z["train_nll"]),
                         "train_steps": int(z["train_steps"])})
    return rows


# ---------------------------------------------------------------- reproduction check
def reproduction_check(held: str = MATCH_IDS[0]) -> dict:
    """One fit: exp_4's training loop with aux off and no touch, exp2_aug's recipe and seed,
    must reproduce exp_2's cached ``exp2_aug`` outer prediction bit-exactly."""
    arrays = [P2.load_tokens(n, kind, r) for n in MATCH_IDS if n != held for r, kind in RECIPE]
    model = X.train(arrays, N.TrainConfig(), X.AUX_OFF)
    cached = P3.cached_exp2("exp2_aug")[f"outer_{held}"]
    out = {"held_out": held, "fits": 1}
    for view in VIEWS:
        probability, boundary = N.predict(model, P2.view_tokens(held, view), N.TrainConfig().window)
        reference_p, reference_b = cached[view][held]
        out[view] = {"max_abs_probability_diff": float(np.abs(probability - reference_p).max()),
                     "max_abs_boundary_diff": float(np.abs(boundary - reference_b).max()),
                     "bit_exact": bool(np.array_equal(probability, reference_p)
                                       and np.array_equal(boundary, reference_b))}
    out["bit_exact"] = all(out[v]["bit_exact"] for v in VIEWS)
    return out


# ---------------------------------------------------------------- predictions
def predictions_for(source: str, cache: dict) -> dict:
    if source not in cache:
        if source in ENSEMBLES:
            members = {arm: predictions_for(arm, cache) for arm in ENSEMBLES[source]}
            cache[source] = P2.ensemble_predictions(members)
        elif source in ARMS:
            cache[source] = ensure_predictions(source, fit=False)
        elif source in EXP2_ARMS:
            cache[source] = P3.cached_exp2(source)
        else:
            raise ValueError(f"Unknown prediction source {source}")
    return cache[source]


def reused_result(variant: str, view: str, label_set: str) -> dict:
    """exp_3's nested result for a reference variant (same code path), read in place."""
    if variant not in REUSED_VARIANTS:
        raise ValueError(f"{variant} is not a reused reference variant")
    result = read_json(exp4_common.C3.RESULTS / f"variant_{variant}__{view}__{label_set}.json")
    return {**result, "reused_from": "exp_3/results"}


def evaluate(variant: str, view: str, label_set: str, cache: dict) -> dict:
    """Nested evaluation of one exp_4 variant (or the reused exp_3 result of a reference)."""
    if variant in REUSED_VARIANTS:
        return reused_result(variant, view, label_set)
    (source,) = VARIANTS[variant]
    result = P3.evaluate_predictions(predictions_for(source, cache), variant, view, label_set)
    return {**result, "source": source}


def result_path(variant: str, view: str, label_set: str):
    return exp4_common.RESULTS / f"variant_{variant}__{view}__{label_set}.json"


_WORKER_CACHE: dict = {}


def evaluate_and_save(task: tuple[str, str, str]) -> tuple[str, str, str, float, str]:
    """Worker entry: evaluate one (variant, view, label set) unless its result file exists."""
    variant, view, label_set = task
    path = result_path(variant, view, label_set)
    if path.exists():
        return variant, view, label_set, 0.0, "cached"
    start = time.time()
    result = evaluate(variant, view, label_set, _WORKER_CACHE)
    exp4_common.C1.write_json(path, result)
    return variant, view, label_set, time.time() - start, "computed"


def spec_table() -> list[dict]:
    return [{"arm": arm, **asdict(spec.aux), "touch": spec.touch, "seed": spec.seed,
             "fingerprint": fingerprint(arm)} for arm, spec in ARMS.items()]
