"""exp_5 nested evaluation (IDSSE only; never fits).

Variants, prediction sources and the result files read in place. Selection is exp_3's
``evaluate_predictions``: temperature and decoder re-selected on the six inner validation matches
against the evaluation label set, the held-out match scored once with the official metric.
"""

from __future__ import annotations

import time

import exp5_common as C  # noqa: I001  (first: env, sys.path, exp_1-exp_4 imports)
import exp1_protocol as P1
import exp2_protocol as P2
import exp3_protocol as P3
import exp5_protocol as P
from exp1_common import read_json, write_json

EXP2_ARMS = ("exp2_aug", "exp2_aug_s1", "exp2_aug_s2")
EXP2_FINGERPRINTS = {arm: P2.fingerprint(arm) for arm in EXP2_ARMS}
ENSEMBLES = {  # per-step mean of p(H, A, D) and p(boundary) over members (exp2_protocol)
    "exp5_touch_ens3": tuple(P.ARMS),
    "exp4_touch_ens3": C.EXP4_TOUCH_ARMS,  # exp_4 cache, read in place
    "exp45_touch_ens6": C.EXP4_TOUCH_ARMS + tuple(P.ARMS),
    "ref_ens3": EXP2_ARMS,  # exp_2 cache, read in place (members of exp3_ref_ens3)
}
EVALUATED = {  # variant -> label sets evaluated by exp_5 (both views each)
    "exp5_touch_ens3": C.S1_LABELS,
    "exp45_touch_ens6": C.S1_LABELS,
    **{arm: C.DECISION_LABELS for arm in P.ARMS},
}
REUSED = {  # variant -> results folder (stored nested results, read in place)
    "exp3_ref_ens3": C.C3.RESULTS,
    "exp2_aug": C.C3.RESULTS,
    "exp4_touch_ens3": C.C4.RESULTS,
}
SOURCE_OF = {**{v: v for v in EVALUATED if v in ENSEMBLES}, **{arm: arm for arm in P.ARMS},
             "exp3_ref_ens3": "ref_ens3", "exp2_aug": "exp2_aug", "exp4_touch_ens3": "exp4_touch_ens3"}
NEVER_ADOPTED = ("exp45_touch_ens6", "exp4_touch_ens3", *P.ARMS)


# ---------------------------------------------------------------- prediction sources
def exp4_touch_predictions(arm: str) -> dict:
    """exp_4's cached touch predictions (SHA-256 frozen in protocol.json), read in place."""
    folder = C.exp4_touch_folder(arm)
    missing = [k for k, *_ in P1.job_keys() if not (folder / f"{k}.npz").exists()]
    if missing:
        raise RuntimeError(f"exp_4 cache incomplete for {arm}: {len(missing)} jobs missing")
    return {key: P3._load_job(folder / f"{key}.npz", names) for key, _, names in P1.job_keys()}


def predictions_for(source: str, cache: dict) -> dict:
    if source not in cache:
        if source in ENSEMBLES:
            members = {arm: predictions_for(arm, cache) for arm in ENSEMBLES[source]}
            cache[source] = P2.ensemble_predictions(members)
        elif source in P.ARMS:
            cache[source] = P.ensure_predictions(source, fit=False)
        elif source in C.EXP4_TOUCH_ARMS:
            cache[source] = exp4_touch_predictions(source)
        elif source in EXP2_ARMS:
            cache[source] = P3.cached_exp2(source)
        else:
            raise ValueError(f"Unknown prediction source {source}")
    return cache[source]


def members_of(variant: str) -> tuple[str, ...]:
    source = SOURCE_OF[variant]
    return ENSEMBLES.get(source, (source,))


def ready(variant: str) -> bool:
    """Every exp_5 member arm has all 28 fits (exp_2 / exp_4 members are cached)."""
    return all(not P.missing_jobs(arm) for arm in members_of(variant) if arm in P.ARMS)


# ---------------------------------------------------------------- results
def result_path(variant: str, view: str, label_set: str):
    folder = REUSED.get(variant, C.RESULTS)
    return folder / f"variant_{variant}__{view}__{label_set}.json"


def read_result(variant: str, view: str, label_set: str) -> dict:
    return read_json(result_path(variant, view, label_set))


def tasks() -> list[tuple[str, str, str]]:
    return [(v, view, ls) for v, labels in EVALUATED.items() for ls in labels for view in C.VIEWS]


def evaluate(variant: str, view: str, label_set: str, cache: dict) -> dict:
    """Nested evaluation of one exp_5 variant (fresh), from cached predictions only."""
    if variant not in EVALUATED or label_set not in EVALUATED[variant]:
        raise ValueError(f"{variant} on {label_set} is not a pre-registered exp_5 evaluation")
    source = SOURCE_OF[variant]
    result = P3.evaluate_predictions(predictions_for(source, cache), variant, view, label_set)
    return {**result, "source": source, "members": list(members_of(variant))}


_WORKER_CACHE: dict = {}


def evaluate_and_save(task: tuple[str, str, str]) -> tuple[str, str, str, float, str]:
    """Worker entry: evaluate one (variant, view, label set) unless its result file exists."""
    variant, view, label_set = task
    path = result_path(variant, view, label_set)
    if path.exists():
        return variant, view, label_set, 0.0, "cached"
    start = time.time()
    result = evaluate(variant, view, label_set, _WORKER_CACHE)
    write_json(path, result)
    return variant, view, label_set, time.time() - start, "computed"


def recheck(variant: str, view: str, label_set: str) -> dict:
    """Recompute a reused result (exp3_ref_ens3 / exp4_touch_ens3) from its cached predictions
    with exp_5's code path; it must equal the stored rows and final config exactly."""
    if variant not in REUSED:
        raise ValueError(f"{variant} is not a reused variant")
    start = time.time()
    fresh = P3.evaluate_predictions(predictions_for(SOURCE_OF[variant], {}), variant, view, label_set)
    stored = read_result(variant, view, label_set)
    rows = fresh["rows"]
    return {"variant": variant, "view": view, "label_set": label_set,
            "rows_equal": rows == stored["rows"], "final_config_equal": fresh["final_config"] == stored["final_config"],
            "mean_spell_f1": sum(r["spell_f1"] for r in rows) / len(rows), "seconds": round(time.time() - start, 1)}
