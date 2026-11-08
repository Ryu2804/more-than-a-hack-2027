"""Nested leave-one-match-out protocol shared by every arm.

For each held-out match h, decoder settings are chosen on the six other matches using
predictions from models that never saw h *or* the validation match v (one model per
unordered pair {h, v}); the outer model is trained on all six and scored on h once.
This mirrors exp_0's frozen protocol so arms are directly comparable.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib
import json
import time

import numpy as np

import exp1_arms as A
import exp1_decode as D
import exp1_net as N
import exp1_tokens as T
from exp1_common import (
    AUG_ROUNDS, JD, MATCH_IDS, PRED_DIR, ROOT, TOKEN_DIR, expand, load_degraded, score, sha256,
)

MODEL_ARMS = {
    "setseq": {"kind": "setseq", "config": N.TrainConfig(), "rounds": (0, *AUG_ROUNDS)},
    "setseq_noaug": {"kind": "setseq", "config": N.TrainConfig(mirror=False), "rounds": (0,)},
    "gbm": {"kind": "gbm", "rounds": (0,)},
}
DECODER_VARIANTS = {
    "setseq_bd": ("setseq", D.GAMMAS),
    "setseq_const": ("setseq", (0.0,)),
    "setseq_argmax": ("setseq", None),
    "setseq_noaug_bd": ("setseq_noaug", D.GAMMAS),
    "gbm_const": ("gbm", (0.0,)),
}
RULE_ARMS = ("baseline", "rasya_c06", "rasya_c08")
# Added after the frozen run selected gamma = 0 in every fold: forces gamma > 0 (still nested)
# to measure what boundary-aware decoding does. Diagnostic only, never a candidate.
DIAGNOSTIC_VARIANTS = {"setseq_bd_forced": ("setseq", (0.5, 1.0, 2.0))}


# ---------------------------------------------------------------- token cache
def token_path(name: str, round_index: int):
    return TOKEN_DIR / f"{name}_r{round_index}.npz"


def build_token_cache() -> list[dict]:
    """Tokens + 5 fps targets for every proxy match and corruption round (aggregates returned)."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MATCH_IDS:
        for round_index in (0, *AUG_ROUNDS):
            match = load_degraded(name, round_index)
            token_set = T.build_tokens(match)
            state, boundary = T.step_targets(match.labels, token_set)
            np.savez(token_path(name, round_index), tokens=token_set.tokens, mask=token_set.mask,
                     glob=token_set.glob, half=token_set.half, indices=token_set.indices,
                     state=state, boundary=boundary)
            rows.append({"match": name, "round": round_index, **JD.degradation_summary(match),
                         "steps_5fps": len(state), "tokens_per_step": float(token_set.mask.sum(1).mean()),
                         "boundary_rate": float(boundary.mean())})
    return rows


@lru_cache(maxsize=32)
def load_tokens(name: str, round_index: int = 0) -> dict:
    with np.load(token_path(name, round_index)) as z:
        return {key: z[key] for key in z.files}


@lru_cache(maxsize=8)
def proxy_truth(name: str) -> dict:
    match = load_degraded(name)
    return {"truth": match.labels, "full_half": match.half}


def build_gbm_features() -> None:
    for name in MATCH_IDS:
        if not (A.GBM_DIR / f"{name}.npz").exists():
            A.gbm_features(name, load_degraded(name))


# ---------------------------------------------------------------- predictions
def fingerprint(arm: str) -> str:
    spec = MODEL_ARMS[arm]
    payload = {
        "arm": arm, "rounds": spec["rounds"], "matches": MATCH_IDS,
        "config": spec["config"].as_dict() if "config" in spec else None,
        "code": {f: sha256(ROOT / f) for f in ("exp1_net.py", "exp1_tokens.py", "exp1_arms.py")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def fit_predict(arm: str, train_names: list[str], predict_items: dict) -> dict:
    """Train one model of ``arm`` on proxy matches; predict arbitrary token/feature items."""
    spec = MODEL_ARMS[arm]
    if spec["kind"] == "setseq":
        arrays = [load_tokens(n, r) for n in train_names for r in spec["rounds"]]
        model = N.train(arrays, spec["config"])
        return {k: N.predict(model, item, spec["config"].window) for k, item in predict_items.items()}
    model = A.gbm_fit([A.gbm_features(n)["x"] for n in train_names],
                      [load_tokens(n)["state"] for n in train_names])
    return {k: (A.gbm_predict(model, item["x"]), None) for k, item in predict_items.items()}


def _proxy_inputs(arm: str, names: list[str]) -> dict:
    if MODEL_ARMS[arm]["kind"] == "setseq":
        return {n: load_tokens(n) for n in names}
    return {n: A.gbm_features(n) for n in names}


def job_keys() -> list[tuple[str, list[str], list[str]]]:
    jobs = [(f"pair_{h}_{v}", [n for n in MATCH_IDS if n not in (h, v)], [h, v])
            for h, v in combinations(MATCH_IDS, 2)]
    jobs += [(f"outer_{h}", [n for n in MATCH_IDS if n != h], [h]) for h in MATCH_IDS]
    return jobs


def ensure_predictions(arm: str, log=print) -> dict:
    """Run (or reuse) every pair and outer fit for one model arm."""
    folder = PRED_DIR / f"{arm}_{fingerprint(arm)}"
    folder.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for key, train_names, predict_names in job_keys():
        path = folder / f"{key}.npz"
        if not path.exists():
            start = time.time()
            result = fit_predict(arm, train_names, _proxy_inputs(arm, predict_names))
            arrays = {f"{n}_probability": p for n, (p, _) in result.items()}
            arrays.update({f"{n}_boundary": b for n, (_, b) in result.items() if b is not None})
            np.savez(path, **arrays)
            log(f"FIT {arm} {key} {time.time() - start:.1f}s")
        with np.load(path) as z:
            predictions[key] = {
                n: (z[f"{n}_probability"], z[f"{n}_boundary"] if f"{n}_boundary" in z.files else None)
                for n in predict_names
            }
    return predictions


def pair_key(a: str, b: str) -> str:
    left, right = sorted((a, b), key=MATCH_IDS.index)
    return f"pair_{left}_{right}"


def make_item(name: str, prediction: tuple) -> dict:
    tokens = load_tokens(name)
    probability, boundary = prediction
    return {"probability": probability, "boundary": boundary, "half": tokens["half"],
            "indices": tokens["indices"], "state": tokens["state"], **proxy_truth(name)}


# ---------------------------------------------------------------- evaluation
def evaluate_variant(variant: str, predictions: dict) -> dict:
    """Nested evaluation of one decoder variant on cached model predictions."""
    _, gammas = {**DECODER_VARIANTS, **DIAGNOSTIC_VARIANTS}[variant]
    rows, selections = [], {}
    for held in MATCH_IDS:
        outer = make_item(held, predictions[f"outer_{held}"][held])
        if gammas is None:
            codes = outer["probability"].argmax(axis=1).astype("int8")
            result = score(outer["truth"], expand(codes, outer["indices"], outer["full_half"]))
            rows.append({"match": held, "variant": variant, **result})
            continue
        validation = [make_item(v, predictions[pair_key(held, v)][v]) for v in MATCH_IDS if v != held]
        config, _ = D.select_config(validation, gammas)
        selections[held] = config.as_dict()
        rows.append({"match": held, "variant": variant, **D.score_item(outer, config), **config.as_dict()})
    final = None
    if gammas is not None:
        oof = [make_item(n, predictions[f"outer_{n}"][n]) for n in MATCH_IDS]
        final = D.select_config(oof, gammas)[0].as_dict()
    return {"rows": rows, "selections": selections, "final_config": final}


def evaluate_rules() -> list[dict]:
    rows = []
    for name in MATCH_IDS:
        match = load_degraded(name)
        predictions = {
            "baseline": A.baseline_arm(match),
            "rasya_c06": A.rasya_arm(match, A.SLOW_RUN_C06),
            "rasya_c08": A.rasya_arm(match, A.SLOW_RUN_C08),
        }
        for arm, prediction in predictions.items():
            rows.append({"match": name, "variant": arm, **score(match.labels, prediction)})
    return rows


def evaluate_ryu() -> list[dict]:
    """ryu's shipped TCN ensemble + decoder, zero-shot on each frozen proxy match."""
    import exp1_ryu as R

    ensemble = R.load_ensemble(str(N.DEVICE))
    rows = []
    for name in MATCH_IDS:
        match = load_degraded(name)
        codes, info = R.ryu_arm(match, ensemble, str(N.DEVICE))
        rows.append({"match": name, "variant": "ryu_final", **score(match.labels, codes),
                     **{f"ryu_{k}": v for k, v in info.items()}})
    return rows
