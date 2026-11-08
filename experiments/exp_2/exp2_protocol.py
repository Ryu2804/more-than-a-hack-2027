"""exp_2 nested protocol: token caches for both views, model arms, ensembles, evaluation.

Identical nesting to exp_1 (``exp1_protocol.job_keys``): 21 pair models give inner
validation predictions, 7 outer models give the held-out predictions. Every fit predicts
*both* views of its held-out matches with the same weights, so view deltas are paired.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
import time
import warnings

import numpy as np
import torch

import exp1_arms as A
import exp1_common as C1
import exp1_decode as D
import exp1_net as N
import exp1_protocol as P1
import exp1_tokens as T
import exp2_hsmm as H
import exp2_views as V
from exp1_common import expand, load_degraded, score, sha256
from exp2_common import BASE_SEED, EXP1, MATCH_IDS, PRED_DIR, ROOT, TOKEN_DIR, VIEWS

RECIPES = {
    "original": ((0, "original"), (1, "original"), (2, "original")),
    "target_aug": ((0, "original"), (1, "target_aug"), (2, "target_aug")),
}
MODEL_ARMS = {
    "exp1_ref": {"recipe": "original", "config": N.TrainConfig()},
    "exp2_aug": {"recipe": "target_aug", "config": N.TrainConfig()},
    "exp2_aug_s1": {"recipe": "target_aug", "config": replace(N.TrainConfig(), seed=BASE_SEED + 1)},
    "exp2_aug_s2": {"recipe": "target_aug", "config": replace(N.TrainConfig(), seed=BASE_SEED + 2)},
}
ENSEMBLE = ("exp2_aug", "exp2_aug_s1", "exp2_aug_s2")
VARIANTS = {  # variant -> (prediction source, decoder)
    "exp1_ref": ("exp1_ref", "bd"),
    "exp2_aug": ("exp2_aug", "bd"),
    "exp2_aug_const": ("exp2_aug", "const"),
    "exp2_aug_hsmm": ("exp2_aug", "hsmm"),
    "exp2_aug_s1": ("exp2_aug_s1", "bd"),
    "exp2_aug_s2": ("exp2_aug_s2", "bd"),
    "exp2_aug_ens3": ("ens3", "bd"),
}
REFERENCE_ARMS = ("baseline", "rasya_c06", "rasya_c08", "ryu_final")
FIT_CODE = ("exp1_net.py", "exp1_tokens.py")


# ---------------------------------------------------------------- token caches
def token_path(name: str, kind: str, round_index: int = 0):
    return TOKEN_DIR / f"{name}_{kind}_r{round_index}.npz"


def _save_tokens(path, match) -> dict:
    token_set = T.build_tokens(match)
    state, boundary = T.step_targets(match.labels, token_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, tokens=token_set.tokens, mask=token_set.mask, glob=token_set.glob,
             half=token_set.half, indices=token_set.indices, state=state, boundary=boundary)
    os.replace(tmp, path)
    return {"steps_5fps": len(state), "tokens_per_step": float(token_set.mask.sum(1).mean())}


def view_stats(match) -> dict:
    """Aggregate ball statistics of one IDSSE match (graded frames; labels only describe)."""
    graded = match.half > 0
    visible = V.ball_visible(match)
    outside = V.outside_mask(match)
    dead, alive = graded & (match.labels == 2), graded & (match.labels < 2)
    return {
        "ball_visible": float(visible[graded].mean()),
        "outside_of_visible": float(outside[graded].sum() / max(1, visible[graded].sum())),
        "visible_on_dead": float(visible[dead].mean()),
        "visible_on_alive": float(visible[alive].mean()),
        "outside_of_visible_dead": float(outside[dead].sum() / max(1, visible[dead].sum())),
    }


def build_token_caches() -> list[dict]:
    """target_like round 0 (held-out view) and target_aug rounds 1-2 (training) per match."""
    rows = []
    for name in MATCH_IDS:
        base = load_degraded(name, 0)
        rows.append({"match": name, "kind": "original", "round": 0, **view_stats(base)})
        view = V.target_like(base)
        path = token_path(name, "target_like", 0)
        extra = _save_tokens(path, view) if not path.exists() else {}
        rows.append({"match": name, "kind": "target_like", "round": 0, **view_stats(view), **extra})
        for round_index in V.AUG_ROUNDS:
            match = load_degraded(name, round_index)
            augmented = V.target_aug(match, np.random.default_rng(V.aug_seed(name, round_index)))
            path = token_path(name, "target_aug", round_index)
            extra = _save_tokens(path, augmented) if not path.exists() else {}
            rows.append({"match": name, "kind": "target_aug", "round": round_index,
                         **view_stats(augmented), **extra})
    return rows


@lru_cache(maxsize=48)
def load_tokens(name: str, kind: str = "original", round_index: int = 0) -> dict:
    if kind == "original":
        return P1.load_tokens(name, round_index)  # exp_1 cache, read in place
    with np.load(token_path(name, kind, round_index)) as z:
        return {key: z[key] for key in z.files}


def view_tokens(name: str, view: str) -> dict:
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view}")
    return load_tokens(name, "original" if view == "original" else "target_like", 0)


# ---------------------------------------------------------------- fits
def fingerprint(arm: str) -> str:
    spec = MODEL_ARMS[arm]
    code = {f: sha256(EXP1 / f) for f in FIT_CODE}
    code["exp2_views.py"] = sha256(ROOT / "exp2_views.py")
    payload = {"arm": arm, "recipe": RECIPES[spec["recipe"]], "config": spec["config"].as_dict(),
               "matches": MATCH_IDS, "views": VIEWS, "code": code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def train_arrays(arm: str, names: list[str]) -> list[dict]:
    recipe = RECIPES[MODEL_ARMS[arm]["recipe"]]
    return [load_tokens(n, kind, r) for n in names for r, kind in recipe]


def fit_model(arm: str, names: list[str]) -> N.SetSeq:
    return N.train(train_arrays(arm, names), MODEL_ARMS[arm]["config"])


def _fit_job(arm: str, train_names: list[str], predict_names: list[str]) -> tuple[dict, dict]:
    model = fit_model(arm, train_names)
    window = MODEL_ARMS[arm]["config"].window
    arrays = {}
    for name in predict_names:
        for view in VIEWS:
            probability, boundary = N.predict(model, view_tokens(name, view), window)
            arrays[f"{view}__{name}_probability"] = probability
            arrays[f"{view}__{name}_boundary"] = boundary
    return arrays, {k: v.detach().cpu() for k, v in model.state_dict().items()}


def prediction_folder(arm: str):
    return PRED_DIR / f"{arm}_{fingerprint(arm)}"


def ensure_predictions(arm: str, log=print) -> dict:
    """Run (or reuse) the 28 pair/outer fits of one arm -> {job: {view: {match: (p, b)}}}."""
    folder = prediction_folder(arm)
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
            log(f"FIT {arm} {key} {time.time() - start:.1f}s", flush=True)
        with np.load(path) as z:
            predictions[key] = {view: {n: (z[f"{view}__{n}_probability"], z[f"{view}__{n}_boundary"])
                                       for n in predict_names} for view in VIEWS}
    return predictions


def cached_fit_count() -> int:
    return sum(1 for _ in PRED_DIR.glob("*/*.pt")) if PRED_DIR.exists() else 0


def ensemble_predictions(members: dict) -> dict:
    """Mean p(H,A,D) and mean p(boundary) over seed arms, per job, view and match."""
    first = next(iter(members.values()))
    out = {}
    for key, views in first.items():
        out[key] = {view: {name: tuple(
            np.mean([members[arm][key][view][name][i] for arm in members], axis=0).astype("float32")
            for i in (0, 1)) for name in names} for view, names in views.items()}
    return out


def predictions_for(source: str, cache: dict) -> dict:
    if source not in cache:
        cache[source] = (ensemble_predictions({arm: predictions_for(arm, cache) for arm in ENSEMBLE})
                         if source == "ens3" else ensure_predictions(source))
    return cache[source]


def reproduction_check(predictions: dict) -> dict:
    """exp1_ref re-fit vs the cached exp_1 SetSeq predictions (original view, outer folds)."""
    folder = EXP1 / "cache" / "predictions" / f"setseq_{P1.fingerprint('setseq')}"
    if not folder.is_dir():
        raise RuntimeError("Cached exp_1 SetSeq predictions not found; refusing to refit exp_1")
    diffs, flips = [], []
    for name in MATCH_IDS:
        with np.load(folder / f"outer_{name}.npz") as z:
            cached = z[f"{name}_probability"]
        refit = predictions[f"outer_{name}"]["original"][name][0]
        diffs.append(float(np.abs(cached - refit).max()))
        flips.append(float(np.mean(cached.argmax(1) != refit.argmax(1))))
    return {"max_abs_probability_diff": max(diffs), "per_match_max_abs_diff": dict(zip(MATCH_IDS, diffs)),
            "argmax_disagreement_share": dict(zip(MATCH_IDS, flips)),
            "bit_exact": all(d == 0.0 for d in diffs)}


# ---------------------------------------------------------------- evaluation
@lru_cache(maxsize=64)
def prior_for(names: tuple[str, ...]) -> H.DurationPrior:
    return H.estimate_prior([(load_tokens(n)["state"], load_tokens(n)["half"]) for n in names])


def make_item(name: str, prediction: tuple, view: str, prior_names: tuple[str, ...]) -> dict:
    tokens = view_tokens(name, view)
    probability, boundary = prediction
    if not np.array_equal(tokens["indices"], load_tokens(name)["indices"]):
        raise RuntimeError("View tokens are not aligned with the original grid")
    return {"probability": probability, "boundary": boundary, "half": tokens["half"],
            "indices": tokens["indices"], "state": tokens["state"], "prior": prior_for(prior_names),
            **P1.proxy_truth(name)}


def others(*excluded: str) -> tuple[str, ...]:
    return tuple(n for n in MATCH_IDS if n not in excluded)


def select(decoder: str, validation: list[dict]):
    if decoder == "hsmm":
        return H.select_config(validation)
    return D.select_config(validation, D.GAMMAS if decoder == "bd" else (0.0,))[0]


def score_with(decoder: str, item: dict, config) -> dict:
    return H.score_item(item, config) if decoder == "hsmm" else D.score_item(item, config)


def evaluate_variant(variant: str, view: str, cache: dict) -> dict:
    """Nested evaluation of one variant on one view; final config from the 7 out-of-fold items."""
    source, decoder = VARIANTS[variant]
    predictions = predictions_for(source, cache)
    rows, selections = [], {}
    for held in MATCH_IDS:
        validation = [make_item(v, predictions[P1.pair_key(held, v)][view][v], view, others(held, v))
                      for v in MATCH_IDS if v != held]
        config = select(decoder, validation)
        selections[held] = config.as_dict()
        outer = make_item(held, predictions[f"outer_{held}"][view][held], view, others(held))
        rows.append({"match": held, "variant": variant, "view": view,
                     **score_with(decoder, outer, config), **config.as_dict()})
    oof = [make_item(n, predictions[f"outer_{n}"][view][n], view, others(n)) for n in MATCH_IDS]
    return {"rows": rows, "selections": selections, "final_config": select(decoder, oof).as_dict(),
            "decoder": decoder, "source": source}


def evaluate_argmax(source: str, view: str, cache: dict) -> list[dict]:
    """Diagnostic: no decoder, so model under-segmentation is separated from decoder effects."""
    predictions = predictions_for(source, cache)
    rows = []
    for held in MATCH_IDS:
        item = make_item(held, predictions[f"outer_{held}"][view][held], view, others(held))
        codes = item["probability"].argmax(axis=1).astype("int8")
        rows.append({"match": held, "variant": f"{source}_argmax", "view": view,
                     **score(item["truth"], expand(codes, item["indices"], item["full_half"]))})
    return rows


def evaluate_reference(view: str) -> list[dict]:
    """baseline, rasya C06/C08 and ryu's shipped pipeline (zero-shot) on one view."""
    import exp1_ryu as R

    ensemble = R.load_ensemble(str(N.DEVICE))
    rows = []
    for name in MATCH_IDS:
        match = load_degraded(name, 0)
        match = V.target_like(match) if view == "target_like" else match
        predictions = {"baseline": A.baseline_arm(match),
                       "rasya_c06": A.rasya_arm(match, A.SLOW_RUN_C06),
                       "rasya_c08": A.rasya_arm(match, A.SLOW_RUN_C08)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions["ryu_final"] = R.ryu_arm(match, ensemble, str(N.DEVICE))[0]
        for arm, prediction in predictions.items():
            rows.append({"match": name, "variant": arm, "view": view, **score(match.labels, prediction)})
    return rows


def view_integrity() -> list[dict]:
    """Frozen proxy still bit-exact (exp_1 check) and the view only removes ball observations."""
    rows = C1.verify_frozen_proxy()
    for row in rows:
        base = load_degraded(row["match"], 0)
        view = V.target_like(base)
        row["view_labels_unchanged"] = bool(np.array_equal(view.labels, base.labels))
        row["view_players_unchanged"] = bool(np.array_equal(view.players, base.players, equal_nan=True))
        same = np.isnan(view.ball) | (view.ball == base.ball)
        row["view_only_removes_ball"] = bool(same.all())
    return rows
