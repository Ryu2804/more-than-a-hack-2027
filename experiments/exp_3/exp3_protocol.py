"""exp_3 nested protocol: label sets, L_ctrl-trained arms, cached exp_2 arms, evaluation.

Nesting is exp_1's (``exp1_protocol.job_keys``): 21 pair models give inner validation
predictions, 7 outer models the held-out ones. Every analysis has an *evaluation label set*:
temperature, decoder selection and the held-out score all use it. exp_2's cached predictions
are read in place and never re-fitted (a missing file is an error, not a fit).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import NamedTuple
import hashlib
import json
import os
import time

import numpy as np
import torch

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
import exp1_decode as D
import exp1_net as N
import exp1_protocol as P1
import exp1_tokens as T
import exp2_protocol as P2
import exp3_coverage as V3
import exp3_labels as L
from exp1_common import expand, load_clean, load_degraded, score, sha256
from exp3_common import BASE_SEED, EXP1, EXP2, MATCH_IDS, PRED_DIR, ROOT, TOKEN_DIR, VIEWS

# Pre-registered fallback (PREREG 2.3): the event sync failed its acceptance criterion.
EVENTS_USED = False
TRAIN_LABEL = "ctrl"
RECIPE = P2.RECIPES["target_aug"]  # round 0 original + rounds 1-2 target_aug
MODEL_ARMS = {
    "exp3_ctrl": N.TrainConfig(),
    "exp3_ctrl_s1": replace(N.TrainConfig(), seed=BASE_SEED + 1),
    "exp3_ctrl_s2": replace(N.TrainConfig(), seed=BASE_SEED + 2),
}
EXP2_ARMS = ("exp1_ref", "exp2_aug", "exp2_aug_s1", "exp2_aug_s2")
EXP2_FINGERPRINTS = {arm: P2.fingerprint(arm) for arm in EXP2_ARMS}
ENSEMBLES = {"ref_ens3": ("exp2_aug", "exp2_aug_s1", "exp2_aug_s2"),
             "ctrl_ens3": ("exp3_ctrl", "exp3_ctrl_s1", "exp3_ctrl_s2")}
VARIANTS = {  # variant -> prediction source (decoder: exp_1 main, D.GAMMAS, for all)
    "exp3_ctrl_ens3": ("ctrl_ens3",),
    "exp3_ref_ens3": ("ref_ens3",),
    "exp3_ctrl": ("exp3_ctrl",),
    "exp3_ctrl_s1": ("exp3_ctrl_s1",),
    "exp3_ctrl_s2": ("exp3_ctrl_s2",),
    "exp2_aug": ("exp2_aug",),
    "exp1_ref": ("exp1_ref",),
}
FIT_CODE = ("exp1_net.py", "exp1_tokens.py")


class Grid(NamedTuple):
    """The two TokenSet fields ``exp1_tokens.step_targets`` reads."""
    indices: np.ndarray
    half: np.ndarray


# ---------------------------------------------------------------- labels
@lru_cache(maxsize=8)
def control_events(name: str) -> tuple:
    return L.match_events(name, load_clean(name))


@lru_cache(maxsize=64)
def frame_labels(name: str, label_set: str) -> np.ndarray:
    """25 fps labels of one proxy match under one label set (read-only array)."""
    truth = P1.proxy_truth(name)
    provider, half = truth["truth"], truth["full_half"]
    if label_set == "provider":
        out = provider.copy()
    elif label_set.startswith("track_"):
        out = L.l_track(provider, half, float(label_set.split("_")[1]))
    elif label_set == "ctrl":
        if EVENTS_USED:
            events, _ = control_events(name)
            out = L.l_ctrl(provider, half, events.index, events.team, use_events=True)
        else:
            out = L.l_ctrl(provider, half, np.zeros(0, "int64"), np.zeros(0, "int64"), use_events=False)
    else:
        raise ValueError(f"Unknown label set {label_set}")
    out.flags.writeable = False
    return out


def step_labels(name: str, label_set: str, tokens: dict) -> tuple[np.ndarray, np.ndarray]:
    """5 fps state and boundary targets of a label set on a token grid."""
    return T.step_targets(frame_labels(name, label_set), Grid(tokens["indices"], tokens["half"]))


def label_table() -> list[dict]:
    """Label-only statistics per match and label set (no model involved)."""
    rows = []
    for name in MATCH_IDS:
        truth = P1.proxy_truth(name)
        provider, half = truth["truth"], truth["full_half"]
        ctrl = frame_labels(name, "ctrl")
        for label_set in ("provider", "track_1", "track_2", "track_3", "ctrl"):
            labels = frame_labels(name, label_set)
            rows.append({"match": name, "label_set": label_set, **L.label_stats(labels, half, provider),
                         "agreement_with_ctrl": L.agreement(labels, ctrl, half)})
    return rows


def label_integrity() -> list[dict]:
    """D frames and frames outside halves untouched; L_ctrl == L_track(2 s) under the fallback."""
    rows = []
    for name in MATCH_IDS:
        truth = P1.proxy_truth(name)
        provider, half = truth["truth"], truth["full_half"]
        row = {"match": name}
        for label_set in ("ctrl", "track_1", "track_2", "track_3"):
            labels = frame_labels(name, label_set)
            row[f"{label_set}_dead_untouched"] = bool(np.array_equal(labels == 2, provider == 2))
            row[f"{label_set}_outside_halves_untouched"] = bool(
                np.array_equal(labels[half <= 0], provider[half <= 0]))
        row["ctrl_equals_track_2"] = bool(np.array_equal(frame_labels(name, "ctrl"),
                                                         frame_labels(name, "track_2")))
        row["degraded_labels_equal_clean"] = bool(np.array_equal(provider, load_clean(name).labels))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- fits
def fingerprint(arm: str) -> str:
    code = {f: sha256(EXP1 / f) for f in FIT_CODE}
    code["exp2_views.py"] = sha256(EXP2 / "exp2_views.py")
    code["exp3_labels.py"] = sha256(ROOT / "exp3_labels.py")
    payload = {"arm": arm, "recipe": RECIPE, "config": MODEL_ARMS[arm].as_dict(), "matches": MATCH_IDS,
               "views": VIEWS, "train_label": TRAIN_LABEL, "events_used": EVENTS_USED, "code": code}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def prediction_folder(arm: str):
    return PRED_DIR / f"{arm}_{fingerprint(arm)}"


def train_arrays(names: list[str]) -> list[dict]:
    """exp2_aug recipe inputs (cached tokens) with L_ctrl state/boundary targets."""
    arrays = []
    for name in names:
        for round_index, kind in RECIPE:
            tokens = P2.load_tokens(name, kind, round_index)
            state, boundary = step_labels(name, TRAIN_LABEL, tokens)
            arrays.append({**tokens, "state": state, "boundary": boundary})
    return arrays


def fit_model(arm: str, names: list[str]) -> N.SetSeq:
    return N.train(train_arrays(names), MODEL_ARMS[arm])


def _fit_job(arm: str, train_names: list[str], predict_names: list[str]) -> tuple[dict, dict]:
    model = fit_model(arm, train_names)
    window = MODEL_ARMS[arm].window
    arrays = {}
    for name in predict_names:
        for view in VIEWS:
            probability, boundary = N.predict(model, P2.view_tokens(name, view), window)
            arrays[f"{view}__{name}_probability"] = probability
            arrays[f"{view}__{name}_boundary"] = boundary
    return arrays, {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _load_job(path, predict_names: list[str]) -> dict:
    with np.load(path) as z:
        return {view: {n: (z[f"{view}__{n}_probability"], z[f"{view}__{n}_boundary"])
                       for n in predict_names} for view in VIEWS}


def ensure_predictions(arm: str, log=print, fit: bool = True) -> dict:
    """Run (or reuse) the 28 pair/outer fits of one exp_3 arm -> {job: {view: {match: (p, b)}}}.
    With ``fit=False`` a missing job is an error (evaluation never starts a fit)."""
    folder = prediction_folder(arm)
    folder.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for key, train_names, predict_names in P1.job_keys():
        path = folder / f"{key}.npz"
        if not path.exists() and not fit:
            raise RuntimeError(f"{arm} {key} not fitted yet; run 02_fits first")
        if not path.exists():
            start = time.time()
            arrays, weights = _fit_job(arm, train_names, predict_names)
            torch.save(weights, folder / f"{key}.pt")
            tmp = folder / f"{key}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            log(f"FIT {arm} {key} {time.time() - start:.1f}s")
        predictions[key] = _load_job(path, predict_names)
    return predictions


def cached_exp2(arm: str) -> dict:
    """exp_2 predictions read in place; refuses (never fits) if anything is missing."""
    if P2.fingerprint(arm) != EXP2_FINGERPRINTS[arm]:
        raise RuntimeError(f"exp_2 fingerprint changed for {arm}")
    folder = P2.prediction_folder(arm)
    missing = [k for k, *_ in P1.job_keys() if not (folder / f"{k}.npz").exists()]
    if missing:
        raise RuntimeError(f"exp_2 cache incomplete for {arm}: {len(missing)} jobs missing")
    return {key: _load_job(folder / f"{key}.npz", names) for key, _, names in P1.job_keys()}


def exp3_fit_count() -> int:
    return sum(1 for _ in PRED_DIR.glob("*/*.pt")) if PRED_DIR.exists() else 0


def predictions_for(source: str, cache: dict) -> dict:
    if source not in cache:
        if source in ENSEMBLES:
            members = {arm: predictions_for(arm, cache) for arm in ENSEMBLES[source]}
            cache[source] = P2.ensemble_predictions(members)
        elif source in MODEL_ARMS:
            cache[source] = ensure_predictions(source, fit=False)
        elif source in EXP2_ARMS:
            cache[source] = cached_exp2(source)
        else:
            raise ValueError(f"Unknown prediction source {source}")
    return cache[source]


# ---------------------------------------------------------------- evaluation
def view_tokens(name: str, view: str) -> dict:
    if view == "coverage":
        return load_coverage_tokens(name)
    return P2.view_tokens(name, view)


def make_item(name: str, prediction: tuple, view: str, label_set: str) -> dict:
    tokens = view_tokens(name, view)
    if not np.array_equal(tokens["indices"], P2.load_tokens(name)["indices"]):
        raise RuntimeError("View tokens are not aligned with the original grid")
    probability, boundary = prediction
    state, _ = step_labels(name, label_set, tokens)
    return {"probability": probability, "boundary": boundary, "half": tokens["half"],
            "indices": tokens["indices"], "state": state, "truth": frame_labels(name, label_set),
            "full_half": P1.proxy_truth(name)["full_half"]}


def evaluate_variant(variant: str, view: str, label_set: str, cache: dict) -> dict:
    """Nested evaluation of one variant on one view against one label set (selection on it too)."""
    (source,) = VARIANTS[variant]
    return {**evaluate_predictions(predictions_for(source, cache), variant, view, label_set), "source": source}


def evaluate_predictions(predictions: dict, variant: str, view: str, label_set: str) -> dict:
    """Nested selection + held-out scores for any {job: {view: {match: (p, b)}}} predictions."""
    rows, selections = [], {}
    for held in MATCH_IDS:
        validation = [make_item(v, predictions[P1.pair_key(held, v)][view][v], view, label_set)
                      for v in MATCH_IDS if v != held]
        config, _ = D.select_config(validation, D.GAMMAS)
        selections[held] = config.as_dict()
        outer = make_item(held, predictions[f"outer_{held}"][view][held], view, label_set)
        rows.append({"match": held, "variant": variant, "view": view, "label_set": label_set,
                     **D.score_item(outer, config), **config.as_dict()})
    oof = [make_item(n, predictions[f"outer_{n}"][view][n], view, label_set) for n in MATCH_IDS]
    final, _ = D.select_config(oof, D.GAMMAS)
    return {"rows": rows, "selections": selections, "final_config": final.as_dict()}


def argmax_rows(variant: str, view: str, label_set: str, cache: dict) -> list[dict]:
    """Diagnostic: no decoder."""
    (source,) = VARIANTS[variant]
    predictions = predictions_for(source, cache)
    rows = []
    for held in MATCH_IDS:
        item = make_item(held, predictions[f"outer_{held}"][view][held], view, label_set)
        codes = item["probability"].argmax(axis=1).astype("int8")
        rows.append({"match": held, "variant": f"{variant}_argmax", "view": view, "label_set": label_set,
                     **score(item["truth"], expand(codes, item["indices"], item["full_half"]))})
    return rows


def reference_predictions(view: str) -> dict:
    """baseline, rasya C06/C08 and ryu (zero-shot) 25 fps predictions on one view, per match."""
    import warnings

    import exp1_arms as A
    import exp1_ryu as R
    import exp2_views as V

    ensemble = R.load_ensemble(str(N.DEVICE))
    out = {}
    for name in MATCH_IDS:
        match = load_degraded(name, 0)
        match = V.target_like(match) if view == "target_like" else match
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ryu = R.ryu_arm(match, ensemble, str(N.DEVICE))[0]
        out[name] = {"baseline": A.baseline_arm(match), "rasya_c06": A.rasya_arm(match, A.SLOW_RUN_C06),
                     "rasya_c08": A.rasya_arm(match, A.SLOW_RUN_C08), "ryu_final": ryu}
    return out


def score_reference(predictions: dict, view: str, label_sets) -> list[dict]:
    return [{"match": name, "variant": arm, "view": view, "label_set": label_set,
             **score(frame_labels(name, label_set), codes)}
            for name, arms in predictions.items() for arm, codes in arms.items() for label_set in label_sets]


# ---------------------------------------------------------------- S2 coverage view
def coverage_path(name: str):
    return TOKEN_DIR / f"{name}_coverage_r0.npz"


def build_coverage_tokens() -> list[dict]:
    rows = []
    for name in MATCH_IDS:
        view, stats = V3.coverage_view(load_degraded(name, 0), load_clean(name).ball,
                                       np.random.default_rng(V3.reveal_seed(name)))
        path = coverage_path(name)
        if not path.exists():
            token_set = T.build_tokens(view)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, tokens=token_set.tokens, mask=token_set.mask, glob=token_set.glob,
                     half=token_set.half, indices=token_set.indices)
            os.replace(tmp, path)
        rows.append({"match": name, **stats})
    return rows


@lru_cache(maxsize=8)
def load_coverage_tokens(name: str) -> dict:
    with np.load(coverage_path(name)) as z:
        return {key: z[key] for key in z.files}


def _member_model(arm: str, key: str) -> tuple[N.SetSeq, int]:
    config = MODEL_ARMS[arm] if arm in MODEL_ARMS else P2.MODEL_ARMS[arm]["config"]
    folder = prediction_folder(arm) if arm in MODEL_ARMS else P2.prediction_folder(arm)
    model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
    model.load_state_dict(torch.load(folder / f"{key}.pt", map_location=N.DEVICE))
    return model.eval(), config.window


def coverage_outer_predictions(source: str) -> dict:
    """Outer-fold predictions of the coverage view from saved weights (no training)."""
    members = ENSEMBLES.get(source, (source,))
    out = {}
    for held in MATCH_IDS:
        parts = [N.predict(*_swap(_member_model(arm, f"outer_{held}"), load_coverage_tokens(held)))
                 for arm in members]
        out[held] = tuple(np.mean([p[i] for p in parts], axis=0).astype("float32") for i in (0, 1))
    return out


def _swap(model_window: tuple, tokens: dict) -> tuple:
    model, window = model_window
    return model, tokens, window


def evaluate_coverage(variant: str, label_set: str, selections: dict) -> list[dict]:
    """Coverage view decoded with each fold's target_like selection (only the input changes)."""
    (source,) = VARIANTS[variant]
    predictions = coverage_outer_predictions(source)
    rows = []
    for held in MATCH_IDS:
        item = make_item(held, predictions[held], "coverage", label_set)
        rows.append({"match": held, "variant": variant, "view": "coverage", "label_set": label_set,
                     **D.score_item(item, D.DecoderConfig(**selections[held]))})
    return rows
