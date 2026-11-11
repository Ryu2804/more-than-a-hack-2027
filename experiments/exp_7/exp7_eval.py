"""exp_7 evaluation on IDSSE (never fits).

* Nested decoder selection (PREREG 3.4): exp_1's grid, temperature per selection, validation predictions of
  the **selection view** (= deployment view), the held-out match scored once on every view decoded with it.
* ``exp3_ref_ens3`` from exp_2's saved weights through exp_6's scoring path (exp_3's stored per-fold
  ``target_like`` selections): the binding recheck against exp_6's stored rows, and the AR(1) report rows.
* ``exp3_ref_ens3_msel`` (PREREG 0, report only): exp_2's saved pair + outer weights, nested selection on the
  measured view, to separate decoder re-selection from robustness training.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

import exp7_common as C  # noqa: I001  (first: environment, sys.path, safe exp_6 imports)
import exp1_decode as D
import exp1_net as N
import exp1_protocol as P1
import exp2_protocol as P2
import exp3_protocol as P3
import exp7_aug as A
import exp7_fit as F
from exp1_common import MATCH_IDS, read_json, write_json

S = C.S6
SELECTION_VIEW = {"target_like": "target_like", "measured": "measured", "stress": "measured", "ar1": "measured",
                  "clean25": "clean25", "clean50": "clean50"}
REF_MSEL = "exp3_ref_ens3_msel"
REF_MSEL_VIEWS = ("measured", "stress", "ar1")
SELECTIONS = {  # variant -> selection views evaluated (each on both label sets)
    C.PRIMARY: ("target_like", "measured", "clean25", "clean50"),
    **{arm: ("target_like", "measured") for arm in F.ARMS},
    REF_MSEL: ("measured",),
}
EXP6_SETTING = {"target_like": "target_like", "measured": "measured", "stress": "stress",
                "clean25": "voi_c0.25_s0_f0", "clean50": "voi_c0.5_s0_f0"}  # exp_6's names of the same views
ROW_TOLERANCE = 1e-12  # exp_6's rows went through a CSV


def score_views(selection_view: str) -> list[str]:
    return [view for view in A.EVAL_VIEWS if SELECTION_VIEW[view] == selection_view]


# ---------------------------------------------------------------- items and nested selection
def make_item(name: str, prediction: tuple, label_set: str) -> dict:
    """exp3_protocol.make_item on the shared 5 fps grid (every view has the original grid)."""
    grid = P2.load_tokens(name)
    probability, boundary = prediction
    if len(probability) != len(grid["indices"]):
        raise RuntimeError("Prediction is not aligned with the original grid")
    state, _ = P3.step_labels(name, label_set, grid)
    return {"probability": probability, "boundary": boundary, "half": grid["half"], "indices": grid["indices"],
            "state": state, "truth": P3.frame_labels(name, label_set),
            "full_half": P1.proxy_truth(name)["full_half"]}


def evaluate(predictions: dict, variant: str, selection_view: str, label_set: str,
             views: list[str] | None = None) -> dict:
    """Nested selection on ``selection_view``; each held-out match scored on every view it selects for."""
    views = views or score_views(selection_view)
    rows, selections = [], {}
    for held in MATCH_IDS:
        validation = [make_item(v, predictions[P1.pair_key(held, v)][selection_view][v], label_set)
                      for v in MATCH_IDS if v != held]
        config, _ = D.select_config(validation, D.GAMMAS)
        selections[held] = config.as_dict()
        for view in views:
            outer = make_item(held, predictions[f"outer_{held}"][view][held], label_set)
            rows.append({"match": held, "variant": variant, "view": view, "selection_view": selection_view,
                         "label_set": label_set, **D.score_item(outer, config), **config.as_dict()})
    oof = [make_item(n, predictions[f"outer_{n}"][selection_view][n], label_set) for n in MATCH_IDS]
    final, _ = D.select_config(oof, D.GAMMAS)
    return {"variant": variant, "selection_view": selection_view, "label_set": label_set, "views": views,
            "rows": rows, "selections": selections, "final_config": final.as_dict()}


# ---------------------------------------------------------------- prediction sources
def _reference_job(arm: str, key: str, names: list[str]) -> dict:
    """One exp_2 member's saved weights on exp_7's simulated views (flag dropped), cached in exp_7."""
    path = C.CACHE / "reference" / f"{arm}_{P2.fingerprint(arm)}" / f"{key}.npz"
    if not path.exists():
        config = P2.MODEL_ARMS[arm]["config"]
        model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
        model.load_state_dict(torch.load(P2.prediction_folder(arm) / f"{key}.pt", map_location=N.DEVICE))
        model.eval()
        arrays = {}
        for name in names:
            for view in REF_MSEL_VIEWS:
                p, b = N.predict(model, A.without_flag(A.view_item(name, view)), config.window)
                arrays[f"{view}__{name}_probability"], arrays[f"{view}__{name}_boundary"] = p, b
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    with np.load(path) as z:
        return {view: {n: (z[f"{view}__{n}_probability"], z[f"{view}__{n}_boundary"]) for n in names}
                for view in REF_MSEL_VIEWS}


def reference_predictions() -> dict:
    members = {arm: {key: _reference_job(arm, key, names) for key, _, names in P1.job_keys()}
               for arm in C.REF_MEMBERS}
    return P2.ensemble_predictions(members)


def predictions_for(source: str, cache: dict) -> dict:
    if source not in cache:
        if source == C.PRIMARY:
            cache[source] = P2.ensemble_predictions({arm: predictions_for(arm, cache) for arm in F.ARMS})
        elif source in F.ARMS:
            cache[source] = F.ensure_predictions(source, fit=False)
        elif source == REF_MSEL:
            cache[source] = reference_predictions()
        else:
            raise ValueError(f"Unknown prediction source {source}")
    return cache[source]


def ready(variant: str) -> bool:
    members = tuple(F.ARMS) if variant == C.PRIMARY else (variant,) if variant in F.ARMS else ()
    return all(not F.missing_jobs(arm) for arm in members)


# ---------------------------------------------------------------- results
def result_path(variant: str, selection_view: str, label_set: str):
    return C.RESULTS / f"variant_{variant}__sel_{selection_view}__{label_set}.json"


def tasks() -> list[tuple[str, str, str]]:
    return [(v, sel, label) for v, sels in SELECTIONS.items() for sel in sels for label in C.LABEL_SETS]


def evaluate_and_save(task: tuple[str, str, str], cache: dict) -> tuple[float, str]:
    variant, selection_view, label_set = task
    path = result_path(*task)
    if path.exists():
        return 0.0, "cached"
    start = time.time()
    members = list(F.ARMS) if variant == C.PRIMARY else list(C.REF_MEMBERS) if variant == REF_MSEL else [variant]
    result = evaluate(predictions_for(variant, cache), variant, selection_view, label_set)
    write_json(path, {**result, "members": members})
    return time.time() - start, "computed"


def evaluation_path_recheck(label_set: str) -> dict:
    """exp_7's ``evaluate`` on exp_2's cached exp3_ref_ens3 target_like predictions reproduces exp_3's rows."""
    predictions = P2.ensemble_predictions({arm: P3.cached_exp2(arm) for arm in C.REF_MEMBERS})
    mine = evaluate(predictions, C.REFERENCE, "target_like", label_set, ["target_like"])
    stored = read_json(C.exp3_result(label_set))
    keys = ("spell_f1", "macro_f1", "n_true", "n_pred", "count_error_ratio")
    equal = all(all(m[k] == s[k] for k in keys) for m, s in zip(mine["rows"], stored["rows"]))
    return {"label_set": label_set, "rows_equal": equal, "selections_equal": mine["selections"] == stored["selections"],
            "final_config_equal": mine["final_config"] == stored["final_config"]}


# ---------------------------------------------------------------- reference through exp_6's path
def reference_rows(views: tuple[str, ...], log=print) -> list[dict]:
    """exp3_ref_ens3 (saved outer weights) on exp_7's view tokens without the flag, decoded with exp_3's stored
    per-fold target_like selections (exp_6's scoring path, no re-selection)."""
    selections = {label: read_json(C.exp3_result(label))["selections"] for label in C.LABEL_SETS}
    window = N.TrainConfig().window
    rows = []
    for name in MATCH_IDS:
        models = [S.load_member(arm, name) for arm in C.REF_MEMBERS]
        for view in views:
            item = A.without_flag(A.view_item(name, view))
            prediction = S.ensemble(models, item, window)
            for label in C.LABEL_SETS:
                rows.append({"variant": C.REFERENCE, "view": view, "match": name, "label_set": label,
                             "selection_view": "exp3_target_like_stored",
                             **S.score_prediction(name, prediction, item, label, selections[label][name])})
            log(f"REF {view} {name}")
        del models
        torch.cuda.empty_cache()
    return rows


def recheck_against_exp6(rows: list[dict]) -> dict:
    """Rows through exp_7's path vs exp_6's stored reference rows (same views, seeds and run lengths)."""
    table = pd.read_csv(C.EXP6_ROWS)
    table = table[table["family"] == "reference"]
    out = {}
    for view, setting in EXP6_SETTING.items():
        mine = [r for r in rows if r["view"] == view]
        diffs = []
        for r in mine:
            ref = table[(table["setting"] == setting) & (table["match"] == r["match"])
                        & (table["label_set"] == r["label_set"])][list(S.ROW_KEYS)].drop_duplicates()
            if len(ref) != 1:  # exp_6 scored target_like twice (decision and VOI passes): identical rows
                raise RuntimeError(f"exp_6 row missing for {view} {r['match']} {r['label_set']}")
            diffs.append(max(abs(float(r[k]) - float(ref[k].iloc[0])) for k in S.ROW_KEYS))
        out[view] = {"rows": len(mine), "max_abs_diff": float(max(diffs)),
                     "equal": bool(len(mine) == 14 and max(diffs) <= ROW_TOLERANCE)}
    out["all_equal"] = all(v["equal"] for v in out.values())
    return out
