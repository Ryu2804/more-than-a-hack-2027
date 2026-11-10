"""Report-only diagnostics (IDSSE only): argmax scores, recall of short possessions and the
generalisation gap (training NLL vs held-out NLL) of every fit.

Nothing here fits a model. The exp_2 reference fits' training NLL is computed from their saved
weights (inference only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import exp4_common  # noqa: F401,I001  (first: puts ../exp_3, ../exp_2, ../exp_1 on sys.path)
import exp1_decode as D
import exp1_net as N
import exp1_protocol as P1
import exp2_protocol as P2
import exp3_diagnostics as G3
import exp3_protocol as P3
import exp4_net as X
import exp4_protocol as P
from exp1_common import expand, read_json, score
from exp4_common import C3, GAP_GROWTH_FLAG, MATCH_IDS, RESULTS, VIEWS

SHORT_S = 2.0
FPS = 25
SOURCE_OF = {**{v: s[0] for v, s in P.VARIANTS.items()}, "exp3_ref_ens3": "ref_ens3", "exp2_aug": "exp2_aug"}
REFERENCE_FAMILY = "exp2_recipe"  # exp2_aug, _s1, _s2 (the members of exp3_ref_ens3)
FAMILY_OF = {**{arm: arm.removesuffix("_s1").removesuffix("_s2") for arm in P.ARMS},
             **{arm: REFERENCE_FAMILY for arm in P.EXP2_ARMS}}


def _result(variant: str, view: str, label_set: str) -> dict:
    folder = C3.RESULTS if variant in P.REUSED_VARIANTS else RESULTS
    return read_json(folder / f"variant_{variant}__{view}__{label_set}.json")


# ---------------------------------------------------------------- argmax and short possessions
def argmax_rows(variant: str, view: str, label_set: str, cache: dict) -> list[dict]:
    """No decoder: per-step argmax of the held-out p(H, A, D)."""
    predictions = P.predictions_for(SOURCE_OF[variant], cache)
    rows = []
    for held in MATCH_IDS:
        item = P3.make_item(held, predictions[f"outer_{held}"][view][held], view, label_set)
        codes = item["probability"].argmax(axis=1).astype("int8")
        rows.append({"match": held, "variant": f"{variant}_argmax", "view": view, "label_set": label_set,
                     **score(item["truth"], expand(codes, item["indices"], item["full_half"]))})
    return rows


def short_recall_rows(variant: str, view: str, cache: dict, label_set: str = "provider") -> list[dict]:
    """Per held-out match: recall (IoU 0.5) of true possessions < 2 s and of all possessions,
    decoded with that fold's own selected config, and without decoder (argmax)."""
    predictions = P.predictions_for(SOURCE_OF[variant], cache)
    selections = _result(variant, view, label_set)["selections"]
    rows = []
    for held in MATCH_IDS:
        probability, boundary = predictions[f"outer_{held}"][view][held]
        item = P3.make_item(held, (probability, boundary), view, label_set)
        codes = D.decode(probability, boundary, item["half"], D.DecoderConfig(**selections[held]))
        spells, decoded = G3._matched(codes, item)
        _, argmax = G3._matched(probability.argmax(axis=1), item)
        decoded, argmax = np.asarray(decoded, dtype=bool), np.asarray(argmax, dtype=bool)
        seconds = np.array([(stop - start) / FPS for start, stop, _ in spells])
        short = seconds < SHORT_S
        rows.append({"match": held, "variant": variant, "view": view, "label_set": label_set,
                     "possessions": int(len(spells)), "short_possessions": int(short.sum()),
                     "recall_all_decoded": float(decoded.mean()), "recall_all_argmax": float(argmax.mean()),
                     "recall_short_decoded": float(decoded[short].mean()) if short.any() else np.nan,
                     "recall_short_argmax": float(argmax[short].mean()) if short.any() else np.nan})
    return rows


# ---------------------------------------------------------------- generalisation gap
def _nll(probability: np.ndarray, state: np.ndarray) -> tuple[float, int]:
    keep = state < 3
    picked = probability[np.flatnonzero(keep), state[keep].astype("int64")].astype("float64")
    return float(-np.log(np.clip(picked, 1e-7, 1.0)).mean()), int(keep.sum())


def heldout_nll_rows(arm: str, cache: dict) -> list[dict]:
    """Raw (T = 1) NLL of every outer and pair prediction of one fitted arm, provider labels."""
    predictions = P.predictions_for(arm, cache)
    rows = []
    for key, _, names in P1.job_keys():
        for name in names:
            state = P2.load_tokens(name)["state"]  # provider 5 fps targets, shared grid of both views
            for view in VIEWS:
                nll, steps = _nll(predictions[key][view][name][0], state)
                rows.append({"arm": arm, "job": key, "kind": key.split("_")[0], "match": name,
                             "view": view, "heldout_nll": nll, "steps": steps})
    return rows


def reference_train_nll(arm: str) -> list[dict]:
    """Training NLL of exp_2's saved fits (``exp2_aug`` recipe), recomputed from their weights."""
    config = P2.MODEL_ARMS[arm]["config"]
    folder = P2.prediction_folder(arm)
    rows = []
    for key, train_names, _ in P1.job_keys():
        model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
        model.load_state_dict(torch.load(folder / f"{key}.pt", map_location=N.DEVICE))
        nll, steps = X.state_nll(model.eval(), P2.train_arrays(arm, train_names), config.window)
        rows.append({"arm": arm, "job": key, "train_nll": nll, "train_steps": steps})
    return rows


def gap_table(train: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """Per arm and per family: mean training NLL over its 28 fits, mean held-out NLL over all its
    outer + pair predictions, gap = held-out - training. Flag a family whose gap on the original
    view exceeds the exp_2 recipe's by more than GAP_GROWTH_FLAG (report only)."""
    pivot = heldout.groupby(["arm", "view"]).heldout_nll.mean().unstack("view")
    table = train.groupby("arm").train_nll.mean().to_frame().join(pivot)
    for view in VIEWS:
        table[f"gap_{view}"] = table[view] - table.train_nll
    table = table.rename(columns={v: f"heldout_nll_{v}" for v in VIEWS})
    table["family"] = [FAMILY_OF[a] for a in table.index]
    family = table.groupby("family").mean(numeric_only=True)
    reference = family.loc[REFERENCE_FAMILY, "gap_original"]
    family["gap_growth_vs_exp2_recipe"] = family.gap_original / reference - 1.0
    family["flag_gap_growth_gt_25pct"] = family.gap_growth_vs_exp2_recipe > GAP_GROWTH_FLAG
    table["gap_growth_vs_exp2_recipe"] = table.gap_original / reference - 1.0
    return pd.concat([table.reset_index().assign(level="arm"),
                      family.reset_index().rename(columns={"family": "arm"}).assign(level="family")],
                     ignore_index=True)
