"""Report-only diagnostics (IDSSE only, never fits): argmax scores, recall of short possessions
and the train -> held-out NLL generalisation gap, with exp_4's definitions.

The exp_2-recipe reference training NLL is exp_4's ``results/ref_train_nll.csv``, read in place
(SHA-256 frozen); its held-out NLL comes from exp_2's cached predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp5_common as C  # noqa: I001  (first: env, sys.path, exp_1-exp_4 imports)
import exp1_decode as D
import exp1_protocol as P1
import exp2_protocol as P2
import exp3_diagnostics as G3
import exp3_protocol as P3
import exp5_eval as E
import exp5_protocol as P
from exp1_common import expand, score

SHORT_S = 2.0
FPS = 25
REFERENCE_FAMILY = "exp2_recipe"  # exp2_aug, _s1, _s2 (the members of exp3_ref_ens3)
FAMILY_OF = {**{arm: C.PRIMARY_FAMILY for arm in P.ARMS}, **{arm: REFERENCE_FAMILY for arm in E.EXP2_ARMS}}


# ---------------------------------------------------------------- argmax and short possessions
def argmax_rows(variant: str, view: str, label_set: str, cache: dict) -> list[dict]:
    """No decoder: per-step argmax of the held-out p(H, A, D)."""
    predictions = E.predictions_for(E.SOURCE_OF[variant], cache)
    rows = []
    for held in C.MATCH_IDS:
        item = P3.make_item(held, predictions[f"outer_{held}"][view][held], view, label_set)
        codes = item["probability"].argmax(axis=1).astype("int8")
        rows.append({"match": held, "variant": f"{variant}_argmax", "view": view, "label_set": label_set,
                     **score(item["truth"], expand(codes, item["indices"], item["full_half"]))})
    return rows


def short_recall_rows(variant: str, view: str, cache: dict, label_set: str = "provider") -> list[dict]:
    """Per held-out match: recall (IoU 0.5) of true possessions < 2 s and of all possessions,
    decoded with that fold's own selected config, and without decoder (argmax)."""
    predictions = E.predictions_for(E.SOURCE_OF[variant], cache)
    selections = E.read_result(variant, view, label_set)["selections"]
    rows = []
    for held in C.MATCH_IDS:
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
    predictions = E.predictions_for(arm, cache)
    rows = []
    for key, _, names in P1.job_keys():
        for name in names:
            state = P2.load_tokens(name)["state"]  # provider 5 fps targets, shared grid of both views
            for view in C.VIEWS:
                nll, steps = _nll(predictions[key][view][name][0], state)
                rows.append({"arm": arm, "job": key, "kind": key.split("_")[0], "match": name,
                             "view": view, "heldout_nll": nll, "steps": steps})
    return rows


def reference_train_nll() -> pd.DataFrame:
    """exp_4's training NLL of exp_2's saved fits (inference only), read in place."""
    frame = pd.read_csv(C.C4.RESULTS / "ref_train_nll.csv")
    if set(frame.arm) != set(E.EXP2_ARMS) or len(frame) != 28 * len(E.EXP2_ARMS):
        raise RuntimeError("exp_4 ref_train_nll.csv does not cover the 3 x 28 exp_2 fits")
    return frame


def gap_table(train: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """exp_4's definition: per arm and family, mean training NLL over its 28 fits, mean held-out
    NLL over its outer + pair predictions, gap = held-out - training. The flag compares the
    family's ``original``-view gap with the exp_2 recipe's (exp_4 code); tl growth is reported too."""
    pivot = heldout.groupby(["arm", "view"]).heldout_nll.mean().unstack("view")
    table = train.groupby("arm").train_nll.mean().to_frame().join(pivot)
    for view in C.VIEWS:
        table[f"gap_{view}"] = table[view] - table.train_nll
    table = table.rename(columns={v: f"heldout_nll_{v}" for v in C.VIEWS})
    table["family"] = [FAMILY_OF[a] for a in table.index]
    family = table.groupby("family").mean(numeric_only=True)
    for view in C.VIEWS:
        reference = family.loc[REFERENCE_FAMILY, f"gap_{view}"]
        family[f"gap_growth_{view}"] = family[f"gap_{view}"] / reference - 1.0
        table[f"gap_growth_{view}"] = table[f"gap_{view}"] / reference - 1.0
    family["flag_gap_growth_gt_25pct"] = family.gap_growth_original > C.GAP_GROWTH_FLAG
    return pd.concat([table.reset_index().assign(level="arm"),
                      family.reset_index().rename(columns={"family": "arm"}).assign(level="family")],
                     ignore_index=True)


def exp4_gap_context() -> pd.DataFrame:
    """exp_4's own family rows (report-only replication context), read in place."""
    frame = pd.read_csv(C.C4.RESULTS / "gap_table.csv")
    return frame[(frame.level == "family") & frame.arm.isin(["exp4_touch", "exp2_recipe"])]
