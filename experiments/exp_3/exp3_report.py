"""exp_3 aggregation: per (label set, view) tables, paired deltas, gates and PREREG decisions.

Rows always carry a ``label_set``; every comparison is made inside one label set, using
exp_2's paired-bootstrap ``delta`` and exp_0 ``gates`` on the filtered rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
import exp2_report as R2
from exp1_common import MATCH_IDS, paired_bootstrap, read_json
from exp3_common import EVAL_LABEL, PRIMARY_VIEW, RESULTS, VIEWS

REFERENCE = "exp3_ref_ens3"
CANDIDATE_IF_H1 = "exp3_ctrl_ens3"
SEEDS = ("exp3_ctrl", "exp3_ctrl_s1", "exp3_ctrl_s2")
S1_LABELS = ("ctrl", "provider", "track_1", "track_3")  # track_2 == ctrl under the fallback
LABELS = {
    "exp3_ctrl_ens3": "H1: mean of 3 L_ctrl-trained seeds, selected on the evaluation labels",
    "exp3_ref_ens3": "H1 reference: exp_2's 3 provider-trained seeds (cached), re-selected",
    "exp3_ctrl": "L_ctrl-trained SetSeq, seed 20261108",
    "exp3_ctrl_s1": "L_ctrl-trained SetSeq, seed 20261109",
    "exp3_ctrl_s2": "L_ctrl-trained SetSeq, seed 20261110",
    "exp2_aug": "exp_2 adopted candidate (cached), re-selected",
    "exp1_ref": "exp_1 SetSeq re-fit (cached), re-selected",
    "exp3_ctrl_ens3_argmax": "diagnostic: exp3_ctrl_ens3 without decoder",
    "exp3_ref_ens3_argmax": "diagnostic: exp3_ref_ens3 without decoder",
    "ryu_final": "ryu shipped TCN + decoder, zero-shot",
    "rasya_c08": "rasya 012 C08 rules",
    "rasya_c06": "rasya 012 C06 rules",
    "baseline": "organisers' starter rule",
}
COUNT_LIMIT = R2.COUNT_LIMIT


def variant_rows(variant: str, view: str, label_set: str) -> list[dict]:
    return read_json(RESULTS / f"variant_{variant}__{view}__{label_set}.json")["rows"]


def final_config(variant: str, view: str = PRIMARY_VIEW, label_set: str = EVAL_LABEL) -> dict:
    return read_json(RESULTS / f"variant_{variant}__{view}__{label_set}.json")["final_config"]


def at(rows: pd.DataFrame, label_set: str) -> pd.DataFrame:
    part = rows[rows.label_set == label_set]
    R2.assert_proxy_only(part)
    return part


def delta(rows: pd.DataFrame, label_set: str, view: str, a: str, b: str) -> dict:
    return {"label_set": label_set, **R2.delta(at(rows, label_set), view, a, b)}


def _row(rows, label_set, view, variant, present) -> dict:
    part = at(rows, label_set)
    group = part[(part.view == view) & (part.variant == variant)].set_index("match").reindex(MATCH_IDS)
    vs_ref = delta(rows, label_set, view, variant, REFERENCE) if variant != REFERENCE else None
    vs_exp2 = delta(rows, label_set, view, variant, "exp2_aug") if variant != "exp2_aug" and "exp2_aug" in present else None
    return {
        "label_set": label_set, "view": view, "variant": variant,
        "mean_spell_f1": group.spell_f1.mean(), "min_spell_f1": group.spell_f1.min(),
        "mean_macro_f1": group.macro_f1.mean(),
        "mean_abs_count_error": group.count_error_ratio.abs().mean(),
        "max_abs_count_error": group.count_error_ratio.abs().max(),
        "mean_count_error": group.count_error_ratio.mean(),
        "delta_vs_ref_ens3": vs_ref["mean_delta"] if vs_ref else 0.0,
        "ci95_low_vs_ref": vs_ref["ci95"][0] if vs_ref else 0.0,
        "ci95_high_vs_ref": vs_ref["ci95"][1] if vs_ref else 0.0,
        "wins_vs_ref": vs_ref["wins"] if vs_ref else 0,
        "delta_vs_exp2_aug": vs_exp2["mean_delta"] if vs_exp2 else np.nan,
        "ci95_low_vs_exp2": vs_exp2["ci95"][0] if vs_exp2 else np.nan,
        "ci95_high_vs_exp2": vs_exp2["ci95"][1] if vs_exp2 else np.nan,
        "count_gate": bool(group.count_error_ratio.abs().max() <= COUNT_LIMIT),
    }


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label_set in S1_LABELS:
        part = at(rows, label_set)
        for view in VIEWS:
            present = set(part[part.view == view].variant)
            out += [_row(rows, label_set, view, v, present) for v in LABELS if v in present]
    return pd.DataFrame(out)


def gate_rows(rows: pd.DataFrame, label_set: str = EVAL_LABEL) -> list[dict]:
    part = at(rows, label_set)
    out = []
    for view in VIEWS:
        present = set(part[part.view == view].variant)
        out += [{"label_set": label_set, **R2.gates(part, view, v)} for v in LABELS
                if v in present and v != "baseline"]
    return out


def seed_spread(rows: pd.DataFrame, view: str, label_set: str = EVAL_LABEL) -> float:
    means = R2._pivot(at(rows, label_set), view)[list(SEEDS)].mean()
    return float(means.max() - means.min())


def decide(rows: pd.DataFrame) -> dict:
    """PREREG section 6, exactly as written; everything except H1 is report-only."""
    h1 = delta(rows, EVAL_LABEL, PRIMARY_VIEW, CANDIDATE_IF_H1, REFERENCE)
    passed = h1["ci95"][0] > 0
    candidate = CANDIDATE_IF_H1 if passed else REFERENCE
    report_only = {
        "H1_original_view": delta(rows, EVAL_LABEL, "original", CANDIDATE_IF_H1, REFERENCE),
        "ctrl_ens3_vs_exp2_aug": [delta(rows, EVAL_LABEL, v, CANDIDATE_IF_H1, "exp2_aug") for v in VIEWS],
        "ref_ens3_vs_exp2_aug": [delta(rows, EVAL_LABEL, v, REFERENCE, "exp2_aug") for v in VIEWS],
        "seed_spread": {v: seed_spread(rows, v) for v in VIEWS},
        "S1_ctrl_ens3_vs_ref_ens3": [delta(rows, ls, v, CANDIDATE_IF_H1, REFERENCE)
                                     for ls in S1_LABELS for v in VIEWS],
    }
    return {"rules": {"H1": {**h1, "rule": "paired 95% CI lower bound > 0", "passed": bool(passed)}},
            "supported": {"H1": bool(passed)}, "candidate_variant": candidate,
            "candidate_final_config": final_config(candidate),
            "candidate_config_selected_on": {"view": PRIMARY_VIEW, "label_set": EVAL_LABEL,
                                             "predictions": "7 out-of-fold outer predictions"},
            "report_only": report_only}


def coverage_shift(rows: pd.DataFrame, coverage: pd.DataFrame) -> list[dict]:
    """S2: coverage minus target_like per variant and label set, paired by match."""
    out = []
    for (variant, label_set), part in coverage.groupby(["variant", "label_set"]):
        base = rows[(rows.variant == variant) & (rows.label_set == label_set) & (rows.view == PRIMARY_VIEW)]
        new, old = part.set_index("match").reindex(MATCH_IDS), base.set_index("match").reindex(MATCH_IDS)
        values = (new.spell_f1 - old.spell_f1).to_numpy()
        if np.isnan(values).any():
            raise ValueError(f"Missing rows for S2 {variant} {label_set}")
        low, high = paired_bootstrap(values.tolist())
        out.append({"variant": variant, "label_set": label_set, "mean_change": float(values.mean()),
                    "ci95": [low, high], "wins": int((values > 0).sum()),
                    "mean_count_error_change": float((new.count_error_ratio - old.count_error_ratio).mean())})
    return out


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe records: NaN (e.g. an arm's delta against itself) becomes null."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def as_markdown(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    return R2.as_markdown(frame, columns, digits)
