"""Report tables from ``results/`` (IDSSE aggregates only; notebook 03)."""

from __future__ import annotations

import pandas as pd

import exp7_common as C  # noqa: I001  (first: environment, sys.path, safe exp_6 imports)
import exp7_decision as DE
import exp7_eval as E
import exp7_fit as F
from exp1_common import MATCH_IDS, read_json

REF_ROWS = C.RESULTS / "ref_rows.csv"


def all_rows() -> pd.DataFrame:
    """Every held-out row: exp_7 evaluations (nested) and exp3_ref_ens3 through exp_6's path (stored selections)."""
    rows = [r for task in E.tasks() for r in read_json(E.result_path(*task))["rows"]]
    frame = pd.DataFrame(rows)
    reference = pd.read_csv(REF_ROWS)
    return pd.concat([frame, reference], ignore_index=True)


def delta_rows(frame: pd.DataFrame) -> list[dict]:
    """Mean spell-F1, count error and paired delta vs exp3_ref_ens3 on target_like (exp_3's stored rows)."""
    out = []
    keys = ["variant", "selection_view", "view", "label_set"]
    for (variant, selection, view, label), group in frame.groupby(keys, sort=False):
        if sorted(group["match"]) != sorted(MATCH_IDS):
            raise RuntimeError(f"Incomplete rows for {variant} {selection} {view} {label}")
        summary = C.DE6.delta_summary(DE.view_deltas(group.to_dict("records"), view, label))
        out.append({"variant": variant, "selection_view": selection, "view": view, "label_set": label,
                    "spell_f1": float(group["spell_f1"].mean()), "macro_f1": float(group["macro_f1"].mean()),
                    "count_error": float(group["count_error_ratio"].mean()),
                    **{k: summary[k] for k in ("mean_delta", "ci_low", "ci_high", "wins")}})
    return out


def seed_spread(deltas: list[dict]) -> list[dict]:
    """Single-seed spell-F1 per (view, label set) and its spread (max - min) next to the ensemble."""
    out = []
    for view in ("target_like", "measured", "stress", "ar1"):
        for label in C.LABEL_SETS:
            seeds = {r["variant"]: r["spell_f1"] for r in deltas
                     if r["variant"] in F.ARMS and r["view"] == view and r["label_set"] == label}
            ensemble = [r["spell_f1"] for r in deltas
                        if r["variant"] == C.PRIMARY and r["view"] == view and r["label_set"] == label]
            if len(seeds) == len(F.ARMS):
                out.append({"view": view, "label_set": label, **seeds, "spread": max(seeds.values()) - min(seeds.values()),
                            "ensemble": ensemble[0] if ensemble else None})
    return out


def markdown(rows: list[dict], digits: int = 4) -> str:
    if not rows:
        return "(none)\n"
    columns = list(rows[0])

    def cell(value: object) -> str:
        return f"{value:+.{digits}f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines) + "\n"
