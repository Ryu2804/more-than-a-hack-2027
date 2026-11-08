"""exp_2 aggregation: per-view tables, paired deltas vs exp1_ref, gates and PREREG decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from exp1_common import MATCH_IDS, QUALITY_GATES, paired_bootstrap
from exp2_common import PRIMARY_VIEW, VIEWS

REFERENCE = "exp1_ref"
LABELS = {
    "exp1_ref": "exp_1 SetSeq re-fit (original training) + exp_1 main decoder",
    "exp2_aug": "H1: SetSeq + target_aug training (rounds 1-2) + exp_1 main decoder",
    "exp2_aug_const": "exp2_aug predictions + constant-penalty Viterbi (gamma 0)",
    "exp2_aug_hsmm": "H2: exp2_aug predictions + explicit-duration (semi-Markov) decoder",
    "exp2_aug_s1": "exp2_aug, training seed +1 (H3 spread only)",
    "exp2_aug_s2": "exp2_aug, training seed +2 (H3 spread only)",
    "exp2_aug_ens3": "H3: mean of 3 seeds' probabilities + exp_1 main decoder",
    "exp1_ref_argmax": "diagnostic: exp1_ref without decoder",
    "exp2_aug_argmax": "diagnostic: exp2_aug without decoder",
    "ryu_final": "ryu shipped TCN + decoder, zero-shot",
    "rasya_c08": "rasya 012 C08 rules",
    "rasya_c06": "rasya 012 C06 rules",
    "baseline": "organisers' starter rule",
}
H1_TRANSFER_MARGIN = 0.0
H1_INDOMAIN_TOLERANCE = -0.02
H2_TOLERANCE = -0.01
COUNT_LIMIT = QUALITY_GATES["max_abs_count_error"]


def _pivot(rows: pd.DataFrame, view: str, column: str = "spell_f1") -> pd.DataFrame:
    part = rows[rows.view == view]
    return part.pivot(index="match", columns="variant", values=column).reindex(MATCH_IDS)


def delta(rows: pd.DataFrame, view: str, a: str, b: str) -> dict:
    pivot = _pivot(rows, view)
    values = (pivot[a] - pivot[b]).to_numpy()
    if np.isnan(values).any():
        raise ValueError(f"Missing per-match rows for {a} or {b} on {view}")
    low, high = paired_bootstrap(values.tolist())
    decision = "better" if low > 0 else "worse" if high < 0 else "inconclusive"
    return {"view": view, "a": a, "b": b, "mean_delta": float(values.mean()), "ci95": [low, high],
            "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "decision": decision}


def gates(rows: pd.DataFrame, view: str, variant: str) -> dict:
    """exp_0 quality gates against the baseline of the same view."""
    main = rows[(rows.view == view) & (rows.variant == variant)].set_index("match").reindex(MATCH_IDS)
    base = rows[(rows.view == view) & (rows.variant == "baseline")].set_index("match").reindex(MATCH_IDS)
    d = delta(rows, view, variant, "baseline")
    checks = {
        "gain_at_least_005": d["mean_delta"] >= QUALITY_GATES["spell_gain"],
        "ci_lower_above_zero": d["ci95"][0] > QUALITY_GATES["ci_low"],
        "macro_regression_at_most_001": float(main.macro_f1.mean() - base.macro_f1.mean())
        >= QUALITY_GATES["macro_regression"],
        "all_match_counts_within_30pct": float(main.count_error_ratio.abs().max()) <= COUNT_LIMIT,
    }
    return {"variant": variant, "view": view, "checks": checks, "pass": all(checks.values()),
            "worst_count_error": float(main.count_error_ratio.abs().max()),
            "worst_count_match": str(main.count_error_ratio.abs().idxmax())}


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for view in VIEWS:
        present = set(rows[rows.view == view].variant)
        for variant in [v for v in LABELS if v in present]:
            group = rows[(rows.view == view) & (rows.variant == variant)].set_index("match").reindex(MATCH_IDS)
            vs_ref = delta(rows, view, variant, REFERENCE) if variant != REFERENCE else None
            out.append({
                "view": view, "variant": variant,
                "mean_spell_f1": group.spell_f1.mean(), "min_spell_f1": group.spell_f1.min(),
                "mean_macro_f1": group.macro_f1.mean(),
                "mean_abs_count_error": group.count_error_ratio.abs().mean(),
                "max_abs_count_error": group.count_error_ratio.abs().max(),
                "mean_count_error": group.count_error_ratio.mean(),
                "delta_vs_exp1_ref": vs_ref["mean_delta"] if vs_ref else 0.0,
                "ci95_low": vs_ref["ci95"][0] if vs_ref else 0.0,
                "ci95_high": vs_ref["ci95"][1] if vs_ref else 0.0,
                "wins_vs_exp1_ref": vs_ref["wins"] if vs_ref else 0,
                "count_gate": bool(group.count_error_ratio.abs().max() <= COUNT_LIMIT),
            })
    return pd.DataFrame(out)


def seed_spread(rows: pd.DataFrame, view: str, seeds=("exp2_aug", "exp2_aug_s1", "exp2_aug_s2")) -> float:
    means = _pivot(rows, view)[list(seeds)].mean()
    return float(means.max() - means.min())


def decide(rows: pd.DataFrame) -> dict:
    """Apply PREREG.md decision rules exactly as written."""
    tl, orig = PRIMARY_VIEW, "original"
    h1a = delta(rows, tl, "exp2_aug", REFERENCE)
    h1b = delta(rows, orig, "exp2_aug", REFERENCE)
    hsmm = rows[(rows.view == tl) & (rows.variant == "exp2_aug_hsmm")]
    h2_delta = delta(rows, tl, "exp2_aug_hsmm", "exp2_aug_const")
    h2_count = float(hsmm.count_error_ratio.abs().max())
    h3 = delta(rows, tl, "exp2_aug_ens3", "exp2_aug")
    spread = seed_spread(rows, tl)
    rules = {
        "H1a_transfer_target_like": {**h1a, "rule": "CI lower > 0",
                                     "passed": h1a["ci95"][0] > H1_TRANSFER_MARGIN},
        "H1b_in_domain_original": {**h1b, "rule": "mean delta >= -0.02",
                                   "passed": h1b["mean_delta"] >= H1_INDOMAIN_TOLERANCE},
        "H2_count_gate_target_like": {"max_abs_count_error": h2_count, "rule": "all 7 matches within +-30%",
                                      "passed": h2_count <= COUNT_LIMIT},
        "H2_non_inferiority_target_like": {**h2_delta, "rule": "CI lower > -0.01 vs constant decoder",
                                           "passed": h2_delta["ci95"][0] > H2_TOLERANCE},
        "H3_ensemble_target_like": {**h3, "seed_spread": spread,
                                    "rule": "|mean delta| > seed spread and CI excludes 0",
                                    "passed": abs(h3["mean_delta"]) > spread
                                    and (h3["ci95"][0] > 0 or h3["ci95"][1] < 0)},
    }
    supported = {
        "H1": rules["H1a_transfer_target_like"]["passed"] and rules["H1b_in_domain_original"]["passed"],
        "H2": rules["H2_count_gate_target_like"]["passed"] and rules["H2_non_inferiority_target_like"]["passed"],
        "H3": rules["H3_ensemble_target_like"]["passed"],
    }
    return {"rules": rules, "supported": supported, "adopt_exp2_candidate": supported["H1"],
            "candidate_variant": "exp2_aug", "candidate_view_for_final_config": tl,
            "secondary_report_only": {
                "H2_original_view": delta(rows, orig, "exp2_aug_hsmm", "exp2_aug_const"),
                "H3_original_view": {**delta(rows, orig, "exp2_aug_ens3", "exp2_aug"),
                                     "seed_spread": seed_spread(rows, orig)},
            }}


def deltas_vs_reference(rows: pd.DataFrame) -> list[dict]:
    out = []
    for view in VIEWS:
        present = set(rows[rows.view == view].variant)
        out += [delta(rows, view, v, REFERENCE) for v in LABELS if v in present and v != REFERENCE]
    return out


def view_shift(rows: pd.DataFrame) -> list[dict]:
    """Per arm: target_like minus original (paired by match) = robustness to the stress view."""
    out = []
    for variant in [v for v in LABELS if v in set(rows.variant)]:
        values = (_pivot(rows, "target_like")[variant] - _pivot(rows, "original")[variant]).to_numpy()
        low, high = paired_bootstrap(values.tolist())
        out.append({"variant": variant, "mean_change": float(values.mean()), "ci95": [low, high]})
    return out


def as_markdown(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                signed = column.startswith(("delta", "ci95", "mean_count"))
                cells.append(f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def assert_proxy_only(rows: pd.DataFrame) -> None:
    if not set(rows.match) <= set(MATCH_IDS):
        raise RuntimeError("Non-proxy rows found in proxy report")
