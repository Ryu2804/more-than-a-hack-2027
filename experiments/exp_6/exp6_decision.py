"""Gates G1-G3, the CI clause, the stress (non-harm) clause and the H1 decision (PREREG 5).

Pure functions; notebook 03 writes their output to ``results/decision.json``.
"""

from __future__ import annotations

import math

import numpy as np

import exp6_common as C
import exp1_common as C1

GATES = {
    "G1": {"min_share": 0.95, "label_margin_px": 8.0, "what": "ATD-visible graded balls projecting into the image"},
    "G2": {"min_precision": 0.90, "min_recall": 0.50, "inlier_m": 2.0, "what": "test blocks, run as if hidden"},
    "G3": {"min_coverage": 0.20, "what": "accepted share of hidden graded frames"},
}
H1_RULE = {
    "label_sets": list(C.LABEL_SETS),
    "ci_low_gt": 0.0,
    "stress_mean_min": -0.005,
    "rule": "G1 and G2 and G3; on both label sets the measured paired 95% CI lower bound > 0 "
            "and the stress-setting mean delta >= -0.005",
}


def _at_least(value: float, threshold: float) -> bool:
    return bool(value is not None and math.isfinite(value) and value >= threshold)


def gate_table(g1_share: float, precision: float, recall: float, coverage: float) -> dict:
    g2 = GATES["G2"]
    return {
        "G1": {"value": g1_share, "threshold": GATES["G1"]["min_share"],
               "pass": _at_least(g1_share, GATES["G1"]["min_share"])},
        "G2": {"precision": precision, "recall": recall, "min_precision": g2["min_precision"],
               "min_recall": g2["min_recall"],
               "pass": _at_least(precision, g2["min_precision"]) and _at_least(recall, g2["min_recall"])},
        "G3": {"value": coverage, "threshold": GATES["G3"]["min_coverage"],
               "pass": _at_least(coverage, GATES["G3"]["min_coverage"])},
    }


def delta_summary(deltas: list[float]) -> dict:
    values = [float(d) for d in deltas]
    low, high = C1.paired_bootstrap(values)
    return {"mean_delta": float(np.mean(values)), "ci_low": low, "ci_high": high,
            "wins": int(sum(d > 0 for d in values)), "matches": len(values), "per_match": values}


def ci_clause(deltas: dict[str, list[float]]) -> dict:
    out = {}
    for label in C.LABEL_SETS:
        summary = delta_summary(deltas[label])
        out[label] = {**summary, "pass": bool(summary["ci_low"] > H1_RULE["ci_low_gt"])}
    return out


def stress_clause(deltas: dict[str, list[float]]) -> dict:
    out = {}
    for label in C.LABEL_SETS:
        summary = delta_summary(deltas[label])
        out[label] = {**summary, "pass": bool(summary["mean_delta"] >= H1_RULE["stress_mean_min"])}
    return out


def decide(gates: dict, ci: dict, stress: dict) -> dict:
    """H1 passes iff every gate, the CI clause on both label sets and the stress clause on both hold."""
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    failed += [f"ci_clause:{label}" for label in C.LABEL_SETS if not ci[label]["pass"]]
    failed += [f"stress_clause:{label}" for label in C.LABEL_SETS if not stress[label]["pass"]]
    return {"gates": gates, "ci_clause": ci, "stress_clause": stress, "h1_passed": not failed,
            "failed": failed, "rule": H1_RULE["rule"]}


def require_h1(decision: dict) -> None:
    """Candidate files may be written only when the stored decision says H1 passed."""
    if decision.get("h1_passed") is not True:
        raise RuntimeError("H1 did not pass: exp_6 writes no candidate file")
