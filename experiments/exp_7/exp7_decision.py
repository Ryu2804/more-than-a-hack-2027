"""H1 clauses, the decision and the BAR-ATH plausibility guard (PREREG 5). Pure functions; notebook 03 writes
``results/decision.json``.

Baseline of every clause: ``exp3_ref_ens3`` on ``target_like``, exp_3's stored rows (read in place).
"""

from __future__ import annotations

import exp7_common as C  # noqa: I001  (first: environment, sys.path, safe exp_6 imports)
from exp1_common import MATCH_IDS, read_json

CLAUSE_VIEW = {"gain": "measured", "no_harm": "target_like", "stress": "stress"}
H1_RULE = {
    "label_sets": list(C.LABEL_SETS),
    "baseline": f"{C.REFERENCE} on target_like, exp_3's stored rows",
    "gain": {"view": "measured", "ci_low_gt": 0.0},
    "no_harm": {"view": "target_like", "mean_min": -0.005},
    "stress": {"view": "stress", "mean_min": -0.005},
    "rule": f"on provider and L_ctrl: {C.PRIMARY} measured - baseline paired 95% CI low > 0, "
            "target_like - baseline mean >= -0.005, stress - baseline mean >= -0.005",
}
PLAUSIBILITY = {"dead_share_prior": 0.407, "tolerance": 0.030}


def baseline_spell(label_set: str) -> dict[str, float]:
    return {r["match"]: r["spell_f1"] for r in read_json(C.exp3_result(label_set))["rows"]}


def view_deltas(rows: list[dict], view: str, label_set: str) -> list[float]:
    """Per-match spell-F1(primary on view) - spell-F1(baseline), in MATCH_IDS order."""
    base = baseline_spell(label_set)
    spell = {}
    for r in rows:
        if r["view"] == view and r["label_set"] == label_set:
            if r["match"] in spell:
                raise RuntimeError(f"Duplicate row {view} {label_set} {r['match']}")
            spell[r["match"]] = r["spell_f1"]
    return [spell[name] - base[name] for name in MATCH_IDS]


def clause_table(deltas: dict[str, dict[str, list[float]]]) -> dict:
    """deltas[label][clause] -> per-clause summaries with pass flags."""
    out = {}
    for label in C.LABEL_SETS:
        out[label] = {}
        for clause in CLAUSE_VIEW:
            summary = C.DE6.delta_summary(deltas[label][clause])
            rule = H1_RULE[clause]
            passed = summary["ci_low"] > rule["ci_low_gt"] if "ci_low_gt" in rule \
                else summary["mean_delta"] >= rule["mean_min"]
            out[label][clause] = {**summary, "view": CLAUSE_VIEW[clause], "pass": bool(passed)}
    return out


def decide(clauses: dict) -> dict:
    """H1 passes iff every clause holds on both label sets."""
    failed = [f"{clause}:{label}" for label in C.LABEL_SETS for clause in CLAUSE_VIEW
              if not clauses[label][clause]["pass"]]
    return {"clauses": clauses, "h1_passed": not failed, "failed": failed, "rule": H1_RULE["rule"]}


def plausible(dead_share: float) -> bool:
    return abs(dead_share - PLAUSIBILITY["dead_share_prior"]) <= PLAUSIBILITY["tolerance"]


def require_h1(decision: dict) -> None:
    """Candidate files may be written only when the stored decision says H1 passed."""
    if decision.get("h1_passed") is not True:
        raise RuntimeError("H1 did not pass: exp_7 writes no candidate file")
