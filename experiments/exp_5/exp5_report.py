"""exp_5 aggregation: per (label set, view) tables, paired deltas, gates, seed spread, decision.

Every comparison is made inside one label set and one view with exp_2's paired match bootstrap
(1000 draws, seed 20261108, 95 % percentile). Only H1 drives the decision (PREREG §5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp5_common as C  # noqa: I001  (first: env, sys.path, exp_1-exp_4 imports)
import exp2_report as R2
import exp3_report as R3
import exp5_eval as E
import exp5_protocol as P
from exp1_common import MATCH_IDS

PRIMARY, REFERENCE, VIEW = C.PRIMARY, C.REFERENCE, C.PRIMARY_VIEW
FAMILY_SEEDS = {C.PRIMARY_FAMILY: tuple(P.ARMS)}
LABELS = {  # display order
    PRIMARY: "H1: touch only, 3 fresh seeds 20261111/12/13 (provider-trained)",
    REFERENCE: "H1 reference: exp_2's 3 provider-trained seeds (cached), re-selected",
    "exp4_touch_ens3": "replication context: exp_4 touch only, seeds 20261108/09/10 (cached)",
    "exp45_touch_ens6": "report only: pooled 6-seed touch ensemble (half its seeds were seen)",
    "exp2_aug": "exp_2 adopted single seed (cached), re-selected",
    **{arm: f"{arm} single seed {spec.seed}" for arm, spec in P.ARMS.items()},
    f"{PRIMARY}_argmax": f"diagnostic: {PRIMARY} without decoder",
    f"{REFERENCE}_argmax": f"diagnostic: {REFERENCE} without decoder",
    "ryu_final": "ryu shipped TCN + decoder, zero-shot",
    "rasya_c08": "rasya 012 C08 rules",
    "baseline": "organisers' starter rule",
}


# ---------------------------------------------------------------- the rule (pure)
def h1_rule(mean_delta: float, ci_low: float, seed_spread: float | None) -> dict:
    """PREREG §5: H1 passes iff the paired CI lower bound > 0 AND mean Δ > the exp5_touch seed spread."""
    ci_clause = bool(ci_low > C.H1_CI_LOW)
    seed_clause = bool(seed_spread is not None and mean_delta > seed_spread)
    return {"ci_clause": {"rule": f"paired 95% CI lower bound > {C.H1_CI_LOW}", "ci_low": ci_low,
                          "passed": ci_clause},
            "seed_clause": {"rule": "mean delta > exp5_touch seed spread (provider, target_like)",
                            "mean_delta": mean_delta, "seed_spread": seed_spread, "passed": seed_clause},
            "passed": ci_clause and seed_clause}


def adoption_plan(h1_passed: bool, ctrl_ci_low: float) -> dict:
    """Adoption tree: provider file iff H1; ctrl file iff H1 and L_ctrl non-inferiority."""
    if not h1_passed:
        return {"write_files": False, "files": {}, "label_sets": [],
                "statement": "H1 failed: no exp_5 file is written; exp_3's two BAR-ATH files stay the team's options"}
    ctrl_ok = bool(ctrl_ci_low > C.CTRL_NONINF_CI_LOW)
    label_sets = ["provider", "ctrl"] if ctrl_ok else ["provider"]
    statement = ("H1 passed: provider and ctrl candidates are written" if ctrl_ok else
                 "H1 passed: provider candidate only; L_ctrl non-inferiority failed, exp_3's L_ctrl candidate stays")
    return {"write_files": True, "variant": PRIMARY, "members": list(P.ARMS), "label_sets": label_sets,
            "files": {ls: f"predictions/exp_5_candidate_{ls}.csv" for ls in label_sets},
            "ctrl_non_inferiority": {"rule": f"L_ctrl paired 95% CI lower bound > {C.CTRL_NONINF_CI_LOW}",
                                     "ci_low": ctrl_ci_low, "passed": ctrl_ok},
            "statement": statement}


# ---------------------------------------------------------------- rows
def available(variant: str, view: str, label_set: str) -> bool:
    return E.result_path(variant, view, label_set).exists()


def variant_rows(variant: str, view: str, label_set: str) -> list[dict]:
    return E.read_result(variant, view, label_set)["rows"]


def collect_rows(extra: list[pd.DataFrame]) -> pd.DataFrame:
    """Every evaluated / reused (variant, view, label set) plus reference and diagnostic rows."""
    parts = list(extra)
    for label_set in C.S1_LABELS:
        for variant in (*E.EVALUATED, *E.REUSED):
            for view in C.VIEWS:
                if available(variant, view, label_set):
                    parts.append(pd.DataFrame(variant_rows(variant, view, label_set)))
    rows = pd.concat(parts, ignore_index=True)
    R2.assert_proxy_only(rows)
    return rows


def delta(rows: pd.DataFrame, label_set: str, view: str, a: str, b: str) -> dict:
    return R3.delta(rows, label_set, view, a, b)


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """exp_3's row summary (Δ vs exp3_ref_ens3 and vs exp2_aug) for every present variant."""
    out = []
    for label_set in C.S1_LABELS:
        part = R3.at(rows, label_set)
        for view in C.VIEWS:
            present = set(part[part.view == view].variant)
            out += [R3._row(rows, label_set, view, v, present) for v in LABELS if v in present]
    return pd.DataFrame(out)


def gate_rows(rows: pd.DataFrame, label_set: str) -> list[dict]:
    part = R3.at(rows, label_set)
    out = []
    for view in C.VIEWS:
        present = set(part[part.view == view].variant)
        out += [{"label_set": label_set, **R2.gates(part, view, v)} for v in LABELS
                if v in present and v != "baseline" and not v.endswith("_argmax")]
    return out


def seed_means(rows: pd.DataFrame, family: str, view: str, label_set: str) -> dict:
    pivot = R2._pivot(R3.at(rows, label_set), view)
    return {arm: float(pivot[arm].mean()) for arm in FAMILY_SEEDS[family] if arm in pivot}


def seed_spreads(rows: pd.DataFrame) -> list[dict]:
    """Max - min of single-seed mean spellF1 per family, label set and view (where evaluated)."""
    out = []
    for label_set in C.DECISION_LABELS:
        for view in C.VIEWS:
            for family in FAMILY_SEEDS:
                means = seed_means(rows, family, view, label_set)
                if len(means) == 3:
                    values = np.array(list(means.values()))
                    out.append({"family": family, "label_set": label_set, "view": view, "seeds": means,
                                "seed_mean": float(values.mean()), "seed_spread": float(values.max() - values.min())})
    return out


def primary_spread(spreads: list[dict]) -> float | None:
    hits = [s["seed_spread"] for s in spreads
            if (s["family"], s["label_set"], s["view"]) == (C.PRIMARY_FAMILY, "provider", VIEW)]
    return hits[0] if hits else None


def replication_rows(rows: pd.DataFrame) -> list[dict]:
    """Per S1 cell: Δ(exp5 vs ref) next to Δ(exp4 vs ref), the paired exp5 − exp4 delta, and the
    pooled 6-seed ensemble vs ref. Report only."""
    out = []
    for label_set in C.S1_LABELS:
        for view in C.VIEWS:
            exp5 = delta(rows, label_set, view, PRIMARY, REFERENCE)
            exp4 = delta(rows, label_set, view, "exp4_touch_ens3", REFERENCE)
            paired = delta(rows, label_set, view, PRIMARY, "exp4_touch_ens3")
            pooled = delta(rows, label_set, view, "exp45_touch_ens6", REFERENCE)
            out.append({"label_set": label_set, "view": view,
                        "exp5_delta": exp5["mean_delta"], "exp5_ci_low": exp5["ci95"][0], "exp5_ci_high": exp5["ci95"][1],
                        "exp5_wins": exp5["wins"],
                        "exp4_delta": exp4["mean_delta"], "exp4_ci_low": exp4["ci95"][0], "exp4_ci_high": exp4["ci95"][1],
                        "exp4_wins": exp4["wins"],
                        "exp5_minus_exp4": paired["mean_delta"], "paired_ci_low": paired["ci95"][0],
                        "paired_ci_high": paired["ci95"][1], "paired_wins": paired["wins"],
                        "ens6_delta": pooled["mean_delta"], "ens6_ci_low": pooled["ci95"][0],
                        "ens6_ci_high": pooled["ci95"][1], "ens6_wins": pooled["wins"]})
    return out


def final_config(variant: str, view: str, label_set: str) -> dict:
    return E.read_result(variant, view, label_set)["final_config"]


def decide(rows: pd.DataFrame) -> dict:
    """PREREG §5, exactly as written; everything except H1 (and the ctrl-file gate) is report-only."""
    provider = delta(rows, "provider", VIEW, PRIMARY, REFERENCE)
    ctrl = delta(rows, "ctrl", VIEW, PRIMARY, REFERENCE)
    spreads = seed_spreads(rows)
    spread = primary_spread(spreads)
    rule = h1_rule(provider["mean_delta"], provider["ci95"][0], spread)
    adoption = adoption_plan(rule["passed"], ctrl["ci95"][0])
    if adoption["write_files"]:
        adoption["configs"] = {ls: final_config(PRIMARY, VIEW, ls) for ls in adoption["label_sets"]}
    report_only = {
        "S1_primary_vs_reference": [delta(rows, ls, v, PRIMARY, REFERENCE) for ls in C.S1_LABELS for v in C.VIEWS],
        "replication": replication_rows(rows),
        "pooled_ens6_vs_reference": [delta(rows, ls, v, "exp45_touch_ens6", REFERENCE)
                                     for ls in C.S1_LABELS for v in C.VIEWS],
        "single_seeds_vs_reference": [delta(rows, ls, v, arm, REFERENCE) for arm in P.ARMS
                                      for ls in C.DECISION_LABELS for v in C.VIEWS],
        "seed_spreads": spreads,
    }
    return {
        "H1": {"passed": rule["passed"], "comparison": [PRIMARY, REFERENCE], "labels": "provider", "view": VIEW,
               "delta": provider, **{k: rule[k] for k in ("ci_clause", "seed_clause")}},
        "ctrl_non_inferiority": {**ctrl, "rule": f"paired 95% CI lower bound > {C.CTRL_NONINF_CI_LOW}",
                                 "passed": bool(ctrl["ci95"][0] > C.CTRL_NONINF_CI_LOW),
                                 "role": "gates the ctrl candidate file only (never H1)"},
        "adoption": adoption,
        "never_adopted": list(E.NEVER_ADOPTED),
        "scope": ("seed replication on the same 7 IDSSE matches that selected this arm: a pass rules out "
                  "seed luck, not selection on these matches or transfer to BAR-ATH"),
        "report_only": report_only,
    }


# ---------------------------------------------------------------- formatting
def per_match(rows: pd.DataFrame, label_set: str, view: str, variants: tuple[str, ...]) -> pd.DataFrame:
    part = R3.at(rows, label_set)
    part = part[(part.view == view) & part.variant.isin(variants)]
    spell = part.pivot(index="match", columns="variant", values="spell_f1").reindex(MATCH_IDS)
    count = part.pivot(index="match", columns="variant", values="count_error_ratio").reindex(MATCH_IDS)
    return spell[list(variants)].round(3).astype(str) + " (" + count[list(variants)].round(2).map("{:+.2f}".format) + ")"


def as_markdown(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    return R2.as_markdown(frame, columns, digits)


def records(frame: pd.DataFrame) -> list[dict]:
    return R3.records(frame)
