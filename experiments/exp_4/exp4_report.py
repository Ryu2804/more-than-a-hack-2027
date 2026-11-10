"""exp_4 aggregation: per (label set, view) tables, paired deltas, gates, seed spread, decision.

Every comparison is made inside one label set and one view with exp_2's paired match
bootstrap (1000 draws, seed 20261108). Only H1 drives a decision (PREREG §5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp4_common  # noqa: F401,I001  (first: puts ../exp_3, ../exp_2, ../exp_1 on sys.path)
import exp2_report as R2
import exp3_report as R3
import exp4_protocol as P
from exp1_common import MATCH_IDS, read_json
from exp4_common import (
    C3, H1_CTRL_CI_LOW, H1_PROVIDER_CI_LOW, PRIMARY, PRIMARY_VIEW, REFERENCE, RESULTS, S1_LABELS, VIEWS,
)

FAMILY_SEEDS = {family: P.ENSEMBLES[f"{family}_ens3"] for family in P.FAMILIES}
FAMILY_SEEDS["exp2_recipe"] = P.EXP2_ARMS
LABELS = {
    "exp4_full_ens3": "H1: touch + aux head, mean of 3 seeds (provider-trained)",
    "exp4_touch_ens3": "ablation: touch features only, 3 seeds",
    "exp4_aux_ens3": "ablation: aux hidden-ball head only, 3 seeds",
    "exp3_ref_ens3": "H1 reference: exp_2's 3 provider-trained seeds (cached), re-selected",
    "exp2_aug": "exp_2 adopted single seed (cached), re-selected",
    **{arm: f"{arm} single seed {spec.seed}" for arm, spec in P.ARMS.items()},
    "exp4_full_ens3_argmax": "diagnostic: exp4_full_ens3 without decoder",
    "exp3_ref_ens3_argmax": "diagnostic: exp3_ref_ens3 without decoder",
    "ryu_final": "ryu shipped TCN + decoder, zero-shot",
    "rasya_c08": "rasya 012 C08 rules",
    "baseline": "organisers' starter rule",
}


def variant_rows(variant: str, view: str, label_set: str) -> list[dict]:
    folder = C3.RESULTS if variant in P.REUSED_VARIANTS else RESULTS
    return read_json(folder / f"variant_{variant}__{view}__{label_set}.json")["rows"]


def available(variant: str, view: str, label_set: str) -> bool:
    folder = C3.RESULTS if variant in P.REUSED_VARIANTS else RESULTS
    return (folder / f"variant_{variant}__{view}__{label_set}.json").exists()


def collect_rows(extra: list[pd.DataFrame]) -> pd.DataFrame:
    """Every evaluated (variant, view, label set) plus reference / diagnostic row frames."""
    parts = list(extra)
    for label_set in S1_LABELS:
        for variant in (*P.VARIANTS, *P.REUSED_VARIANTS):
            for view in VIEWS:
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
    for label_set in S1_LABELS:
        part = R3.at(rows, label_set)
        for view in VIEWS:
            present = set(part[part.view == view].variant)
            out += [R3._row(rows, label_set, view, v, present) for v in LABELS if v in present]
    return pd.DataFrame(out)


def gate_rows(rows: pd.DataFrame, label_set: str) -> list[dict]:
    part = R3.at(rows, label_set)
    out = []
    for view in VIEWS:
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
    for label_set in S1_LABELS:
        for view in VIEWS:
            for family in FAMILY_SEEDS:
                means = seed_means(rows, family, view, label_set)
                if len(means) == 3:
                    values = np.array(list(means.values()))
                    out.append({"family": family, "label_set": label_set, "view": view, "seeds": means,
                                "seed_mean": float(values.mean()), "seed_spread": float(values.max() - values.min())})
    return out


def _spread(spreads: list[dict], family: str, label_set: str, view: str) -> float | None:
    hits = [s["seed_spread"] for s in spreads if (s["family"], s["label_set"], s["view"]) == (family, label_set, view)]
    return hits[0] if hits else None


def decide(rows: pd.DataFrame) -> dict:
    """PREREG §5, exactly as written; everything except H1 is report-only."""
    provider = delta(rows, "provider", PRIMARY_VIEW, PRIMARY, REFERENCE)
    ctrl = delta(rows, "ctrl", PRIMARY_VIEW, PRIMARY, REFERENCE)
    provider_pass = provider["ci95"][0] > H1_PROVIDER_CI_LOW
    ctrl_pass = ctrl["ci95"][0] > H1_CTRL_CI_LOW
    passed = bool(provider_pass and ctrl_pass)
    spreads = seed_spreads(rows)
    spread = _spread(spreads, "exp4_full", "provider", PRIMARY_VIEW)
    noise = spread is not None and abs(provider["mean_delta"]) < spread
    report_only = {
        "H1_original_view": [delta(rows, ls, "original", PRIMARY, REFERENCE) for ls in ("provider", "ctrl")],
        "ablations_vs_reference": [delta(rows, ls, v, a, REFERENCE) for a in ("exp4_touch_ens3", "exp4_aux_ens3")
                                   for ls in ("provider", "ctrl") for v in VIEWS
                                   if available(a, v, ls)],
        "full_vs_ablations": [delta(rows, ls, PRIMARY_VIEW, PRIMARY, a) for a in ("exp4_touch_ens3", "exp4_aux_ens3")
                              for ls in ("provider", "ctrl") if available(a, PRIMARY_VIEW, ls)],
        "S1_full_vs_reference": [delta(rows, ls, v, PRIMARY, REFERENCE) for ls in S1_LABELS for v in VIEWS],
        "single_seeds_vs_reference": [delta(rows, ls, PRIMARY_VIEW, arm, REFERENCE)
                                      for arm in FAMILY_SEEDS["exp4_full"] for ls in ("provider", "ctrl")],
        "seed_spreads": spreads,
    }
    return {
        "rules": {
            "H1_provider": {**provider, "rule": f"paired 95% CI lower bound > {H1_PROVIDER_CI_LOW}",
                            "passed": bool(provider_pass)},
            "H1_ctrl_non_inferiority": {**ctrl, "rule": f"paired 95% CI lower bound > {H1_CTRL_CI_LOW}",
                                        "passed": bool(ctrl_pass)},
        },
        "H1": {"passed": passed, "comparison": [PRIMARY, REFERENCE], "view": PRIMARY_VIEW},
        "seed_honesty": {"exp4_full_seed_spread_provider_target_like": spread,
                         "provider_delta_smaller_than_seed_spread": bool(noise),
                         "statement": ("Δ is smaller than the seed spread: called noise in REPORT.md"
                                       if noise else "Δ is at least the seed spread")},
        "adoption": ({"write_files": True, "variant": PRIMARY, "members": list(FAMILY_SEEDS["exp4_full"]),
                      "files": {"provider": "predictions/exp_4_candidate_provider.csv",
                                "ctrl": "predictions/exp_4_candidate_ctrl.csv"},
                      "configs": {ls: final_config(PRIMARY, PRIMARY_VIEW, ls) for ls in ("provider", "ctrl")}}
                     if passed else
                     {"write_files": False, "statement": "H1 failed: no exp_4 files are written; exp_3's two "
                                                         "BAR-ATH files stay the team's options"}),
        "report_only": report_only,
    }


def final_config(variant: str, view: str, label_set: str) -> dict:
    folder = C3.RESULTS if variant in P.REUSED_VARIANTS else RESULTS
    return read_json(folder / f"variant_{variant}__{view}__{label_set}.json")["final_config"]


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
