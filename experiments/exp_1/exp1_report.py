"""Aggregate proxy results: per-arm means, paired bootstrap deltas, quality gates."""

from __future__ import annotations

import pandas as pd

from exp1_common import MATCH_IDS, QUALITY_GATES, RESULTS, paired_bootstrap, write_json

MAIN = "setseq_bd"
LABELS = {
    "baseline": "starter rule: nearest player + +-50 majority (organisers)",
    "ryu_final": "ryu final: shipped TCN (Metrica+SkillCorner) + rate-calibrated 2-state Viterbi, zero-shot",
    "rasya_c06": "rasya 012 C06: touch-hold + out_or_slow2 dead cue + merge150",
    "rasya_c08": "rasya 012 C08: touch-hold + long-slow dead cue + merge150",
    "gbm_const": "exp_0: hand features + LightGBM + constant-penalty Viterbi",
    "setseq_argmax": "exp_1 SetSeq, no decoder (argmax)",
    "setseq_const": "exp_1 SetSeq + constant-penalty Viterbi (gamma=0)",
    "setseq_noaug_bd": "exp_1 SetSeq without corruption/mirror augmentation + BD decoder",
    "setseq_bd": "exp_1 SetSeq + boundary-aware Viterbi (main)",
    "setseq_bd_forced": "post-hoc diagnostic: SetSeq + boundary-aware Viterbi with gamma>0 forced",
}
ABLATIONS = (
    ("boundary_aware_decoding", "setseq_bd", "setseq_const"),
    ("temporal_decoder", "setseq_const", "setseq_argmax"),
    ("augmentation", "setseq_bd", "setseq_noaug_bd"),
    ("set_model_vs_gbm_same_decoder", "setseq_const", "gbm_const"),
    ("forced_boundary_decoding_posthoc", "setseq_bd_forced", "setseq_const"),
)


def _pivot(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    return rows.pivot(index="match", columns="variant", values=column).reindex(MATCH_IDS)


def _delta(rows: pd.DataFrame, a: str, b: str) -> dict:
    pivot = _pivot(rows, "spell_f1")
    delta = (pivot[a] - pivot[b]).to_numpy()
    low, high = paired_bootstrap(delta.tolist())
    decision = "supported" if low > 0 else "worse" if high < 0 else "inconclusive"
    return {"a": a, "b": b, "mean_delta": float(delta.mean()), "ci95": [low, high],
            "wins": int((delta > 0).sum()), "decision": decision}


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for variant in [v for v in LABELS if v in set(rows.variant)]:
        group = rows[rows.variant == variant].set_index("match").reindex(MATCH_IDS)
        delta = _delta(rows, variant, "baseline") if variant != "baseline" else None
        out.append({
            "variant": variant,
            "description": LABELS[variant],
            "mean_spell_f1": group.spell_f1.mean(),
            "min_spell_f1": group.spell_f1.min(),
            "mean_macro_f1": group.macro_f1.mean(),
            "mean_abs_count_error": group.count_error_ratio.abs().mean(),
            "max_abs_count_error": group.count_error_ratio.abs().max(),
            "delta_vs_baseline": delta["mean_delta"] if delta else 0.0,
            "ci95_low": delta["ci95"][0] if delta else 0.0,
            "ci95_high": delta["ci95"][1] if delta else 0.0,
        })
    return pd.DataFrame(out).sort_values("mean_spell_f1", ascending=False).reset_index(drop=True)


def gates(rows: pd.DataFrame, variant: str = MAIN) -> dict:
    main = rows[rows.variant == variant].set_index("match").reindex(MATCH_IDS)
    base = rows[rows.variant == "baseline"].set_index("match").reindex(MATCH_IDS)
    delta = _delta(rows, variant, "baseline")
    macro = float(main.macro_f1.mean() - base.macro_f1.mean())
    checks = {
        "gain_at_least_005": delta["mean_delta"] >= QUALITY_GATES["spell_gain"],
        "ci_lower_above_zero": delta["ci95"][0] > QUALITY_GATES["ci_low"],
        "macro_regression_at_most_001": macro >= QUALITY_GATES["macro_regression"],
        "all_match_counts_within_30pct": float(main.count_error_ratio.abs().max())
        <= QUALITY_GATES["max_abs_count_error"],
    }
    return {"variant": variant, "delta": delta, "macro_delta": macro,
            "checks": checks, "quality_gate_pass": all(checks.values())}


def build_report(rows: pd.DataFrame) -> dict:
    assert_proxy_only(rows)
    table = summary_table(rows)
    present = set(rows.variant)
    pairwise = [_delta(rows, MAIN, other) for other in LABELS if other != MAIN and other in present]
    ablations = [{"ablation": name, **_delta(rows, a, b)} for name, a, b in ABLATIONS
                 if {a, b} <= present]
    report = {
        "bootstrap": "paired match bootstrap, 1000 draws, seed 20261108, 95% percentile interval",
        "table": table.to_dict(orient="records"),
        "pairwise_main_vs": pairwise,
        "ablations": ablations,
        "gates": gates(rows),
    }
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "comparison_table.csv", index=False)
    rows.to_csv(RESULTS / "per_match_results.csv", index=False)
    write_json(RESULTS / "proxy_report.json", report)
    return report


def per_match_matrix(rows: pd.DataFrame, column: str = "spell_f1") -> pd.DataFrame:
    order = [v for v in LABELS if v in set(rows.variant)]
    return _pivot(rows, column)[order]


def as_markdown(table: pd.DataFrame) -> str:
    columns = ["variant", "mean_spell_f1", "mean_macro_f1", "mean_abs_count_error",
               "max_abs_count_error", "delta_vs_baseline", "ci95_low", "ci95_high"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in table[columns].iterrows():
        cells = [row["variant"]] + [f"{row[c]:+.4f}" if c.startswith(("delta", "ci95"))
                                    else f"{row[c]:.4f}" for c in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def assert_proxy_only(rows: pd.DataFrame) -> None:
    """Proxy reports must only ever contain the seven external matches."""
    if not set(rows.match) <= set(MATCH_IDS):
        raise RuntimeError("Non-proxy rows found in proxy report")
