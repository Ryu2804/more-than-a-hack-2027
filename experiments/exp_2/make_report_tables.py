"""Render the REPORT.md tables from results/ (aggregates only) into results/report_tables.md."""

from __future__ import annotations

import json

import pandas as pd

from exp2_common import RESULTS, VIEWS  # first: puts ../exp_1 on sys.path
import exp2_report as R  # noqa: E402


def _fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def main_table(table: pd.DataFrame) -> str:
    lines = []
    for view in VIEWS:
        part = table[table.view == view]
        lines += [f"**{view}**", "",
                  "| arm | mean spellF1 | min | macroF1 | mean / max abs count err | Δ vs exp1_ref [95% CI] | wins | count gate |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            delta = "–" if r.variant == R.REFERENCE else (
                f"{_fmt(r.delta_vs_exp1_ref, True)} [{_fmt(r.ci95_low, True)}, {_fmt(r.ci95_high, True)}]")
            wins = "–" if r.variant == R.REFERENCE else f"{int(r.wins_vs_exp1_ref)}/7"
            lines.append(f"| `{r.variant}` | {_fmt(r.mean_spell_f1)} | {_fmt(r.min_spell_f1)} | "
                         f"{_fmt(r.mean_macro_f1)} | {_fmt(r.mean_abs_count_error)} / {_fmt(r.max_abs_count_error)} | "
                         f"{delta} | {wins} | {'pass' if r.count_gate else 'FAIL'} |")
        lines.append("")
    return "\n".join(lines)


def rules_table(decision: dict) -> str:
    lines = ["| rule | Δ [95% CI] / statistic | result |", "|---|---|---|"]
    for name, rule in decision["rules"].items():
        if "mean_delta" in rule:
            low, high = rule["ci95"]
            stat = f"{_fmt(rule['mean_delta'], True)} [{_fmt(low, True)}, {_fmt(high, True)}], wins {rule['wins']}/7"
            if "seed_spread" in rule:
                stat += f", seed spread {rule['seed_spread']:.3f}"
        else:
            stat = f"max abs count error {rule['max_abs_count_error']:.3f}"
        lines.append(f"| {name} ({rule['rule']}) | {stat} | {'**passed**' if rule['passed'] else '**failed**'} |")
    lines.append(f"\nSupported: {decision['supported']}; adopt exp_2 candidate: {decision['adopt_exp2_candidate']}")
    return "\n".join(lines)


def per_match(rows: pd.DataFrame, variants: list[str]) -> str:
    out = []
    for view in VIEWS:
        part = rows[rows.view == view]
        f1 = part.pivot(index="match", columns="variant", values="spell_f1")[variants]
        counts = part.pivot(index="match", columns="variant", values="count_error_ratio")[variants]
        out += [f"**{view}** (spellF1; count error in brackets)", "",
                "| match | " + " | ".join(f"`{v}`" for v in variants) + " |",
                "|---" * (len(variants) + 1) + "|"]
        for match in f1.index:
            cells = [f"{f1.loc[match, v]:.3f} ({counts.loc[match, v]:+.2f})" for v in variants]
            out.append(f"| {match} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    table = pd.read_csv(RESULTS / "comparison_table.csv")
    rows = pd.read_csv(RESULTS / "per_match_results.csv")
    decision = json.loads((RESULTS / "decision.json").read_text())
    report = json.loads((RESULTS / "proxy_report.json").read_text())
    shift = "\n".join(f"* `{s['variant']}`: {s['mean_change']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
                      for s in report["view_shift"])
    gates = pd.DataFrame([{"view": g["view"], "variant": g["variant"], **g["checks"], "pass": g["pass"],
                           "worst": f"{g['worst_count_match']} {g['worst_count_error']:.2f}"}
                          for g in report["gates"]])
    parts = ["## Main table", main_table(table), "## Rules", rules_table(decision),
             "## Per match", per_match(rows, ["exp1_ref", "exp2_aug", "exp2_aug_hsmm", "exp2_aug_ens3", "ryu_final"]),
             "## View shift (target_like - original)", shift, "## Gates", gates.to_markdown(index=False)]
    (RESULTS / "report_tables.md").write_text("\n\n".join(parts) + "\n")
    print("wrote", RESULTS / "report_tables.md")


if __name__ == "__main__":
    main()
