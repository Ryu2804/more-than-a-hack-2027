"""Markdown tables for REPORT.md, built from ``results/*.json`` only (aggregates; never per-frame data).

Run ``python exp6_report.py`` to write ``results/report_tables.md``.
"""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = ("provider", "ctrl")
NAMES = {"provider": "provider", "ctrl": "`L_ctrl`"}


def _read(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _f(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "–"
    return f"{value:.{digits}f}"


def _delta(summary: dict) -> str:
    return (f"{summary['mean_delta']:+.4f} [{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}], "
            f"{summary['wins']}/{summary['matches']}")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def detector_tables(q: dict) -> list[str]:
    out = ["### Detector quality (BAR-ATH, ATD as reference)"]
    rows = [[name, q[name]["frames"], q[name]["accepted"], _f(q[name]["precision"]), _f(q[name]["recall"]),
             _f(q[name]["sigma_m"]), _f(q[name]["fp_rate"])] for name in ("val", "test")]
    out.append(table(["blocks", "frames", "accepted", "precision@2 m", "recall@2 m", "σ̂ (m, per axis)", "f̂"], rows))
    runs = q["run_lengths"]
    out.append(f"τ = {q['tau']} · ĉ = {_f(q['coverage'])} ({q['accepted_hidden_frames']} of {q['hidden_frames']} "
               f"hidden graded frames) · on-path share of hidden frames {_f(q['on_path_share_hidden'])} · "
               f"visibility ATD {_f(q['ball_visibility_graded']['atd'])} → with recovery "
               f"{_f(q['ball_visibility_graded']['atd_plus_recovered'])}")
    if runs.get("runs"):
        out.append(table(["accepted runs", "frames", "mean", "median", "p10", "p90", "max", "frames in runs ≥ 25"],
                         [[runs["runs"], runs["frames"], f"{runs['mean']:.1f}", runs["median"], runs["p10"],
                           runs["p90"], runs["max"], _f(runs["share_frames_in_runs_ge_25"])]]))
    out.append(table(["τ", "val precision", "val recall", "val accepted", "ĉ (hidden)"],
                     [[r["tau"], _f(r["precision"]), _f(r["recall"]), r["accepted"],
                       _f(q["coverage_by_tau"][str(r["tau"])])] for r in q["tau_table_val"]]))
    out.append(table(["nearest ATD player (m)", "test frames", "accepted", "precision@2 m", "recall@2 m", "σ̂ (m)"],
                     [[r["nearest_player_m"], r["frames"], r["accepted"], _f(r["precision"]), _f(r["recall"]),
                       _f(r["sigma_m"])] for r in q["test_by_nearest_player"]]))
    b = q["bridge"]
    out.append(table(["bridge", "n", "median (m)", "p90 (m)", "≤ 2 m", "≤ 5 m"],
                     [[k, b[k].get("n", 0), _f(b[k].get("median_m")), _f(b[k].get("p90_m")),
                       _f(b[k].get("share_le_2m")), _f(b[k].get("share_le_5m"))] for k in ("gap_start", "gap_end", "both")]))
    d = q["detector"]
    out.append(f"Training: {d['train_frames']} train-block frames, {d['samples_per_pass']} samples per pass, "
               f"{d['train_seconds']:.0f} s; inference on {d['inferred_frames']} frames, {d['inference_seconds']:.0f} s.")
    return out


def recheck_tables(r: dict) -> list[str]:
    s2 = r["s2_coverage"]
    rows = [["S(c = 0) tokens = exp_2 cached `target_like` tokens (7 matches)", r["tokens"]["all_equal"]],
            ["S(c = 0) rows = exp_3 stored `exp3_ref_ens3` rows (provider + `L_ctrl`)",
             f"{r['target_like_rows']['rows_equal']} (max abs diff {r['target_like_rows']['max_abs_diff']:.2g}; "
             f"max abs p diff vs cache {r['target_like_rows']['max_cached_probability_diff']:.2g})"],
            ["exp_3 S2 view regenerated = exp_3 S2 rows", f"{s2['rows_equal']} (max abs diff {s2['max_abs_diff']:.2g})"]]
    rows += [[f"S2 mean Δ, {NAMES[label]} (mine / stored)",
              f"{s2[f'delta_{label}']['mine']:+.4f} / {s2[f'delta_{label}']['stored']:+.4f}"] for label in LABELS]
    t = r["touch_target_like_rows"]
    rows.append(["`exp5_touch_ens3` `target_like` rows = exp_5 stored rows (context)",
                 f"{t['rows_equal']} (max abs diff {t['max_abs_diff']:.2g})"])
    return ["### Simulator rechecks (IDSSE)", table(["check", "result"], rows)]


def decision_tables(d: dict) -> list[str]:
    g = d["gates"]
    out = ["### Decision (`results/decision.json`)",
           table(["gate", "value", "threshold", "pass"],
                 [["G1 projection", _f(g["G1"]["value"]), f"≥ {g['G1']['threshold']}", g["G1"]["pass"]],
                  ["G2 precision@2 m (test)", _f(g["G2"]["precision"]), f"≥ {g['G2']['min_precision']}", g["G2"]["pass"]],
                  ["G2 recall@2 m (test)", _f(g["G2"]["recall"]), f"≥ {g['G2']['min_recall']}", g["G2"]["pass"]],
                  ["G3 coverage ĉ", _f(g["G3"]["value"]), f"≥ {g['G3']['threshold']}", g["G3"]["pass"]]])]
    rows = []
    for label in LABELS:
        rows.append([NAMES[label], "measured", _delta(d["ci_clause"][label]), "CI low > 0", d["ci_clause"][label]["pass"]])
        rows.append([NAMES[label], "stress", _delta(d["stress_clause"][label]), "mean ≥ −0.005",
                     d["stress_clause"][label]["pass"]])
    out.append(table(["label set", "setting", "Δ spell-F1 [95% CI], wins", "clause", "pass"], rows))
    s = d["settings"]
    out.append("Settings: " + "; ".join(f"{k} c = {v['coverage']:.3f}, σ = {v['noise']:.3f} m, f = {v['fp']:.3f}"
                                         for k, v in s.items()))
    out.append(f"**H1 passed: {d['h1_passed']}**" + (f" · failed: {', '.join(d['failed'])}" if d["failed"] else ""))
    return out


def voi_table(v: dict) -> list[str]:
    if not v or not v.get("computable"):
        return ["### VOI grid", "Not computable (no accepted run on BAR-ATH)."]
    rows = [[f"{c['coverage']:g}", f"{c['noise']:.3f}", f"{c['fp']:.3f}"] + [_delta(c[label]) for label in LABELS]
            for c in v["grid"]]
    return ["### VOI grid (report only)", table(["c", "σ (m)", "f", "provider Δ", "`L_ctrl` Δ"], rows)]


def touch_table(t: dict) -> list[str]:
    if not t:
        return []
    rows = [[NAMES[label], _delta(t["measured"][label])] for label in LABELS]
    return ["### `exp5_touch_ens3` context (report only)", table(["label set", "measured Δ [95% CI], wins"], rows),
            f"Recheck vs exp_5 stored `target_like` rows: {t['recheck_target_like']['rows_equal']}"]


def target_tables(t: dict) -> list[str]:
    if not t:
        return []
    b = t["build"]
    rows = []
    for key, a in b["aggregates"].items():
        rows.append([key, _f(a["dead_share"]), f"{a['possessions_per_min']:.2f}", f"{a['mean_possession_s']:.1f}",
                     a["possession_count"], _f(a["mean_p_dead"]), a.get("reproduces_exp3_file", "–")])
    out = ["### BAR-ATH aggregates (label-free)", f"Fill: {b['fill']}",
           table(["input__decoder", "dead share", "possessions/min", "mean possession (s)", "possessions",
                  "mean p(dead)", "= exp_3 file"], rows)]
    if "rebuild" in t:
        out.append(f"Rebuild byte-identical: {t['rebuild']['all_byte_identical']} · files: {t['rebuild']['files']}")
    return out


def main() -> None:
    parts: list[str] = []
    for name, builder in (("detector_quality.json", detector_tables), ("sim_rechecks.json", recheck_tables),
                          ("decision.json", decision_tables), ("voi_grid.json", voi_table),
                          ("touch_context.json", touch_table), ("target_candidates.json", target_tables)):
        data = _read(name)
        if data is not None:
            parts += builder(data)
    (RESULTS / "report_tables.md").write_text("\n\n".join(parts) + "\n")
    print("wrote results/report_tables.md")


if __name__ == "__main__":
    main()
