"""Render the README figures into image/ (PNG, web width).

Every number comes from committed evidence: the submitted CSV (carried_by_admin.csv, the final method
retrained from scratch), the public ground truth, the official baseline CSVs and each experiment's
results/*.json. Data handling is shared with the technical report (docs/report/make_figures.py).
The explainability figures (fig10-fig14) are the matplotlib outputs of the executed E06 notebook
(final_notebook.ipynb), copied unchanged.

    python image/make_figures.py

Palette: Okabe-Ito (colour-blind safe); every colour is paired with a second cue (hatching,
marker shape or a direct label).
"""
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "experiments"

spec = importlib.util.spec_from_file_location("report_figures", ROOT / "docs/report/make_figures.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)  # loads GT, frozen CSV, baselines; draws nothing
# the README describes the submitted file: the same method retrained from scratch
SUBMITTED_TEXT = (ROOT / "carried_by_admin.csv").read_text()
R.w5 = R.labels_csv(ROOT / "carried_by_admin.csv")

# Okabe-Ito
BLUE, ORANGE, VERM, GREEN, SKY, PURPLE, BLACK, GREY = (
    "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#000000", "#999999")
C_H, C_A, C_D = VERM, BLUE, "#BBBBBB"
DPI = 200
mpl.rcdefaults()
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=DPI)
    plt.close(fig)


def load(exp, name):
    return json.loads((EXP / exp / "results" / name).read_text())


# ------------------------------------------------------------------ 1. pipeline
def fig_pipeline():
    """The paper's Fig. 1, drawn by the report's own code and style, saved as a README PNG."""
    orig = R.save
    R.save = lambda fig, name: (fig.savefig(HERE / "fig01_pipeline.png", dpi=300), plt.close(fig))
    try:
        with mpl.rc_context(R.RC):
            R.fig_pipeline()
    finally:
        R.save = orig


# ------------------------------------------------------------------ 2. public timeline
def fig_timeline():
    lo, hi = R.lo, R.hi
    rows = [("Ground truth", R.gt), ("Final method (wave5)", R.w5), ("Official baseline smooth50", R.b50)]
    fig, ax = plt.subplots(figsize=(12, 2.9), layout="constrained")
    style = {"H": dict(fc=C_H), "A": dict(fc=C_A), "D": dict(fc=C_D, hatch="////", ec="white")}
    for k, (name, lab) in enumerate(rows):
        y = len(rows) - 1 - k
        for l, s, n in R.runs(lab[lo:hi + 1]):
            if l in style:
                ax.broken_barh([((lo + s) / 1500, n / 1500)], (y - 0.38, 0.76), lw=0, **style[l])
    ax.set(yticks=range(3), yticklabels=[r[0] for r in rows][::-1], xlim=(lo / 1500, (hi + 1) / 1500),
           xlabel="Tactical video time (min)", title="Public 5-minute window: labels per frame")
    ax.grid(False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(fc=C_H, label="H  home (Barcelona) in possession"), Patch(fc=C_A, label="A  away (Athletic) in possession"),
                       Patch(fc=C_D, hatch="////", ec="white", label="D  dead ball")],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    save(fig, "fig02_public_timeline")


# ------------------------------------------------------------------ 3. scoreboard
def fig_scoreboard(board):
    order = ["baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125", "wave5"]
    names = ["Baseline raw", "Baseline ±25", "Baseline ±50 (official floor)", "Baseline ±125", "Final method (wave5)"]
    b = board.set_index("submission").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.5, 1]))
    y = np.arange(len(order))
    for i in y:
        ax[0].plot([b.macro_f1.iloc[i], b.spell_f1_mean.iloc[i]], [i, i], color=GREY, lw=1.2, zorder=1)
    ax[0].scatter(b.macro_f1, y, s=60, marker="s", color=ORANGE, label="macroF1 (per frame)", zorder=2)
    ax[0].scatter(b.spell_f1_mean, y, s=70, marker="o", color=BLUE, label="spellF1 (ranking metric)", zorder=3)
    for i in y:
        ax[0].text(b.spell_f1_mean.iloc[i], i - 0.32, f"{b.spell_f1_mean.iloc[i]:.3f}", ha="center", va="top", fontsize=8.5, color=BLUE)
    ax[0].set(yticks=y, yticklabels=names, xlim=(0, 1), ylim=(-0.8, len(order) - 0.5), xlabel="Score on the public window",
              title="Public 5-minute GT (official scorer)")
    ax[0].legend(loc="lower right")
    ax[1].barh(y, b.poss, color=[GREY] * 4 + [BLUE], height=0.6)
    ax[1].axvline(17, color=VERM, ls="--", lw=1.2)
    ax[1].text(19, -0.55, "17 GT possessions", color=VERM, fontsize=8.5, va="center")
    for i in y:
        ax[1].text(b.poss.iloc[i] + 2, i, str(int(b.poss.iloc[i])), va="center", fontsize=8.5)
    ax[1].set(yticks=y, yticklabels=[], xlim=(0, 150), ylim=(-0.8, len(order) - 0.5), xlabel="Predicted possessions in the window", title="Segmentation count")
    save(fig, "fig03_public_scoreboard")


# ------------------------------------------------------------------ 4. decision forest
def fig_forest():
    rows = []
    dev = load("E06_wave5", "wave5_development.json")
    sel = next(r for r in dev["records"] if r["family"] == "short_gru25" and r["decoder"]["name"] == "dead45")
    rows.append(("E06 wave5: short-loss PP-TCN + BiGRU / dead45", sel["gain_vs_control"], None, None, 0.01, True))
    w6 = load("E07_wave6", "wave6_development.json")
    b6 = next(r for r in w6["records"] if r["name"] == w6["best_new"])
    rows.append(("E07 wave6: dead hysteresis (best of 36)", b6["gain_vs_control"], b6["paired_bootstrap"]["lower"], b6["paired_bootstrap"]["upper"], 0.005, w6["promoted"]))
    for exp, name, label in (("E09_cross_branch", "cross_branch_results.json", "E09 cross-branch: triple combination"),
                             ("E10_temporal_methods", "temporal_methods_results.json", "E10 temporal: duration HSMM"),
                             ("E11_spell_training", "spell_training_results.json", "E11 spell training: event sampling"),
                             ("E12_event_learning", "event_learning_results.json", "E12 event learning: balanced loss")):
        d = load(exp, name); p = d["primary"]["paired_vs_control"]
        rows.append((label, p["mean_delta"], p["lower"], p["upper"], 0.005, d["primary_promoted_for_research"]))
    d = load("E13_provider_generalization", "provider_generalization_results.json")
    m = next(r for r in d["records"] if r["family"] == "modality_dropout")["summary"]["clean"]["paired_vs_control"]
    rows.append(("E13 provider transfer: modality dropout", m["mean_delta"], m["lower"], m["upper"], 0.01, d["primary_promoted_for_research"]))

    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    y = np.arange(len(rows))[::-1]
    for yi, (label, g, lo, hi, thr, prom) in zip(y, rows):
        col = GREEN if prom else VERM
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.2, solid_capstyle="butt")
        ax.scatter([g], [yi], s=80, marker="D" if prom else "X", color=col, zorder=3, edgecolor="white", lw=0.6)
        ax.plot([thr, thr], [yi - 0.35, yi + 0.35], color=BLACK, lw=1.4)
        ax.text(0.031, yi, f"{g:+.4f}" + ("" if lo is not None else "  (no CI recorded)"), va="center", fontsize=8.5)
    ax.axvline(0, color=GREY, lw=1, ls="--")
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xlim=(-0.022, 0.045),
           xlabel="Primary minus its own control, mean spellF1 (95% paired match-bootstrap CI)",
           title="Pre-registered decisions: gain of each campaign's primary over its control")
    ax.legend(handles=[Line2D([], [], marker="D", color=GREEN, ls="-", label="promoted"),
                       Line2D([], [], marker="X", color=VERM, ls="-", label="not promoted"),
                       Line2D([], [], color=BLACK, marker="|", ms=12, ls="none", label="pre-registered minimum gain")],
              loc="upper left")
    fig.text(0.01, -0.02, "Controls differ per campaign (cohorts: E06/E07 three providers; E09–E12 seven IDSSE LOSO folds; "
             "E13 cross-provider transfer). Compare a row only with its own threshold.", fontsize=8, color="#555555")
    save(fig, "fig08_decision_forest")


# ------------------------------------------------------------------ 5. development grid
def fig_grid():
    dev = load("E06_wave5", "wave5_development.json")
    F, D = R.FAMILY_ORDER, R.DECODER_ORDER
    grid = np.full((len(F), len(D)), np.nan); elig = np.zeros_like(grid, bool)
    for r in dev["records"]:
        i, j = F.index(r["family"]), D.index(r["decoder"]["name"])
        grid[i, j] = r["external_mean_spellF1"]; elig[i, j] = r["eligible"]
    fig, ax = plt.subplots(figsize=(10, 6.2), layout="constrained")
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    thr = np.nanpercentile(grid, 60)
    for i in range(len(F)):
        for j in range(len(D)):
            ax.text(j, i, f"{grid[i, j]:.3f}" + ("*" if elig[i, j] else ""), ha="center", va="center", fontsize=8,
                    color="black" if grid[i, j] > thr else "white", weight="bold" if elig[i, j] else "normal")
    bi = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((bi[1] - .5, bi[0] - .5), 1, 1, fill=False, ec=VERM, lw=2.5))
    ci = (F.index("old_gru25"), D.index("wave4"))
    ax.add_patch(plt.Rectangle((ci[1] - .5, ci[0] - .5), 1, 1, fill=False, ec="white", lw=2, ls="--"))
    ax.set(xticks=range(len(D)), yticks=range(len(F)), xticklabels=D, yticklabels=F, xlabel="Decoder", ylabel="Probability family",
           title="E06 development sweep: 96 configurations (equal-provider spellF1)")
    ax.grid(False); ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="Development spellF1", shrink=0.85)
    fig.text(0.01, -0.03, "* passes the pre-registered adoption rule (6 of 96).  Solid red box: selected short_gru25/dead45 (0.670).  "
             "Dashed white box: wave4 control (0.656).", fontsize=8.5, color="#333333")
    save(fig, "fig04_development_grid")


# ------------------------------------------------------------------ 6. provider + holdout with per-unit points
def fig_provider():
    dev = load("E06_wave5", "wave5_development.json")
    rec = {(r["family"], r["decoder"]["name"]): r for r in dev["records"]}
    sel, ctl = rec[("short_gru25", "dead45")], rec[("old_gru25", "wave4")]
    ho = R.holdout_cohorts()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), layout="constrained")
    provs = ["idsse", "skillcorner", "metrica"]
    pm = lambda r: {d["source"]: d["spellF1"] for d in r["provider_means"]}
    cats = ["IDSSE", "SkillCorner", "Metrica", "Equal mean"]
    o = [pm(ctl)[p] for p in provs] + [ctl["external_mean_spellF1"]]
    n = [pm(sel)[p] for p in provs] + [sel["external_mean_spellF1"]]
    order = ["IDSSE (2)", "SkillCorner (6) clean", "SkillCorner (6) stress", "Metrica G3 (2 cond.)"]
    for a, labels, oo, nn, title in ((ax[0], cats, o, n, "Development (grouped folds, 23 scenarios)"),
                                     (ax[1], ["IDSSE\n(2 matches)", "SkillCorner\n(6) clean", "SkillCorner\n(6) ball-drop", "Metrica G3\n(2 cond.)"],
                                      [ho[c]["previous_wave4"] for c in order], [ho[c]["selected"] for c in order],
                                      "Post-freeze holdout (previously exposed)")):
        x = np.arange(len(labels))
        for xi, a0, a1 in zip(x, oo, nn):
            a.plot([xi, xi], [a0, a1], color=GREY, lw=1.5, zorder=1)
            a.text(xi + 0.08, (a0 + a1) / 2, f"{a1 - a0:+.3f}", fontsize=8.5, va="center", color=GREEN if a1 >= a0 else VERM)
        a.scatter(x, oo, marker="o", s=60, facecolor="white", edgecolor=GREY, lw=1.5, label="wave4 control", zorder=2)
        a.scatter(x, nn, marker="D", s=60, color=BLUE, label="wave5 (final)", zorder=3)
        a.set(xticks=x, xticklabels=labels, ylabel="Mean spellF1", title=title, ylim=(0.56, 0.73))
    ax[0].legend(loc="lower right")
    save(fig, "fig05_provider_holdout")


# ------------------------------------------------------------------ 7. duration recall
def fig_recall():
    d = load("E06_wave5", "wave5_duration_diagnostics.json")["records"]
    cohorts = {"IDSSE (2)": lambda r: r["match"] in ("J03WQQ", "J03WR9"),
               "SkillCorner (6) clean": lambda r: r["match"] in R.HOLDOUT_SC and r["regime"] == "clean",
               "SkillCorner (6) ball-drop": lambda r: r["match"] in R.HOLDOUT_SC and r["regime"] == "stress",
               "Metrica G3 (2 cond.)": lambda r: r["match"] == "metrica3"}
    buckets = [("lt_4s", "< 4 s"), ("4_to_lt_15s", "4–15 s"), ("gte_15s", "≥ 15 s")]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6), layout="constrained", sharey=True)
    for a, (cname, f) in zip(axs, cohorts.items()):
        x = np.arange(3); w = 0.38
        for k, (cand, col, hatch, lab) in enumerate((("previous_wave4", "#CCCCCC", "", "wave4 control"), ("selected", BLUE, "", "wave5 (final)"))):
            rs = [r for r in d if r["candidate"] == cand and f(r)]
            g = [sum(r["duration_buckets"][b]["gt_count"] for r in rs) for b, _ in buckets]
            m50 = [sum(r["duration_buckets"][b]["at_iou"]["0.5"]["matched_count"] for r in rs) for b, _ in buckets]
            m75 = [sum(r["duration_buckets"][b]["at_iou"]["0.75"]["matched_count"] for r in rs) for b, _ in buckets]
            a.bar(x + (k - 0.5) * w, np.divide(m50, g), w, color=col, edgecolor=BLACK, lw=0.5, label=lab)
            a.scatter(x + (k - 0.5) * w, np.divide(m75, g), marker="_", s=120, color=VERM, lw=2, zorder=3,
                      label="recall at IoU 0.75" if k == 1 else None)
        a.set(xticks=x, xticklabels=[f"{l}\nn={gi}" for (_, l), gi in zip(buckets, g)], title=cname, ylim=(0, 1))
    axs[0].set_ylabel("Matched recall at IoU 0.5")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("E06 holdout: recall of GT possessions by duration (bars IoU 0.5, red ticks IoU 0.75)", weight="bold", fontsize=11)
    save(fig, "fig06_duration_recall")


# ------------------------------------------------------------------ 8. fair benchmark with raw points
def fig_benchmark():
    d = load("E08_fair_idsse", "fair_idsse_results.json")
    fams = [("short_tcn", "Short-loss PP-TCN\n(Farras)"), ("setseq", "SetSeq\n(Jeremy)"), ("pp_tcn", "PP-TCN\n(Ryu)"), ("hgb", "HGB symmetric\n(Farras wave2)")]
    rows = [r for r in d["rows"] if r["mode"] == "shared_decoder"]
    matches = sorted({r["match"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    vals = {f: [next(r["spell_f1_mean"] for r in rows if r["family"] == f and r["match"] == m) for m in matches] for f, _ in fams}
    for mi, m in enumerate(matches):
        ax.plot(range(len(fams)), [vals[f][mi] for f, _ in fams], color="#DDDDDD", lw=0.8, zorder=1)
    for i, (f, _) in enumerate(fams):
        ax.scatter([i] * len(matches), vals[f], s=22, color=GREY, zorder=2)
        mean = np.mean(vals[f])
        ax.plot([i - 0.25, i + 0.25], [mean, mean], color=BLUE if i == 0 else BLACK, lw=2.5, zorder=3)
        ax.text(i + 0.28, mean, f"{mean:.3f}", va="center", fontsize=9, color=BLUE if i == 0 else BLACK)
    ax.set(xticks=range(len(fams)), xticklabels=[n for _, n in fams], ylabel="spellF1 on held-out match", ylim=(0, 0.75),
           title="E08 shared benchmark: 7 IDSSE leave-one-match-out folds, same decoder (dead45)")
    fig.text(0.01, -0.03, "Grey dots: one held-out match each (lines join the same match). Bars: mean over 7 matches. "
             "Equal update budget (3 seeds × 500 updates); adapted families, not each branch's exact submission.", fontsize=8, color="#555555")
    save(fig, "fig09_fair_benchmark")


# ------------------------------------------------------------------ 9. output anatomy
def fig_anatomy(report, dur):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    gt_d = R.possession_durations(R.gt, [(R.lo, R.hi)]); w5_full = R.possession_durations(R.w5, R.halves)
    bins = np.logspace(np.log10(0.2), np.log10(200), 24)
    ax[0].hist(w5_full, bins=bins, weights=np.full(len(w5_full), 1 / len(w5_full)), color=BLUE, alpha=0.75,
               label=f"wave5, full match (n={len(w5_full)})")
    ax[0].hist(gt_d, bins=bins, weights=np.full(len(gt_d), 1 / len(gt_d)), histtype="step", color=VERM, lw=2,
               label=f"GT, public window (n={len(gt_d)})")
    ax[0].axvline(4, color=BLACK, ls=":", lw=1); ax[0].text(4.4, 0.97, "4 s: short-loss threshold", fontsize=8, transform=ax[0].get_xaxis_transform(), va="top")
    ax[0].set(xscale="log", xlabel="Possession duration (s, log scale)", ylabel="Fraction of possessions", title="Possession durations")
    ax[0].set_ylim(0, 0.26)
    ax[0].legend(loc="upper left", fontsize=8, framealpha=0.95)
    x = np.arange(3); w = 0.27
    pub_gt = [np.mean(R.gt[R.lo:R.hi + 1] == l) for l in "HAD"]
    pub_w5 = [np.mean(R.w5[R.lo:R.hi + 1] == l) for l in "HAD"]
    full = [dur["full_share"][k] for k in "HAD"]
    ax[1].bar(x - w, pub_gt, w, color=VERM, label="GT, public")
    ax[1].bar(x, pub_w5, w, color=BLUE, label="wave5, public")
    ax[1].bar(x + w, full, w, color="#CCCCCC", hatch="///", edgecolor="#777777", label="wave5, full match")
    ax[1].set(xticks=x, xticklabels=["H", "A", "D"], ylabel="Share of graded frames", title="Label shares", ylim=(0, 0.6))
    ax[1].legend(fontsize=8)
    cm = np.array(report["confusion"])[:3, :3]; nm = cm / cm.sum(1, keepdims=True)
    ax[2].imshow(nm, cmap="Blues", vmin=0, vmax=1); ax[2].grid(False)
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, f"{cm[i, j]}\n{nm[i, j]:.1%}", ha="center", va="center", fontsize=8.5, color="white" if nm[i, j] > 0.5 else BLACK)
    ax[2].set(xticks=range(3), yticks=range(3), xticklabels=list("HAD"), yticklabels=list("HAD"),
              xlabel="Predicted", ylabel="Ground truth", title="Frame confusion (public)")
    save(fig, "fig07_output_anatomy")


def score_submitted():
    """Official scorer on the public GT for the submitted CSV and the four baselines."""
    import subprocess, sys, tempfile
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "sub"; sub.mkdir()
        (sub / "wave5.csv").write_text(SUBMITTED_TEXT)
        for b in ("raw", "smooth25", "smooth50", "smooth125"):
            (sub / f"baseline_{b}.csv").write_bytes((ROOT / f"evaluation/results/baseline_{b}.csv").read_bytes())
        subprocess.run([sys.executable, str(ROOT / "evaluation/run_evaluation.py"), "--results", str(sub),
                        "--out", tmp, "--no-detail"], check=True, capture_output=True)
        board = pd.read_csv(Path(tmp) / "scoreboard.csv")
        rep = json.loads((Path(tmp) / "reports" / "wave5.json").read_text())
    return board, rep


def export_notebook_figures():
    """Copy the explainability figures of the executed E06 notebook (if present) into image/."""
    import base64
    nb_path = ROOT / "final_notebook.ipynb"
    if not nb_path.exists():
        print("final_notebook.ipynb not found; keeping existing fig10-fig14"); return
    wanted = {"## [9] Anatomy of a decision": "fig10_decision_anatomy", "## [11] Which inputs drive": "fig11_integrated_gradients",
              "## [12] How far in time": "fig12_temporal_receptive_field", "## [13] Which players": "fig13_player_attention",
              "## [15] Where the three seeds": "fig14_seed_disagreement",
              "## [10] Counterfactual stage ablation": "fig15_stage_ablation", "## [14] Feature-group occlusion": "fig16_occlusion",
              "## [16] Error anatomy": "fig17_error_anatomy"}
    cells = json.loads(nb_path.read_text())["cells"]
    for i, c in enumerate(cells):
        head = "".join(c["source"]) if c["cell_type"] == "markdown" else ""
        for key, name in wanted.items():
            if key in head:
                code = next(x for x in cells[i + 1:] if x["cell_type"] == "code")
                png = next(o["data"]["image/png"] for o in code["outputs"] if "data" in o and "image/png" in o["data"])
                (HERE / f"{name}.png").write_bytes(base64.b64decode(png))
                whiten(HERE / f"{name}.png")


def whiten(path, cream=(0xF6, 0xF7, 0xEB)):
    """The notebook draws on its cream palette background; README figures use plain white."""
    from PIL import Image
    im = np.array(Image.open(path).convert("RGB")).astype(int)
    im[(np.abs(im - cream) <= 2).all(-1)] = 255
    Image.fromarray(im.astype(np.uint8)).save(path)


def main():
    board, report = score_submitted()
    full = np.concatenate([R.w5[a:b + 1] for a, b in R.halves])
    durinfo = {"full_share": {l: float(np.mean(full == l)) for l in "HAD"}}
    fig_pipeline(); fig_timeline(); fig_scoreboard(board); fig_forest(); fig_grid()
    fig_provider(); fig_recall(); fig_benchmark(); fig_anatomy(report, durinfo)
    export_notebook_figures()
    print("wrote", sorted(p.name for p in HERE.glob("*.png")))


if __name__ == "__main__":
    main()
