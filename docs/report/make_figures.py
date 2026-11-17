"""Generate every figure and data table used by docs/report/report.typ.

Inputs are files already in the repository: the submitted CSV (carried_by_admin.csv, the method
retrained from scratch) and the frozen wave5 candidate, read from the locked release commit with
`git show`. Figures describe the submitted file; the frozen one is re-scored for comparison.
Run from anywhere:  python docs/report/make_figures.py
"""
from __future__ import annotations

import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "experiments" / "E06_wave5" / "results"
METHOD = json.loads((ROOT / "docs" / "final_method.json").read_text())
RELEASE = METHOD["source"]["release_commit"]
FPS = 25

# One palette for the whole report: home, away, dead; control vs selected.
C_H, C_A, C_D = "#e94f37", "#393e41", "#c9c9c9"
C_OLD, C_NEW = "#9aa5ab", "#1f6f8b"
COL1, COL2 = 3.45, 7.16  # IEEE single / double column width, inches

RC = {  # also reused by image/make_figures.py for the README copy of the pipeline figure
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}
mpl.rcParams.update(RC)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def labels_csv(text_or_path):
    df = pd.read_csv(text_or_path, dtype={"label": str})
    assert list(df.columns) == ["frame", "label"] and (df.frame.values == np.arange(len(df))).all()
    return df.label.values


def runs(lab):
    """Maximal runs of identical labels -> (label, start, length)."""
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(lab)]))
    return [(lab[s], s, e - s) for s, e in zip(starts, ends)]


# ---------------------------------------------------------------- data
meta = json.loads((ROOT / "evaluation/gt/BAR-ATH_possession_gt_public_meta.json").read_text())
gt = labels_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
RETRAIN = json.loads((RES / "wave5_retrain_verification.json").read_text())
frozen_text = subprocess.check_output(
    ["git", "-C", str(ROOT), "show", f"{RELEASE}:{METHOD['artifact']['path']}"], text=True)
w5_text = (ROOT / "carried_by_admin.csv").read_text()   # submitted file
w5 = labels_csv(io.StringIO(w5_text))
b50 = labels_csv(ROOT / "evaluation/results/baseline_smooth50.csv")
assert len(w5) == METHOD["artifact"]["rows"] == len(gt) == len(b50)

lo, hi = meta["covered_video_frames"][0]
win = slice(lo, hi + 1)
halves = [(a - meta["frame_offsets"][h], b - meta["frame_offsets"][h])
          for h, (a, b) in meta["half_id_frames"].items()]


# ---------------------------------------------------------------- Fig. 1 pipeline
def fig_pipeline():
    """Four stage panels: input -> representation -> models -> decoding. Boxes carry a title and at most
    one short line; dimensions and details sit on the connectors or under the boxes."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(COL2, 2.45))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 200), ylim=(7, 70)); ax.axis("off")
    INK, LINE = "#1a1a1a", dict(color="#1a1a1a", lw=1.0, solid_capstyle="butt", zorder=4)

    def panel(x, w, title, ec, fc):
        ax.add_patch(FancyBboxPatch((x, 8), w, 60, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc=fc, ec=ec, lw=0.9, ls=(0, (4, 2))))
        ax.text(x + w / 2, 64.6, title, ha="center", va="center", weight="bold", fontsize=6.8, color=INK)

    def box(x, y, w, h, title, sub=None, fc="white", ec=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=0.8, zorder=3))
        cy = y + h / 2
        if sub:
            ax.text(x + w / 2, cy + 1.9, title, ha="center", va="center", weight="bold", fontsize=5.5, color=INK, zorder=6)
            ax.text(x + w / 2, cy - 2.2, sub, ha="center", va="center", fontsize=5.0, color=INK, zorder=6)
        else:
            ax.text(x + w / 2, cy, title, ha="center", va="center", weight="bold", fontsize=5.5, color=INK,
                    linespacing=1.1, zorder=6)


    def path(*pts, label=None, at=0):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1], ys[:-1], **LINE)
        ax.annotate("", pts[-1], pts[-2], zorder=4, arrowprops=dict(
            arrowstyle="-|>,head_length=0.45,head_width=0.22", lw=1.0, color=INK, shrinkA=0, shrinkB=0))
        if label:
            (x0, y0), (x1, y1) = pts[at], pts[at + 1]
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 2.6, label, ha="center", va="center", fontsize=4.7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=6)

    def line(*pts):
        xs, ys = zip(*pts); ax.plot(xs, ys, **LINE)

    def dot(x, y):
        ax.plot(x, y, "o", ms=2.6, color=INK, zorder=5)

    # ---------------- 1. input
    panel(1, 33, "Input", "#8c8c8c", "#f4f4f4")
    box(4, 36, 27, 24, "Automatic tracking")
    ax.texts[-1].set_position((17.5, 56.5))
    ax.add_patch(Rectangle((7.5, 39), 20, 12, fc="#e3f0dc", ec="#5b8c4a", lw=0.6, zorder=5))
    ax.plot([17.5, 17.5], [39, 51], color="#5b8c4a", lw=0.5, zorder=5)
    rng = np.random.default_rng(3)
    for c, n in ((C_H, 7), (C_A, 7)):
        ax.plot(rng.uniform(8.5, 26.5, n), rng.uniform(40, 50, n), "o", ms=1.6, color=c, zorder=6)
    ax.plot(19.5, 45, "o", ms=1.8, mfc="white", mec=INK, mew=0.5, zorder=6)
    box(4, 14, 27, 13, "Resample + mirror", "25 → 5 fps · 2 views")
    path((17.5, 36), (17.5, 27))
    # ---------------- 2. representation
    panel(37, 50, "Representation", "#1f6f8b", "#eaf3f6")
    for y, title, fc in ((51, "Team shape", "#d9ecf2"), (35, "Ball + validity", "#fbe3de"),
                         (19, "Player set", "#ece4f2")):
        box(42, y, 20.5, 10, title, fc=fc)
    line((31, 20.5), (39, 20.5)); line((39, 20.5), (39, 56)); dot(39, 24); dot(39, 40)
    for y in (56, 40, 24):
        path((39, y), (42, y))
    for y, lab in ((56, "23"), (40, "1+5"), (24, "48×10")):
        line((62.5, y), (70, y)); ax.text(66.3, y + 1.2, lab, ha="center", va="bottom", fontsize=4.4, color=INK)
    line((70, 24), (70, 56)); dot(70, 40)
    grid = mpl.colormaps["viridis"](np.random.default_rng(7).uniform(0.15, 0.95, (5, 5)))
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((72.5 + 2.4 * j, 34 + 2.4 * i), 2.4, 2.4, fc=grid[i, j], ec="white", lw=0.3, zorder=5))
    path((70, 40), (72.5, 40))
    ax.text(78.5, 30.6, "29 ch × T\n+ player set", ha="center", va="center", fontsize=4.9, weight="bold", color=INK)
    # ---------------- 3. models
    panel(90, 33, "Models", "#2a9d6f", "#eaf6f0")
    box(92.5, 46, 28, 12, "Short-loss PP-TCN", "3 seeds × 2 views", fc="#d9ecf2", ec="#1f6f8b")
    box(92.5, 20, 28, 12, "Set-sequence BiGRU", "2-layer, hidden 32", fc="#fdebd0", ec="#c47f1a")
    line((84.5, 40), (89, 40)); line((89, 26), (89, 52)); dot(89, 40)
    path((89, 52), (92.5, 52)); path((89, 26), (92.5, 26))
    # ---------------- 4. decoding and output: one straight chain, rate from above, output below
    panel(126, 73, "Decoding (dead45) and output", "#c0392b", "#fbeeec")
    ax.add_patch(Circle((130.5, 39), 2.2, fc="white", ec=INK, lw=0.9, zorder=5))
    ax.text(130.5, 39.1, "+", ha="center", va="center", fontsize=7, weight="bold", zorder=6)
    path((120.5, 52), (130.5, 52), (130.5, 41.2)); path((120.5, 26), (130.5, 26), (130.5, 36.8))
    for y, w in ((47, "0.75"), (31, "0.25")):
        ax.text(131.6, y, w, ha="left", va="center", fontsize=4.9, color=INK)
    box(135.5, 34, 17, 10, "Dead mask", r"$p_D > 0.45$")
    box(156, 34, 19, 10, "Team evidence", r"$\kappa \log p_T$")
    box(178.5, 34, 19.5, 10, "Interval-reset\nViterbi", fc="#fadbd8", ec="#c0392b")
    box(178.5, 51, 19.5, 10, "Rate ρ", "3.64 / min", fc="#fff6dc", ec="#c47f1a")
    box(178.5, 15, 19.5, 10, "Merge < 1 s", "→ 25 fps")
    path((132.7, 39), (135.5, 39)); path((152.5, 39), (156, 39)); path((175, 39), (178.5, 39))
    path((188.25, 51), (188.25, 44)); ax.text(189.7, 47.5, "λ", ha="left", va="center", fontsize=5.4, color=INK)
    path((188.25, 34), (188.25, 25))
    # output document icon, left of the merge step
    ax.add_patch(Polygon([(163, 12), (171, 12), (171, 26), (168.6, 28.4), (163, 28.4)], closed=True,
                         fc="white", ec=INK, lw=0.8, zorder=5))
    for k, c in enumerate((C_H, C_A, C_D, C_H)):
        ax.add_patch(Rectangle((164.2, 14 + 3 * k), 5.6, 1.9, fc=c, ec="none", zorder=6))
    path((178.5, 20), (171, 20))
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------- Fig. 2 public window
def fig_public_timeline():
    rows = [("Ground truth", gt[win]), ("Submitted (ours)", w5[win]), ("Baseline smooth50", b50[win])]
    t0 = lo / FPS / 60
    fig, ax = plt.subplots(figsize=(COL2, 1.45))
    cmap = {"H": C_H, "A": C_A, "D": C_D, "X": "white"}
    for k, (name, lab) in enumerate(rows):
        y = len(rows) - 1 - k
        for l, s, n in runs(lab):
            ax.broken_barh([((lo + s) / FPS / 60, n / FPS / 60)], (y - 0.36, 0.72),
                           facecolors=cmap[l], edgecolor="none",
                           hatch="////" if l == "D" else None)
    ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows][::-1],
           xlim=(t0, (hi + 1) / FPS / 60), xlabel="Tactical video time (min)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(fc=C_H, label="H (home, Barcelona)"), Patch(fc=C_A, label="A (away, Athletic)"),
                       Patch(fc=C_D, hatch="////", label="D (dead ball)")],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "fig_public_timeline")


# ---------------------------------------------------------------- Fig. 3 possession durations
def possession_durations(lab, blocks):
    out = []
    for a, b in blocks:
        out += [n / FPS for l, _, n in runs(lab[a:b + 1]) if l in ("H", "A")]
    return np.array(out)


def fig_durations():
    gt_d = possession_durations(gt, [(lo, hi)])
    w5_pub = possession_durations(w5, [(lo, hi)])
    w5_full = possession_durations(w5, halves)
    full = np.concatenate([w5[a:b + 1] for a, b in halves])
    share = {l: float(np.mean(full == l)) for l in "HAD"}

    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.75), gridspec_kw=dict(width_ratios=[1.6, 1]))
    bins = np.logspace(np.log10(0.2), np.log10(200), 22)
    ax[0].hist(w5_full, bins=bins, weights=np.full(len(w5_full), 1 / len(w5_full)), color=C_NEW, alpha=0.85,
               label=f"submitted, full match (n={len(w5_full)})")
    ax[0].hist(gt_d, bins=bins, weights=np.full(len(gt_d), 1 / len(gt_d)), histtype="step", color=C_H, lw=1.1,
               label=f"GT, public window (n={len(gt_d)})")
    ax[0].axvline(4, color="k", ls=":", lw=0.7)
    ax[0].text(4.4, 0.215, "4 s", fontsize=6)
    ax[0].set(xscale="log", xlabel="Possession duration (s)", ylabel="Fraction of possessions", ylim=(0, 0.25))
    ax[0].legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=5.8)
    ax[0].text(0.97, 0.95, "(a)", transform=ax[0].transAxes, ha="right", va="top", weight="bold")

    gt_win = gt[win]
    pub_gt = [float(np.mean(gt_win == l)) for l in "HAD"]
    pub_w5 = [float(np.mean(w5[win] == l)) for l in "HAD"]
    x = np.arange(3); w = 0.27
    ax[1].bar(x - w, pub_gt, w, color=C_H, label="GT pub.")
    ax[1].bar(x, pub_w5, w, color=C_NEW, label="submitted pub.")
    ax[1].bar(x + w, [share[l] for l in "HAD"], w, color=C_OLD, label="submitted full")
    ax[1].set(xticks=x, xticklabels=list("HAD"), ylabel="Share of graded frames", ylim=(0, 0.75))
    ax[1].legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), frameon=False, fontsize=5.6, handlelength=1.0)
    ax[1].text(0.97, 0.95, "(b)", transform=ax[1].transAxes, ha="right", va="top", weight="bold")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_durations")
    return dict(gt_mean=gt_d.mean(), w5_pub_n=len(w5_pub), w5_full_n=len(w5_full),
                w5_full_median=float(np.median(w5_full)), w5_full_mean=float(w5_full.mean()),
                w5_full_lt4=float(np.mean(w5_full < 4)), gt_lt4=int(np.sum(gt_d < 4)),
                full_share=share, n_graded_full=int(len(full)))


# ---------------------------------------------------------------- Fig. 4 development grid
FAMILY_ORDER = ["old_pp", "old_gru25", "boundary_tcn", "boundary_gru25", "short_tcn", "short_gru25",
                "masked_tcn", "masked_gru25", "new_ensemble", "new_ensemble_gru25", "quality_old", "quality_new"]
DECODER_ORDER = ["wave4", "postmerge", "live_reset", "boundary_reset", "dead1", "dead45", "dead55", "no_merge"]


def fig_dev_grid():
    dev = json.loads((RES / "wave5_development.json").read_text())
    grid = np.full((len(FAMILY_ORDER), len(DECODER_ORDER)), np.nan)
    elig = np.zeros_like(grid, bool)
    for r in dev["records"]:
        i, j = FAMILY_ORDER.index(r["family"]), DECODER_ORDER.index(r["decoder"]["name"])
        grid[i, j] = r["external_mean_spellF1"]; elig[i, j] = r["eligible"]
    assert np.isfinite(grid).all() and elig.sum() == 6
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    assert (FAMILY_ORDER[best[0]], DECODER_ORDER[best[1]]) == ("short_gru25", "dead45")
    ctrl = grid[FAMILY_ORDER.index("old_gru25"), DECODER_ORDER.index("wave4")]

    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    im = ax.imshow(grid, cmap="cividis", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v:.3f}"[1:], ha="center", va="center", fontsize=5.2,
                    color="white" if v < np.nanpercentile(grid, 55) else "black",
                    weight="bold" if elig[i, j] else "normal")
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False, ec=C_H, lw=1.4))
    ci = (FAMILY_ORDER.index("old_gru25"), DECODER_ORDER.index("wave4"))
    ax.add_patch(plt.Rectangle((ci[1] - .5, ci[0] - .5), 1, 1, fill=False, ec="white", lw=1.0, ls="--"))
    ax.set(xticks=range(len(DECODER_ORDER)), yticks=range(len(FAMILY_ORDER)),
           yticklabels=FAMILY_ORDER, xlabel="Decoder", ylabel="Probability family")
    ax.set_xticklabels(DECODER_ORDER, rotation=40, ha="right")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Equal-provider dev. spellF1"); cb.outline.set_linewidth(0.4)
    save(fig, "fig_dev_grid")
    return dict(control=float(ctrl), best=float(grid[best]), n_eligible=int(elig.sum()),
                min=float(np.nanmin(grid)), max=float(np.nanmax(grid)))


# ---------------------------------------------------------------- Fig. 5 provider / holdout
HOLDOUT_SC = ["skillcorner_2010085", "skillcorner_2011166", "skillcorner_2013725",
              "skillcorner_2015213", "skillcorner_2016236", "skillcorner_2017461"]


def holdout_cohorts():
    h = json.loads((RES / "wave5_holdout.json").read_text())["results"]
    by = defaultdict(dict)
    for r in h:
        if r["candidate"] not in ("selected", "previous_wave4"):
            continue
        if r["source"] == "idsse":
            c = "IDSSE (2)"
        elif r["source"] == "metrica":
            c = "Metrica G3 (2 cond.)"
        else:
            c = "SkillCorner (6) clean" if r["regime"] == "clean" else "SkillCorner (6) stress"
        by[c].setdefault(r["candidate"], []).append(r["spell_f1_mean"])
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in by.items()}


def fig_provider_holdout():
    dev = json.loads((RES / "wave5_development.json").read_text())
    rec = {(r["family"], r["decoder"]["name"]): r for r in dev["records"]}
    sel, ctl = rec[("short_gru25", "dead45")], rec[("old_gru25", "wave4")]
    provs = ["idsse", "skillcorner", "metrica"]

    def pmeans(r):
        pm = r["provider_means"]
        return {d["source"]: d["spellF1"] for d in pm}
    ps, pc = pmeans(sel), pmeans(ctl)

    ho = holdout_cohorts()
    order = ["IDSSE (2)", "SkillCorner (6) clean", "SkillCorner (6) stress", "Metrica G3 (2 cond.)"]
    ticks = ["IDSSE\n(2 matches)", "SkillCorner\n(6) clean", "SkillCorner\n(6) ball-drop", "Metrica G3\n(2 cond.)"]

    fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.85))
    names = ["IDSSE", "SkillCorner", "Metrica", "Equal mean"]
    old = [pc[p] for p in provs] + [ctl["external_mean_spellF1"]]
    new = [ps[p] for p in provs] + [sel["external_mean_spellF1"]]
    for a, title, cats, o, n in ((ax[0], "Development (grouped folds), 23 scenarios", names, old, new),
                                 (ax[1], "Post-freeze holdout (historically exposed)", ticks,
                                  [ho[c]["previous_wave4"] for c in order], [ho[c]["selected"] for c in order])):
        x = np.arange(len(cats)); w = 0.36
        a.bar(x - w / 2, o, w, color=C_OLD, label="control wave4 (old_gru25/wave4)")
        a.bar(x + w / 2, n, w, color=C_NEW, label="selected wave5 (short_gru25/dead45)")
        for xi, oi, ni in zip(x, o, n):
            d = ni - oi
            a.text(xi + w / 2, ni + 0.004, f"{d:+.3f}", ha="center", va="bottom", fontsize=5.6,
                   color=C_NEW if d >= 0 else C_H)
        a.set(xticks=x, ylim=(0.55, 0.73), title=title, ylabel="Mean spellF1")
        a.set_xticklabels(cats, fontsize=6.2)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2,
               frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_provider_holdout")
    return dict(dev_sel=ps, dev_ctl=pc, holdout=ho)


# ---------------------------------------------------------------- Fig. 6 duration-bucket recall
def fig_duration_recall():
    d = json.loads((RES / "wave5_duration_diagnostics.json").read_text())["records"]
    cohorts = {
        "IDSSE (2)": lambda r: r["match"] in ("J03WQQ", "J03WR9"),
        "SC clean (6)": lambda r: r["match"] in HOLDOUT_SC and r["regime"] == "clean",
        "SC stress (6)": lambda r: r["match"] in HOLDOUT_SC and r["regime"] == "stress",
        "Metrica G3 (2)": lambda r: r["match"] == "metrica3",
    }
    buckets = [("lt_4s", "< 4 s"), ("4_to_lt_15s", "4–15 s"), ("gte_15s", "≥ 15 s")]
    pooled = {}
    for cname, f in cohorts.items():
        for cand in ("previous_wave4", "selected"):
            rs = [r for r in d if r["candidate"] == cand and f(r)]
            for b, _ in buckets:
                for iou in ("0.5", "0.75"):
                    g = sum(r["duration_buckets"][b]["gt_count"] for r in rs)
                    m = sum(r["duration_buckets"][b]["at_iou"][iou]["matched_count"] for r in rs)
                    pooled[(cname, cand, b, iou)] = (m, g)
    # Must reproduce the short-spell table in wave5/RESULTS.md (IoU 0.75, < 4 s).
    assert pooled[("IDSSE (2)", "previous_wave4", "lt_4s", "0.75")] == (55, 360)
    assert pooled[("IDSSE (2)", "selected", "lt_4s", "0.75")] == (64, 360)
    assert pooled[("SC clean (6)", "selected", "lt_4s", "0.75")] == (116, 535)
    assert pooled[("Metrica G3 (2)", "selected", "lt_4s", "0.75")] == (49, 254)

    fig, axs = plt.subplots(1, 4, figsize=(COL2, 1.55), sharey=True)
    for a, cname in zip(axs, cohorts):
        x = np.arange(len(buckets)); w = 0.36
        for k, (cand, col, lab) in enumerate((("previous_wave4", C_OLD, "control wave4"),
                                              ("selected", C_NEW, "selected wave5"))):
            vals = [pooled[(cname, cand, b, "0.5")] for b, _ in buckets]
            r = [m / g for m, g in vals]
            a.bar(x + (k - 0.5) * w, r, w, color=col, label=lab)
            v75 = [pooled[(cname, cand, b, "0.75")] for b, _ in buckets]
            a.scatter(x + (k - 0.5) * w, [m / g for m, g in v75], marker="_", s=40, color="k", lw=1.0,
                      label="IoU 0.75" if k == 1 else None, zorder=3)
        n = [pooled[(cname, "selected", b, "0.5")][1] for b, _ in buckets]
        a.set_xticks(x); a.set_xticklabels([f"{l}\nn={k}" for (_, l), k in zip(buckets, n)], fontsize=5.8)
        a.set_title(cname)
    axs[0].set(ylabel="Matched recall (IoU 0.5)", ylim=(0, 1))
    fig.tight_layout(w_pad=0.4)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
               frameon=False)
    save(fig, "fig_duration_recall")
    return {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in pooled.items()}


# ---------------------------------------------------------------- Fig. 7 public error anatomy
def fig_public_confusion(report):
    cm = np.array(report["confusion"])[:3, :3]  # rows GT H,A,D ; cols pred H,A,D
    fig, ax = plt.subplots(figsize=(1.75, 1.6))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=5.8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=list("HAD"), yticklabels=list("HAD"),
           xlabel="Predicted", ylabel="Ground truth")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig_public_confusion")


# ---------------------------------------------------------------- public scoring (official scorer)
def score_public():
    import sys
    import tempfile
    sys.path.insert(0, str(ROOT / "evaluation"))
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "sub"; sub.mkdir()
        (sub / "wave5.csv").write_text(w5_text)
        (sub / "frozen.csv").write_text(frozen_text)
        for b in ("raw", "smooth25", "smooth50", "smooth125"):
            (sub / f"baseline_{b}.csv").write_bytes((ROOT / f"evaluation/results/baseline_{b}.csv").read_bytes())
        subprocess.run([sys.executable, str(ROOT / "evaluation/run_evaluation.py"), "--results", str(sub),
                        "--out", tmp, "--no-detail"], check=True, capture_output=True)
        board = pd.read_csv(Path(tmp) / "scoreboard.csv")
        rep = json.loads((Path(tmp) / "reports" / "wave5.json").read_text())
        rep_frozen = json.loads((Path(tmp) / "reports" / "frozen.json").read_text())
        trans = pd.read_csv(Path(tmp) / "reports" / "wave5_transitions.csv")
    w = board[board.submission == "wave5"].iloc[0]
    pub = METHOD["metrics"]["competition_public"]
    assert abs(rep_frozen["spell_f1_mean"] - pub["spellF1"]) < 1e-9 and abs(rep_frozen["macro_f1"] - pub["macroF1"]) < 1e-9
    sub_pub = RETRAIN["public_diagnostic"]
    assert abs(rep["spell_f1_mean"] - sub_pub["spellF1"]) < 1e-9 and abs(rep["macro_f1"] - sub_pub["macroF1"]) < 1e-9
    assert abs(w.spell_f1_mean - sub_pub["spellF1"]) < 1e-4
    cols = ["submission", "spell_f1_mean", "macro_f1", "poss", "dur", "kappa", "trans_f1_25", "mean_abs_latency"]
    return board[board.submission != "frozen"][cols], rep, trans  # frozen scored only as a consistency check


def crop_error_iou():
    """Panel A of the notebook's error-anatomy figure; panel B repeats Fig. 7."""
    from PIL import Image
    im = Image.open(ROOT / "image/fig15_error_anatomy.png")
    im.crop((0, 0, int(im.width * 0.655), im.height)).save(FIG / "fig_error_iou.png")


def main():
    board, rep, trans = score_public()
    fig_pipeline()
    fig_public_timeline()
    dur = fig_durations()
    grid = fig_dev_grid()
    prov = fig_provider_holdout()
    rec = fig_duration_recall()
    fig_public_confusion(rep)
    crop_error_iou()
    out = dict(
        public_board=board.to_dict(orient="records"),
        public_report={k: rep[k] for k in ("accuracy", "kappa", "alive_dead_f1", "per_class", "n_gt_spells",
                                           "n_pred_spells", "mean_spell_duration_s", "gt_mean_spell_duration_s",
                                           "confusion")},
        public_transitions=trans.status.value_counts().to_dict(),
        durations=dur, dev_grid=grid, provider=prov, duration_recall=rec,
    )
    (HERE / "report_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("public_transitions", "durations", "dev_grid")}, indent=1, default=float))
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
