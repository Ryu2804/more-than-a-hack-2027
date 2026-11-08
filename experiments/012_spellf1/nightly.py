#!/usr/bin/env python3
"""Overnight loop for the spellF1 project: runs a small queue of independent, expensive checks.

Design goals: idempotent (state.json), per-task isolation (one failure never kills the loop), no
interactive input, and a single readable report in the morning. It never writes a submission CSV,
never changes rules, and treats the 5-minute window as a sanity check only, never as a selector.

Usage:
    python nightly.py                 # run every pending task
    python nightly.py --only NAME     # run one task (even if done, with --force)
    python nightly.py --hours 6       # stop starting new tasks after N hours
    python nightly.py --status
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))

import harness as H  # noqa: E402
import phase2 as P2  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402
from deployable import build_prediction, merge_short_dead  # noqa: E402

OUT = os.path.join(HERE, "out", "nightly")
STATE = os.path.join(OUT, "state.json")
LOG = os.path.join(OUT, "nightly.log")
REPORT = os.path.join(OUT, "MORNING_REPORT.md")
MERGE_K = 150
SIGMAS = (0.5, 1.0, 2.0)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(line):
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write("[{}] {}\n".format(now(), line))


def load_state():
    if os.path.isfile(STATE):
        with open(STATE) as handle:
            return json.load(handle)
    return {"done": [], "failed": {}, "started": now()}


def save_state(state):
    with open(STATE, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def write_result(name, payload):
    path = os.path.join(OUT, "results_{}.json".format(name))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


# ---------------------------------------------------------------- IDSSE helpers

def idsee_arms(name, merge_k=MERGE_K):
    ctx = T.match_context(name)
    npz = H.load(name)
    cue, _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
    held = T.hold(ctx["dmin"], ctx["team"], 0.8, 30)
    return ctx, npz, cue, held


def score_idsee(ctx, pred):
    result = pm.spell_scores(ctx["gt"], pred, ctx["runs"])
    return {"f1": float(result["mean_f1"]), "n_pred": int(result["n_pred"])}


# ---------------------------------------------------------------------- tasks

def task_confirm_sota():
    rows = {}
    for name in H.CONFIRM:
        ctx, npz, cue, held = idsee_arms(name)
        pred = build_prediction(ctx, cue, held, MERGE_K)
        rows[name] = score_idsee(ctx, pred)
        log("confirm_sota {} f1={:.4f}".format(name, rows[name]["f1"]))
    values = [row["f1"] for row in rows.values()]
    return {
        "task": "confirm_sota",
        "pipeline": "touch_hold r0.8 m30 + cue out_or_slow2 dead + merge 150",
        "matches": rows,
        "mean": mean(values),
        "ci_note": "two matches only; no CI quoted, direction only",
    }


def task_confirm_ceiling():
    rows = {}
    for name in H.CONFIRM:
        ctx, npz, cue, held = idsee_arms(name)
        gt = ctx["gt"]
        oracle_team = gt.copy()
        oracle_team[oracle_team == 3] = -1
        oracle_dead = gt == 2
        arms = {
            "cue_dead_pred_team": build_prediction(ctx, cue, held, MERGE_K),
            "cue_dead_oracle_team": build_prediction(ctx, cue, oracle_team, 0),
            "oracle_dead_pred_team": build_prediction(ctx, oracle_dead, held, 0),
            "oracle_dead_oracle_team": build_prediction(ctx, oracle_dead, oracle_team, 0),
        }
        rows[name] = {key: score_idsee(ctx, pred) for key, pred in arms.items()}
        log("confirm_ceiling {} {}".format(
            name, {k: round(v["f1"], 4) for k, v in rows[name].items()}))
    summary = {}
    for key in rows[H.CONFIRM[0]]:
        summary[key] = mean([rows[m][key]["f1"] for m in rows])
    return {"task": "confirm_ceiling", "matches": rows, "mean": summary,
            "note": "oracle arms are ceilings, not deployable"}


def task_barath_dead_grid():
    import barath as B
    data_dir = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "..", "data")
    context = B.load_context(data_dir)
    gt = B.public_gt(data_dir)
    n = context["frames"]
    low, high = B.WINDOW
    graded = np.zeros(n, dtype=bool)
    graded[low:high + 1] = True

    import phase2b as P2B
    cue, answered = P2B.cue_of(context["ball_x"], context["ball_y"], context["present"],
                               context["index"], context["speed"], graded)

    missing = ~context["present"]
    slow = (context["speed"] < 2.0) & (context["speed"] >= 0)
    slow25 = P2.run_mask(slow, 25) & graded
    outside = answered & ((np.abs(context["ball_x"]) > 52.6) |
                          (np.abs(context["ball_y"]) > 34.1))
    out15 = P2.run_mask(outside, 15) & graded

    held = T.hold(context["dmin"], context["team"], B.R_M, B.MINLEN)
    nearest_codes = np.where(context["team"] >= 0, context["team"], -1).astype(np.int8)
    smooth = B.smooth_codes(nearest_codes, B.SMOOTH_W)

    deads = {
        "meaning_missing": missing,
        "cue_all_frame": cue,
        "slow25_only": slow25,
        "out15_only": out15,
        "missing_or_slow25": missing | slow25,
        "missing_or_out15": missing | out15,
    }
    codes = {"touch_hold": held, "nearest_smooth50": smooth}
    arms = {}
    for code_name, code in codes.items():
        for merge_k in (0, 50, 150):
            base_code = merge_short_dead(code, merge_k) if merge_k else code
            for dead_name, dead in deads.items():
                pred = np.where(dead & graded, 2,
                                np.where(base_code >= 0, base_code, 2)).astype(np.int8)
                key = "{}_k{}_dead_{}".format(code_name, merge_k, dead_name)
                arms[key] = B.score_window(gt, pred, low, high)
    arms["touch_hold_k150_dead_oracle"] = B.score_window(
        gt, np.where((gt == 2) & graded, 2,
                     np.where(merge_short_dead(held, 150) >= 0,
                              merge_short_dead(held, 150), 2)).astype(np.int8), low, high)
    best = sorted(arms.items(), key=lambda kv: -kv[1]["spell_f1"])[:10]
    for key, row in best:
        log("barath {} {:.4f}".format(key, row["spell_f1"]))
    return {
        "task": "barath_dead_grid",
        "window": list(B.WINDOW),
        "note": "5-minute sanity only; never a selector",
        "ball_present_fraction": float(context["present"].mean()),
        "arms": arms,
        "top": [{"arm": key, "spell_f1": row["spell_f1"], "n_pred": row["n_pred"]}
                for key, row in best],
    }


def task_metrica_check():
    base = os.path.join(ROOT, "..", "data", "external", "metrica")
    listing = {}
    for root, _dirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        if rel.count(os.sep) > 2:
            continue
        entries = []
        for name in sorted(files)[:20]:
            path = os.path.join(root, name)
            entries.append({"name": name, "bytes": os.path.getsize(path)})
        if entries:
            listing[rel] = entries
    return {"task": "metrica_check", "base_exists": os.path.isdir(base),
            "listing": listing,
            "note": "availability and schema only; no pipeline is built here"}


def task_noise_price():
    rows = {}
    for name in H.DEV:
        ctx = T.match_context(name)
        npz = H.load(name)
        bx, by = npz["ball_x"], npz["ball_y"]
        cue, _, _ = P2.cue_masks(npz, bx, by)
        held = T.hold(ctx["dmin"], ctx["team"], 0.8, 30)
        base = score_idsee(ctx, build_prediction(ctx, cue, held, MERGE_K))["f1"]
        entry = {"baseline": base}
        rng = np.random.default_rng(20261108)
        for sigma in SIGMAS:
            noisy = dict(npz)
            noise_x = rng.standard_normal(npz["player_x"].shape) * sigma
            noise_y = rng.standard_normal(npz["player_y"].shape) * sigma
            noisy["player_x"] = (npz["player_x"] + noise_x).astype(np.float32)
            noisy["player_y"] = (npz["player_y"] + noise_y).astype(np.float32)
            answered, index, _ = P2.nearest(noisy, bx, by)
            team = np.where(answered, H.sides(noisy)[index], -1).astype(np.int8)
            if ctx["flipped"]:
                team = np.where(team >= 0, 1 - team, -1).astype(np.int8)
            dmin = np.where(answered, np.min(np.hypot(
                noisy["player_x"] - bx[None, :], noisy["player_y"] - by[None, :]), axis=0),
                np.inf)
            held_n = T.hold(dmin, team, 0.8, 30)
            cue_n, _, _ = P2.cue_masks(noisy, bx, by)
            entry["sigma_{:g}".format(sigma)] = score_idsee(
                ctx, build_prediction(ctx, cue_n, held_n, MERGE_K))["f1"]
        rows[name] = entry
        log("noise_price {} {}".format(
            name, {k: round(v, 4) for k, v in entry.items()}))
    summary = {"baseline": mean([r["baseline"] for r in rows.values()])}
    for sigma in SIGMAS:
        key = "sigma_{:g}".format(sigma)
        summary[key] = mean([r[key] for r in rows.values()])
    summary["delta_sigma_1"] = summary["sigma_1"] - summary["baseline"]
    summary["delta_sigma_2"] = summary["sigma_2"] - summary["baseline"]
    return {"task": "noise_price", "matches": rows, "mean": summary,
            "note": "player position noise only; ball and cue thresholds unchanged"}


def task_summary():
    pieces = []
    files = sorted(os.listdir(OUT))
    data = {}
    for name in files:
        if name.startswith("results_") and name.endswith(".json"):
            with open(os.path.join(OUT, name)) as handle:
                data[name] = json.load(handle)
    lines = ["# Morning report", "", "Generated {}.".format(now()), ""]
    if "results_confirm_sota.json" in data:
        row = data["results_confirm_sota.json"]
        lines.append("## Confirmation of the frozen deployable pipeline")
        lines.append("")
        lines.append("Mean spellF1 on J03WN1 + J03WR9: {:.4f} (two matches, direction only).".format(
            row["mean"]))
        for name, arm in row["matches"].items():
            lines.append("- {}: {:.4f} ({} predicted spells)".format(
                name, arm["f1"], arm["n_pred"]))
        lines.append("")
    if "results_confirm_ceiling.json" in data:
        row = data["results_confirm_ceiling.json"]
        lines.append("## Confirmation ceilings")
        lines.append("")
        for key, value in row["mean"].items():
            lines.append("- {}: {:.4f}".format(key, value))
        lines.append("")
    if "results_barath_dead_grid.json" in data:
        row = data["results_barath_dead_grid.json"]
        lines.append("## BAR-ATH dead-rule grid (5-minute sanity, not a selector)")
        lines.append("")
        for item in row["top"][:8]:
            lines.append("- {}: {:.4f} ({} predicted spells)".format(
                item["arm"], item["spell_f1"], item["n_pred"]))
        lines.append("")
    if "results_noise_price.json" in data:
        row = data["results_noise_price.json"]
        lines.append("## Price of player position noise (IDSSE dev mean)")
        lines.append("")
        for key in ("baseline", "sigma_0.5", "sigma_1", "sigma_2"):
            lines.append("- {}: {:.4f}".format(key, row["mean"][key]))
        lines.append("- delta sigma 1 m: {:.4f}, sigma 2 m: {:.4f}".format(
            row["mean"]["delta_sigma_1"], row["mean"]["delta_sigma_2"]))
        lines.append("")
    if "results_metrica_check.json" in data:
        row = data["results_metrica_check.json"]
        lines.append("## Metrica availability")
        lines.append("")
        lines.append("- folders with files: {}".format(len(row["listing"])))
        lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- Deployable SOTA remains touch_hold r0.8/m30 + cue dead + merge 150 at 0.5263 "
                 "(IDSSE LOSO); oracle dead 0.6150; published baseline 0.2433.")
    lines.append("- No submission CSV was written; no candidate was selected by window numbers.")
    header = "\n".join(lines)
    for name, payload in data.items():
        pieces.append("### " + name + "\n\n```json\n" +
                      json.dumps(payload, indent=2, sort_keys=True)[:4000] + "\n```\n")
    with open(REPORT, "w") as handle:
        handle.write(header + "\n## Raw results\n\n" + "\n".join(pieces))
    return {"task": "summary", "report": REPORT}


TASKS = [
    ("confirm_sota", task_confirm_sota),
    ("confirm_ceiling", task_confirm_ceiling),
    ("barath_dead_grid", task_barath_dead_grid),
    ("metrica_check", task_metrica_check),
    ("noise_price", task_noise_price),
    ("summary", task_summary),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run one task by name")
    parser.add_argument("--hours", type=float, default=6.0, help="stop starting tasks after N hours")
    parser.add_argument("--force", action="store_true", help="re-run even if already done")
    parser.add_argument("--status", action="store_true", help="print state and exit")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    state = load_state()
    if args.status:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    started = time.time()
    for name, function in TASKS:
        if args.only and name != args.only:
            continue
        if name in state["done"] and not args.force:
            log("skip {} (done)".format(name))
            continue
        if time.time() - started > args.hours * 3600 and not args.only:
            log("stop: hour budget reached before {}".format(name))
            break
        log("start {}".format(name))
        try:
            payload = function()
            payload["finished"] = now()
            payload["runtime_s"] = round(time.time() - started, 1)
            path = write_result(name, payload)
            if name not in state["done"]:
                state["done"].append(name)
            state["failed"].pop(name, None)
            log("done {} -> {}".format(name, path))
        except Exception as error:  # noqa: BLE001 - one task must never kill the loop
            state["failed"][name] = str(error)
            log("FAIL {}: {}\n{}".format(name, error, traceback.format_exc()))
        save_state(state)
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
