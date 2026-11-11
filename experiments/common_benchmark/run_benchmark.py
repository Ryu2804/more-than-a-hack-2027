"""Run the common benchmark over a list of IDSSE matches.

Expected files:
  <gt_dir>/<MATCH>_gt.csv
  <pred_dir>/<MATCH>_pred.csv  (or <MATCH>.csv)

Usage:
  python run_benchmark.py --matches J03WMX,J03WOH --gt-dir /path/gt --pred-dir /path/pred --out results/run1
"""

import argparse
import csv
import json
from pathlib import Path

from score_wrapper import score_pair


def find_file(directory, match, suffixes):
    directory = Path(directory)
    for suffix in suffixes:
        candidate = directory / f"{match}{suffix}"
        if candidate.exists():
            return candidate
    # fallback: any csv containing the match id
    for candidate in sorted(directory.glob("*.csv")):
        if match in candidate.name:
            return candidate
    raise FileNotFoundError(f"no file for {match} in {directory}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", required=True, help="comma-separated match ids")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    matches = [m.strip() for m in args.matches.split(",") if m.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for match in matches:
        gt_path = find_file(args.gt_dir, match, ["_gt.csv", "_possession_gt.csv"])
        pred_path = find_file(args.pred_dir, match, ["_pred.csv", ".csv"])
        metrics = score_pair(gt_path, pred_path, bootstrap_n=args.bootstrap)
        metrics["match"] = match
        rows.append(metrics)
        (out / f"{match}.json").write_text(json.dumps(metrics, indent=2) + "\n")

    summary = {
        "matches": matches,
        "n_matches": len(matches),
        "mean_spell_f1": float(sum(r["spell_f1_mean"] for r in rows) / len(rows)),
        "mean_macro_f1": float(sum(r["macro_f1"] for r in rows) / len(rows)),
        "mean_spell_count_error": float(sum(r["spell_count_error"] for r in rows) / len(rows)),
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match", "macro_f1", "spell_f1_mean", "n_gt_spells", "n_pred_spells", "spell_count_error", "mean_spell_duration_s"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(json.dumps({k: summary[k] for k in ("n_matches", "mean_spell_f1", "mean_macro_f1", "mean_spell_count_error")}, indent=2))


if __name__ == "__main__":
    main()