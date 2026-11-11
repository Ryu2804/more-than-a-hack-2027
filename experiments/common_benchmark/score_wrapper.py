"""Thin wrapper around the official possession scorer for the common benchmark."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import possession_metrics as pm  # noqa: E402


def read_codes(path):
    frame = pm.read_label_csv(str(path))
    return pm.label_codes(frame)


def score_pair(gt_path, pred_path, bootstrap_n=0):
    gt = read_codes(gt_path)
    pred = read_codes(pred_path)
    result = pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)
    return {
        "macro_f1": float(result["macro_f1"]),
        "spell_f1_mean": float(result["spell_f1_mean"]),
        "spell_f1_headline": float(result["spell_f1_at_headline"]),
        "n_gt_spells": int(result["n_gt_spells"]),
        "n_pred_spells": int(result["n_pred_spells"]),
        "spell_count_error": int(result["spell_count_error"]),
        "mean_spell_duration_s": float(result["mean_spell_duration_s"]),
        "gt_mean_spell_duration_s": float(result["gt_mean_spell_duration_s"]),
        "bootstrap_low": float(result["bootstrap_low"]),
        "bootstrap_high": float(result["bootstrap_high"]),
        "spell_bootstrap_low": float(result["spell_bootstrap_low"]),
        "spell_bootstrap_high": float(result["spell_bootstrap_high"]),
        "n_frames": int(result["n_frames"]),
        "n_graded": int(result["n_graded"]),
    }


def score_to_json(gt_path, pred_path, out_path, bootstrap_n=0):
    metrics = score_pair(gt_path, pred_path, bootstrap_n=bootstrap_n)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics