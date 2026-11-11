"""BAR-ATH candidates (PREREG 5), only after H1 passed (``results/decision.json``).

Fill the recovered ball on accepted hidden graded frames (1 cm quantised), predict with exp_2's three
cached full-data ``exp3_ref_ens3`` members, decode with exp_3's two final configs, write and validate
both CSVs, report label-free aggregates with and without the recovered ball, and rebuild from a
detector retrained from scratch. Aggregates only are returned; BAR-ATH arrays stay in-process.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

import exp6_common as C
import exp6_decision as DE
import exp1_common as C1

CANDIDATES = {"ctrl": C.PREDICTIONS_OUT / "exp_6_candidate_ctrl.csv",
              "provider": C.PREDICTIONS_OUT / "exp_6_candidate_provider.csv"}
REBUILT = {label: path.with_name(path.stem + "_rebuild.csv") for label, path in CANDIDATES.items()}
DECODER_SOURCES = {"ctrl": "target_candidate.json", "provider": "target_hedge_provider.json"}
QUANTUM_DECIMALS = 2  # 1 cm


def decoder(label: str) -> dict:
    return C1.read_json(C.C3.RESULTS / DECODER_SOURCES[label])["decoder"]


def write_candidate(path: Path, codes: np.ndarray, decision: dict) -> str:
    """Official-validator write, refused unless the stored decision says H1 passed."""
    DE.require_h1(decision)
    import exp1_target as X1

    return X1.write_csv(path, codes)


def fill(match, recovered: dict) -> tuple[object, dict]:
    """Recovered positions on accepted, ATD-hidden graded frames only; ATD-visible frames untouched."""
    accepted = recovered["accepted"].astype(bool)
    frames = recovered["frame"][accepted]
    xy = np.round(np.stack([recovered["x"], recovered["y"]], axis=1)[accepted], QUANTUM_DECIMALS)
    hidden = (match.half > 0) & ~np.isfinite(match.ball).all(axis=1)
    use = hidden[frames]
    ball = match.ball.copy()
    ball[frames[use]] = xy[use].astype(ball.dtype)
    graded = match.half > 0
    return replace(match, ball=ball), {
        "filled_frames": int(use.sum()), "accepted_visible_frames_left_unchanged": int((~use).sum()),
        "ball_visibility_graded_before": float(np.isfinite(match.ball[graded]).all(axis=1).mean()),
        "ball_visibility_graded_after": float(np.isfinite(ball[graded]).all(axis=1).mean())}


def _tactical_matches_exp3(codes: np.ndarray, label: str) -> bool:
    """Without the recovered ball the pipeline must reproduce exp_3's own file byte for byte."""
    import exp1_target as X1

    path = C.CACHE / f"tactical_check_{label}.csv"
    digest = X1.write_csv(path, codes)
    path.unlink()
    return digest == C1.read_json(C.C3.RESULTS / DECODER_SOURCES[label])["sha256"]


def build(decision: dict, recovered_path: Path, paths: dict[str, Path], log: Callable = print) -> dict:
    """Both candidate files + label-free aggregates with and without the recovered ball."""
    DE.require_h1(decision)
    import exp1_target as X1
    import exp3_target as E3T

    match = X1.load_target_match()
    with np.load(recovered_path) as z:
        recovered = {key: z[key] for key in z.files}
    filled, fill_stats = fill(match, recovered)
    models = {arm: E3T.full_model(arm, use_cache=True) for arm in C.MEMBERS}
    out: dict = {"fill": fill_stats, "files": {}, "aggregates": {}}
    for source, current in (("tactical", match), ("with_recovered", filled)):
        item = E3T.token_item(current)
        output = E3T.ensemble_output(models, item)
        for label in C.LABEL_SETS:
            codes = E3T.codes_for(output, item, current, decoder(label))
            summary = {**E3T.aggregates(codes, current), "mean_p_dead": float(output[0][:, 2].mean()),
                       "decoder": decoder(label)}
            if source == "tactical":
                summary["reproduces_exp3_file"] = _tactical_matches_exp3(codes, label)
            else:
                out["files"][label] = {"file": paths[label].name, "format_valid": True,
                                       "sha256": write_candidate(paths[label], codes, decision)}
            out["aggregates"][f"{source}__{label}"] = summary
            log(f"{source} {label}: possessions/min {summary['possessions_per_min']:.2f}, "
                f"dead share {summary['dead_share']:.3f}")
    return out


def rebuild(decision: dict, first: dict, log: Callable = print) -> dict:
    """Retrain the detector from scratch, re-run inference + linking, rebuild both files."""
    DE.require_h1(decision)
    import exp6_recover as R

    summary = R.recover(C.REBUILD_DIR, log)
    second = build(decision, C.REBUILD_DIR / "recovered_ball.npz", REBUILT, log)
    files = {label: {"first_sha256": first["files"][label]["sha256"], "second_sha256": second["files"][label]["sha256"],
                     "byte_identical": first["files"][label]["sha256"] == second["files"][label]["sha256"]}
             for label in C.LABEL_SETS}
    return {"detector_state_sha256": summary["detector"]["state_sha256"],
            "recovered_sha256": summary["recovered_sha256"], "files": files,
            "all_byte_identical": all(f["byte_identical"] for f in files.values())}
