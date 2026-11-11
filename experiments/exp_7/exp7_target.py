"""BAR-ATH candidates (PREREG 5), only after H1 passed (``results/decision.json``). Label-free, aggregates only.

Input: the tactical ATD (fusion off) with exp_6's recovered ball on accepted hidden graded frames only
(``exp6_target.fill``, 1 cm) and ``ball_recovered`` = 1 there. Model: three full-data ``exp7_robust`` fits.
Decoders: the final measured-view configs. Plausibility guard on the graded dead share. Rebuild: exp_6's detector
retrained in a fresh folder + three fresh full-data fits must reproduce every written file byte for byte.
BAR-ATH arrays stay in-process; only aggregates are returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

import exp7_common as C  # noqa: I001  (first: environment, sys.path, safe exp_6 imports)
import exp1_net as N
import exp1_target as X1
import exp3_target as E3T
import exp7_aug as A
import exp7_decision as DE
import exp7_fit as F
from exp1_common import MATCH_IDS

CANDIDATES = {"ctrl": C.PREDICTIONS_OUT / "exp_7_candidate_ctrl.csv",
              "provider": C.PREDICTIONS_OUT / "exp_7_candidate_provider.csv"}
REBUILT = {label: path.with_name(path.stem + "_rebuild.csv") for label, path in CANDIDATES.items()}


def model_path(arm: str) -> Path:
    return C.TARGET_MODEL_DIR / f"{arm}_{F.fingerprint(arm)}.pt"


def full_model(arm: str, use_cache: bool = True) -> N.SetSeq:
    """Full-data member (all 7 proxy matches). ``use_cache=False`` fits afresh and saves nothing."""
    spec = F.ARMS[arm]
    path = model_path(arm)
    if use_cache and path.exists():
        model = F.new_model(spec)
        model.load_state_dict(torch.load(path, map_location=N.DEVICE))
        return model.eval()
    model = F.fit_model(spec, list(MATCH_IDS))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return model


def target_fit_count() -> int:
    return sum(1 for _ in C.TARGET_MODEL_DIR.glob("*.pt")) if C.TARGET_MODEL_DIR.exists() else 0


def ensemble_output(models: dict, item: dict) -> tuple[np.ndarray, np.ndarray]:
    outputs = [N.predict(model, item, F.ARMS[arm].config.window) for arm, model in models.items()]
    return (np.mean([o[0] for o in outputs], axis=0).astype("float32"),
            np.mean([o[1] for o in outputs], axis=0).astype("float32"))


def inputs(recovered_path: Path) -> tuple[dict, dict, dict]:
    """{source: match}, {source: flagged token item} for the tactical input and the recovered-ball input."""
    match = X1.load_target_match()
    with np.load(recovered_path) as z:
        recovered = {key: z[key] for key in z.files}
    filled, fill_stats = C.T6.fill(match, recovered)
    matches = {"tactical": match, "with_recovered": filled}
    items = {source: A.token_item(m, A.recovered_mask(match, m), labels=False) for source, m in matches.items()}
    fill_stats["flagged_steps"] = int(items["with_recovered"]["glob"][:, -1].sum())
    return matches, items, fill_stats


def write_candidate(path: Path, codes: np.ndarray, decision: dict) -> str:
    """Official-validator write, refused unless the stored decision says H1 passed."""
    DE.require_h1(decision)
    return X1.write_csv(path, codes)


def guarded_file(label: str, codes: np.ndarray, summary: dict, path: Path, decision: dict) -> dict:
    """Write the file only if its graded dead share passes the plausibility guard."""
    dead_share = summary["dead_share"]
    if not DE.plausible(dead_share):
        return {"written": False, "dead_share": dead_share,
                "reason": f"dead share outside {DE.PLAUSIBILITY['dead_share_prior']} +- {DE.PLAUSIBILITY['tolerance']}"}
    return {"written": True, "dead_share": dead_share, "file": path.name, "format_valid": True,
            "sha256": write_candidate(path, codes, decision), "label_set": label}


def build(decision: dict, recovered_path: Path, paths: dict[str, Path], models: dict, log: Callable = print) -> dict:
    """Aggregates with and without the recovered ball; guarded candidate files on the recovered-ball input."""
    DE.require_h1(decision)
    matches, items, fill_stats = inputs(recovered_path)
    out: dict = {"fill": fill_stats, "aggregates": {}, "files": {}}
    for source in ("tactical", "with_recovered"):
        output = ensemble_output(models, items[source])
        for label in C.LABEL_SETS:
            config = decision["final_configs"][label]
            codes = E3T.codes_for(output, items[source], matches[source], config)
            summary = {**E3T.aggregates(codes, matches[source]), "mean_p_dead": float(output[0][:, 2].mean()),
                       "decoder": config}
            out["aggregates"][f"{source}__{label}"] = summary
            if source == "with_recovered":
                out["files"][label] = guarded_file(label, codes, summary, paths[label], decision)
            log(f"{source} {label}: possessions/min {summary['possessions_per_min']:.2f}, "
                f"dead share {summary['dead_share']:.3f}")
    return out


def rebuild(decision: dict, first: dict, log: Callable = print) -> dict:
    """exp_6's detector retrained in a fresh folder + three fresh full-data fits -> every written file again."""
    DE.require_h1(decision)
    R = C.import_exp6("exp6_recover")
    summary = R.recover(C.REBUILD_DIR, log)
    models = {arm: full_model(arm, use_cache=False) for arm in F.ARMS}
    second = build(decision, C.REBUILD_DIR / "recovered_ball.npz", REBUILT, models, log)
    files = {}
    for label, info in first["files"].items():
        if info["written"]:
            again = second["files"][label]
            files[label] = {"first_sha256": info["sha256"], "second_sha256": again.get("sha256"),
                            "byte_identical": info["sha256"] == again.get("sha256")}
    return {"detector_state_sha256": summary["detector"]["state_sha256"],
            "recovered_sha256": summary["recovered_sha256"], "independent_fits": len(models), "files": files,
            "all_byte_identical": all(f["byte_identical"] for f in files.values())}
