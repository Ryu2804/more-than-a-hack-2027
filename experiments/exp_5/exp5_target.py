"""BAR-ATH (only if H1 passed), label-free and aggregate-only: candidate CSVs and their rebuild.

Hygiene: BAR-ATH arrays stay inside the notebook-04 process; only aggregates are returned.
The public ground truth is never opened (exp_1 spent the single public check). Fusion is exp_3's
pinned rule + replay guard, by import (tactical input if the rule rejects fusion).
"""

from __future__ import annotations

import numpy as np
import torch

import exp5_common as C  # noqa: I001  (first: env, sys.path, exp_1-exp_4 imports)
import exp1_net as N
import exp1_target as X1
import exp3_fusion as F
import exp3_target as E3
import exp3_touch as S
import exp5_protocol as P
from exp1_common import JD, read_json, sha256

MEMBERS = tuple(P.ARMS)  # exp5_touch seeds 20261111 / 12 / 13
FILES = {ls: C.PREDICTIONS_OUT / f"exp_5_candidate_{ls}.csv" for ls in C.DECISION_LABELS}
REBUILT = {ls: C.PREDICTIONS_OUT / f"exp_5_candidate_{ls}_reproduced.csv" for ls in C.DECISION_LABELS}


def decision() -> dict:
    return read_json(C.RESULTS / "decision.json")


def inputs() -> tuple[object, dict, dict]:
    """Model input match (fused iff the pinned rule accepts both halves), its token item with the
    4 touch columns, and the fusion report (aggregates)."""
    tactical = X1.load_target_match()
    fused, report = F.fuse(tactical, JD.load_target("broadcast"))
    match = fused if report["use_fused_input"] else tactical
    item = E3.token_item(match)
    touch = S.touch_features(match).astype("float16")
    if len(touch) != len(item["glob"]):
        raise RuntimeError("Touch features are not aligned with the token grid")
    item = {**item, "glob": np.concatenate([item["glob"], touch], axis=1)}
    summary = {k: v for k, v in report.items() if k != "halves"}
    return match, item, {**summary, "input": "fused" if report["use_fused_input"] else "tactical"}


def model_path(arm: str):
    return C.TARGET_MODEL_DIR / f"{arm}_{P.fingerprint(arm)}.pt"


def full_model(arm: str, use_cache: bool = True):
    """Full-data member (all 7 proxy matches). ``use_cache=False`` fits afresh and saves nothing."""
    spec = P.ARMS[arm]
    path = model_path(arm)
    if use_cache and path.exists():
        model = P.new_model(spec)
        model.load_state_dict(torch.load(path, map_location=N.DEVICE))
        return model.eval()
    model, _ = P.fit_spec(spec, list(C.MATCH_IDS))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return model


def target_fit_count() -> int:
    return sum(1 for _ in C.TARGET_MODEL_DIR.glob("*.pt")) if C.TARGET_MODEL_DIR.exists() else 0


def ensemble_output(models: dict, item: dict) -> tuple[np.ndarray, np.ndarray]:
    outputs = [N.predict(model, item, P.ARMS[arm].config.window) for arm, model in models.items()]
    return (np.mean([o[0] for o in outputs], axis=0).astype("float32"),
            np.mean([o[1] for o in outputs], axis=0).astype("float32"))


def build_all(match, item: dict, models: dict, paths: dict = FILES) -> dict:
    """Every file the adoption tree allows, from one ensemble output; each decoder is the final
    config selected on the 7 target_like out-of-fold predictions against its label set."""
    plan = decision()
    if not plan["H1"]["passed"] or not plan["adoption"]["write_files"]:
        raise RuntimeError("H1 failed: exp_5 writes no BAR-ATH file")
    output = ensemble_output(models, item)
    out: dict = {}
    for label_set in plan["adoption"]["label_sets"]:
        config = read_json(C.RESULTS / f"variant_{C.PRIMARY}__{C.PRIMARY_VIEW}__{label_set}.json")["final_config"]
        if config != plan["adoption"]["configs"][label_set]:
            raise RuntimeError("Final config differs from decision.json")
        codes = E3.codes_for(output, item, match, config)
        digest = X1.write_csv(paths[label_set], codes)  # runs the official validator
        out[label_set] = {"variant": C.PRIMARY, "members": list(MEMBERS), "selected_on": label_set,
                          "file": paths[label_set].name, "sha256": digest, "format_valid": True,
                          "decoder": config, "public_labels_used": False, **E3.aggregates(codes, match)}
    out["mean_p_dead"] = float(output[0][:, 2].mean())
    return out


def reproduce(match, item: dict) -> dict:
    """Independent fits of the 3 members (no cached weights) -> every file byte-identical."""
    models = {arm: full_model(arm, use_cache=False) for arm in MEMBERS}
    second = build_all(match, item, models, REBUILT)
    label_sets = decision()["adoption"]["label_sets"]
    out: dict = {"independent_fits": len(models)}
    for label_set in label_sets:
        first = sha256(FILES[label_set])
        out[label_set] = {"first_sha256": first, "second_sha256": second[label_set]["sha256"],
                          "byte_identical": first == second[label_set]["sha256"]}
    out["byte_identical"] = all(out[ls]["byte_identical"] for ls in label_sets)
    return out


def exp3_files() -> dict:
    """exp_3's two BAR-ATH files (aggregates already on disk), for comparison only."""
    keys = ("dead_share", "possessions_per_min", "mean_possession_s", "possession_count", "sha256")
    out = {}
    for name in ("target_candidate.json", "target_hedge_provider.json"):
        summary = read_json(C.C3.RESULTS / name)
        out[name] = {k: summary.get(k) for k in keys}
    return out
