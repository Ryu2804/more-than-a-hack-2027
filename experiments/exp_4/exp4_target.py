"""BAR-ATH (only if H1 passed), label-free and aggregate-only: two candidate CSVs and their rebuild.

Hygiene: BAR-ATH arrays stay inside the notebook-04 process; only aggregates are returned.
The public ground truth is never opened (exp_1 spent the single public check). Fusion is exp_3's
pinned rule + replay guard, by import.
"""

from __future__ import annotations

import numpy as np
import torch

import exp4_common  # noqa: F401,I001  (first: puts ../exp_3, ../exp_2, ../exp_1 on sys.path)
import exp1_net as N
import exp1_target as X1
import exp3_fusion as F
import exp3_target as E3
import exp3_touch as S
import exp4_protocol as P
from exp1_common import JD, read_json, sha256
from exp4_common import C3, MATCH_IDS, PREDICTIONS_OUT, PRIMARY, PRIMARY_VIEW, RESULTS, TARGET_MODEL_DIR

MEMBERS = P.ENSEMBLES[PRIMARY]  # exp4_full seeds 0-2
LABEL_SETS = ("provider", "ctrl")
FILES = {ls: PREDICTIONS_OUT / f"exp_4_candidate_{ls}.csv" for ls in LABEL_SETS}
REBUILT = {ls: PREDICTIONS_OUT / f"exp_4_candidate_{ls}_reproduced.csv" for ls in LABEL_SETS}


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
    return match, item, {**report, "input": "fused" if report["use_fused_input"] else "tactical"}


def model_path(arm: str):
    return TARGET_MODEL_DIR / f"{arm}_{P.fingerprint(arm)}.pt"


def full_model(arm: str, use_cache: bool = True):
    """Full-data member (all 7 proxy matches). ``use_cache=False`` always fits afresh and saves nothing."""
    path = model_path(arm)
    if use_cache and path.exists():
        model = P.new_model(arm)
        model.load_state_dict(torch.load(path, map_location=N.DEVICE))
        return model.eval()
    model, _ = P.fit_model(arm, list(MATCH_IDS))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return model


def ensemble_output(models: dict, item: dict) -> tuple[np.ndarray, np.ndarray]:
    outputs = [N.predict(model, item, P.ARMS[arm].config.window) for arm, model in models.items()]
    return (np.mean([o[0] for o in outputs], axis=0).astype("float32"),
            np.mean([o[1] for o in outputs], axis=0).astype("float32"))


def final_config(label_set: str) -> dict:
    return read_json(RESULTS / f"variant_{PRIMARY}__{PRIMARY_VIEW}__{label_set}.json")["final_config"]


def build_all(match, item: dict, models: dict, paths: dict = FILES) -> dict:
    """Both candidate files from one ensemble output; decoders selected on the 7 target_like OOF
    predictions against provider labels and against L_ctrl respectively."""
    decision = read_json(RESULTS / "decision.json")
    if not decision["H1"]["passed"]:
        raise RuntimeError("H1 failed: exp_4 writes no BAR-ATH file")
    output = ensemble_output(models, item)
    out = {}
    for label_set in LABEL_SETS:
        config = final_config(label_set)
        if config != decision["adoption"]["configs"][label_set]:
            raise RuntimeError("Final config differs from decision.json")
        codes = E3.codes_for(output, item, match, config)
        digest = X1.write_csv(paths[label_set], codes)
        out[label_set] = {"variant": PRIMARY, "members": list(MEMBERS), "selected_on": label_set,
                          "file": paths[label_set].name, "sha256": digest, "format_valid": True,
                          "decoder": config, "public_labels_used": False, **E3.aggregates(codes, match)}
    out["mean_p_dead"] = float(output[0][:, 2].mean())
    return out


def reproduce(match, item: dict) -> dict:
    """Independent fits of the 3 members (no cached weights) -> both files byte-identical."""
    models = {arm: full_model(arm, use_cache=False) for arm in MEMBERS}
    second = build_all(match, item, models, REBUILT)
    out: dict = {"independent_fits": len(models)}
    for label_set in LABEL_SETS:
        first = sha256(FILES[label_set])
        out[label_set] = {"first_sha256": first, "second_sha256": second[label_set]["sha256"],
                          "byte_identical": first == second[label_set]["sha256"]}
    out["byte_identical"] = all(out[ls]["byte_identical"] for ls in LABEL_SETS)
    return out


def exp3_files() -> dict:
    """exp_3's two BAR-ATH files (aggregates already on disk), for comparison only."""
    keys = ("dead_share", "possessions_per_min", "mean_possession_s", "possession_count", "sha256")
    out = {}
    for name in ("target_candidate.json", "target_hedge_provider.json"):
        summary = read_json(C3.RESULTS / name)
        out[name] = {k: summary.get(k) for k in keys}
    return out
