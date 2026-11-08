"""BAR-ATH, label-free and aggregate-only: plausibility check and (if adopted) the candidate CSV.

Hygiene: BAR-ATH arrays stay inside the notebook-05 process; only aggregates are returned.
The public ground truth is never opened (exp_1 spent the single public check).
"""

from __future__ import annotations

import numpy as np
import torch

import exp1_decode as D
import exp1_net as N
import exp1_target as X1
import exp1_tokens as T
import exp2_hsmm as H
import exp2_protocol as P
from exp1_common import expand, read_json, sha256
from exp2_common import EXP1, MATCH_IDS, RESULTS, ROOT, TARGET_MODEL_DIR

CANDIDATE = ROOT / "predictions" / "exp_2_candidate.csv"
REPRODUCED = ROOT / "predictions" / "exp_2_candidate_reproduced.csv"
PUBLISHED_DEAD_SHARE = 0.407  # La Liga effective playing time prior used by the team (ryu run.py)
FPS = 25


def load_target() -> tuple[object, dict]:
    match = X1.load_target_match()
    token_set = T.build_tokens(match)
    item = {"tokens": token_set.tokens, "mask": token_set.mask, "glob": token_set.glob,
            "half": token_set.half, "indices": token_set.indices}
    return match, item


def full_model(arm: str, use_cache: bool = True) -> N.SetSeq:
    """SetSeq trained on all seven proxy matches with the arm's recipe (cached unless asked not to)."""
    path = TARGET_MODEL_DIR / f"{arm}_{P.fingerprint(arm)}.pt"
    config = P.MODEL_ARMS[arm]["config"]
    if use_cache and path.exists():
        model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
        model.load_state_dict(torch.load(path, map_location=N.DEVICE))
        return model.eval()
    model = P.fit_model(arm, list(MATCH_IDS))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return model


def predict(model: N.SetSeq, item: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    return N.predict(model, item, P.MODEL_ARMS[arm]["config"].window)


def final_config(variant: str, view: str = "target_like") -> dict:
    return read_json(RESULTS / f"variant_{variant}__{view}.json")["final_config"]


def _finish(codes: np.ndarray, item: dict, match) -> np.ndarray:
    full = expand(codes, item["indices"], match.half)
    full[X1.HALF2_START:X1.KICKOFF_2ND] = 2
    return full.astype("int8")


def codes_bd(probability, boundary, item, match, config: dict) -> np.ndarray:
    codes = D.decode(probability, boundary, item["half"], D.DecoderConfig(**config))
    return _finish(codes, item, match)


def codes_hsmm(probability, item, match, config: dict) -> np.ndarray:
    prior = P.prior_for(tuple(MATCH_IDS))
    codes = H.decode(probability, item["half"], prior, H.HsmmConfig(**config))
    return _finish(codes, item, match)


def _rates(graded: float, dead_share: float, alive_frames: float, count: int) -> dict:
    minutes = graded / FPS / 60
    return {"dead_share": float(dead_share), "possessions_per_min": float(count / minutes),
            "mean_possession_s": float(alive_frames / max(1, count) / FPS),
            "dead_share_minus_prior": float(dead_share - PUBLISHED_DEAD_SHARE)}


def aggregates(codes: np.ndarray, half: np.ndarray) -> dict:
    base = X1.aggregates(codes, half)
    shares, graded = base["share_graded"], base["graded_frames"]
    alive = (shares["H"] + shares["A"]) * graded
    return {**base, **_rates(graded, shares["D"], alive, base["possession_count"])}


def exp1_aggregates() -> dict:
    """exp_1's shipped candidate, from its aggregate summary (no BAR-ATH re-read)."""
    summary = read_json(EXP1 / "results" / "target_summary.json")
    shares, graded = summary["share_graded"], summary["graded_frames"]
    count = summary["possession_count"]
    return {"graded_frames": graded, "share_graded": shares, "possession_count": count,
            **_rates(graded, shares["D"], (shares["H"] + shares["A"]) * graded, count)}


def plausibility(match, item, models: dict) -> dict:
    """Label-free aggregates for exp_1 vs the exp_2 candidates; report only, never selection."""
    outputs = {arm: predict(model, item, arm) for arm, model in models.items()}
    p_ens = np.mean([outputs[a][0] for a in P.ENSEMBLE], axis=0)
    b_ens = np.mean([outputs[a][1] for a in P.ENSEMBLE], axis=0)
    p0, b0 = outputs["exp2_aug"]
    candidates = {
        "exp2_aug": codes_bd(p0, b0, item, match, final_config("exp2_aug")),
        "exp2_aug_hsmm": codes_hsmm(p0, item, match, final_config("exp2_aug_hsmm")),
        "exp2_aug_ens3": codes_bd(p_ens, b_ens, item, match, final_config("exp2_aug_ens3")),
    }
    result: dict[str, object] = {"exp1_setseq_shipped": exp1_aggregates()}
    result.update({name: aggregates(codes, match.half) for name, codes in candidates.items()})
    result["published_dead_share_prior"] = PUBLISHED_DEAD_SHARE
    result["ball_coverage_graded"] = float(np.isfinite(match.ball[match.half > 0]).all(axis=1).mean())
    result["mean_p_dead_exp2_aug"] = float(p0[:, 2].mean())
    return result


def build_candidate(match, item, model, path=CANDIDATE) -> dict:
    """exp2_aug candidate CSV; the pre-registered decision must have adopted it."""
    decision = read_json(RESULTS / "decision.json")
    if not decision["adopt_exp2_candidate"]:
        raise RuntimeError("PREREG adoption rule did not adopt an exp_2 candidate")
    probability, boundary = predict(model, item, "exp2_aug")
    config = final_config("exp2_aug")
    codes = codes_bd(probability, boundary, item, match, config)
    digest = X1.write_csv(path, codes)
    return {"sha256": digest, "format_valid": True, "decoder": config,
            "public_labels_used": False, **aggregates(codes, match.half)}


def reproduce(match, item) -> dict:
    """Independent second fit (no cached weights) must give a byte-identical CSV."""
    first = sha256(CANDIDATE)
    second = build_candidate(match, item, full_model("exp2_aug", use_cache=False), REPRODUCED)["sha256"]
    return {"first_sha256": first, "second_sha256": second, "byte_identical": first == second}
