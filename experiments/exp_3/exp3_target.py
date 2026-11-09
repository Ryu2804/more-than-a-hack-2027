"""BAR-ATH, label-free and aggregate-only: fusion, plausibility, candidate CSV and its rebuild.

Hygiene: BAR-ATH arrays stay inside the notebook-05 process; only aggregates are returned.
The public ground truth is never opened (exp_1 spent the single public check).
"""

from __future__ import annotations

import numpy as np
import torch

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
import exp1_decode as D
import exp1_net as N
import exp1_target as X1
import exp1_tokens as T
import exp2_protocol as P2
import exp2_target as E2
import exp3_fusion as F
import exp3_protocol as P
from exp1_common import JD, expand, read_json, sha256
from exp3_common import EXP2, MATCH_IDS, RESULTS, ROOT, TARGET_MODEL_DIR

CANDIDATE = ROOT / "predictions" / "exp_3_candidate.csv"
REPRODUCED = ROOT / "predictions" / "exp_3_candidate_reproduced.csv"
RYU_POSSESSIONS_PER_MIN = 2.97  # ryu's event-derived sources (planning prompt), comparison only
MEMBERS = {"exp3_ctrl_ens3": P.ENSEMBLES["ctrl_ens3"], "exp3_ref_ens3": P.ENSEMBLES["ref_ens3"]}


def token_item(match) -> dict:
    token_set = T.build_tokens(match)
    return {"tokens": token_set.tokens, "mask": token_set.mask, "glob": token_set.glob,
            "half": token_set.half, "indices": token_set.indices}


def inputs() -> tuple[object, dict, dict]:
    """Tactical match, {"tactical": item, "fused": item or None} and the fusion report."""
    tactical = X1.load_target_match()
    broadcast = JD.load_target("broadcast")
    fused, report = F.fuse(tactical, broadcast)
    items = {"tactical": token_item(tactical), "fused": token_item(fused) if report["use_fused_input"] else None}
    return tactical, items, report


def _config(arm: str):
    return P.MODEL_ARMS[arm] if arm in P.MODEL_ARMS else P2.MODEL_ARMS[arm]["config"]


def _cached_path(arm: str):
    if arm in P.MODEL_ARMS:
        return TARGET_MODEL_DIR / f"{arm}_{P.fingerprint(arm)}.pt"
    return EXP2 / "cache" / "target_models" / f"{arm}_{P2.fingerprint(arm)}.pt"


def full_model(arm: str, use_cache: bool = True) -> N.SetSeq:
    """Full-data member (all 7 proxy matches). exp_2 members are read from exp_2's cache only;
    ``use_cache=False`` always fits afresh (independent rebuild) and saves nothing."""
    config = _config(arm)
    path = _cached_path(arm)
    if use_cache and path.exists():
        model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
        model.load_state_dict(torch.load(path, map_location=N.DEVICE))
        return model.eval()
    if use_cache and arm not in P.MODEL_ARMS:
        raise RuntimeError(f"exp_2 full-data model missing for {arm}; exp_3 never refits it silently")
    model = P.fit_model(arm, list(MATCH_IDS)) if arm in P.MODEL_ARMS else P2.fit_model(arm, list(MATCH_IDS))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return model


def ensemble_output(models: dict, item: dict) -> tuple[np.ndarray, np.ndarray]:
    outputs = [N.predict(model, item, _config(arm).window) for arm, model in models.items()]
    return (np.mean([o[0] for o in outputs], axis=0).astype("float32"),
            np.mean([o[1] for o in outputs], axis=0).astype("float32"))


def final_config(variant: str) -> dict:
    return read_json(RESULTS / f"variant_{variant}__target_like__ctrl.json")["final_config"]


def codes_for(output: tuple, item: dict, match, config: dict) -> np.ndarray:
    probability, boundary = output
    codes = expand(D.decode(probability, boundary, item["half"], D.DecoderConfig(**config)),
                   item["indices"], match.half)
    codes[X1.HALF2_START:X1.KICKOFF_2ND] = 2
    return codes.astype("int8")


def aggregates(codes: np.ndarray, match) -> dict:
    out = E2.aggregates(codes, match.half)
    out["possessions_per_min_minus_ryu"] = out["possessions_per_min"] - RYU_POSSESSIONS_PER_MIN
    return out


def exp2_candidate_aggregates() -> dict:
    summary = read_json(EXP2 / "results" / "target_candidate.json")
    keys = ("graded_frames", "share_graded", "possession_count", "dead_share", "possessions_per_min",
            "mean_possession_s", "dead_share_minus_prior", "sha256")
    return {k: summary[k] for k in keys}


def plausibility(match, items: dict, models: dict) -> dict:
    """Label-free aggregates: exp_1 / exp_2 shipped files and both exp_3 ensembles on both inputs."""
    result: dict[str, object] = {"exp1_setseq_shipped": E2.exp1_aggregates(),
                                 "exp2_candidate_exp2_aug": exp2_candidate_aggregates()}
    for variant, members in MEMBERS.items():
        config = final_config(variant)
        for source, item in items.items():
            if item is None:
                continue
            output = ensemble_output({a: models[a] for a in members}, item)
            result[f"{variant}__{source}"] = {**aggregates(codes_for(output, item, match, config), match),
                                              "mean_p_dead": float(output[0][:, 2].mean()), "decoder": config}
    result["published_dead_share_prior"] = E2.PUBLISHED_DEAD_SHARE
    result["ryu_possessions_per_min"] = RYU_POSSESSIONS_PER_MIN
    return result


def build_candidate(match, items: dict, models: dict, fusion: dict, path=CANDIDATE) -> dict:
    """The adoption tree's candidate (results/decision.json), on the fused input iff accepted."""
    decision = read_json(RESULTS / "decision.json")
    variant = decision["candidate_variant"]
    source = "fused" if fusion["use_fused_input"] else "tactical"
    config = final_config(variant)
    if config != decision["candidate_final_config"]:
        raise RuntimeError("Final config differs from decision.json")
    output = ensemble_output({a: models[a] for a in MEMBERS[variant]}, items[source])
    codes = codes_for(output, items[source], match, config)
    digest = X1.write_csv(path, codes)
    return {"variant": variant, "members": list(MEMBERS[variant]), "input": source, "sha256": digest,
            "format_valid": True, "decoder": config, "public_labels_used": False, **aggregates(codes, match)}


HEDGE = ROOT / "predictions" / "exp_3_hedge_provider.csv"
HEDGE_REPRODUCED = ROOT / "predictions" / "exp_3_hedge_provider_reproduced.csv"
HEDGE_VARIANT, HEDGE_LABEL = "exp3_ref_ens3", "provider"


def hedge_config() -> dict:
    """Post-hoc hedge (not pre-registered): the same ensemble, decoder selected on provider labels."""
    return read_json(RESULTS / f"variant_{HEDGE_VARIANT}__target_like__{HEDGE_LABEL}.json")["final_config"]


def build_hedge(match, items: dict, models: dict, fusion: dict, path=HEDGE) -> dict:
    source = "fused" if fusion["use_fused_input"] else "tactical"
    config = hedge_config()
    output = ensemble_output({a: models[a] for a in MEMBERS[HEDGE_VARIANT]}, items[source])
    codes = codes_for(output, items[source], match, config)
    digest = X1.write_csv(path, codes)
    return {"variant": HEDGE_VARIANT, "selected_on": HEDGE_LABEL, "pre_registered": False,
            "members": list(MEMBERS[HEDGE_VARIANT]), "input": source, "sha256": digest, "format_valid": True,
            "decoder": config, "public_labels_used": False, **aggregates(codes, match)}


def reproduce_hedge(match, items: dict, fusion: dict) -> dict:
    """Independent fits of the 3 members (no cached weights) -> byte-identical hedge CSV."""
    models = {arm: full_model(arm, use_cache=False) for arm in MEMBERS[HEDGE_VARIANT]}
    second = build_hedge(match, items, models, fusion, HEDGE_REPRODUCED)["sha256"]
    first = sha256(HEDGE)
    return {"first_sha256": first, "second_sha256": second, "byte_identical": first == second,
            "independent_fits": len(models)}


def reproduce(match, items: dict, fusion: dict) -> dict:
    """Independent fits of the candidate's 3 members (no cached weights) -> byte-identical CSV."""
    variant = read_json(RESULTS / "decision.json")["candidate_variant"]
    models = {arm: full_model(arm, use_cache=False) for arm in MEMBERS[variant]}
    second = build_candidate(match, items, models, fusion, REPRODUCED)["sha256"]
    first = sha256(CANDIDATE)
    return {"first_sha256": first, "second_sha256": second, "byte_identical": first == second,
            "independent_fits": len(models)}
