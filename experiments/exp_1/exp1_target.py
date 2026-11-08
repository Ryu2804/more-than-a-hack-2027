"""BAR-ATH inference with the frozen exp_1 pipeline. Aggregate outputs only.

Hygiene: BAR-ATH arrays stay in this process. Nothing per-frame is printed or written
except the submission CSV itself, which lives in a private ``predictions/`` folder.
The public 5-minute ground truth is opened once, after the predictor is frozen, as a
correctness check that is never used to choose anything.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from lxml import etree

import exp1_decode as D
import exp1_protocol as P
import exp1_tokens as T
from exp1_common import (
    JD, MATCH_IDS, PM, RESULTS, ROOT, TARGET_REPO, expand, read_json, sha256, write_json,
)

N_FRAMES = 199319
HALF2_START = 110808  # published half boundary
KICKOFF_2ND = 113742  # published 2nd-half kick-off; 110808-113741 are dead ball
PRE_KICKOFF_PERIOD = 3
PRED_OUT = ROOT / "predictions"
CANDIDATE = PRED_OUT / "exp_1_setseq_bd.csv"
REPRODUCED = PRED_OUT / "exp_1_setseq_bd_reproduced.csv"
MARKER = RESULTS / "public_check_started.json"
META = TARGET_REPO / "data" / "tactical" / "BAR-ATH_tactical_atd.xml"


def team_audit() -> dict:
    """Counts of track slots per metadata team id (aggregate metadata, no tracking values)."""
    tree = etree.parse(str(META))
    team_ids = [t.get("id") for t in tree.iter("Team")]
    slots = [p.get("teamId") for p in tree.iter("Player")]
    return {
        "metadata_teams": len(team_ids),
        "slots": len(slots),
        "slots_per_team_rank": [sum(s == t for s in slots) for t in team_ids],
        "slots_outside_first_two_teams": sum(s not in team_ids[:2] for s in slots),
    }


def load_target_match() -> "JD.Match":
    """Tactical tracking via the frozen loader; non-team slots dropped; pre-kick-off split off."""
    match = JD.load_target("tactical")
    tree = etree.parse(str(META))
    team_ids = [t.get("id") for t in tree.iter("Team")][:2]
    keep = np.array([p.get("teamId") in team_ids for p in tree.iter("Player")])
    players = match.players.copy()
    players[:, ~keep] = np.nan
    half = match.half.copy()
    half[HALF2_START:KICKOFF_2ND] = PRE_KICKOFF_PERIOD
    return replace(match, players=players, half=half)


def final_config() -> D.DecoderConfig:
    stored = read_json(RESULTS / "variant_setseq_bd.json")["final_config"]
    return D.DecoderConfig(**stored)


def predict_codes(match) -> tuple[np.ndarray, dict]:
    """Train SetSeq on all seven proxy matches, decode BAR-ATH, return 25 fps codes."""
    token_set = T.build_tokens(match)
    item = {"tokens": token_set.tokens, "mask": token_set.mask, "glob": token_set.glob,
            "half": token_set.half}
    probability, boundary = P.fit_predict("setseq", list(MATCH_IDS), {"target": item})["target"]
    codes = expand(D.decode(probability, boundary, token_set.half, final_config()),
                   token_set.indices, match.half)
    codes[HALF2_START:KICKOFF_2ND] = 2
    graded = match.half > 0
    stats = {
        "tokens_per_step": float(token_set.mask.sum(1).mean()),
        "steps_at_token_cap": float((token_set.mask.sum(1) == T.MAX_TOKENS).mean()),
        "ball_coverage_graded": float(np.isfinite(match.ball[graded]).all(axis=1).mean()),
        "mean_p_boundary": float(boundary.mean()),
    }
    return codes.astype("int8"), stats


def write_csv(path, codes: np.ndarray) -> str:
    if len(codes) != N_FRAMES or not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError("Invalid target prediction schema")
    path.parent.mkdir(mode=0o700, exist_ok=True)
    letters = np.array(list("HADX"))[codes]
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": letters}).to_csv(
        path, index=False, lineterminator="\n")
    error = PM.validate_submission(PM.read_label_csv(path), N_FRAMES)
    if error:
        raise ValueError("Official format validation failed")
    return sha256(path)


def aggregates(codes: np.ndarray, half: np.ndarray) -> dict:
    graded = half > 0
    runs = [(int(i[0]), int(i[-1]) + 1) for p in (1, 2, PRE_KICKOFF_PERIOD)
            for i in [np.flatnonzero(half == p)] if len(i)]
    return {
        "graded_frames": int(graded.sum()),
        "share_graded": {k: float(np.mean(codes[graded] == c)) for c, k in enumerate("HAD")},
        "possession_count": len(PM.possession_spells(codes, runs)),
        "abstain_frames_graded": int(np.sum(codes[graded] == 3)),
    }


def build_candidate(path=CANDIDATE) -> dict:
    if MARKER.exists() and path == CANDIDATE:
        raise RuntimeError("Predictor frozen after the single-use public check")
    match = load_target_match()
    codes, stats = predict_codes(match)
    digest = write_csv(path, codes)
    summary = {"sha256": digest, "format_valid": True, **stats, **aggregates(codes, match.half),
               "team_audit": team_audit(), "decoder": final_config().as_dict(),
               "public_labels_used": False}
    return summary


def reproduce() -> dict:
    """Independent second fit + inference must give a byte-identical CSV."""
    first = read_json(RESULTS / "target_summary.json")["sha256"]
    second = build_candidate(REPRODUCED)["sha256"]
    result = {"first_sha256": first, "second_sha256": second, "byte_identical": first == second}
    write_json(RESULTS / "target_reproduction.json", result)
    return result


def public_sanity_once() -> dict:
    """One look at the public 5-minute window after freezing; never used for selection."""
    if (RESULTS / "public_sanity.json").exists():
        return read_json(RESULTS / "public_sanity.json")
    if MARKER.exists():
        raise RuntimeError("Public check already attempted; it cannot be repeated")
    frozen = {name: sha256(ROOT / name) for name in sorted(p.name for p in ROOT.glob("exp1_*.py"))}
    write_json(MARKER, {"predictor_sources": frozen, "candidate_sha256": sha256(CANDIDATE),
                        "decoder": final_config().as_dict()})
    public = PM.label_codes(PM.read_label_csv(TARGET_REPO / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
    predicted = PM.label_codes(PM.read_label_csv(CANDIDATE))
    result = PM.evaluate(public, predicted, bootstrap_n=1000)
    keep = ("macro_f1", "spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high",
            "n_gt_spells", "n_pred_spells", "alive_dead_f1", "team_acc_both_alive")
    report = {"purpose": "sanity check, not used for selection", **{k: result[k] for k in keep}}
    write_json(RESULTS / "public_sanity.json", report)
    return report
