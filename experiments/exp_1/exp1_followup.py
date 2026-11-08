"""Pre-registered follow-up (PREREG_followup.md): centred boundary-aware decoding.

Switch cost i -> j at step t: max(0, lambda_ij - gamma * (log b_t - log mean(b))); staying is free.
gamma = 0 is the constant-penalty decoder. Reuses the cached exp_1 SetSeq predictions; nothing
is retrained for the proxy. BAR-ATH outputs are aggregates only and never use public labels.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numba import njit

import exp1_decode as D
import exp1_protocol as P
import exp1_target as X
import exp1_tokens as T
from exp1_common import MATCH_IDS, RESULTS, ROOT, expand, read_json, score, sha256

PREREG = ROOT / "PREREG_followup.md"
PROTOCOL = RESULTS / "followup_protocol.json"
CANDIDATE = ROOT / "predictions" / "exp_1_followup_cbd.csv"
REPRODUCED = ROOT / "predictions" / "exp_1_followup_cbd_reproduced.csv"
MAIN_ARM = "setseq_cbd_wide"


def verify_prereg() -> dict:
    protocol = read_json(PROTOCOL)
    if sha256(PREREG) != protocol["prereg_sha256"]:
        raise RuntimeError("Pre-registration changed after freezing")
    return protocol


@njit(cache=True)
def _viterbi_centred(logp, adjust, lambda_team, lambda_dead):
    n = logp.shape[0]
    back = np.zeros((n, 3), dtype=np.int8)
    cost = logp[0].copy()
    for t in range(1, n):
        new = np.empty(3)
        for j in range(3):
            best, arg = -1e300, 0
            for i in range(3):
                if i == j:
                    value = cost[i]
                else:
                    penalty = lambda_dead if (i == 2 or j == 2) else lambda_team
                    value = cost[i] - max(0.0, penalty - adjust[t])
                if value > best:
                    best, arg = value, i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        cost = new
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax(cost))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path


def decode_centred(probability: np.ndarray, boundary: np.ndarray, half: np.ndarray,
                   config: D.DecoderConfig) -> np.ndarray:
    if probability.shape != (len(half), 3) or boundary.shape != (len(half),):
        raise ValueError("Probabilities and boundary must align with periods")
    b = np.clip(boundary.astype(np.float64), D.EPS, 1 - D.EPS)
    adjust = config.gamma * (np.log(b) - np.log(b.mean()))
    logp = np.log(np.clip(D.calibrate(probability.astype(np.float64), config.temperature), 1e-6, 1.0))
    output = np.empty(len(half), dtype=np.int8)
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        output[index] = _viterbi_centred(logp[index], adjust[index],
                                         config.lambda_team, config.lambda_dead)
    return output


def score_item(item: dict, config: D.DecoderConfig) -> dict:
    codes = decode_centred(item["probability"], item["boundary"], item["half"], config)
    return score(item["truth"], expand(codes, item["indices"], item["full_half"]))


def select(validation: list[dict], gammas: list[float], lambdas: list[float]):
    temperature = D.fit_temperature([v["probability"] for v in validation],
                                    [v["state"] for v in validation])
    best, best_value = None, -np.inf
    for gamma, lambda_team, lambda_dead in product(gammas, lambdas, lambdas):
        config = D.DecoderConfig(temperature, lambda_team, lambda_dead, gamma)
        value = D.objective([score_item(v, config) for v in validation])
        if value > best_value + 1e-12:
            best, best_value = config, value
    if best is None:
        raise RuntimeError("Follow-up grid produced no finite objective")
    return best


def evaluate(variant: str) -> dict:
    """Nested evaluation identical to exp_1, with the centred decoder and the frozen wide grid."""
    protocol = verify_prereg()
    gammas, lambdas = protocol["arms"][variant], protocol["lambdas"]
    predictions = P.ensure_predictions("setseq")
    rows, selections = [], {}
    for held in MATCH_IDS:
        validation = [P.make_item(v, predictions[P.pair_key(held, v)][v]) for v in MATCH_IDS if v != held]
        config = select(validation, gammas, lambdas)
        selections[held] = config.as_dict()
        outer = P.make_item(held, predictions[f"outer_{held}"][held])
        rows.append({"match": held, "variant": variant, **score_item(outer, config), **config.as_dict()})
    oof = [P.make_item(n, predictions[f"outer_{n}"][n]) for n in MATCH_IDS]
    final = select(oof, gammas, lambdas).as_dict()
    return {"rows": rows, "selections": selections, "final_config": final}


def decide(rows) -> dict:
    """Apply the three pre-registered decision rules (paired 95% CI lower bound > 0)."""
    import exp1_report as R

    rules = {
        "rule1_centring_helps": R._delta(rows, MAIN_ARM, "setseq_const_wide"),
        "rule2_grid_edge_mattered": R._delta(rows, "setseq_const_wide", "setseq_const"),
        "rule3_beats_frozen_exp1": R._delta(rows, MAIN_ARM, "setseq_bd"),
    }
    gates = {arm: R.gates(rows, arm)["checks"] for arm in (MAIN_ARM, "setseq_const_wide")}
    return {
        "rules": rules,
        "passed": {name: rule["ci95"][0] > 0 for name, rule in rules.items()},
        "adopt_followup_candidate": rules["rule3_beats_frozen_exp1"]["ci95"][0] > 0,
        "count_gate": gates,
    }


def build_candidate(path=CANDIDATE) -> dict:
    """BAR-ATH follow-up CSV (only if decision rule 3 adopted it); aggregates only."""
    decision = read_json(RESULTS / "followup_decision.json")
    if not decision["adopt_followup_candidate"]:
        raise RuntimeError("Decision rule 3 did not adopt a follow-up candidate")
    config = D.DecoderConfig(**read_json(RESULTS / f"variant_{MAIN_ARM}.json")["final_config"])
    match = X.load_target_match()
    token_set = T.build_tokens(match)
    item = {"tokens": token_set.tokens, "mask": token_set.mask, "glob": token_set.glob,
            "half": token_set.half}
    probability, boundary = P.fit_predict("setseq", list(MATCH_IDS), {"target": item})["target"]
    codes = expand(decode_centred(probability, boundary, token_set.half, config),
                   token_set.indices, match.half)
    codes[X.HALF2_START:X.KICKOFF_2ND] = 2
    digest = X.write_csv(path, codes)
    return {"sha256": digest, "format_valid": True, **X.aggregates(codes, match.half),
            "decoder": config.as_dict(), "public_labels_used": False}
