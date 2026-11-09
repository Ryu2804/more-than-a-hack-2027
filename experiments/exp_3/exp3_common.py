"""exp_3 shared paths and pre-registration checks.

exp_1 and exp_2 are reused by import (``../exp_2`` on ``sys.path``, which puts ``../exp_1``
there too), never copied or edited. exp_2's cached predictions and full-data models are read
in place and must match exp_2's current fingerprints. exp_3 writes only inside its own folder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
EXP2 = ROOT.parent / "exp_2"
if str(EXP2) not in sys.path:
    sys.path.insert(0, str(EXP2))

import exp2_common as C2  # noqa: E402  (puts ../exp_1 on sys.path, verifies the judge)
import exp1_common as C1  # noqa: E402

EXP1 = C2.EXP1
CACHE = ROOT / "cache"
TOKEN_DIR = CACHE / "tokens"
PRED_DIR = CACHE / "predictions"
TARGET_MODEL_DIR = CACHE / "target_models"
RESULTS = ROOT / "results"
PREREG = ROOT / "PREREG.md"
PROTOCOL = RESULTS / "protocol.json"
FUSION_PIN = ROOT / "judge" / "fusion_pin.json"
FUSION_SNAPSHOT = ROOT / "judge" / "exp0_fusion_candidate.py"

MATCH_IDS = C1.MATCH_IDS
BASE_SEED = C1.BASE_SEED
VIEWS = C2.VIEWS  # ("original", "target_like")
PRIMARY_VIEW = "target_like"
EVAL_LABEL = "ctrl"  # primary evaluation label set (L_ctrl)
LABEL_SETS = ("ctrl", "provider", "track_1", "track_2", "track_3")


def frozen_constants() -> dict:
    """Every number PREREG.md fixes, read from the code that implements it."""
    import exp1_decode as D
    import exp1_net as N
    import exp3_coverage as V3
    import exp3_fusion as F
    import exp3_labels as L
    import exp3_protocol as P

    constants = {
        "labels": {"tau_hold_s": L.TAU_HOLD_S, "event_margin_s": L.EVENT_MARGIN_S,
                   "track_taus_s": L.TRACK_TAUS_S, "events_used": P.EVENTS_USED,
                   "play_evaluations": L.CONTROLLED_PLAY_EVALUATIONS,
                   "claim_types": L.CONTROLLED_CLAIM_TYPES, "tackle_results": L.CONTROLLED_TACKLE_RESULTS,
                   "snap": {"before": L.SNAP_BEFORE, "after": L.SNAP_AFTER,
                            "max_distance_m": L.SNAP_MAX_DISTANCE_M},
                   "sync_acceptance": {"median_abs_max": L.SYNC_MEDIAN_MAX, "p90_abs_max": L.SYNC_P90_MAX},
                   "label_sets": LABEL_SETS, "eval_label": EVAL_LABEL},
        "train_config": N.TrainConfig().as_dict(),
        "recipe": P.RECIPE,
        "model_arms": {k: {"seed": v.seed} for k, v in P.MODEL_ARMS.items()},
        "variants": {k: list(v) for k, v in P.VARIANTS.items()},
        "ensembles": {k: list(v) for k, v in P.ENSEMBLES.items()},
        "exp2_cached_fingerprints": P.EXP2_FINGERPRINTS,
        "bd_lambdas": D.LAMBDAS, "bd_gammas": D.GAMMAS,
        "coverage": {"reveal_share": V3.REVEAL_SHARE, "seed_offset": V3.SEED_OFFSET},
        "fusion": {"replay_speed_mps": F.REPLAY_SPEED, "pin": json.loads(FUSION_PIN.read_text())},
        "h1": {"comparison": ["exp3_ctrl_ens3", "exp3_ref_ens3"], "view": PRIMARY_VIEW,
               "label": EVAL_LABEL, "rule": "paired 95% bootstrap CI lower bound > 0"},
        "bootstrap": {"draws": C1.BOOTSTRAP_DRAWS, "seed": BASE_SEED, "interval": 0.95},
        "views": VIEWS, "primary_view": PRIMARY_VIEW,
    }
    return json.loads(json.dumps(constants, default=C1._json_default))


def code_hashes() -> dict:
    """exp_3 code that defines labels, views, fusion and the fit recipe (informational record)."""
    names = ("exp3_labels.py", "exp3_coverage.py", "exp3_fusion.py", "exp3_protocol.py")
    hashes = {name: C1.sha256(ROOT / name) for name in names}
    hashes["judge/exp0_fusion_candidate.py"] = C1.sha256(FUSION_SNAPSHOT)
    return hashes


def freeze_prereg() -> dict:
    """Record PREREG.md's SHA-256 (+ the constants it fixes) once, before any exp_3 fit."""
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; use verify_prereg()")
    if PRED_DIR.exists() and any(PRED_DIR.iterdir()):
        raise RuntimeError("Predictions exist already; the pre-registration must precede every fit")
    protocol = {
        "prereg_sha256": C1.sha256(PREREG),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": C1.verify_judge(),
        "code_sha256_at_freeze": code_hashes(),
        "constants": frozen_constants(),
    }
    C1.write_json(PROTOCOL, protocol)
    return protocol


def verify_prereg() -> dict:
    """Fail loudly if PREREG.md, the judge, the fusion pin or any pre-registered constant changed."""
    import exp3_fusion as F

    protocol = C1.read_json(PROTOCOL)
    if C1.sha256(PREREG) != protocol["prereg_sha256"]:
        raise RuntimeError("PREREG.md changed after freezing; diagnose before rerunning")
    if C1.verify_judge() != protocol["judge"]:
        raise RuntimeError("Frozen judge changed after freezing")
    F.verify_pin()
    if frozen_constants() != protocol["constants"]:
        raise RuntimeError("Pre-registered constants differ from the code; diagnose before rerunning")
    return protocol


def public_check_spent() -> bool:
    """exp_3 never opens the public ground truth; exp_1 spent the single check."""
    return C2.public_check_spent()
