"""exp_2 shared paths and pre-registration checks.

exp_1 is reused by import (``../exp_1`` on ``sys.path``), never copied: its frozen judge,
token cache, network, decoder and reference arms are hash-checked through ``exp1_common``.
exp_2 writes only inside its own folder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
EXP1 = ROOT.parent / "exp_1"
if str(EXP1) not in sys.path:
    sys.path.insert(0, str(EXP1))

import exp1_common as C1  # noqa: E402  (verifies the frozen judge snapshot at import)

CACHE = ROOT / "cache"
TOKEN_DIR = CACHE / "tokens"
PRED_DIR = CACHE / "predictions"
TARGET_MODEL_DIR = CACHE / "target_models"
RESULTS = ROOT / "results"
PREREG = ROOT / "PREREG.md"
PROTOCOL = RESULTS / "protocol.json"
PUBLIC_MARKER = EXP1 / "results" / "public_check_started.json"

MATCH_IDS = C1.MATCH_IDS
BASE_SEED = C1.BASE_SEED
VIEWS = ("original", "target_like")
PRIMARY_VIEW = "target_like"


def frozen_constants() -> dict:
    """Every number PREREG.md fixes, read from the code that implements it."""
    import exp1_decode as D
    import exp1_net as N
    import exp2_hsmm as H
    import exp2_protocol as P
    import exp2_views as V

    constants = {
        "target_like": {"k_out_frames": V.K_OUT, "slow_speed": V.SLOW_SPEED,
                        "slow_window_frames": V.SLOW_WINDOW, "slow_min_frames": V.SLOW_MIN_FRAMES,
                        "nan_speed": V.NAN_SPEED},
        "target_aug": {"rounds": V.AUG_ROUNDS, "k_max_frames": V.AUG_K_MAX,
                       "dead_min_frames": V.AUG_DEAD_MIN, "dead_probability": V.AUG_DEAD_PROB,
                       "dead_margin_frames": V.AUG_DEAD_MARGIN, "seed_offset": V.AUG_SEED_OFFSET},
        "train_config": N.TrainConfig().as_dict(),
        "model_arms": {k: {"recipe": v["recipe"], "seed": v["config"].seed}
                       for k, v in P.MODEL_ARMS.items()},
        "variants": {k: list(v) for k, v in P.VARIANTS.items()},
        "ensemble": P.ENSEMBLE,
        "bd_lambdas": D.LAMBDAS, "bd_gammas": D.GAMMAS,
        "hsmm": {"d_max": H.D_MAX, "weights": H.DURATION_WEIGHTS, "lambdas": H.HSMM_LAMBDAS},
        "bootstrap": {"draws": C1.BOOTSTRAP_DRAWS, "seed": BASE_SEED, "interval": 0.95},
        "views": VIEWS, "primary_view": PRIMARY_VIEW,
    }
    return json.loads(json.dumps(constants, default=C1._json_default))


def freeze_prereg() -> dict:
    """Record PREREG.md's SHA-256 (+ the constants it fixes) once, before any exp_2 fit."""
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; use verify_prereg()")
    if PRED_DIR.exists() and any(PRED_DIR.iterdir()):
        raise RuntimeError("Predictions exist already; the pre-registration must precede every fit")
    protocol = {
        "prereg_sha256": C1.sha256(PREREG),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": C1.verify_judge(),
        "constants": frozen_constants(),
    }
    C1.write_json(PROTOCOL, protocol)
    return protocol


def verify_prereg() -> dict:
    """Fail loudly if PREREG.md, the judge or any pre-registered constant changed."""
    protocol = C1.read_json(PROTOCOL)
    if C1.sha256(PREREG) != protocol["prereg_sha256"]:
        raise RuntimeError("PREREG.md changed after freezing; diagnose before rerunning")
    if C1.verify_judge() != protocol["judge"]:
        raise RuntimeError("Frozen judge changed after freezing")
    if frozen_constants() != protocol["constants"]:
        raise RuntimeError("Pre-registered constants differ from the code; diagnose before rerunning")
    return protocol


def public_check_spent() -> bool:
    """exp_2 never opens the public ground truth; exp_1 spent the single check."""
    return PUBLIC_MARKER.exists()
