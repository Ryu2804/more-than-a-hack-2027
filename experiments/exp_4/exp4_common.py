"""exp_4 shared paths, constants and pre-registration checks.

exp_1-exp_3 are reused by import (``../exp_3`` on ``sys.path``; ``exp3_common`` adds ``../exp_2``
and ``../exp_1``), never copied or edited. exp_2 / exp_3 caches and result files are read in
place and hash-checked. exp_4 writes only inside its own folder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
EXP3 = ROOT.parent / "exp_3"
if str(EXP3) not in sys.path:
    sys.path.insert(0, str(EXP3))

import exp3_common as C3  # noqa: E402  (puts ../exp_2 and ../exp_1 on sys.path, verifies the judge)
import exp1_common as C1  # noqa: E402

EXP1, EXP2 = C3.EXP1, C3.EXP2
CACHE = ROOT / "cache"
PRED_DIR = CACHE / "predictions"
AUX_DIR = CACHE / "aux"
TARGET_MODEL_DIR = CACHE / "target_models"
RESULTS = ROOT / "results"
PREDICTIONS_OUT = ROOT / "predictions"
PREREG = ROOT / "PREREG.md"
PROTOCOL = RESULTS / "protocol.json"

MATCH_IDS = C1.MATCH_IDS
BASE_SEED = C1.BASE_SEED
VIEWS = C3.VIEWS  # ("original", "target_like")
PRIMARY_VIEW = "target_like"
TRAIN_LABEL = "provider"
DECISION_LABELS = ("provider", "ctrl")  # H1: provider superiority + L_ctrl non-inferiority
S1_LABELS = ("provider", "track_1", "ctrl", "track_3")  # track_2 == ctrl (exp_3 fallback)
H1_PROVIDER_CI_LOW = 0.0
H1_CTRL_CI_LOW = -0.01
GAP_GROWTH_FLAG = 0.25  # flag an arm whose generalisation gap exceeds the reference by > 25 %
FIT_CAP = 280
REFERENCE = "exp3_ref_ens3"
PRIMARY = "exp4_full_ens3"


def reused_result_files() -> dict:
    """exp_3 result files read in place (reference arms, every S1 label set, both views)."""
    import exp4_protocol as P

    return {f"{v}__{view}__{label}": C3.RESULTS / f"variant_{v}__{view}__{label}.json"
            for v in P.REUSED_VARIANTS for view in VIEWS for label in S1_LABELS}


def frozen_constants() -> dict:
    """Every number PREREG.md fixes, read from the code that implements it."""
    import exp1_decode as D
    import exp1_net as N
    import exp3_touch as S
    import exp4_net as X
    import exp4_protocol as P

    constants = {
        "train_config": N.TrainConfig().as_dict(),
        "recipe": P.RECIPE,
        "train_label": TRAIN_LABEL,
        "arms": {k: v.as_dict() for k, v in P.ARMS.items()},
        "ensembles": {k: list(v) for k, v in P.ENSEMBLES.items()},
        "variants": {k: list(v) for k, v in P.VARIANTS.items()},
        "reused_variants": list(P.REUSED_VARIANTS),
        "aux": {"on": X.AUX_ON.as_dict(), "off": X.AUX_OFF.as_dict(), "head_outputs": X.AUX_OUTPUTS},
        "touch": {"names": S.TOUCH_NAMES, "window_frames": S.WINDOW, "dist_scale": S.DIST_SCALE,
                  "dist_cap": S.DIST_CAP, "dv_scale": S.DV_SCALE, "dv_cap": S.DV_CAP},
        "fingerprints": {arm: P.fingerprint(arm) for arm in P.ARMS},
        "exp2_cached_fingerprints": P.EXP2_FINGERPRINTS,
        "reused_result_sha256": {k: C1.sha256(p) for k, p in reused_result_files().items()},
        "bd_lambdas": D.LAMBDAS, "bd_gammas": D.GAMMAS,
        "views": VIEWS, "primary_view": PRIMARY_VIEW, "s1_labels": S1_LABELS,
        "h1": {"comparison": [PRIMARY, REFERENCE], "view": PRIMARY_VIEW,
               "provider_rule": f"paired 95% CI lower bound > {H1_PROVIDER_CI_LOW}",
               "ctrl_rule": f"paired 95% CI lower bound > {H1_CTRL_CI_LOW}",
               "provider_ci_low": H1_PROVIDER_CI_LOW, "ctrl_ci_low": H1_CTRL_CI_LOW},
        "bootstrap": {"draws": C1.BOOTSTRAP_DRAWS, "seed": BASE_SEED, "interval": 0.95},
        "gap_growth_flag": GAP_GROWTH_FLAG,
        "fit_cap": FIT_CAP,
    }
    return json.loads(json.dumps(constants, default=C1._json_default))


def code_hashes() -> dict:
    """exp_4 code at freezing time (informational record; fit code is also in the fingerprints)."""
    names = ("exp4_common.py", "exp4_net.py", "exp4_aux.py", "exp4_protocol.py")
    return {name: C1.sha256(ROOT / name) for name in names}


def freeze_prereg() -> dict:
    """Record PREREG.md's SHA-256 (+ the constants it fixes) once, before any exp_4 fit."""
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
    """exp_4 never opens the public ground truth; exp_1 spent the single check."""
    return C3.public_check_spent()
