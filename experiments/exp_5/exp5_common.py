"""exp_5 shared paths, constants and pre-registration checks.

exp_0-exp_4 are reused by import (``../exp_4`` on ``sys.path``; ``exp4_common`` adds ``../exp_3``,
``../exp_2`` and ``../exp_1``), never copied or edited. Their caches and result files are read in
place and SHA-256-checked against ``results/protocol.json``. exp_5 writes only inside its own folder.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
# exp1_common defaults NUMBA_CACHE_DIR to exp_1/cache/numba: force every cache write into exp_5.
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "cache" / "numba")
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # before CUDA initialises (exp1_net needs it)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

EXP4 = ROOT.parent / "exp_4"
if str(EXP4) not in sys.path:
    sys.path.insert(0, str(EXP4))

from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import exp4_common as C4  # noqa: E402  (puts ../exp_3, ../exp_2, ../exp_1 on sys.path)

C3 = C4.C3
C1 = C4.C1
EXP1, EXP2, EXP3 = C4.EXP1, C4.EXP2, C4.EXP3
EXPERIMENTS = ROOT.parent
CACHE = ROOT / "cache"
PRED_DIR = CACHE / "predictions"
TARGET_MODEL_DIR = CACHE / "target_models"
LOG_DIR = CACHE / "logs"
RESULTS = ROOT / "results"
PREDICTIONS_OUT = ROOT / "predictions"
PREREG = ROOT / "PREREG.md"
PROTOCOL = RESULTS / "protocol.json"

MATCH_IDS = C1.MATCH_IDS
BASE_SEED = C1.BASE_SEED  # bootstrap seed 20261108 (exp_2 / exp_3 / exp_4)
VIEWS = C3.VIEWS  # ("original", "target_like")
PRIMARY_VIEW = "target_like"
TRAIN_LABEL = "provider"
DECISION_LABELS = ("provider", "ctrl")  # H1 on provider; ctrl gates only the ctrl file
S1_LABELS = ("provider", "track_1", "ctrl", "track_3")
PRIMARY = "exp5_touch_ens3"
REFERENCE = "exp3_ref_ens3"
PRIMARY_FAMILY = "exp5_touch"
H1_CI_LOW = 0.0  # clause 1: paired 95% CI lower bound > 0
CTRL_NONINF_CI_LOW = -0.01  # ctrl file only if the L_ctrl paired CI lower bound > -0.01
GAP_GROWTH_FLAG = 0.25
FIT_CAP = 100
PLANNED_FITS = {"reproduction_check": 1, "proxy": 84, "full_data": 3, "rebuild": 3}

EXP4_TOUCH_ARMS = ("exp4_touch", "exp4_touch_s1", "exp4_touch_s2")
EXP4_RESULT_FILES = ("ref_train_nll.csv", "gap_table.csv")
FOLDER_PATTERN = re.compile(r"^(?P<arm>.+)_(?P<fp>[0-9a-f]{12})$")
WATCHED = ("exp_0", "exp_1", "exp_2", "exp_3", "exp_4")  # exp_5 never writes here


# ---------------------------------------------------------------- exp_4 caches read in place
def exp4_touch_folder(arm: str) -> Path:
    """exp_4's prediction folder of one touch arm (exactly one ``<arm>_<12 hex>`` folder)."""
    if arm not in EXP4_TOUCH_ARMS:
        raise ValueError(f"{arm} is not an exp_4 touch arm")
    hits = [p for p in sorted(C4.PRED_DIR.iterdir())
            if (m := FOLDER_PATTERN.match(p.name)) and m["arm"] == arm and p.is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one exp_4 cache folder for {arm}, found {len(hits)}")
    return hits[0]


def job_names() -> list[str]:
    import exp1_protocol as P1

    return [key for key, *_ in P1.job_keys()]


def exp4_touch_cache_sha256() -> dict:
    """SHA-256 of every exp_4 touch prediction (.npz) that exp_5 reads in place."""
    out = {}
    for arm in EXP4_TOUCH_ARMS:
        folder = exp4_touch_folder(arm)
        out[arm] = {"folder": folder.name,
                    "npz": {key: C1.sha256(folder / f"{key}.npz") for key in job_names()}}
    return out


def reused_result_files() -> dict:
    """Result files read in place: exp_3 reference arms (+ context rows), exp_4 touch ensemble,
    exp_4's reference training NLL and gap table."""
    files = {f"exp_3/{v}__{view}__{label}": C3.RESULTS / f"variant_{v}__{view}__{label}.json"
             for v in ("exp3_ref_ens3", "exp2_aug") for view in VIEWS for label in S1_LABELS}
    files["exp_3/reference_rows.csv"] = C3.RESULTS / "reference_rows.csv"
    files.update({f"exp_4/exp4_touch_ens3__{view}__{label}":
                  C4.RESULTS / f"variant_exp4_touch_ens3__{view}__{label}.json"
                  for view in VIEWS for label in S1_LABELS})
    files.update({f"exp_4/{name}": C4.RESULTS / name for name in EXP4_RESULT_FILES})
    return files


# ---------------------------------------------------------------- pre-registration
def frozen_constants() -> dict:
    """Every number PREREG.md fixes, read from the code that implements it."""
    import exp1_decode as D
    import exp1_net as N
    import exp3_touch as S
    import exp4_net as X
    import exp5_eval as E
    import exp5_protocol as P

    constants = {
        "train_config": N.TrainConfig().as_dict(),
        "recipe": P.RECIPE,
        "train_label": TRAIN_LABEL,
        "arms": {k: v.as_dict() for k, v in P.ARMS.items()},
        "fit_order": list(P.FIT_ORDER),
        "fingerprints": {arm: P.fingerprint(arm) for arm in P.ARMS},
        "fit_code": sorted(P.FIT_CODE),
        "ensembles": {k: list(v) for k, v in E.ENSEMBLES.items()},
        "evaluated_variants": {k: list(v) for k, v in E.EVALUATED.items()},
        "reused_variants": {k: str(v.relative_to(EXPERIMENTS)) for k, v in E.REUSED.items()},
        "never_adopted": list(E.NEVER_ADOPTED),
        "aux": X.AUX_OFF.as_dict(),
        "touch": {"names": S.TOUCH_NAMES, "window_frames": S.WINDOW, "dist_scale": S.DIST_SCALE,
                  "dist_cap": S.DIST_CAP, "dv_scale": S.DV_SCALE, "dv_cap": S.DV_CAP},
        "reproduction_check": P.REPRO,
        "exp2_cached_fingerprints": E.EXP2_FINGERPRINTS,
        "exp4_touch_cache_sha256": exp4_touch_cache_sha256(),
        "reused_result_sha256": {k: C1.sha256(p) for k, p in reused_result_files().items()},
        "bd_lambdas": D.LAMBDAS, "bd_gammas": D.GAMMAS,
        "views": VIEWS, "primary_view": PRIMARY_VIEW, "s1_labels": S1_LABELS,
        "decision_labels": DECISION_LABELS,
        "h1": {"comparison": [PRIMARY, REFERENCE], "labels": "provider", "view": PRIMARY_VIEW,
               "ci_clause": f"paired 95% CI lower bound > {H1_CI_LOW}",
               "seed_clause": f"mean delta > seed spread (max - min single-seed mean) of {PRIMARY_FAMILY}, "
                              f"provider, {PRIMARY_VIEW}",
               "ci_low": H1_CI_LOW},
        "ctrl_file_rule": f"L_ctrl {PRIMARY_VIEW} paired 95% CI lower bound > {CTRL_NONINF_CI_LOW}",
        "ctrl_noninf_ci_low": CTRL_NONINF_CI_LOW,
        "bootstrap": {"draws": C1.BOOTSTRAP_DRAWS, "seed": BASE_SEED, "interval": 0.95},
        "gap_growth_flag": GAP_GROWTH_FLAG,
        "fit_cap": FIT_CAP, "planned_fits": PLANNED_FITS,
    }
    return json.loads(json.dumps(constants, default=C1._json_default))


def code_hashes() -> dict:
    """exp_5 code at freezing time (informational; fit code is also in the fingerprints)."""
    names = ("exp5_common.py", "exp5_protocol.py", "exp5_eval.py", "exp5_diagnostics.py",
             "exp5_report.py", "exp5_target.py")
    return {name: C1.sha256(ROOT / name) for name in names}


def predictions_exist() -> bool:
    return PRED_DIR.exists() and any(PRED_DIR.iterdir())


def freeze_prereg() -> dict:
    """Record PREREG.md's SHA-256 (+ the constants it fixes) once, before any exp_5 fit."""
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; use verify_prereg()")
    if predictions_exist() or (RESULTS / "repro_check.json").exists():
        raise RuntimeError("exp_5 fits exist already; the pre-registration must precede every fit")
    protocol = {
        "prereg_sha256": C1.sha256(PREREG),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": C1.verify_judge(),
        "code_sha256_at_freeze": code_hashes(),
        "constants": frozen_constants(),
    }
    C1.write_json(PROTOCOL, protocol)
    return protocol


def refreeze_prereg() -> dict:
    """Re-freeze (PREREG header): allowed only while no exp_5 prediction exists; the old
    protocol is kept as ``results/protocol_superseded_N.json``."""
    if predictions_exist() or (RESULTS / "repro_check.json").exists():
        raise RuntimeError("exp_5 fits exist: re-freezing is no longer allowed")
    number = 1 + sum(1 for _ in RESULTS.glob("protocol_superseded_*.json"))
    os.replace(PROTOCOL, RESULTS / f"protocol_superseded_{number}.json")
    return freeze_prereg()


def verify_prereg() -> dict:
    """Fail loudly if PREREG.md, the judge, a pre-registered constant or a reused file changed."""
    protocol = C1.read_json(PROTOCOL)
    if C1.sha256(PREREG) != protocol["prereg_sha256"]:
        raise RuntimeError("PREREG.md changed after freezing; diagnose before rerunning")
    if C1.verify_judge() != protocol["judge"]:
        raise RuntimeError("Frozen judge changed after freezing")
    if frozen_constants() != protocol["constants"]:
        raise RuntimeError("Pre-registered constants or reused files differ; diagnose before rerunning")
    return protocol


def public_check_spent() -> bool:
    """exp_5 never opens the public ground truth; exp_1 spent the single check."""
    return C3.public_check_spent()


# ---------------------------------------------------------------- write guard
def tree_digest(folder: Path) -> dict:
    """File count + digest of (path, size, mtime) under a folder; reads no file content."""
    digest, count = hashlib.sha256(), 0
    for path in sorted(p for p in folder.rglob("*") if p.is_file() or p.is_symlink()):
        stat = path.lstat()
        digest.update(f"{path.relative_to(folder)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        count += 1
    return {"files": count, "digest": digest.hexdigest()}


def watched_folders() -> dict:
    """exp_0-exp_4 and the competition data folder (exp_5 must write none of them)."""
    folders = {name: EXPERIMENTS / name for name in WATCHED}
    folders["data"] = C1.TARGET_REPO / "data"
    return folders


def write_manifest() -> dict:
    return {name: tree_digest(path) for name, path in watched_folders().items()}
