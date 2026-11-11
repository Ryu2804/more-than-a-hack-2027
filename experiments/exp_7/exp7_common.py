"""exp_7 shared paths, safe exp_6 imports, reused-file registry, pre-registration checks and write guard.

exp_1-exp_3 and exp_6 are reused by import, never copied or edited. Importing anything from exp_6 imports
``exp6_common``, which re-forces NUMBA_CACHE_DIR / MPLCONFIGDIR into ``exp_6/cache``. Order here:
1. force exp_7's environment; 2. import ``exp1_decode`` (numba fixes its cache directory to exp_7);
3. import ``exp6_sim`` / ``exp6_target`` / ``exp6_decision``; 4. restore exp_7's environment.
exp_6's caches and results are read in place and SHA-256-checked; exp_6 is in the write guard.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
ENV = {
    "NUMBA_CACHE_DIR": str(ROOT / "cache" / "numba"),
    "MPLCONFIGDIR": str(ROOT / "cache" / "matplotlib"),
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # before CUDA initialises
    "PYTHONDONTWRITEBYTECODE": "1",
}
os.environ.update(ENV)
sys.dont_write_bytecode = True

EXPERIMENTS = ROOT.parent
EXP3, EXP6 = EXPERIMENTS / "exp_3", EXPERIMENTS / "exp_6"
if str(EXP3) not in sys.path:
    sys.path.insert(0, str(EXP3))

from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import exp3_common as C3  # noqa: E402  (puts ../exp_2 and ../exp_1 on sys.path; verifies the judge)
import exp1_common as C1  # noqa: E402
import exp1_decode  # noqa: E402,F401  (numba imported now: its cache directory is exp_7's)
import numba  # noqa: E402

EXP1, EXP2 = C3.EXP1, C3.EXP2
if str(EXP6) not in sys.path:
    sys.path.append(str(EXP6))  # appended: exp_1-exp_3 modules keep their own folders first


def import_exp6(name: str):
    """Import an exp_6 module, then restore exp_7's environment (exp6_common overwrites it)."""
    try:
        return importlib.import_module(name)
    finally:
        os.environ.update(ENV)


S6 = import_exp6("exp6_sim")
T6 = import_exp6("exp6_target")
DE6 = import_exp6("exp6_decision")
C6 = sys.modules["exp6_common"]
if Path(numba.config.CACHE_DIR) != Path(ENV["NUMBA_CACHE_DIR"]):
    raise RuntimeError("numba cache directory is not exp_7's")

CACHE = ROOT / "cache"
TOKEN_DIR = CACHE / "tokens"
PRED_DIR = CACHE / "predictions"
TARGET_MODEL_DIR = CACHE / "target_models"
REBUILD_DIR = CACHE / "rebuild"
LOG_DIR = CACHE / "logs"
RESULTS = ROOT / "results"
PREDICTIONS_OUT = ROOT / "predictions"
PREREG = ROOT / "PREREG.md"
PROTOCOL = RESULTS / "protocol.json"

MATCH_IDS = C1.MATCH_IDS
BASE_SEED = C1.BASE_SEED  # 20261108: bootstrap seed and seed base
LABEL_SETS = ("provider", "ctrl")  # H1 needs both
TRAIN_LABEL = "provider"
REFERENCE = "exp3_ref_ens3"
REF_MEMBERS = C6.MEMBERS  # exp_2's exp2_aug seeds, averaged in this order
PRIMARY = "exp7_robust_ens3"
FIT_CAP = 100
PLANNED_FITS = {"proxy": 84, "full_data": 3, "rebuild": 3}
WATCHED = ("exp_0", "exp_1", "exp_2", "exp_3", "exp_4", "exp_5", "exp_6")  # exp_7 never writes here

# exp_6 inputs read in place
RUN_LENGTHS = EXP6 / "cache" / "run_lengths.npy"
RECOVERED = EXP6 / "cache" / "recovered_ball.npz"
DETECTOR_QUALITY = EXP6 / "results" / "detector_quality.json"
EXP6_DECISION = EXP6 / "results" / "decision.json"
EXP6_ROWS = EXP6 / "results" / "sim_rows_03.csv"
EXP6_CODE = ("exp6_common.py", "exp6_sim.py", "exp6_target.py", "exp6_decision.py", "exp6_recover.py",
             "exp6_data.py", "exp6_detector.py", "exp6_linker.py")


# ---------------------------------------------------------------- reused files (read in place)
def exp3_result(label_set: str) -> Path:
    return C3.RESULTS / f"variant_{REFERENCE}__target_like__{label_set}.json"


def measured_inputs() -> dict:
    """exp_6's measured (c, sigma, f), exactly as its notebook 03 built them."""
    quality = C1.read_json(DETECTOR_QUALITY)
    return {"coverage": quality["coverage"], "sigma_m": quality["test"]["sigma_m"],
            "fp_rate": quality["test"]["fp_rate"]}


def reused_files() -> dict[str, Path]:
    """Every file exp_7 reads from exp_1 / exp_2 / exp_3 / exp_6 (SHA-256 frozen at freezing time)."""
    import exp1_protocol as P1
    import exp2_protocol as P2

    files: dict[str, Path] = {}
    for name in MATCH_IDS:
        files[f"exp_1/tokens/{name}_r0"] = P1.token_path(name, 0)
        files[f"exp_2/tokens/{name}_target_like_r0"] = P2.token_path(name, "target_like", 0)
        for arm in REF_MEMBERS:
            files[f"exp_2/{arm}/outer_{name}.pt"] = C6.exp2_outer_weight(arm, name)
    for arm in REF_MEMBERS:
        for key, *_ in P1.job_keys():
            if key.startswith("pair_"):
                files[f"exp_2/{arm}/{key}.pt"] = P2.prediction_folder(arm) / f"{key}.pt"
    for label_set in LABEL_SETS:
        files[f"exp_3/{REFERENCE}__target_like__{label_set}"] = exp3_result(label_set)
    for name in ("target_candidate.json", "target_hedge_provider.json"):
        files[f"exp_3/{name}"] = C3.RESULTS / name
    for path in (RUN_LENGTHS, RECOVERED, DETECTOR_QUALITY, EXP6_DECISION, EXP6_ROWS):
        files[f"exp_6/{path.relative_to(EXP6)}"] = path
    for name in EXP6_CODE:
        files[f"exp_6/{name}"] = EXP6 / name
    return files


def reused_sha256() -> dict[str, str]:
    return {key: C1.sha256(path) for key, path in reused_files().items()}


def token_cache_sha256() -> dict[str, str]:
    """exp_7's own token caches (built before freezing, never rebuilt afterwards)."""
    return {p.name: C1.sha256(p) for p in sorted(TOKEN_DIR.glob("*.npz"))} if TOKEN_DIR.exists() else {}


# ---------------------------------------------------------------- pre-registration
def frozen_constants() -> dict:
    """Every number PREREG.md fixes, read from the code that implements it."""
    import exp1_decode as D
    import exp1_net as N
    import exp7_aug as A
    import exp7_decision as DE
    import exp7_eval as E
    import exp7_fit as F

    constants = {
        "train_config": N.TrainConfig().as_dict(),
        "arms": {arm: F.ARMS[arm].as_dict() for arm in F.ARMS},
        "fingerprints": {arm: F.fingerprint(arm) for arm in F.ARMS},
        "fit_code": sorted(F.FIT_CODE),
        "recipe": A.RECIPE, "train_label": TRAIN_LABEL, "flag": A.FLAG_NAME,
        "train_ranges": A.TRAIN_RANGES, "train_seed_offset": A.TRAIN_SEED_OFFSET,
        "train_seeds": {f"{n}_r{r}": A.train_seed(n, r) for n in MATCH_IDS for r in A.SIM_ROUNDS},
        "eval_seeds": {n: S6.sim_seed(n) for n in MATCH_IDS},
        "eval_settings": {k: v.as_dict() for k, v in A.eval_settings().items()},
        "ar1_rho": A.AR1_RHO, "views": list(A.EVAL_VIEWS),
        "selection_view": E.SELECTION_VIEW, "evaluated": {k: list(v) for k, v in E.SELECTIONS.items()},
        "ensemble": list(F.ARMS), "reference_members": list(REF_MEMBERS),
        "bd_lambdas": D.LAMBDAS, "bd_gammas": D.GAMMAS,
        "h1": DE.H1_RULE, "plausibility": DE.PLAUSIBILITY,
        "bootstrap": {"draws": C1.BOOTSTRAP_DRAWS, "seed": BASE_SEED, "interval": 0.95},
        "fit_cap": FIT_CAP, "planned_fits": PLANNED_FITS,
        "reused_sha256": reused_sha256(),
        "token_cache_sha256": token_cache_sha256(),
    }
    return json.loads(json.dumps(constants, default=C1._json_default))


def code_hashes() -> dict:
    """exp_7 code at freezing time (informational; fit code is also in the fingerprints)."""
    return {p.name: C1.sha256(p) for p in sorted(ROOT.glob("exp7_*.py"))}


def fits_exist() -> bool:
    return any(PRED_DIR.glob("*/*.pt")) if PRED_DIR.exists() else False


def freeze_prereg() -> dict:
    """Record PREREG.md's SHA-256 (+ the constants it fixes) once, before any exp_7 fit or scoring."""
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; use verify_prereg()")
    if fits_exist() or any(RESULTS.glob("variant_*.json")):
        raise RuntimeError("exp_7 fits or scores exist; the pre-registration must precede them")
    if not token_cache_sha256():
        raise RuntimeError("Token caches missing: build them before freezing")
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
    """Allowed only while no exp_7 fit exists; the old protocol is kept as protocol_superseded_N.json."""
    if fits_exist():
        raise RuntimeError("exp_7 fits exist: re-freezing is no longer allowed")
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
        raise RuntimeError("Pre-registered constants, token caches or reused files differ; diagnose first")
    return protocol


def public_check_spent() -> bool:
    """exp_7 never opens the public ground truth; exp_1 spent the single check."""
    return C3.public_check_spent()


def finite_or_none(value: object) -> object:
    """JSON-safe copy: NaN / inf floats become None (``C1.write_json`` refuses NaN)."""
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float):  # numpy float64 is a float subclass
        return value if math.isfinite(value) else None
    return value


def write_results(path: Path, value: object) -> None:
    C1.write_json(path, finite_or_none(value))


# ---------------------------------------------------------------- write guard
def tree_digest(folder: Path) -> dict:
    """File count + digest of (path, size, mtime) under a folder; reads no file content."""
    digest, count = hashlib.sha256(), 0
    if folder.exists():
        for path in sorted(p for p in folder.rglob("*") if p.is_file() or p.is_symlink()):
            stat = path.lstat()
            digest.update(f"{path.relative_to(folder)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
            count += 1
    return {"files": count, "digest": digest.hexdigest()}


def watched_folders() -> dict[str, Path]:
    """exp_0-exp_6 and the competition data folder(s) (exp_7 must write none of them)."""
    folders = {name: EXPERIMENTS / name for name in WATCHED}
    folders["data"] = C1.TARGET_REPO / "data"
    if C1.METRIC_REPO != C1.TARGET_REPO:
        folders["data_metric_repo"] = C1.METRIC_REPO / "data"
    return folders


def write_manifest(folders: dict[str, Path] | None = None) -> dict:
    return {name: tree_digest(path) for name, path in (folders or watched_folders()).items()}


def guard_record(path: Path) -> dict:
    """Write the manifest once (freezing time) and return it."""
    if not path.exists():
        C1.write_json(path, {"taken_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "folders": write_manifest()})
    return C1.read_json(path)


def guard_check(before_path: Path, out_path: Path) -> dict:
    before = C1.read_json(before_path)
    now = write_manifest()
    folders = {name: {"unchanged": now.get(name) == state, "files_before": state["files"],
                      "files_now": now.get(name, {}).get("files")}
               for name, state in before["folders"].items()}
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "since": before["taken_at_utc"], "all_unchanged": all(f["unchanged"] for f in folders.values()),
              "folders": folders}
    C1.write_json(out_path, result)
    return result
