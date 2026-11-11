"""exp_6 shared paths, reused-file registry, pre-registration checks and write guard.

exp_1-exp_3 are reused by import (``../exp_3`` on ``sys.path``; ``exp3_common`` adds ``../exp_2`` and
``../exp_1``), never copied or edited. exp_2's saved weights, exp_3's stored results and exp_5's saved
touch weights are read in place and SHA-256-checked against ``results/protocol.json``. exp_4 / exp_5
modules are never imported (their ``*_common`` modules redirect cache directories). exp_6 writes only
inside its own folder.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
# exp1_common defaults NUMBA_CACHE_DIR to exp_1/cache/numba: force every cache write into exp_6 first.
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "cache" / "numba")
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # before CUDA initialises
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

EXP3 = ROOT.parent / "exp_3"
if str(EXP3) not in sys.path:
    sys.path.insert(0, str(EXP3))

from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import exp3_common as C3  # noqa: E402  (puts ../exp_2 and ../exp_1 on sys.path; verifies the judge)
import exp1_common as C1  # noqa: E402

EXP1, EXP2 = C3.EXP1, C3.EXP2
EXPERIMENTS = ROOT.parent
EXP5 = EXPERIMENTS / "exp_5"
CACHE = ROOT / "cache"
LOG_DIR = CACHE / "logs"
DETECTOR_DIR = CACHE / "detector"
INFER_DIR = CACHE / "infer"
RECOVERED = CACHE / "recovered_ball.npz"
REBUILD_DIR = CACHE / "rebuild"
RESULTS = ROOT / "results"
PREDICTIONS_OUT = ROOT / "predictions"
PREREG = ROOT / "PREREG.md"
PROTOCOL = RESULTS / "protocol.json"
VIDEO = C1.TARGET_REPO / "data" / "tactical" / "BAR-ATH_tactical.mp4"

MATCH_IDS = C1.MATCH_IDS
BASE_SEED = C1.BASE_SEED  # 20261108: detector seed, bootstrap seed, simulator base seed
LABEL_SETS = ("provider", "ctrl")  # H1 needs both (intersection-union)
REFERENCE = "exp3_ref_ens3"
MEMBERS = ("exp2_aug", "exp2_aug_s1", "exp2_aug_s2")  # exp3_ref_ens3, averaged in this order
TOUCH_CONTEXT = "exp5_touch_ens3"
TOUCH_MEMBERS = ("exp5_touch", "exp5_touch_s1", "exp5_touch_s2")
FOLDER_PATTERN = re.compile(r"^(?P<arm>.+)_(?P<fp>[0-9a-f]{12})$")
WATCHED = ("exp_0", "exp_1", "exp_2", "exp_3", "exp_4", "exp_5")  # exp_6 never writes here


# ---------------------------------------------------------------- reused files (read in place)
def exp2_outer_weight(arm: str, held: str) -> Path:
    import exp2_protocol as P2

    return P2.prediction_folder(arm) / f"outer_{held}.pt"


def exp2_full_weight(arm: str) -> Path:
    import exp2_protocol as P2

    return EXP2 / "cache" / "target_models" / f"{arm}_{P2.fingerprint(arm)}.pt"


def exp5_touch_folder(arm: str) -> Path:
    """exp_5's prediction folder of one touch arm (exactly one ``<arm>_<12 hex>`` folder)."""
    if arm not in TOUCH_MEMBERS:
        raise ValueError(f"{arm} is not an exp_5 touch arm")
    hits = [p for p in sorted((EXP5 / "cache" / "predictions").iterdir())
            if (m := FOLDER_PATTERN.match(p.name)) and m["arm"] == arm and p.is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one exp_5 cache folder for {arm}, found {len(hits)}")
    return hits[0]


def exp3_result(variant: str, label_set: str) -> Path:
    return C3.RESULTS / f"variant_{variant}__target_like__{label_set}.json"


def exp5_result(label_set: str) -> Path:
    return EXP5 / "results" / f"variant_{TOUCH_CONTEXT}__target_like__{label_set}.json"


def reused_files() -> dict[str, Path]:
    """Every file exp_6 reads from exp_2 / exp_3 / exp_5 (SHA-256 frozen at freezing time)."""
    import exp2_protocol as P2

    files: dict[str, Path] = {}
    for arm in MEMBERS:
        for held in MATCH_IDS:
            files[f"exp_2/{arm}/outer_{held}.pt"] = exp2_outer_weight(arm, held)
        files[f"exp_2/target_models/{arm}"] = exp2_full_weight(arm)
    for held in MATCH_IDS:
        files[f"exp_2/tokens/{held}_target_like_r0"] = P2.token_path(held, "target_like", 0)
    for label_set in LABEL_SETS:
        files[f"exp_3/{REFERENCE}__target_like__{label_set}"] = exp3_result(REFERENCE, label_set)
        files[f"exp_5/{TOUCH_CONTEXT}__target_like__{label_set}"] = exp5_result(label_set)
    for name in ("coverage_rows.csv", "target_candidate.json", "target_hedge_provider.json"):
        files[f"exp_3/{name}"] = C3.RESULTS / name
    for arm in TOUCH_MEMBERS:
        folder = exp5_touch_folder(arm)
        for held in MATCH_IDS:
            files[f"exp_5/{folder.name}/outer_{held}.pt"] = folder / f"outer_{held}.pt"
    return files


def reused_sha256() -> dict[str, str]:
    return {key: C1.sha256(path) for key, path in reused_files().items()}


# ---------------------------------------------------------------- pre-registration
def frozen_constants() -> dict:
    """Every number PREREG.md fixes, read from the code that implements it."""
    import exp2_protocol as P2
    import exp6_data as DA
    import exp6_decision as DE
    import exp6_detector as DT
    import exp6_linker as LK
    import exp6_sim as S

    constants = {
        "splits": {"mod": DA.SPLIT_MOD, "train": DA.TRAIN_RESIDUES, "val": DA.VAL_RESIDUE,
                   "test": DA.TEST_RESIDUE, "label_margin_px": DA.LABEL_MARGIN_PX,
                   "pitch_m": DA.PITCH, "image": [DA.IMAGE_W, DA.IMAGE_H], "frames": DA.N_FRAMES},
        "detector": DT.DetectorConfig().as_dict(),
        "linker": LK.LinkerConfig().as_dict(),
        "tau": {"grid": LK.TAU_GRID, "min_precision": LK.TAU_MIN_PRECISION, "fallback": LK.TAU_FALLBACK,
                "inlier_m": LK.INLIER_M},
        "simulator": {"seed_offset": S.SEED_OFFSET, "fp_noise_m": S.FP_NOISE_M,
                      "stress": S.STRESS_RULE, "voi_coverage": S.VOI_COVERAGE},
        "gates": DE.GATES, "h1": DE.H1_RULE,
        "members": MEMBERS, "touch_members": TOUCH_MEMBERS, "label_sets": LABEL_SETS,
        "exp2_fingerprints": {arm: P2.fingerprint(arm) for arm in MEMBERS},
        "bootstrap": {"draws": C1.BOOTSTRAP_DRAWS, "seed": BASE_SEED, "interval": 0.95},
        "reused_sha256": reused_sha256(),
    }
    return json.loads(json.dumps(constants, default=C1._json_default))


def code_hashes() -> dict:
    """exp_6 code at freezing time (informational record)."""
    return {p.name: C1.sha256(p) for p in sorted(ROOT.glob("exp6_*.py"))}


def started() -> bool:
    """True once a detector checkpoint, an inference chunk or a simulated-view result exists."""
    used = [d for d in (DETECTOR_DIR, INFER_DIR, REBUILD_DIR) if d.exists() and any(d.iterdir())]
    return bool(used) or any(RESULTS.glob("sim_*.json"))


def freeze_prereg() -> dict:
    """Record PREREG.md's SHA-256 (+ the constants it fixes) once, before any training or scoring."""
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; use verify_prereg()")
    if started():
        raise RuntimeError("exp_6 training or simulated scoring exists; the pre-registration must precede it")
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
    """Allowed only while no detector was trained and no simulated view was scored."""
    if started():
        raise RuntimeError("exp_6 training or simulated scoring exists: re-freezing is no longer allowed")
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
    """exp_6 never opens the public ground truth; exp_1 spent the single check."""
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
    """exp_0-exp_5 and the competition data folder(s) (exp_6 must write none of them)."""
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
