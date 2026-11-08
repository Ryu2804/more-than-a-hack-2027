"""Shared paths, frozen judge snapshot and scoring helpers for exp_1.

exp_1 never imports live exp_0 code: exp_0 is still being edited, so the degrader and
feature code it depends on are frozen as verbatim copies in ``judge/`` and checked by
SHA-256 at import. Only external IDSSE data is ever printed; BAR-ATH stays local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
METRIC_MARKER = Path("evaluation") / "possession_metrics.py"
DATA_MARKER = Path("data") / "tactical" / "BAR-ATH_tactical_atd.txt"


def _find_repo(marker: Path) -> Path | None:
    """Competition repo holding ``marker``: $MTAH_REPO first, then each ancestor (or its
    ``more-than-a-hack-2027`` child). Works both standalone and inside repo/experiments/."""
    candidates = [Path(os.environ["MTAH_REPO"])] if os.environ.get("MTAH_REPO") else []
    for ancestor in ROOT.parents:
        candidates += [ancestor, ancestor / "more-than-a-hack-2027"]
    return next((c for c in candidates if (c / marker).is_file()), None)


METRIC_REPO = _find_repo(METRIC_MARKER)
if METRIC_REPO is None:
    raise RuntimeError("Competition repository not found; set MTAH_REPO")
# BAR-ATH data may live in a different clone than the one this folder sits in.
TARGET_REPO = _find_repo(DATA_MARKER) or METRIC_REPO
JUDGE_DIR = ROOT / "judge"
PROXY_DIR = ROOT / "cache" / "proxy"
TOKEN_DIR = ROOT / "cache" / "tokens"
PRED_DIR = ROOT / "cache" / "predictions"
RESULTS = ROOT / "results"

os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "cache" / "numba"))
sys.dont_write_bytecode = True

FROZEN_HASHES = {
    "judge/experiment_data.py": "e2b4034baa721ec7f0faf820626e6bb164325b55bf44595607612ab4c2cff7f1",
    "judge/experiment_model.py": "243dfe93ba4aa051d271bd59f21091c972abdf02baae58ac37423139b5333fbd",
    "official_metric": "341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d",
}
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
BASE_SEED = 20261108
AUG_ROUNDS = (1, 2)  # extra corruption seeds for training matches only
BOOTSTRAP_DRAWS = 1000
QUALITY_GATES = {
    "spell_gain": 0.05,
    "ci_low": 0.0,
    "macro_regression": -0.01,
    "max_abs_count_error": 0.30,
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_judge() -> dict:
    """Fail loudly if the frozen judge snapshot or official metric changed."""
    observed = {
        "judge/experiment_data.py": sha256(JUDGE_DIR / "experiment_data.py"),
        "judge/experiment_model.py": sha256(JUDGE_DIR / "experiment_model.py"),
        "official_metric": sha256(METRIC_REPO / METRIC_MARKER),
    }
    if observed != FROZEN_HASHES:
        raise RuntimeError("Frozen judge snapshot changed; diagnose before rerunning")
    return observed


def _import_judge():
    verify_judge()
    if str(JUDGE_DIR) not in sys.path:
        sys.path.insert(0, str(JUDGE_DIR))
    import experiment_data as data_module  # noqa: E402  (frozen snapshot)

    # experiment_model loads the official metric from experiment_data.REPO at import time,
    # and the snapshot computes REPO relative to judge/, so point it at the resolved repo first.
    data_module.REPO = METRIC_REPO
    import experiment_model as model_module  # noqa: E402

    data_module.REPO = TARGET_REPO  # target loader reads BAR-ATH from the shared repo
    return data_module, model_module


JD, JM = _import_judge()
PM = JM.PM  # official possession_metrics module, imported unchanged
Match = JD.Match
FPS = JD.FPS


def match_seed(name: str, round_index: int = 0) -> int:
    """Round 0 is the frozen evaluation corruption used by exp_0; others are augmentation."""
    return BASE_SEED + 1000 * round_index + MATCH_IDS.index(name)


def load_clean(name: str) -> "Match":
    return JD.read_match(PROXY_DIR / f"{name}_clean.npz")


def load_degraded(name: str, round_index: int = 0) -> "Match":
    """Round 0 comes from the frozen copy; augmentation rounds are regenerated."""
    if round_index == 0:
        return JD.read_match(PROXY_DIR / f"{name}_degraded.npz")
    return JD.degrade(load_clean(name), match_seed(name, round_index))


def verify_frozen_proxy() -> list[dict]:
    """The copied degraded matches must equal the frozen degrader applied to clean data."""
    rows = []
    for name in MATCH_IDS:
        clean, frozen = load_clean(name), load_degraded(name)
        regenerated = JD.degrade(clean, match_seed(name))
        same = bool(
            np.array_equal(frozen.ball, regenerated.ball, equal_nan=True)
            and np.array_equal(frozen.players, regenerated.players, equal_nan=True)
            and np.array_equal(frozen.labels, clean.labels)
        )
        if not same:
            raise RuntimeError(f"Frozen proxy copy for {name} does not match degrader")
        rows.append({"match": name, "frozen_copy_matches_degrader": same})
    return rows


def score(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return JM.score(truth, prediction)


def paired_bootstrap(values: list[float], draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    return JM.paired_bootstrap(list(values), draws)


def sample_indices(half: np.ndarray) -> np.ndarray:
    """5 fps grid inside every graded period (same grid as exp_0)."""
    periods = [p for p in np.unique(half) if p > 0]
    return np.concatenate([np.flatnonzero(half == p)[:: FPS // 5] for p in periods])


def expand(prediction: np.ndarray, indices: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Forward-fill 5 fps codes to 25 fps; ungraded frames become X (3)."""
    position = np.searchsorted(indices, np.arange(len(half)), side="right") - 1
    output = prediction[np.clip(position, 0, len(indices) - 1)]
    return np.where(half > 0, output, 3).astype("int8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
