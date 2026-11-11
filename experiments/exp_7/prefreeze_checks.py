"""Pre-freeze work allowed by PROMPT / PREREG (IDSSE only): token caches, import checks, a tiny synthetic fit.

No exp_7 fit on IDSSE and no scoring of any view. Writes ``results/prefreeze_checks.json`` and
``results/token_caches.json`` (aggregates). Run once before notebook 00 freezes the pre-registration.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp7_common as C  # noqa: E402,I001
import numpy as np  # noqa: E402

import exp1_net as N  # noqa: E402
import exp7_aug as A  # noqa: E402
import exp7_fit as F  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tiny_fit_seconds() -> float:
    """Synthetic random tokens, exp_7's flag width; times one tiny SetSeq fit."""
    rng = np.random.default_rng(0)
    steps, width = 3000, len(N.GLOBAL_NAMES) + 1
    item = {"tokens": rng.normal(size=(steps, 28, 10)).astype("float16"), "mask": rng.random((steps, 28)) < 0.8,
            "glob": rng.normal(size=(steps, width)).astype("float16"), "half": np.ones(steps, "int8"),
            "state": rng.integers(0, 3, steps).astype("int8"), "boundary": np.zeros(steps, "int8")}
    start = time.time()
    with A.flag_globals():
        N.train([item], replace(N.TrainConfig(), epochs=1))
    return round(time.time() - start, 2)


def main() -> int:
    if C.PROTOCOL.exists():
        raise RuntimeError("Already frozen: pre-freeze checks must precede the freeze")
    if C.fits_exist():
        raise RuntimeError("exp_7 fits exist")
    before = C.write_manifest()
    start = time.time()
    rows = A.build_token_caches()
    C.write_results(C.RESULTS / "token_caches.json", {"built_at_utc": now(), "rows": rows})
    unsimulated = {f"{n}_r{r}": A.unsimulated_equals_exp2(n, r) for n in C.MATCH_IDS for r in A.SIM_ROUNDS}
    checks = {
        "checked_at_utc": now(), "token_cache_seconds": round(time.time() - start, 1),
        "token_cache_files": len(C.token_cache_sha256()),
        "token_cache_mb": round(sum(p.stat().st_size for p in C.TOKEN_DIR.glob("*.npz")) / 2**20, 1),
        "unsimulated_target_aug_equals_exp2": unsimulated, "all_unsimulated_equal": all(unsimulated.values()),
        "fingerprints": {arm: F.fingerprint(arm) for arm in F.ARMS},
        "tiny_synthetic_fit_seconds": tiny_fit_seconds(),
        "write_guard_unchanged": C.write_manifest() == before,
    }
    C.write_results(C.RESULTS / "prefreeze_checks.json", checks)
    print({k: v for k, v in checks.items() if k != "unsimulated_target_aug_equals_exp2"}, flush=True)
    return 0 if checks["all_unsimulated_equal"] and checks["write_guard_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
