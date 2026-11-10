"""Optional second fit process (IDSSE only): runs ``exp4_protocol.ensure_predictions`` for the
given arms with the same code, fingerprints and per-job locks as notebook 01, so no job is ever
fitted twice. Notebook 01 then finds these jobs cached. Usage: python fit_worker.py ARM [ARM ...]
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp4_common as C  # noqa: E402
import exp4_protocol as P  # noqa: E402


def main(arms: list[str]) -> int:
    C.verify_prereg()
    unknown = [a for a in arms if a not in P.ARMS]
    if unknown:
        raise ValueError(f"Unknown arms {unknown}")
    log_path = C.RESULTS / "fit_log_worker.txt"

    def log(message: str, **_: object) -> None:
        print(message, flush=True)
        with log_path.open("a") as handle:
            handle.write(message + "\n")

    for arm in arms:
        if P.fit_count() + 28 > C.FIT_CAP:
            log(f"STOP budget: {P.fit_count()} fits cached, cap {C.FIT_CAP}")
            return 1
        start = time.time()
        P.ensure_predictions(arm, log=log)
        log(f"WORKER ARM DONE {arm} {time.time() - start:.0f}s, fits cached {P.fit_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
