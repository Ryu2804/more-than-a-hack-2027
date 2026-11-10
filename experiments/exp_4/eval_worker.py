"""Optional evaluation helper (IDSSE only, never fits): runs notebook 02's own
``exp4_protocol.evaluate_and_save`` tasks as soon as every member arm of a task has all 28 fits,
so evaluation overlaps the remaining fits. Notebook 02 then finds the result files ("cached").
Usage: python eval_worker.py [workers]
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp4_common as C  # noqa: E402
import exp4_protocol as P  # noqa: E402

POLL_S = 120


def tasks() -> list[tuple[str, str, str]]:
    out = [(v, view, ls) for v in P.ENSEMBLE_VARIANTS for ls in C.S1_LABELS for view in C.VIEWS]
    return out + [(arm, view, ls) for arm in P.ARMS for ls in C.DECISION_LABELS for view in C.VIEWS]


def complete(arm: str) -> bool:
    folder = P.prediction_folder(arm)
    return all((folder / f"{key}.npz").exists() for key, *_ in P.P1.job_keys())


def ready(task: tuple[str, str, str]) -> bool:
    (source,) = P.VARIANTS[task[0]]
    return all(complete(arm) for arm in P.ENSEMBLES.get(source, (source,)))


def main(workers: int) -> int:
    C.verify_prereg()
    todo = [t for t in tasks() if not P.result_path(*t).exists()]
    context = multiprocessing.get_context("fork")
    while todo:
        batch = [t for t in todo if ready(t)]
        if batch:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                for variant, view, label_set, seconds, tag in pool.map(P.evaluate_and_save, batch):
                    print(f"EVAL {label_set} {view} {variant} {tag} {seconds:.0f}s", flush=True)
        todo = [t for t in todo if not P.result_path(*t).exists()]
        if todo and not batch:
            time.sleep(POLL_S)
    print("EVAL WORKER DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
