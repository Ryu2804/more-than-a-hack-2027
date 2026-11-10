"""Optional evaluation helper (IDSSE only, never fits): runs notebook 02's own
``exp5_eval.evaluate_and_save`` tasks as soon as every exp_5 member arm of a task has all 28 fits,
so evaluation overlaps the remaining fits. Notebook 02 then finds the result files ("cached").

Memory guard (exp_4 hit an OOM kill): a batch starts only while MemAvailable >= 4 GB.
Usage: python eval_worker.py [workers<=3]
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp5_common as C  # noqa: E402,I001
import exp5_eval as E  # noqa: E402

POLL_S = 120
MIN_AVAILABLE_MB = 4096
MAX_WORKERS = 3


def available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not found")


def main(workers: int) -> int:
    workers = min(workers, MAX_WORKERS)
    C.verify_prereg()
    todo = [t for t in E.tasks() if not E.result_path(*t).exists()]
    context = multiprocessing.get_context("fork")
    while todo:
        batch = [t for t in todo if E.ready(t[0])]
        if batch and available_mb() < MIN_AVAILABLE_MB:
            print(f"EVAL WAIT memory {available_mb()} MB < {MIN_AVAILABLE_MB} MB", flush=True)
            batch = []
        if batch:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                for variant, view, label_set, seconds, tag in pool.map(E.evaluate_and_save, batch):
                    print(f"EVAL {label_set} {view} {variant} {tag} {seconds:.0f}s", flush=True)
        todo = [t for t in todo if not E.result_path(*t).exists()]
        if todo and not batch:
            time.sleep(POLL_S)
    print("EVAL WORKER DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else MAX_WORKERS))
