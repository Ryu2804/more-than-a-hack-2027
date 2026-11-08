#!/usr/bin/env python3
import argparse
import json
import os
import resource
import sys
import time
import traceback

try:
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3, 8 * 1024 ** 3))
except Exception:
    pass

LOOP_DIR = os.path.dirname(os.path.abspath(__file__))
SPELL_DIR = os.path.dirname(LOOP_DIR)
ROOT = os.path.dirname(os.path.dirname(SPELL_DIR))
sys.path.insert(0, LOOP_DIR)
sys.path.insert(0, SPELL_DIR)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "games"))

import tasks  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--loop-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--error-path", required=True)
    args = parser.parse_args()

    ctx = {
        "loop_dir": args.loop_dir,
        "run_dir": args.run_dir,
        "root": ROOT,
        "spell_dir": SPELL_DIR,
        "data_dir": os.path.join(ROOT, "..", "data"),
        "dev": tasks.DEV,
        "confirm": tasks.CONFIRM,
    }
    t0 = time.time()
    try:
        result = tasks.run_task(args.task, ctx)
        result["runtime_s"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        with open(args.result_path, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print("OK")
        return 0
    except Exception:
        os.makedirs(os.path.dirname(args.error_path), exist_ok=True)
        with open(args.error_path, "w") as handle:
            handle.write(traceback.format_exc())
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
