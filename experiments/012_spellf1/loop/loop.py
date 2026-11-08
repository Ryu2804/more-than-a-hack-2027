#!/usr/bin/env python3
"""Deterministic long-run loop for the possession project.

Usage:
    python loop.py --start 8
    python loop.py --status
    python loop.py --stop
    python loop.py --report
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

LOOP_DIR = os.path.dirname(os.path.abspath(__file__))
SPELL_DIR = os.path.dirname(LOOP_DIR)
ROOT = os.path.dirname(os.path.dirname(SPELL_DIR))
PY = os.path.join(ROOT, ".venv", "bin", "python")
RUN_ONE = os.path.join(LOOP_DIR, "run_one.py")

sys.path.insert(0, LOOP_DIR)
import tasks  # noqa: E402

DEV = tasks.DEV
CONFIRM = tasks.CONFIRM
STATE_PATH = os.path.join(LOOP_DIR, "STATE.json")
INDEX_PATH = os.path.join(LOOP_DIR, "INDEX.tsv")
SUMMARY_PATH = os.path.join(LOOP_DIR, "SUMMARY.md")
HEARTBEAT_PATH = os.path.join(LOOP_DIR, "HEARTBEAT")
STOP_PATH = os.path.join(LOOP_DIR, "STOP")
LOCK_PATH = os.path.join(LOOP_DIR, "loop.lock")
LOG_PATH = os.path.join(LOOP_DIR, "loop.log")
MORNING_REPORT_PATH = os.path.join(LOOP_DIR, "MORNING_REPORT.md")
RUNS_DIR = os.path.join(LOOP_DIR, "runs")
CACHE_DIR = os.path.join(LOOP_DIR, "cache")
SOTA_PATH = os.path.join(ROOT, "..", "..", "research_loop", "riset_methodology",
                        "experiments", "long_run_2026-11-07", "SOTA.md")

TASK_ORDER = tasks.TASK_ORDER

# Tasks that measure or consolidate, but cannot raise SOTA. They must not consume the
# anti-churn budget, otherwise a long run stops early after a series of confirmations.
INVENTORY_TASKS = {"confirm_sota", "confirm_ceiling", "barath_dead_grid",
                   "metrica_check", "noise_price", "consolidate", "barath_transfer_final",
                   "confirm_challenger", "barath_medp_dead", "noise_price_ball"}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(line):
    with open(LOG_PATH, "a") as handle:
        handle.write("[{}] {}\n".format(now(), line))


def load_state():
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH) as handle:
            state = json.load(handle)
        if not isinstance(state.get("failed"), dict):
            state["failed"] = {}
        if not isinstance(state.get("failed_tasks"), list):
            state["failed_tasks"] = []
        if not isinstance(state.get("done"), list):
            state["done"] = []
        if not isinstance(state.get("incomplete"), list):
            state["incomplete"] = []
        if not isinstance(state.get("attempts"), dict):
            state["attempts"] = {}
        if not isinstance(state.get("runs"), list):
            state["runs"] = []
        return state
    return {
        "started": now(),
        "started_ts": time.time(),
        "hours": 8,
        "done": [],
        "failed": {},
        "failed_tasks": [],
        "incomplete": [],
        "attempts": {},
        "consecutive_failures": 0,
        "cycles_without_sota": 0,
        "sota_deployable": 0.5263,
        "sota_oracle": 0.6150,
        "sota_updates": [],
        "runs": [],
    }


def save_state(state):
    with open(STATE_PATH, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def disk_free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def acquire_lock():
    handle = open(LOCK_PATH, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("loop already running (lock held)")
        sys.exit(1)
    return handle


def write_heartbeat(task, status):
    with open(HEARTBEAT_PATH, "w") as handle:
        handle.write("{} task={} status={}\n".format(now(), task, status))


def ensure_index_header():
    if not os.path.isfile(INDEX_PATH):
        with open(INDEX_PATH, "w") as handle:
            handle.write("time\ttask\tstatus\tmetric\tdelta\tpath\n")


def append_index(time_str, task, status, metric, delta, path):
    ensure_index_header()
    with open(INDEX_PATH, "a") as handle:
        handle.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
            time_str, task, status, metric, delta, path))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_deps(run_dir, task_name):
    code_files = []
    for d in (LOOP_DIR, SPELL_DIR):
        for name in os.listdir(d):
            if name.endswith(".py"):
                code_files.append(os.path.join(d, name))
    # evaluation + phase2 modules
    for extra in (os.path.join(ROOT, "evaluation", "possession_metrics.py"),
                  os.path.join(ROOT, "experiments", "011_ball_value", "phase2.py"),
                  os.path.join(ROOT, "experiments", "011_ball_value", "phase2b.py")):
        if os.path.isfile(extra):
            code_files.append(extra)
    code_hashes = {}
    for path in sorted(set(code_files)):
        try:
            code_hashes[os.path.relpath(path, ROOT)] = sha256_file(path)
        except Exception:
            pass

    data_files = []
    parsed = os.path.join(ROOT, "..", "data", "external", "idsee", "parsed")
    if os.path.isdir(parsed):
        for name in sorted(os.listdir(parsed)):
            p = os.path.join(parsed, name)
            if os.path.isfile(p):
                st = os.stat(p)
                data_files.append({"path": os.path.relpath(p, ROOT),
                                  "bytes": st.st_size, "mtime": int(st.st_mtime)})
    metrica = os.path.join(ROOT, "..", "data", "external", "metrica")
    if os.path.isdir(metrica):
        for root, _dirs, files in os.walk(metrica):
            for name in files:
                p = os.path.join(root, name)
                st = os.stat(p)
                data_files.append({"path": os.path.relpath(p, ROOT),
                                  "bytes": st.st_size, "mtime": int(st.st_mtime)})

    try:
        pip_out = subprocess.check_output([os.path.join(ROOT, ".venv", "bin", "pip"), "freeze"],
                                         text=True, stderr=subprocess.DEVNULL)
    except Exception:
        pip_out = ""
    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_rev = "unknown"

    deps = {
        "code_hashes": code_hashes,
        "data_files": data_files,
        "python": sys.version.split()[0],
        "pip_freeze": pip_out.splitlines(),
        "git_rev": git_rev,
        "seed": 20261108,
        "task": task_name,
    }
    path = os.path.join(run_dir, "deps.json")
    with open(path, "w") as handle:
        json.dump(deps, handle, indent=2, sort_keys=True)
    return path


def make_decision_json(run_dir, task_name):
    spec = tasks.TASKS[task_name]
    decision = dict(spec["decision"])
    path = os.path.join(run_dir, "decision.json")
    with open(path, "w") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
    return path, decision


def run_gate_pre(run_dir):
    decision_path = os.path.join(run_dir, "decision.json")
    target = os.path.join(run_dir, "gate_pre.txt")
    gate = os.path.abspath(os.path.join(ROOT, "..", "research_loop", "riset_methodology",
                                       "experiments", "long_run_2026-11-07", "gate.py"))
    result = subprocess.run([PY, gate, "--pre", decision_path],
                            capture_output=True, text=True)
    lines = []
    if os.path.isfile(target):
        with open(target) as handle:
            lines = handle.read().splitlines()
    if not lines:
        lines = ["PASS" if result.returncode == 0 else "BLOCK"]
    status = lines[0]
    return status, "\n".join(lines)


def run_gate_post(run_dir):
    decision_path = os.path.join(run_dir, "decision.json")
    results_path = os.path.join(run_dir, "results.json")
    target = os.path.join(run_dir, "gate_post.txt")
    gate = os.path.abspath(os.path.join(ROOT, "..", "research_loop", "riset_methodology",
                                       "experiments", "long_run_2026-11-07", "gate.py"))
    result = subprocess.run([PY, gate, "--post", decision_path, results_path],
                            capture_output=True, text=True)
    with open(target) as handle:
        lines = handle.read().splitlines()
    status = lines[0] if lines else ("PASS" if result.returncode == 0 else "BLOCK")
    return status, "\n".join(lines)


def extract_primary_metric(result):
    # returns float candidate mean if possible
    if isinstance(result, dict):
        if "mean" in result and isinstance(result["mean"], (int, float)):
            return float(result["mean"])
        if "mean" in result and isinstance(result["mean"], dict):
            for key in ("longslow", "reset", "hybrid", "teamlearn", "touch_hold_deployable"):
                if key in result["mean"] and isinstance(result["mean"][key], (int, float)):
                    return float(result["mean"][key])
            # fallback: first numeric
            for v in result["mean"].values():
                if isinstance(v, (int, float)):
                    return float(v)
        if "primary_value" in result:
            return float(result["primary_value"])
        if "loso_mean" in result:
            return float(result["loso_mean"])
    return None


def extract_wins(result):
    if not isinstance(result, dict):
        return None
    if "wins" in result and isinstance(result["wins"], int):
        return result["wins"]
    if "mean" in result and isinstance(result["mean"], dict) and "wins" in result["mean"]:
        w = result["mean"]["wins"]
        if isinstance(w, int):
            return w
    return None


def extract_delta(result):
    if not isinstance(result, dict):
        return None, None
    d = result.get("delta")
    if isinstance(d, dict):
        mean = d.get("mean")
        ci = d.get("ci90")
        return mean, ci
    if "mean" in result and isinstance(result["mean"], dict):
        dd = result["mean"]
        mean = dd.get("delta")
        ci = dd.get("delta_ci90")
        return mean, ci
    return None, None


def generate_report(run_dir, task_name, decision, result, status, gate_pre, gate_post):
    lines = []
    lines.append("# Report: {}".format(task_name))
    lines.append("")
    lines.append("Question: {}".format(decision.get("question")))
    lines.append("")
    lines.append("Mechanism: {}".format(decision.get("mechanism")))
    lines.append("")
    lines.append("Arms: {}".format(", ".join(
        a.get("name") if isinstance(a, dict) else str(a) for a in decision.get("arms", []))))
    lines.append("")
    lines.append("Configuration: split={}, primary_metric={}".format(
        decision.get("split"), decision.get("primary_metric")))
    lines.append("")
    lines.append("## Per-match / results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result, indent=2, sort_keys=True)[:6000])
    lines.append("```")
    lines.append("")
    mean_val = extract_primary_metric(result)
    dmean, dci = extract_delta(result)
    lines.append("Primary mean: {}".format(mean_val))
    lines.append("Delta: {} CI90: {}".format(dmean, dci))
    lines.append("Wins: {}".format(extract_wins(result)))
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("Status: {}".format(status))
    lines.append("gate_pre: {}".format(gate_pre))
    lines.append("gate_post: {}".format(gate_post))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in ("decision.json", "gate_pre.txt", "deps.json", "results.json",
                 "gate_post.txt", "report.md", "runtime.txt"):
        lines.append("- {}".format(name))
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("See SUMMARY.md for next step.")
    path = os.path.join(run_dir, "report.md")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def maybe_update_sota(task_name, result, state, run_dir):
    eligible = task_name in ("dead_threshold_loso", "touch_reset_variant",
                             "teamlearn_v2", "hybrid_reassert")
    if not eligible:
        return False
    mean_val = extract_primary_metric(result)
    wins = extract_wins(result)
    dmean, dci = extract_delta(result)
    if mean_val is None or wins is None or dmean is None or dci is None:
        return False
    if mean_val < state["sota_deployable"] + 0.04:
        return False
    if wins < 4:
        return False
    if not (isinstance(dci, (list, tuple)) and len(dci) == 2 and dci[0] > 0):
        return False
    if dmean < 0.04:
        return False
    # update SOTA.md
    try:
        with open(SOTA_PATH, "a") as handle:
            handle.write("\n## Update from loop run {}\n".format(run_dir))
            handle.write("- task: {}\n".format(task_name))
            handle.write("- candidate spellF1: {:.4f}\n".format(mean_val))
            handle.write("- delta: {:.4f} CI90 [{} , {}]\n".format(dmean, dci[0], dci[1]))
            handle.write("- wins: {}/5\n".format(wins))
            handle.write("- evidence: {}\n".format(run_dir))
    except Exception:
        pass
    state["sota_deployable"] = mean_val
    state["sota_updates"].append({
        "task": task_name, "mean": mean_val, "delta": dmean, "ci90": dci,
        "wins": wins, "run_dir": run_dir, "time": now(),
    })
    return True


def stop_conditions(state, hours):
    if os.path.isfile(STOP_PATH):
        return True, "STOP file present"
    if time.time() - state["started_ts"] > hours * 3600:
        return True, "budget hours reached"
    if state.get("consecutive_failures", 0) >= 3:
        return True, "3 consecutive task failures"
    if disk_free_gb(LOOP_DIR) < 2.0:
        return True, "disk free < 2 GB"
    if state.get("cycles_without_sota", 0) >= 12:
        return True, "12 cycles without new SOTA"
    return False, ""


def select_task(state, only=None, force=False, attempted=None):
    done = set(state.get("done", []))
    failed = set(state.get("failed_tasks", []))
    incomplete = list(state.get("incomplete", []))
    attempted = set(attempted or ())
    if force and only:
        if only in done:
            done.discard(only)
            state["done"] = [x for x in state["done"] if x != only]
    candidates = []
    if incomplete:
        for name in incomplete:
            if name not in done and name not in failed and name not in attempted:
                candidates.append(name)
    for name in TASK_ORDER:
        if only and name != only:
            continue
        if name in done or name in failed or name in attempted:
            continue
        if name in candidates:
            continue
        candidates.append(name)
    return candidates[0] if candidates else None


def write_summary(state):
    lines = []
    lines.append("# Summary")
    lines.append("")
    lines.append("Started: {}".format(state.get("started")))
    lines.append("Done: {}".format(", ".join(state.get("done", []))))
    lines.append("Failed: {}".format(", ".join(state.get("failed_tasks", []))))
    lines.append("SOTA deployable: {:.4f}".format(state.get("sota_deployable", 0.5263)))
    lines.append("SOTA updates: {}".format(len(state.get("sota_updates", []))))
    lines.append("Cycles without SOTA: {}".format(state.get("cycles_without_sota", 0)))
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for run in state.get("runs", []):
        lines.append("- {} {} {} {}".format(run.get("time"), run.get("task"),
                                            run.get("status"), run.get("path")))
    path = SUMMARY_PATH
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def write_morning_report(state):
    lines = []
    lines.append("# MORNING REPORT")
    lines.append("")
    lines.append("Generated: {}".format(now()))
    lines.append("")
    lines.append("SOTA deployable: {:.4f}".format(state.get("sota_deployable", 0.5263)))
    lines.append("Tasks done: {}".format(len(state.get("done", []))))
    lines.append("Tasks failed: {}".format(len(state.get("failed_tasks", []))))
    lines.append("")
    lines.append("## SOTA updates")
    lines.append("")
    for u in state.get("sota_updates", []):
        lines.append("- {} {:.4f} delta {:.4f} wins {}/5 run {}".format(
            u["task"], u["mean"], u["delta"], u["wins"], u["run_dir"]))
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for run in state.get("runs", []):
        lines.append("- {} {} {} metric={} delta={} path={}".format(
            run.get("time"), run.get("task"), run.get("status"),
            run.get("metric"), run.get("delta"), run.get("path")))
    lines.append("")
    lines.append("No submission CSV was written. No candidate frozen.")
    with open(MORNING_REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines))


def do_status():
    state = load_state()
    print(json.dumps(state, indent=2, sort_keys=True))
    if os.path.isfile(HEARTBEAT_PATH):
        print("HEARTBEAT:", open(HEARTBEAT_PATH).read().strip())


def do_stop():
    with open(STOP_PATH, "w") as handle:
        handle.write("stop requested {}\n".format(now()))
    print("STOP requested")


def do_report():
    if os.path.isfile(MORNING_REPORT_PATH):
        print(open(MORNING_REPORT_PATH).read())
    else:
        write_morning_report(load_state())
        print(open(MORNING_REPORT_PATH).read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hours", type=float, default=None)
    args = parser.parse_args()

    if args.status:
        do_status()
        return 0
    if args.stop:
        do_stop()
        return 0
    if args.report:
        do_report()
        return 0

    hours = args.start if args.start is not None else (args.hours if args.hours is not None else 8.0)
    hours = min(hours, 24.0)
    if hours <= 0:
        hours = 8.0

    lock_handle = acquire_lock()
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    state = load_state()
    state["hours"] = hours
    # budget is per process launch, not inherited from a previous run
    state["started_ts"] = time.time()
    state["started"] = now()
    # remove stale STOP on fresh start
    if os.path.isfile(STOP_PATH) and args.start is not None:
        os.remove(STOP_PATH)
    save_state(state)

    log("loop start hours={}".format(hours))
    attempted = set()

    try:
        while True:
            stop, reason = stop_conditions(state, hours)
            if stop:
                log("stop: {}".format(reason))
                break

            task_name = select_task(state, only=args.only, force=args.force,
                                    attempted=attempted)
            if task_name is None:
                log("queue empty: all tasks done or failed")
                break
            attempted.add(task_name)

            run_id = "{}__{}".format(time.strftime("%Y%m%d-%H%M%S"), task_name)
            run_dir = os.path.join(RUNS_DIR, run_id)
            os.makedirs(run_dir, exist_ok=True)

            # decision
            decision_path, decision = make_decision_json(run_dir, task_name)

            # gate pre
            gate_pre_status, gate_pre_text = run_gate_pre(run_dir)
            with open(os.path.join(run_dir, "gate_pre.txt"), "w") as handle:
                handle.write(gate_pre_status + "\n")
                handle.write(gate_pre_text + "\n")
            if gate_pre_status != "PASS":
                log("gate_pre BLOCK {} {}".format(task_name, gate_pre_status))
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
                save_state(state)
                write_heartbeat(task_name, "gate_pre_block")
                continue

            if args.dry_run:
                log("dry-run: {} decision+gate_pre ok".format(task_name))
                print("dry-run ok", task_name, gate_pre_status)
                break

            # deps
            compute_deps(run_dir, task_name)

            # cmd
            result_path = os.path.join(run_dir, "results.json")
            error_path = os.path.join(run_dir, "stderr.txt")
            cmd = [PY, RUN_ONE, "--task", task_name, "--loop-dir", LOOP_DIR,
                   "--run-dir", run_dir, "--result-path", result_path,
                   "--error-path", error_path]
            log("start task {}".format(task_name))
            write_heartbeat(task_name, "running")
            t0 = time.time()
            status = "done"
            result = None
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=LOOP_DIR)
                with open(os.path.join(run_dir, "stdout.txt"), "w") as handle:
                    handle.write(proc.stdout or "")
                with open(os.path.join(run_dir, "stderr.txt"), "a") as handle:
                    handle.write(proc.stderr or "")
                if proc.returncode != 0 or not os.path.isfile(result_path):
                    raise RuntimeError("task failed rc={} stderr={}".format(
                        proc.returncode, (proc.stderr or "")[:500]))
                with open(result_path) as handle:
                    result = json.load(handle)
            except subprocess.TimeoutExpired:
                status = "timeout"
                log("timeout {}".format(task_name))
            except Exception as exc:
                status = "failed"
                log("fail {} {}".format(task_name, str(exc)[:300]))

            runtime = round(time.time() - t0, 1)
            with open(os.path.join(run_dir, "runtime.txt"), "w") as handle:
                handle.write("{}\n".format(runtime))

            if status != "done" or result is None:
                # mark incomplete retry
                attempts = state.setdefault("attempts", {})
                attempts[task_name] = attempts.get(task_name, 0) + 1
                if attempts[task_name] >= 2:
                    state.setdefault("failed_tasks", []).append(task_name)
                    state["failed"][task_name] = status
                    state["incomplete"] = [x for x in state.get("incomplete", []) if x != task_name]
                else:
                    if task_name not in state.get("incomplete", []):
                        state.setdefault("incomplete", []).append(task_name)
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
                save_state(state)
                append_index(now(), task_name, status, "", "", run_dir)
                write_heartbeat(task_name, status)
                write_summary(state)
                time.sleep(2)
                continue

            # add runtime to result if missing
            if "runtime_s" not in result:
                result["runtime_s"] = runtime
            with open(result_path, "w") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)

            # gate post
            gate_post_status, gate_post_text = run_gate_post(run_dir)
            with open(os.path.join(run_dir, "gate_post.txt"), "w") as handle:
                handle.write(gate_post_status + "\n")
                handle.write(gate_post_text + "\n")

            if gate_post_status != "PASS":
                status = "failed_gate_post"

            # report
            generate_report(run_dir, task_name, decision, result, status,
                            gate_pre_status, gate_post_status)

            # update state
            if task_name not in state.get("done", []):
                state.setdefault("done", []).append(task_name)
            state["failed"].pop(task_name, None)
            state["incomplete"] = [x for x in state.get("incomplete", []) if x != task_name]
            state["consecutive_failures"] = 0
            metric_val = extract_primary_metric(result)
            dmean, dci = extract_delta(result)
            delta_str = "" if dmean is None else "{:.4f}".format(dmean)
            metric_str = "" if metric_val is None else "{:.4f}".format(metric_val)
            state.setdefault("runs", []).append({
                "time": now(), "task": task_name, "status": status,
                "metric": metric_str, "delta": delta_str, "path": run_dir,
            })
            append_index(now(), task_name, status, metric_str, delta_str, run_dir)

            # SOTA update
            updated = maybe_update_sota(task_name, result, state, run_dir)
            if updated:
                state["cycles_without_sota"] = 0
                log("SOTA updated by {}".format(task_name))
            else:
                if task_name not in INVENTORY_TASKS:
                    state["cycles_without_sota"] = state.get("cycles_without_sota", 0) + 1

            save_state(state)
            write_summary(state)
            write_heartbeat(task_name, status)
            log("done {} status={} metric={}".format(task_name, status, metric_str))

            time.sleep(2)
    finally:
        write_summary(state)
        write_morning_report(state)
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_handle.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
