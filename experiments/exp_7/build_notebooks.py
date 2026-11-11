"""Write the exp_7 notebooks and their kernel spec (never reads executed outputs).

Pass notebook names to rebuild only those, so rebuilding one never wipes another's outputs.
Notebook 04 reads BAR-ATH (aggregates only): never open it executed; read results/*.json.
"""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP7_PYTHON", sys.executable)
KERNEL = ROOT / "kernels" / "exp7"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_7 (.venv-torch)",
    "language": "python",
    "env": {
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(ROOT / "cache/numba"),
        "MPLCONFIGDIR": str(ROOT / "cache/matplotlib"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "8",
    },
}, indent=2))

IMPORTS = """from pathlib import Path
import os, sys, json, time, subprocess
from datetime import datetime, timezone
ROOT = Path.cwd()
if not (ROOT / 'exp7_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_7')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import exp7_common as C
import exp1_common as C1
import numpy as np
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)
def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
def make_log(name):
    path = C.LOG_DIR / f'{name}_progress.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    def log(message, **_):
        print(message, flush=True)
        with path.open('a') as handle:
            handle.write(f'{now()} {message}\\n')
    return log
"""
VERIFY = """protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""
GUARD = """guard = C.guard_check(C.RESULTS / 'write_guard_before.json', C.RESULTS / GUARD_FILE)
print('nothing written under exp_0-exp_6 / data since freezing:', guard['all_unchanged'],
      {k: v['unchanged'] for k, v in guard['folders'].items()})
if not guard['all_unchanged']:
    raise RuntimeError('Write guard changed: diagnose before continuing')
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


NOTEBOOKS = {
    "00_protocol.ipynb": [
        md("""# exp_7 · 00 protocol — freeze, tests, rechecks (IDSSE only)
Freezes `PREREG.md` (SHA-256 + code-derived constants + SHA-256 of every reused file and of exp_7's token caches)
before any exp_7 fit or scoring, records the write guard, runs the synthetic tests, then (after freezing) scores
`exp3_ref_ens3` from saved weights on exp_7's views through exp_6's path: the `target_like`, measured, stress and
clean 25 / 50 % rows must equal exp_6's stored rows (binding); the AR(1) rows are report only."""),
        code(IMPORTS + """
if not C.PROTOCOL.exists():
    C.freeze_prereg()
    print('PREREG frozen now (before any exp_7 fit or scoring)')
""" + VERIFY + """
before = C.guard_record(C.RESULTS / 'write_guard_before.json')
print('write guard recorded:', {k: v['files'] for k, v in before['folders'].items()})
print('public check spent by exp_1 (never repeated):', C.public_check_spent())
pre = C1.read_json(C.RESULTS / 'prefreeze_checks.json')
print('pre-freeze:', {k: v for k, v in pre.items() if k != 'unsimulated_target_aug_equals_exp2'})"""),
        code("""result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
                        cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
summary = [line for line in result.stdout.strip().splitlines() if 'passed' in line or 'failed' in line][-1]
C1.write_json(C.RESULTS / 'tests.json', {'run_at_utc': now(), 'returncode': result.returncode, 'summary': summary})
print(summary)
if result.returncode:
    raise RuntimeError('exp_7 synthetic tests failed')"""),
        code("""import exp7_eval as E
log = make_log('00')
rows = E.reference_rows(('target_like', 'measured', 'stress', 'clean25', 'clean50', 'ar1'), log)
pd.DataFrame(rows).to_csv(C.RESULTS / 'ref_rows.csv', index=False)
recheck = E.recheck_against_exp6(rows)
C1.write_json(C.RESULTS / 'rechecks.json', {'checked_at_utc': now(), 'reference_vs_exp6_rows': recheck})
print(recheck)
if not recheck['all_equal']:
    raise RuntimeError('Binding recheck failed: exp_7 views do not reproduce exp_6 reference rows')
table = pd.DataFrame(rows).groupby(['view', 'label_set'])[['spell_f1', 'count_error_ratio']].mean().round(4)
print(table)"""),
    ],
    "01_fits.ipynb": [
        md("""# exp_7 · 01 fits — 3 seeds × 28 nested jobs (IDSSE only)
Resumable (one `.npz` + `.pt` per job, cached by fingerprint). Every job predicts all six evaluation views of its
held-out matches with the same weights. Results: `results/fits.json`."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_01.json'
""" + GUARD + """
import exp7_fit as F
log = make_log('01')"""),
        code("""summary = {}
for arm in F.ARMS:
    start = time.time()
    missing = len(F.missing_jobs(arm))
    F.ensure_predictions(arm, log)
    summary[arm] = {'fingerprint': F.fingerprint(arm), 'seed': F.ARMS[arm].seed, 'fitted_now': missing,
                    'seconds': round(time.time() - start, 1), 'jobs_missing_after': len(F.missing_jobs(arm))}
    log(f'ARM {arm} done {summary[arm]}')
count = F.proxy_fit_count()
C1.write_json(C.RESULTS / 'fits.json', {'finished_at_utc': now(), 'arms': summary, 'proxy_fits': count,
                                        'planned': C.PLANNED_FITS, 'cap': C.FIT_CAP})
print('proxy fits:', count)
if count > C.PLANNED_FITS['proxy'] or any(s['jobs_missing_after'] for s in summary.values()):
    raise RuntimeError('Unexpected fit count or missing jobs')"""),
    ],
    "02_evaluation.ipynb": [
        md("""# exp_7 · 02 evaluation — nested decoder selection (IDSSE only, never fits)
1. **Recheck (binding).** exp_7's `evaluate` on exp_2's cached `exp3_ref_ens3` `target_like` predictions reproduces
   exp_3's stored rows, selections and final configs.
2. Every pre-registered (variant, selection view, label set): selection view = deployment view; the held-out match is
   scored on every view decoded with that selection. Results: `results/variant_*__sel_*__*.json`."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_02.json'
""" + GUARD + """
import exp7_eval as E
log = make_log('02')"""),
        code("""rechecks = [E.evaluation_path_recheck(label) for label in C.LABEL_SETS]
C1.write_json(C.RESULTS / 'eval_recheck.json', {'checked_at_utc': now(), 'rechecks': rechecks})
print(rechecks)
if not all(r['rows_equal'] and r['selections_equal'] and r['final_config_equal'] for r in rechecks):
    raise RuntimeError('Binding recheck failed: exp_7 evaluation path does not reproduce exp_3')"""),
        code("""cache, timings = {}, []
for task in E.tasks():
    seconds, tag = E.evaluate_and_save(task, cache)
    timings.append({'task': list(task), 'seconds': round(seconds, 1), 'tag': tag})
    rows = C1.read_json(E.result_path(*task))['rows']
    means = pd.DataFrame(rows).groupby('view')['spell_f1'].mean().round(4).to_dict()
    log(f'EVAL {task} {tag} {seconds:.0f}s {means}')
C1.write_json(C.RESULTS / 'evaluation.json', {'finished_at_utc': now(), 'tasks': timings})"""),
    ],
    "03_decision.ipynb": [
        md("""# exp_7 · 03 decision — H1 and report tables (IDSSE only)
H1 = on provider **and** `L_ctrl`: `exp7_robust_ens3` measured − `exp3_ref_ens3` `target_like` (exp_3's stored rows)
paired 95 % CI low > 0; `target_like` mean Δ ≥ −0.005; stress mean Δ ≥ −0.005. Measured and stress use the
measured-view selection; `target_like` uses the `target_like` selection. Results: `results/decision.json`."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_03.json'
""" + GUARD + """
import exp7_decision as DE
import exp7_eval as E
import exp7_report as R"""),
        code("""results = {(sel, label): C1.read_json(E.result_path(C.PRIMARY, sel, label))
           for sel in ('target_like', 'measured') for label in C.LABEL_SETS}
deltas = {label: {'gain': DE.view_deltas(results[('measured', label)]['rows'], 'measured', label),
                  'no_harm': DE.view_deltas(results[('target_like', label)]['rows'], 'target_like', label),
                  'stress': DE.view_deltas(results[('measured', label)]['rows'], 'stress', label)}
          for label in C.LABEL_SETS}
decision = DE.decide(DE.clause_table(deltas))
decision.update({'decided_at_utc': now(), 'primary': C.PRIMARY, 'baseline': DE.H1_RULE['baseline'],
                 'final_configs': {label: results[('measured', label)]['final_config'] for label in C.LABEL_SETS},
                 'files_written': False,
                 'candidate_files': 'written by 04_target' if decision['h1_passed'] else 'none: H1 failed'})
C.write_results(C.RESULTS / 'decision.json', decision)
for label in C.LABEL_SETS:
    for clause, value in decision['clauses'][label].items():
        print(label, clause, {k: (round(value[k], 4) if isinstance(value[k], float) else value[k])
                              for k in ('mean_delta', 'ci_low', 'ci_high', 'wins', 'pass')})
print('H1 passed:', decision['h1_passed'], '| failed:', decision['failed'])"""),
        code("""frame = R.all_rows()
rows = R.delta_rows(frame)
spread = R.seed_spread(rows)
C.write_results(C.RESULTS / 'summary.json', {'computed_at_utc': now(), 'deltas': rows, 'seed_spread': spread})
(C.RESULTS / 'report_tables.md').write_text('## Views vs exp3_ref_ens3 target_like\\n\\n' + R.markdown(rows)
                                            + '\\n## Seed spread\\n\\n' + R.markdown(spread))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(pd.DataFrame(spread).round(4).to_string(index=False))"""),
        code("""final = C.guard_check(C.RESULTS / 'write_guard_before.json', C.RESULTS / 'guard_final.json')
print('final write guard unchanged:', final['all_unchanged'])"""),
    ],
    "04_target.ipynb": [
        md("""# exp_7 · 04 target — only if H1 passed (BAR-ATH, aggregates only)
Three full-data `exp7_robust` fits; tactical ATD with exp_6's recovered ball on accepted hidden graded frames and
`ball_recovered` = 1 there; final measured-view configs; plausibility guard (dead share 0.407 ± 0.030); validator;
rebuild with exp_6's detector retrained in a fresh folder + three fresh fits. **Reads BAR-ATH: never open this
notebook executed.** Results: `results/target_candidates.json`."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_04.json'
""" + GUARD + """
import exp7_decision as DE
import exp7_fit as F
import exp7_target as TG
log = make_log('04')
decision = C1.read_json(C.RESULTS / 'decision.json')
DE.require_h1(decision)"""),
        code("""models = {arm: TG.full_model(arm) for arm in F.ARMS}
first = TG.build(decision, C.RECOVERED, TG.CANDIDATES, models, log)
C.write_results(C.RESULTS / 'target_candidates.json', {'built_at_utc': now(), 'first': first,
                                                     'target_fits': TG.target_fit_count()})
print({label: {k: v for k, v in info.items() if k != 'sha256'} for label, info in first['files'].items()})"""),
        code("""rebuild = TG.rebuild(decision, first, log)
C.write_results(C.RESULTS / 'target_candidates.json', {'built_at_utc': now(), 'first': first, 'rebuild': rebuild,
                                                     'target_fits': TG.target_fit_count()})
print('rebuild byte-identical:', rebuild['all_byte_identical'])
final = C.guard_check(C.RESULTS / 'write_guard_before.json', C.RESULTS / 'guard_final.json')
print('final write guard unchanged:', final['all_unchanged'])"""),
    ],
}


def main(names: list[str]) -> None:
    for name in names or NOTEBOOKS:
        notebook = nbf.v4.new_notebook()
        notebook.metadata["kernelspec"] = {"name": "exp7", "display_name": "exp_7 (.venv-torch)", "language": "python"}
        notebook.cells = NOTEBOOKS[name]
        nbf.write(notebook, ROOT / name)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
