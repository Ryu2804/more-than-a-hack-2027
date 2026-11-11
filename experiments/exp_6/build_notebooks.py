"""Write the exp_6 notebooks and their kernel spec (never reads executed outputs).

Pass notebook names to rebuild only those, so rebuilding one never wipes another's outputs.
Notebooks 00, 02 and 04 read BAR-ATH (aggregates only): never open them executed; read results/*.json.
"""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP6_PYTHON", sys.executable)
KERNEL = ROOT / "kernels" / "exp6"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_6 (.venv-torch)",
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
if not (ROOT / 'exp6_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_6')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import exp6_common as C
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
print('nothing written under exp_0-exp_5 / data since freezing:', guard['all_unchanged'],
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
        md("""# exp_6 · 00 protocol — freeze, tests, G1
Freezes `PREREG.md` (SHA-256 + code-derived constants + SHA-256 of every reused file) before any detector
training or simulated-view scoring, records the write guard, runs the synthetic tests and computes G1 from
ATD + homography only (no frame). **Reads BAR-ATH ATD: never open this notebook executed.**"""),
        code(IMPORTS + """
if not C.PROTOCOL.exists():
    C.freeze_prereg()
    print('PREREG frozen now (before any detector training or simulated-view scoring)')
""" + VERIFY + """
before = C.guard_record(C.RESULTS / 'write_guard_before.json')
print('write guard recorded:', {k: v['files'] for k, v in before['folders'].items()})
print('public check spent by exp_1 (never repeated):', C.public_check_spent())"""),
        code("""result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
                        cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
summary = result.stdout.strip().splitlines()[-1]
C1.write_json(C.RESULTS / 'tests.json', {'run_at_utc': now(), 'returncode': result.returncode, 'summary': summary})
print(summary)
if result.returncode:
    raise RuntimeError('exp_6 synthetic tests failed')"""),
        code("""import exp6_data as DA
inputs = DA.load_inputs()
summary = DA.input_summary(inputs)
C1.write_json(C.RESULTS / 'g1.json', {**summary['g1'], 'computed_at_utc': now()})
C1.write_json(C.RESULTS / 'inputs_summary.json', summary)
print('G1 (aggregate):', summary['g1'])
print('splits (aggregate):', summary['splits'])
del inputs"""),
    ],
    "01_simulator.ipynb": [
        md("""# exp_6 · 01 simulator rechecks (IDSSE only)
1. **Recheck (binding).** S(c = 0) tokens equal exp_2's cached `target_like` tokens, and its rows equal exp_3's
   stored `exp3_ref_ens3` `target_like` rows on provider and `L_ctrl`.
2. **Recheck (report).** exp_3's S2 coverage view regenerated in memory reproduces exp_3's S2 rows and deltas.
3. **Context recheck (report).** `exp5_touch_ens3` through the same path vs exp_5's stored rows.
The VOI grid runs in 03 (PREREG §0 deviation: it needs σ̂, f̂ and the run lengths from 02)."""),
        code(IMPORTS + VERIFY + """
import exp6_sim as S
log = make_log('01')"""),
        code("""rows = S.evaluate([S.BASE, S.Setting(S.S2_RECHECK, 0.12, 0.0, 0.0)], np.array([1]), 'reference', log)
touch_rows = S.evaluate([S.BASE], np.array([1]), 'touch', log)
rechecks = {'checked_at_utc': now(), 'tokens': S.recheck_tokens(), 'target_like_rows': S.recheck_base(rows),
            's2_coverage': S.recheck_s2(rows), 'touch_target_like_rows': S.recheck_base(touch_rows, 'touch')}
C1.write_json(C.RESULTS / 'sim_rechecks.json', rechecks)
pd.DataFrame(rows + touch_rows).to_csv(C.RESULTS / 'sim_rows_01.csv', index=False)
for key, value in rechecks.items():
    print(key, value)
if not (rechecks['tokens']['all_equal'] and rechecks['target_like_rows']['rows_equal']):
    raise RuntimeError('Binding recheck failed: S(0) does not reproduce exp_3 target_like')"""),
    ],
    "02_detector.ipynb": [
        md("""# exp_6 · 02 detector — train, tau on val, quality on test, coverage (BAR-ATH)
Streams the tactical video (sequential decode passes), trains the 3-frame U-Net on train blocks, infers on
hidden + val + test frames, links, picks tau on val, measures quality once on test and coverage on hidden
graded frames. Resumable (per-pass checkpoints, inference chunks). **Reads BAR-ATH video and ATD: aggregates
only; never open this notebook executed.** Results: `results/detector_quality.json`."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_02.json'
""" + GUARD + """
import exp6_recover as R
log = make_log('02')"""),
        code("""summary = R.recover(C.CACHE, log)
C.write_results(C.RESULTS / 'detector_quality.json', {**summary, 'computed_at_utc': now()})
print('tau:', summary['tau'])
print('val:', summary['val'])
print('test:', summary['test'])
print('coverage of hidden graded frames:', summary['coverage'], '| runs:', summary['run_lengths'])
print('bridge:', summary['bridge'])"""),
    ],
    "03_decision.ipynb": [
        md("""# exp_6 · 03 decision — measured and stress settings, H1, VOI grid, touch context
H1 = G1 & G2 & G3 & (measured paired 95 % CI low > 0 on provider **and** `L_ctrl`) & (stress mean Δ ≥ −0.005 on
both). Saved exp_2 weights, exp_3's per-fold `target_like` decoder selections; no fit, no re-selection.
Reads only `results/*.json` and `cache/run_lengths.npy` from 02 (aggregate run lengths)."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_03.json'
""" + GUARD + """
import exp6_decision as DE
import exp6_sim as S
log = make_log('03')
quality = C1.read_json(C.RESULTS / 'detector_quality.json')
q = {'coverage': quality['coverage'], 'sigma_m': quality['test']['sigma_m'], 'fp_rate': quality['test']['fp_rate']}
lengths = np.load(C.CACHE / 'run_lengths.npy')
draw = lengths if len(lengths) else np.array([1])
print('measured:', S.measured(q), '| stress:', S.stress(q), '| accepted runs:', len(lengths))"""),
        code("""settings = [S.BASE, S.measured(q), S.stress(q)]
rows = S.evaluate(settings, draw, 'reference', log)
measured = {label: S.deltas(rows, 'measured', label) for label in C.LABEL_SETS}
stressed = {label: S.deltas(rows, 'stress', label) for label in C.LABEL_SETS}
g1 = C1.read_json(C.RESULTS / 'g1.json')['share']
gates = DE.gate_table(g1, quality['test']['precision'], quality['test']['recall'], quality['coverage'])
decision = DE.decide(gates, DE.ci_clause(measured), DE.stress_clause(stressed))
decision.update({'decided_at_utc': now(), 'tau': quality['tau'],
                 'settings': {'measured': S.measured(q).as_dict(), 'stress': S.stress(q).as_dict()},
                 'files_written': False,
                 'candidate_files': 'written by 04_target' if decision['h1_passed'] else 'none: H1 failed'})
C.write_results(C.RESULTS / 'decision.json', decision)
print('gates:', {k: v['pass'] for k, v in gates.items()})
for label in C.LABEL_SETS:
    print(label, 'measured', {k: decision['ci_clause'][label][k] for k in ('mean_delta', 'ci_low', 'ci_high', 'pass')},
          '| stress', {k: decision['stress_clause'][label][k] for k in ('mean_delta', 'pass')})
print('H1 passed:', decision['h1_passed'], '| failed:', decision['failed'])"""),
        code("""voi = S.voi_grid(q) if len(lengths) else []
voi_rows = S.evaluate([S.BASE] + voi, draw, 'reference', log) if voi else []
grid = [{**s.as_dict(), **{label: DE.delta_summary(S.deltas(voi_rows, s.name, label)) for label in C.LABEL_SETS}}
        for s in voi]
C.write_results(C.RESULTS / 'voi_grid.json', {'computed_at_utc': now(), 'report_only': True,
                                            'computable': bool(voi), 'grid': grid})
for cell in grid:
    print(cell['name'], {label: round(cell[label]['mean_delta'], 4) for label in C.LABEL_SETS})"""),
        code("""touch_rows = S.evaluate([S.BASE, S.measured(q)], draw, 'touch', log)
touch = {'report_only': True, 'recheck_target_like': S.recheck_base(touch_rows, 'touch'),
         'measured': {label: DE.delta_summary(S.deltas(touch_rows, 'measured', label, 'touch')) for label in C.LABEL_SETS}}
C.write_results(C.RESULTS / 'touch_context.json', {'computed_at_utc': now(), **touch})
pd.DataFrame(rows + voi_rows + touch_rows).to_csv(C.RESULTS / 'sim_rows_03.csv', index=False)
print('touch context:', {label: round(touch['measured'][label]['mean_delta'], 4) for label in C.LABEL_SETS},
      touch['recheck_target_like'])"""),
    ],
    "04_target.ipynb": [
        md("""# exp_6 · 04 target — only if H1 passed
Fills the recovered ball on accepted hidden graded frames (1 cm), runs exp_2's three full-data members,
decodes with exp_3's `L_ctrl` final config and provider hedge config, writes and validates both files, then
retrains the detector from scratch and rebuilds both files byte-identically. **Reads BAR-ATH: aggregates only;
never open this notebook executed.** Results: `results/target_candidates.json`."""),
        code(IMPORTS + VERIFY + """
GUARD_FILE = 'guard_04.json'
""" + GUARD + """
import exp6_decision as DE
import exp6_target as TG
log = make_log('04')
decision = C1.read_json(C.RESULTS / 'decision.json')
DE.require_h1(decision)"""),
        code("""first = TG.build(decision, C.RECOVERED, TG.CANDIDATES, log)
C.write_results(C.RESULTS / 'target_candidates.json', {'build': first, 'computed_at_utc': now()})
rebuild = TG.rebuild(decision, first, log)
C.write_results(C.RESULTS / 'target_candidates.json', {'build': first, 'rebuild': rebuild, 'computed_at_utc': now()})
decision.update({'files_written': True, 'files': {k: v['sha256'] for k, v in first['files'].items()},
                 'rebuild_byte_identical': rebuild['all_byte_identical']})
C.write_results(C.RESULTS / 'decision.json', decision)
print('fill:', first['fill'])
print('files:', first['files'])
print('rebuild byte-identical:', rebuild['all_byte_identical'])"""),
    ],
}


def main(names: list[str]) -> None:
    for name in names or list(NOTEBOOKS):
        notebook = nbf.v4.new_notebook()
        notebook.cells = NOTEBOOKS[name]
        notebook.metadata["kernelspec"] = {"name": "exp6", "display_name": "exp_6 (.venv-torch)", "language": "python"}
        nbf.write(notebook, ROOT / name)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
