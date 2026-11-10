"""Write the exp_5 notebooks and their kernel spec (never reads executed outputs).

Pass notebook names to rebuild only those, so rebuilding one never wipes another's outputs.
"""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP5_PYTHON", sys.executable)
KERNEL = ROOT / "kernels" / "exp5"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_5 (.venv-torch)",
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
import os, sys, json, time
from datetime import datetime, timezone
ROOT = Path.cwd()
if not (ROOT / 'exp5_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_5')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import exp5_common as C
import numpy as np
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)
import exp1_common as C1
"""
SETUP = IMPORTS + """protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""
SETUP_FREEZE = IMPORTS + """if not C.PROTOCOL.exists():
    C.freeze_prereg()
    print('PREREG frozen now (before any exp_5 fit)')
protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
guard_path = C.RESULTS / 'write_guard_before.json'
if not guard_path.exists():
    C1.write_json(guard_path, {'taken_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                               'folders': C.write_manifest()})
print('write guard (exp_0-exp_4, data/):', {k: v['files'] for k, v in C1.read_json(guard_path)['folders'].items()})
"""
LOG = """log_path = C.RESULTS / 'fit_log.txt'
def log(message, **_):
    print(message, flush=True)
    with log_path.open('a') as handle:
        handle.write(message + '\\n')
"""
GUARD = """before = C1.read_json(C.RESULTS / 'write_guard_before.json')
now = C.write_manifest()
guard = {name: {'unchanged': now[name] == state, 'files_before': state['files'], 'files_now': now[name]['files']}
         for name, state in before['folders'].items()}
C1.write_json(C.RESULTS / GUARD_FILE, {'checked_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                       'since': before['taken_at_utc'], 'folders': guard})
print('nothing written under exp_0-exp_4 / data/ since freezing:', all(g['unchanged'] for g in guard.values()), guard)
"""

INTRO = """# exp_5 — touch-only SetSeq, fresh-seed replication (pre-registered in `PREREG.md`)

**Question (H1).** exp_4's report-only ablation `exp4_touch_ens3` (SetSeq + exp_3's 4 touch features, no aux head)
beat the exp_2 3-seed ensemble `exp3_ref_ens3` on provider labels (`target_like` +0.015 [+0.007, +0.027]). Is that
seed luck? exp_5 re-runs the identical recipe with **fresh seeds 20261111 / 12 / 13** on the same 7 IDSSE matches.

**H1 passes iff both hold** (provider labels, `target_like`, nested re-selection): the paired 95 % CI lower bound of
Δ = `exp5_touch_ens3` − `exp3_ref_ens3` is > 0, **and** mean Δ exceeds the `exp5_touch` single-seed spread on that cell.

**Scope.** Same 7 matches that selected this arm: a pass rules out seed luck, not selection on these matches.
"""

NOTEBOOKS = {
    "00_protocol.ipynb": [
        ("markdown", INTRO),
        ("code", SETUP_FREEZE + "import exp5_protocol as P, exp5_eval as E, exp2_protocol as P2, exp3_protocol as P3, exp3_touch as S"),
        ("markdown", "## 1. Unit tests (synthetic arrays)\nArm table, nothing written under exp_0–exp_4 (tiny fit + evaluation with patched exp_5 paths), `fit=False` refusal, the two-clause decision rule, the ctrl-file gate, fingerprints."),
        ("code", """import subprocess
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
                        capture_output=True, text=True, cwd=ROOT)
print(result.stdout.strip().splitlines()[-1])
assert result.returncode == 0, 'unit tests failed'"""),
        ("markdown", "## 2. Inputs read in place\nexp_2 caches (fingerprint-checked), exp_3 touch-feature caches, provider training targets (cached tokens' state / boundary must equal `exp3_protocol.step_labels(·, 'provider')`), and exp_4's touch caches (SHA-256 of all 84 `.npz` verified by `verify_prereg`)."),
        ("code", """for arm, fp in E.EXP2_FINGERPRINTS.items():
    missing = [k for k in C.job_names() if not (P2.prediction_folder(arm) / f'{k}.npz').exists()]
    print(f'exp_2 {arm:12s} fingerprint {fp} | missing jobs {len(missing)}')
    assert not missing and P2.fingerprint(arm) == fp
jobs = [(n, k, r) for n in C.MATCH_IDS for r, k in P.RECIPE]
assert all(S.feature_path(n, k, r).exists() for n, k, r in jobs + [(n, 'target_like', 0) for n in C.MATCH_IDS])
same = []
for name, kind, round_index in jobs:
    tokens = P2.load_tokens(name, kind, round_index)
    state, boundary = P3.step_labels(name, 'provider', tokens)
    same.append(bool(np.array_equal(state, tokens['state']) and np.array_equal(boundary, tokens['boundary'])))
print('touch caches present for', len(jobs) + len(C.MATCH_IDS), 'arrays | provider targets equal cached targets:', all(same), f'({len(same)} arrays)')
assert all(same)
for arm in C.EXP4_TOUCH_ARMS:
    folder = C.exp4_touch_folder(arm)
    print(f'exp_4 {arm:14s} {folder.name} | npz {sum(1 for _ in folder.glob("*.npz"))} (sha256 frozen)')"""),
        ("markdown", "## 3. Reused results: same code path\n`exp3_ref_ens3` (exp_3) and `exp4_touch_ens3` (exp_4) are read from stored nested results. One cell of each (provider, `target_like`) is recomputed here from the cached predictions and must match exactly."),
        ("code", """checks = [E.recheck('exp3_ref_ens3', 'target_like', 'provider'), E.recheck('exp4_touch_ens3', 'target_like', 'provider')]
C1.write_json(C.RESULTS / 'reference_recheck.json', checks)
for check in checks:
    print(check)
assert all(c['rows_equal'] and c['final_config_equal'] for c in checks)"""),
        ("code", """print('exp_5 proxy fits so far:', P.proxy_fit_count(), '| planned', C.PLANNED_FITS, '| cap', C.FIT_CAP)
print(pd.DataFrame([{'arm': a, 'seed': s.seed, 'touch': s.touch, 'aux_enabled': s.aux.enabled,
                     'config_is_default': s.config == P.N.TrainConfig(seed=s.seed), 'fingerprint': P.fingerprint(a)}
                    for a, s in P.ARMS.items()]).to_string(index=False))"""),
    ],
    "01_fits.ipynb": [
        ("markdown", "# 01 — Reproduction check, then fits (3 seeds × 28 nested fits)\n\n`exp4_touch` recipe (`exp2_aug` rounds, provider targets, 4 touch columns, aux off, `TrainConfig()`), seeds 20261111 → 12 → 13. Each fit saves its weights and final training NLL and predicts both views of its held-out matches. Cached by fingerprint with per-job locks. Progress in `results/fit_log.txt`."),
        ("code", SETUP + "import exp5_protocol as P, exp1_net as N\nprint('device', N.DEVICE, '| config', N.TrainConfig().as_dict())"),
        ("markdown", "## Reproduction check (1 fit; must pass before any proxy fit)\nexp_5's code path with `exp4_touch`'s seed 20261108, outer fold holding out J03WMX, against exp_4's cached prediction for that fold (read in place). Bit-exact expected; max |Δp| > 1e-3 stops the run."),
        ("code", LOG + """path = C.RESULTS / 'repro_check.json'
if path.exists():
    repro = C1.read_json(path)
else:
    t0 = time.time()
    repro = P.reproduction_check()
    repro['seconds'] = round(time.time() - t0, 1)
    C1.write_json(path, repro)
    log(f"REPRO CHECK {repro['arm']} outer_{repro['held_out']} bit_exact={repro['bit_exact']} passed={repro['passed']} {repro['seconds']}s")
print(json.dumps(repro, indent=1))
assert repro['passed'], 'reproduction check failed: stop and diagnose before any proxy fit'"""),
        ("code", """t0 = time.time()
reserve = C.PLANNED_FITS['full_data'] + C.PLANNED_FITS['rebuild']
for arm in P.FIT_ORDER:
    remaining = len(P.missing_jobs(arm))
    assert P.proxy_fit_count() + 1 + remaining + reserve <= C.FIT_CAP, 'fit cap reached'
    start = time.time()
    P.ensure_predictions(arm, log=log)
    log(f'ARM DONE {arm} {time.time() - start:.0f}s, proxy fits cached {P.proxy_fit_count()}')
log(f'ALL FITS DONE {time.time() - t0:.0f}s, exp_5 proxy fits cached {P.proxy_fit_count()} (+1 reproduction check)')"""),
        ("code", """train = pd.DataFrame([r for arm in P.ARMS for r in P.train_nll_rows(arm)])
print(train.groupby('arm').train_nll.agg(['mean', 'min', 'max', 'count']).round(4).to_string())"""),
    ],
    "02_evaluation.ipynb": [
        ("markdown", "# 02 — Nested evaluation (never fits)\n\n`exp5_touch_ens3` and the pooled `exp45_touch_ens6` on every S1 label set (provider, `L_track(1 s)`, `L_ctrl`, `L_track(3 s)`) and both views; single seeds on provider and `L_ctrl`, both views. Temperature and decoder are re-selected on the six inner validation matches against each label set; the held-out match is scored once. `exp3_ref_ens3` / `exp2_aug` (exp_3) and `exp4_touch_ens3` (exp_4) are read from stored results. Then argmax diagnostics, recall of possessions < 2 s and the NLL generalisation gap."),
        ("code", SETUP + """import exp5_protocol as P, exp5_eval as E, exp5_diagnostics as G
for arm in P.ARMS:
    assert not P.missing_jobs(arm), f'{arm}: jobs missing; run 01_fits first'"""),
        ("code", """import multiprocessing
from concurrent.futures import ProcessPoolExecutor
tasks = E.tasks()
t0 = time.time()
with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('fork')) as pool:  # RAM ~3 GB free
    for variant, view, label_set, seconds, tag in pool.map(E.evaluate_and_save, tasks):
        frame = pd.DataFrame(E.read_result(variant, view, label_set)['rows'])
        print(f'{label_set:8s} {view:12s} {variant:17s} spellF1 {frame.spell_f1.mean():.4f}  max|cnt| {frame.count_error_ratio.abs().max():.3f}  gamma {sorted(set(frame.gamma))}  ({tag} {seconds:.0f}s)', flush=True)
print(f'{len(tasks)} evaluations, {time.time() - t0:.0f}s')"""),
        ("markdown", "## Diagnostics: argmax (no decoder) and recall of true possessions < 2 s (provider labels)"),
        ("code", """cache = {}
argmax = pd.DataFrame([r for v in (C.PRIMARY, C.REFERENCE) for ls in C.DECISION_LABELS for view in C.VIEWS
                       for r in G.argmax_rows(v, view, ls, cache)])
argmax.to_csv(C.RESULTS / 'argmax_rows.csv', index=False)
print(argmax.groupby(['label_set', 'view', 'variant'])[['spell_f1', 'count_error_ratio']].mean().round(4).to_string())
recall_variants = (C.PRIMARY, C.REFERENCE, 'exp4_touch_ens3', 'exp45_touch_ens6', 'exp2_aug')
recall = pd.DataFrame([r for v in recall_variants for view in C.VIEWS for r in G.short_recall_rows(v, view, cache)])
recall.to_csv(C.RESULTS / 'short_recall.csv', index=False)
print(recall.groupby(['view', 'variant'])[['recall_short_decoded', 'recall_short_argmax', 'recall_all_decoded']].mean().round(4).to_string())"""),
        ("markdown", "## Generalisation gap (provider labels, raw p, T = 1)\nTraining NLL of every exp_5 fit (logged at fit time) and of the exp_2 recipe (exp_4's `ref_train_nll.csv`, read in place) against the held-out NLL of their outer and pair predictions."),
        ("code", """t0 = time.time()
train = pd.concat([pd.DataFrame([r for arm in P.ARMS for r in P.train_nll_rows(arm)]), G.reference_train_nll()],
                  ignore_index=True)
heldout = pd.DataFrame([r for arm in (*P.ARMS, *E.EXP2_ARMS) for r in G.heldout_nll_rows(arm, cache)])
train.to_csv(C.RESULTS / 'train_nll.csv', index=False)
heldout.to_csv(C.RESULTS / 'heldout_nll.csv', index=False)
gap = G.gap_table(train, heldout)
gap.to_csv(C.RESULTS / 'gap_table.csv', index=False)
print(gap.round(4).to_string(index=False))
print(f'{time.time() - t0:.0f}s')"""),
    ],
    "03_comparison.ipynb": [
        ("markdown", "# 03 — Comparison and the pre-registered decision\n\nPaired match bootstrap (1000 draws, seed 20261108, 95 %). H1: `exp5_touch_ens3` − `exp3_ref_ens3`, provider labels, `target_like`: CI lower bound > 0 **and** mean Δ > `exp5_touch` seed spread. The ctrl file additionally needs `L_ctrl` non-inferiority (CI lower bound > −0.01). Everything else is report-only."),
        ("code", SETUP + "import exp5_protocol as P, exp5_eval as E, exp5_report as R, exp5_diagnostics as G\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"),
        ("code", """reference = pd.read_csv(C.C3.RESULTS / 'reference_rows.csv')
reference = reference[reference.label_set.isin(C.S1_LABELS)]
rows = R.collect_rows([reference, pd.read_csv(C.RESULTS / 'argmax_rows.csv')])
rows.to_csv(C.RESULTS / 'per_match_results.csv', index=False)
table = R.summary_table(rows)
table.to_csv(C.RESULTS / 'comparison_table.csv', index=False)
cols = ['variant', 'mean_spell_f1', 'min_spell_f1', 'mean_macro_f1', 'mean_count_error', 'max_abs_count_error',
        'delta_vs_ref_ens3', 'ci95_low_vs_ref', 'ci95_high_vs_ref', 'wins_vs_ref', 'count_gate']
for label_set in C.S1_LABELS:
    for view in C.VIEWS:
        part = table[(table.label_set == label_set) & (table.view == view)]
        print(f'== {label_set} / {view}'); print(R.as_markdown(part, cols)); print()"""),
        ("markdown", "## Per match, `target_like` (spellF1, count error in brackets)"),
        ("code", """show = (C.PRIMARY, C.REFERENCE, 'exp4_touch_ens3', 'exp45_touch_ens6', *P.ARMS)
for label_set in C.DECISION_LABELS:
    print(f'== {label_set}'); print(R.per_match(rows, label_set, 'target_like', show).to_string()); print()"""),
        ("markdown", "## exp_0 quality gates (vs the same view's baseline), provider and `L_ctrl`"),
        ("code", """gates = R.gate_rows(rows, 'provider') + R.gate_rows(rows, 'ctrl')
print(pd.DataFrame([{'label_set': g['label_set'], 'view': g['view'], 'variant': g['variant'], **g['checks'], 'pass': g['pass'],
                     'worst_count_error': round(g['worst_count_error'], 3), 'worst_match': g['worst_count_match']}
                    for g in gates]).to_string(index=False))"""),
        ("markdown", "## Pre-registered decision (H1: CI clause and seed-spread clause), adoption tree"),
        ("code", """decision = R.decide(rows)
C1.write_json(C.RESULTS / 'decision.json', decision)
h1 = decision['H1']
d = h1['delta']
print(f"H1 {C.PRIMARY} - {C.REFERENCE}, provider / target_like: {d['mean_delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] wins {d['wins']}/7")
print('  CI clause  :', h1['ci_clause'])
print('  seed clause:', h1['seed_clause'])
print('H1', 'PASSED' if h1['passed'] else 'FAILED')
c = decision['ctrl_non_inferiority']
print(f"L_ctrl non-inferiority (ctrl file only): {c['mean_delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] -> {'pass' if c['passed'] else 'fail'}")
print('adoption:', json.dumps(decision['adoption']))"""),
        ("markdown", "## Replication (report only): exp_5 vs exp_4 touch, per cell; pooled 6-seed ensemble; single seeds"),
        ("code", """replication = pd.DataFrame(decision['report_only']['replication'])
replication.to_csv(C.RESULTS / 'replication_table.csv', index=False)
print(replication.round(4).to_string(index=False))
ro = decision['report_only']
for item in ro['single_seeds_vs_reference']:
    print(f"  {item['label_set']:8s} {item['view']:12s} {item['a']:14s} - {item['b']}: {item['mean_delta']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}] wins {item['wins']}")
print(pd.DataFrame([{k: v for k, v in s.items() if k != 'seeds'} | s['seeds'] for s in ro['seed_spreads']]).round(4).to_string(index=False))"""),
        ("markdown", "## Generalisation gap and short-possession recall (report only)"),
        ("code", """gap = pd.read_csv(C.RESULTS / 'gap_table.csv')
print(gap[gap.level == 'family'].round(4).to_string(index=False))
exp4_gap = G.exp4_gap_context()
print('exp_4 context (read in place):'); print(exp4_gap.round(4).to_string(index=False))
recall = pd.read_csv(C.RESULTS / 'short_recall.csv')
recall_summary = recall.groupby(['view', 'variant'])[['recall_short_decoded', 'recall_short_argmax', 'recall_all_decoded']].mean()
print(recall_summary.round(4).to_string())
argmax = pd.read_csv(C.RESULTS / 'argmax_rows.csv')
report = {'prereg_sha256': protocol['prereg_sha256'], 'bootstrap': 'paired match bootstrap, 1000 draws, seed 20261108, 95% percentile',
          'table': R.records(table), 'gates': gates, 'decision': decision, 'replication': R.records(replication),
          'gap_table': R.records(gap), 'exp4_gap_context': R.records(exp4_gap),
          'short_recall': R.records(recall_summary.reset_index()),
          'argmax': R.records(argmax.groupby(['label_set', 'view', 'variant'])[['spell_f1', 'count_error_ratio']].mean().reset_index()),
          'repro_check': C1.read_json(C.RESULTS / 'repro_check.json'),
          'reference_recheck': C1.read_json(C.RESULTS / 'reference_recheck.json')}
C1.write_json(C.RESULTS / 'proxy_report.json', report)"""),
        ("markdown", "## Figure (proxy only)"),
        ("code", """order = [C.REFERENCE, 'exp4_touch_ens3', C.PRIMARY, 'exp45_touch_ens6']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label_set, marker in zip(C.S1_LABELS, ('o', 's', '^', 'v')):
    part = table[(table.label_set == label_set) & (table.view == 'target_like')].set_index('variant').reindex(order)
    axes[0].plot(order, part.mean_spell_f1, marker=marker, label=label_set)
    axes[1].plot(order, part.delta_vs_ref_ens3, marker=marker, label=label_set)
axes[0].set_ylabel('mean spellF1, target_like (7 held-out matches)'); axes[0].set_title('spellF1 by arm and label set')
axes[1].axhline(0, color='0.5', lw=0.8); axes[1].set_ylabel('Δ vs exp3_ref_ens3'); axes[1].set_title('paired Δ (point estimate)')
for ax in axes:
    ax.tick_params(axis='x', rotation=20); ax.legend(title='labels')
fig.tight_layout(); fig.savefig(C.RESULTS / 'fig_exp5.png', dpi=130)
print('saved', C.RESULTS / 'fig_exp5.png')"""),
        ("markdown", "## Write guard: nothing under exp_0–exp_4 or `data/` changed since freezing"),
        ("code", "GUARD_FILE = 'write_guard_03.json'\n" + GUARD),
    ],
    "04_target.ipynb": [
        ("markdown", "# 04 — BAR-ATH (only if H1 passed; aggregates only)\n\nIf H1 failed, nothing is read or written for BAR-ATH. If it passed: 3 full-data `exp5_touch` fits, exp_3's fusion rule (tactical input if rejected), `exp_5_candidate_provider.csv` (decoder selected on provider labels) and, only if `L_ctrl` non-inferiority holds, `exp_5_candidate_ctrl.csv`; official validator; byte-identical rebuild from 3 independent fits; label-free plausibility. No public-window check."),
        ("code", SETUP + "import exp5_protocol as P\nassert C.public_check_spent(), 'exp_1 public check marker missing'"),
        ("code", """decision = C1.read_json(C.RESULTS / 'decision.json')
if not decision['H1']['passed']:
    summary = {'H1_passed': False, 'files_written': [], 'bar_ath_read': False,
               'ci_clause': decision['H1']['ci_clause'], 'seed_clause': decision['H1']['seed_clause'],
               'statement': decision['adoption']['statement']}
    C1.write_json(C.RESULTS / 'target_none.json', summary)
    print(json.dumps(summary, indent=1))
else:
    import exp5_target as T
    missing_full = sum(not T.model_path(a).exists() for a in T.MEMBERS)
    used = P.proxy_fit_count() + 1 + T.target_fit_count()
    assert used + missing_full + C.PLANNED_FITS['rebuild'] <= C.FIT_CAP, 'fit cap reached'
    t0 = time.time()
    match, item, fusion = T.inputs()
    models = {arm: T.full_model(arm) for arm in T.MEMBERS}
    summary = T.build_all(match, item, models)
    summary['fusion'] = fusion
    summary['reproduction'] = T.reproduce(match, item)
    summary['exp3_files'] = T.exp3_files()
    summary['files_written'] = [summary[ls]['file'] for ls in decision['adoption']['label_sets']]
    summary['fits_used'] = {'proxy': P.proxy_fit_count(), 'reproduction_check': 1, 'full_data': T.target_fit_count(),
                            'rebuild': summary['reproduction']['independent_fits']}
    C1.write_json(C.RESULTS / 'target_candidates.json', summary)
    keys = ('file', 'sha256', 'dead_share', 'possessions_per_min', 'mean_possession_s', 'possession_count', 'decoder')
    for ls in decision['adoption']['label_sets']:
        print(ls, json.dumps({k: summary[ls].get(k) for k in keys}))
    print('input:', fusion['input'], '| reproduction byte-identical:', summary['reproduction']['byte_identical'], f'| {time.time() - t0:.0f}s')
print('exp_5 proxy fits cached:', P.proxy_fit_count(), '| + 1 reproduction check')"""),
        ("code", "GUARD_FILE = 'write_guard_04.json'\n" + GUARD),
    ],
}

selected = sys.argv[1:] or list(NOTEBOOKS)
for name, cells in NOTEBOOKS.items():
    if name not in selected:
        continue
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "exp5", "display_name": "exp_5 (.venv-torch)", "language": "python"}
    notebook.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
                      for kind, src in cells]
    nbf.write(notebook, ROOT / name)
    print("wrote", name, len(cells), "cells")
