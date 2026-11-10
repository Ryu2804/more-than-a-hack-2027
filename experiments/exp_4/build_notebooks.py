"""Write the exp_4 notebooks and their kernel spec (never reads executed outputs).

Pass notebook names to rebuild only those, so rebuilding one never wipes another's outputs.
"""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP4_PYTHON", sys.executable)
KERNEL = ROOT / "kernels" / "exp4"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_4 (.venv-torch)",
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
ROOT = Path.cwd()
if not (ROOT / 'exp4_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_4')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)
import exp4_common as C
import exp1_common as C1
"""
SETUP = IMPORTS + """protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""
SETUP_FREEZE = IMPORTS + """if not C.PROTOCOL.exists():
    C.freeze_prereg()
    print('PREREG frozen now (before any exp_4 fit)')
protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""
LOG = """log_path = C.RESULTS / 'fit_log.txt'
def log(message, **_):
    print(message, flush=True)
    with log_path.open('a') as handle:
        handle.write(message + '\\n')
"""

INTRO = """# exp_4 — SetSeq + touch features + auxiliary hidden-ball head (pre-registered in `PREREG.md`)

**Question (H1).** exp_3 showed that team assignment, not dead/alive, is the bottleneck: short possessions are
missed even with the ball visible, and half the misses sit in ball gaps. Does SetSeq with the 4 sub-step touch
features **and** a training-only auxiliary head that predicts the hidden clean ball and the team nearest to it
(BallRadar-style) beat exp_2's 3-seed ensemble (`exp3_ref_ens3`) on provider labels, while staying non-inferior
on `L_ctrl`? 3 seeds per arm, provider training labels, nested leave-one-match-out on 7 degraded IDSSE matches.

| component | exp_4 | closest prior work |
|---|---|---|
| auxiliary supervision | clean ball (x, y) + nearest team, training matches only, unused at inference | none in the team; BallRadar (Kim et al., KDD 2023) predicts the ball from players |
| touch features | exp_3's 4 features, now 3 seeds on provider labels | exp_3 S3: 1 seed, `L_ctrl` labels |
| overfitting controls | frozen prereg, fixed `TrainConfig()`, NLL generalisation gap per fit, seed honesty | exp_1–exp_3: prereg + nested selection |
"""

NOTEBOOKS = {
    "00_protocol.ipynb": [
        ("markdown", INTRO),
        ("code", SETUP_FREEZE + "import exp4_protocol as P, exp4_aux as A, exp2_protocol as P2, exp3_protocol as P3, exp3_touch as S"),
        ("markdown", "## 1. Unit tests (synthetic arrays)\nBit-exact equivalence with `exp1_net.train` (aux off, no touch), aux targets from training matches only, mirror rules, `fit=False` refusal, job locks."),
        ("code", """import subprocess
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'],
                        capture_output=True, text=True, cwd=ROOT)
print(result.stdout.strip().splitlines()[-1])
assert result.returncode == 0, 'unit tests failed'"""),
        ("markdown", "## 2. Inputs read in place\nexp_2 caches (fingerprint-checked), exp_3 touch-feature caches, and the provider training targets: the cached tokens' own state / boundary must equal `exp3_protocol.step_labels(·, 'provider')` on every training array."),
        ("code", """for arm, fp in P.EXP2_FINGERPRINTS.items():
    missing = [k for k, *_ in P.P1.job_keys() if not (P2.prediction_folder(arm) / f'{k}.npz').exists()]
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
assert all(same)"""),
        ("markdown", "## 3. Auxiliary targets (clean IDSSE data)\nOn every training array and both held-out views, where the observed ball is visible the aux ball target must equal the observed `ball_x` / `ball_y` globals up to float16 rounding (this checks the normalisation and the flip sign). Reported: share of hidden-ball steps that get a ball target, and how often the clean nearest team equals the provider possession team on alive steps (descriptive only)."),
        ("code", """rows = []
for name in C.MATCH_IDS:
    arrays = [(kind, r, P2.load_tokens(name, kind, r)) for r, kind in P.RECIPE]
    arrays += [(view, 0, P2.view_tokens(name, view)) for view in C.VIEWS]
    for kind, r, tokens in arrays:
        rows.append({'match': name, 'kind': kind, 'round': r, **A.consistency(tokens, name, r)})
    rows[-1]['team_agrees_with_provider_alive'] = A.team_agreement(name, P2.load_tokens(name)['state'])
aux = pd.DataFrame(rows)
aux.to_csv(C.RESULTS / 'aux_integrity.csv', index=False)
print(aux.groupby('kind')[['visible_steps', 'max_abs_diff_visible', 'hidden_steps', 'hidden_with_ball_target', 'team_target_defined', 'steps_sign_differs_from_clean']].agg(['mean', 'max']).round(4).to_string())
print('team agreement on alive steps (mean of 7):', round(aux.team_agrees_with_provider_alive.mean(), 4))
assert aux.max_abs_diff_visible.max() <= 2e-3 and (aux.visible_without_clean_ball == 0).all()"""),
        ("markdown", "## 4. Reference code path\n`exp3_ref_ens3` is read from exp_3's stored nested results. One of them is recomputed here with the same code and must match exactly."),
        ("code", """t0 = time.time()
fresh = P3.evaluate_predictions(P.predictions_for('ref_ens3', {}), 'exp3_ref_ens3', 'target_like', 'provider')
reused = P.reused_result('exp3_ref_ens3', 'target_like', 'provider')
check = {'rows_equal': fresh['rows'] == reused['rows'], 'final_config_equal': fresh['final_config'] == reused['final_config'],
         'mean_spell_f1': float(np.mean([r['spell_f1'] for r in fresh['rows']])), 'seconds': round(time.time() - t0, 1)}
C1.write_json(C.RESULTS / 'reference_recheck.json', check)
print(check)
assert check['rows_equal'] and check['final_config_equal']"""),
        ("code", """print('exp_4 fits so far:', P.fit_count())
print(pd.DataFrame(P.spec_table()).to_string(index=False))"""),
    ],
    "01_fits.ipynb": [
        ("markdown", "# 01 — Fits (3 arms × 3 seeds × 28 nested fits)\n\n`exp2_aug` recipe, provider targets, `TrainConfig()`. Order: `exp4_full` (3 seeds), `exp4_touch`, `exp4_aux`. Each fit saves its weights, its final training NLL, and predicts both views of its held-out matches. Cached by fingerprint; a per-job lock lets an optional second process (`fit_worker.py`) share the work without fitting any job twice. Progress in `results/fit_log.txt`."),
        ("code", SETUP + "import exp4_protocol as P, exp1_net as N\nprint('device', N.DEVICE, '| config', N.TrainConfig().as_dict())"),
        ("markdown", "## Reproduction check (1 fit, report only)\nexp_4's training loop with aux off and no touch columns, exp2_aug's recipe and seed, outer fold of the first match, against exp_2's cached `exp2_aug` prediction. Bit-exact expected; max |Δp| > 1e-3 stops the run."),
        ("code", LOG + """path = C.RESULTS / 'repro_check.json'
if path.exists():
    repro = C1.read_json(path)
else:
    t0 = time.time()
    repro = P.reproduction_check()
    repro['seconds'] = round(time.time() - t0, 1)
    C1.write_json(path, repro)
    log(f"REPRO CHECK {repro['held_out']} bit_exact={repro['bit_exact']} {repro['seconds']}s")
print(json.dumps(repro, indent=1))
assert max(repro[v]['max_abs_probability_diff'] for v in C.VIEWS) <= 1e-3"""),
        ("code", """t0 = time.time()
for arm in P.FIT_ORDER:
    remaining = sum(1 for k, *_ in P.P1.job_keys() if not (P.prediction_folder(arm) / f'{k}.npz').exists())
    assert P.fit_count() + 1 + remaining <= C.FIT_CAP, 'fit cap reached'
    start = time.time()
    P.ensure_predictions(arm, log=log)
    log(f'ARM DONE {arm} {time.time() - start:.0f}s, fits cached {P.fit_count()}')
log(f'ALL FITS DONE {time.time() - t0:.0f}s, exp_4 proxy fits cached {P.fit_count()} (+1 reproduction check)')"""),
        ("code", """train = pd.DataFrame([r for arm in P.ARMS for r in P.train_nll_rows(arm)])
print(train.groupby('arm').train_nll.agg(['mean', 'min', 'max', 'count']).round(4).to_string())"""),
    ],
    "02_evaluation.ipynb": [
        ("markdown", "# 02 — Nested evaluation (never fits)\n\nEnsembles on every S1 label set (provider, `L_track(1 s)`, `L_ctrl`, `L_track(3 s)`) and both views; single seeds on provider and `L_ctrl`, both views. Temperature and decoder are re-selected on the six inner validation matches against each label set; the held-out match is scored once. `exp3_ref_ens3` / `exp2_aug` come from exp_3's stored results. Then argmax diagnostics, recall of possessions < 2 s, and the NLL generalisation gap."),
        ("code", SETUP + "import exp4_protocol as P, exp4_diagnostics as G\nfor arm in P.ARMS:\n    missing = [k for k, *_ in P.P1.job_keys() if not (P.prediction_folder(arm) / f'{k}.npz').exists()]\n    assert not missing, f'{arm}: {len(missing)} jobs missing; run 01_fits first'"),
        ("code", """import multiprocessing
from concurrent.futures import ProcessPoolExecutor
tasks = [(v, view, ls) for v in P.ENSEMBLE_VARIANTS for ls in C.S1_LABELS for view in C.VIEWS]
tasks += [(arm, view, ls) for arm in P.ARMS for ls in C.DECISION_LABELS for view in C.VIEWS]
t0 = time.time()
with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
    for variant, view, label_set, seconds, tag in pool.map(P.evaluate_and_save, tasks):
        frame = pd.DataFrame(C1.read_json(P.result_path(variant, view, label_set))['rows'])
        print(f'{label_set:8s} {view:12s} {variant:16s} spellF1 {frame.spell_f1.mean():.4f}  max|cnt| {frame.count_error_ratio.abs().max():.3f}  gamma {sorted(set(frame.gamma))}  ({tag} {seconds:.0f}s)', flush=True)
print(f'{len(tasks)} evaluations, {time.time() - t0:.0f}s')"""),
        ("markdown", "## Diagnostics: argmax (no decoder) and recall of true possessions < 2 s (provider labels)"),
        ("code", """cache = {}
variants = (*P.ENSEMBLE_VARIANTS, 'exp3_ref_ens3')
argmax = pd.DataFrame([r for v in variants for ls in C.DECISION_LABELS for view in C.VIEWS
                       for r in G.argmax_rows(v, view, ls, cache)])
argmax.to_csv(C.RESULTS / 'argmax_rows.csv', index=False)
print(argmax.groupby(['label_set', 'view', 'variant'])[['spell_f1', 'count_error_ratio']].mean().round(4).to_string())
recall = pd.DataFrame([r for v in (*variants, 'exp2_aug') for view in C.VIEWS for r in G.short_recall_rows(v, view, cache)])
recall.to_csv(C.RESULTS / 'short_recall.csv', index=False)
print(recall.groupby(['view', 'variant'])[['recall_short_decoded', 'recall_short_argmax', 'recall_all_decoded']].mean().round(4).to_string())"""),
        ("markdown", "## Generalisation gap (provider labels, raw p, T = 1)\nTraining NLL of every fit (exp_4: logged at fit time; exp_2 recipe: recomputed from its saved weights, inference only) against the held-out NLL of its outer and pair predictions."),
        ("code", """t0 = time.time()
train = pd.DataFrame([r for arm in P.ARMS for r in P.train_nll_rows(arm)])
ref_path = C.RESULTS / 'ref_train_nll.csv'
if not ref_path.exists():
    pd.DataFrame([r for arm in P.EXP2_ARMS for r in G.reference_train_nll(arm)]).to_csv(ref_path, index=False)
train = pd.concat([train, pd.read_csv(ref_path)], ignore_index=True)
heldout = pd.DataFrame([r for arm in (*P.ARMS, *P.EXP2_ARMS) for r in G.heldout_nll_rows(arm, cache)])
train.to_csv(C.RESULTS / 'train_nll.csv', index=False)
heldout.to_csv(C.RESULTS / 'heldout_nll.csv', index=False)
gap = G.gap_table(train, heldout)
gap.to_csv(C.RESULTS / 'gap_table.csv', index=False)
print(gap.round(4).to_string(index=False))
print(f'{time.time() - t0:.0f}s')"""),
    ],
    "03_comparison.ipynb": [
        ("markdown", "# 03 — Comparison and the pre-registered decision\n\nPaired match bootstrap (1000 draws, seed 20261108, 95 %). H1: `exp4_full_ens3` − `exp3_ref_ens3`, `target_like`; provider CI lower bound > 0 **and** `L_ctrl` CI lower bound > −0.01, each with its own nested re-selection. Everything else is report-only."),
        ("code", SETUP + "import exp4_protocol as P, exp4_report as R\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"),
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
        ("code", """show = ('exp4_full_ens3', 'exp3_ref_ens3', 'exp4_touch_ens3', 'exp4_aux_ens3', 'exp2_aug')
for label_set in C.DECISION_LABELS:
    print(f'== {label_set}'); print(R.per_match(rows, label_set, 'target_like', show).to_string()); print()"""),
        ("markdown", "## exp_0 quality gates (vs the same view's baseline), provider and `L_ctrl`"),
        ("code", """gates = R.gate_rows(rows, 'provider') + R.gate_rows(rows, 'ctrl')
print(pd.DataFrame([{'label_set': g['label_set'], 'view': g['view'], 'variant': g['variant'], **g['checks'], 'pass': g['pass'],
                     'worst_count_error': round(g['worst_count_error'], 3), 'worst_match': g['worst_count_match']}
                    for g in gates]).to_string(index=False))"""),
        ("markdown", "## Pre-registered decision (H1), seed honesty, adoption tree"),
        ("code", """decision = R.decide(rows)
C1.write_json(C.RESULTS / 'decision.json', decision)
for name, rule in decision['rules'].items():
    print(f"{name:26s} {rule['mean_delta']:+.4f} [{rule['ci95'][0]:+.4f}, {rule['ci95'][1]:+.4f}] wins {rule['wins']}/7 -> {'pass' if rule['passed'] else 'FAIL'}")
print('H1', 'PASSED' if decision['H1']['passed'] else 'FAILED', '|', decision['seed_honesty']['statement'],
      '| seed spread', decision['seed_honesty']['exp4_full_seed_spread_provider_target_like'])
print('adoption:', json.dumps(decision['adoption']))
ro = decision['report_only']
for item in ro['H1_original_view'] + ro['ablations_vs_reference'] + ro['full_vs_ablations'] + ro['S1_full_vs_reference'] + ro['single_seeds_vs_reference']:
    print(f"  {item['label_set']:8s} {item['view']:12s} {item['a']:16s} - {item['b']:16s} {item['mean_delta']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}] wins {item['wins']}")
print(pd.DataFrame([{k: v for k, v in s.items() if k != 'seeds'} for s in ro['seed_spreads']]).round(4).to_string(index=False))"""),
        ("markdown", "## Generalisation gap and short-possession recall (report only)"),
        ("code", """gap = pd.read_csv(C.RESULTS / 'gap_table.csv')
print(gap[gap.level == 'family'].round(4).to_string(index=False))
recall = pd.read_csv(C.RESULTS / 'short_recall.csv')
recall_summary = recall.groupby(['view', 'variant'])[['recall_short_decoded', 'recall_short_argmax', 'recall_all_decoded']].mean()
print(recall_summary.round(4).to_string())
report = {'prereg_sha256': protocol['prereg_sha256'], 'bootstrap': 'paired match bootstrap, 1000 draws, seed 20261108, 95% percentile',
          'table': R.records(table), 'gates': gates, 'decision': decision,
          'gap_table': R.records(gap), 'short_recall': R.records(recall_summary.reset_index()),
          'repro_check': C1.read_json(C.RESULTS / 'repro_check.json')}
C1.write_json(C.RESULTS / 'proxy_report.json', report)"""),
        ("markdown", "## Figure (proxy only)"),
        ("code", """order = ['exp3_ref_ens3', 'exp4_touch_ens3', 'exp4_aux_ens3', 'exp4_full_ens3']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label_set, marker in zip(C.S1_LABELS, ('o', 's', '^', 'v')):
    part = table[(table.label_set == label_set) & (table.view == 'target_like')].set_index('variant').reindex(order)
    axes[0].plot(order, part.mean_spell_f1, marker=marker, label=label_set)
    axes[1].plot(order, part.delta_vs_ref_ens3, marker=marker, label=label_set)
axes[0].set_ylabel('mean spellF1, target_like (7 held-out matches)'); axes[0].set_title('spellF1 by arm and label set')
axes[1].axhline(0, color='0.5', lw=0.8); axes[1].set_ylabel('Δ vs exp3_ref_ens3'); axes[1].set_title('paired Δ (point estimate)')
for ax in axes:
    ax.tick_params(axis='x', rotation=25); ax.legend(title='labels')
fig.tight_layout(); fig.savefig(C.RESULTS / 'fig_exp4.png', dpi=130)
print('saved', C.RESULTS / 'fig_exp4.png')"""),
    ],
    "04_target.ipynb": [
        ("markdown", "# 04 — BAR-ATH (only if H1 passed; aggregates only)\n\nIf H1 failed, nothing is read or written for BAR-ATH. If it passed: 3 full-data `exp4_full` fits, exp_3's fusion rule (tactical input if rejected), two candidate files (decoder selected on provider labels / on `L_ctrl`), official validator, byte-identical rebuild from 3 independent fits, label-free plausibility. No public-window check."),
        ("code", SETUP + "import exp4_protocol as P\nassert C.public_check_spent(), 'exp_1 public check marker missing'"),
        ("code", """decision = C1.read_json(C.RESULTS / 'decision.json')
if not decision['H1']['passed']:
    summary = {'H1_passed': False, 'files_written': [], 'bar_ath_read': False,
               'statement': decision['adoption']['statement']}
    C1.write_json(C.RESULTS / 'target_none.json', summary)
    print(json.dumps(summary, indent=1))
else:
    import exp4_target as E
    t0 = time.time()
    match, item, fusion = E.inputs()
    models = {arm: E.full_model(arm) for arm in E.MEMBERS}
    summary = E.build_all(match, item, models)
    summary['fusion'] = {k: v for k, v in fusion.items() if k != 'halves'}
    summary['reproduction'] = E.reproduce(match, item)
    summary['exp3_files'] = E.exp3_files()
    C1.write_json(C.RESULTS / 'target_candidates.json', summary)
    keys = ('file', 'sha256', 'dead_share', 'possessions_per_min', 'mean_possession_s', 'possession_count', 'decoder')
    for ls in E.LABEL_SETS:
        print(ls, json.dumps({k: summary[ls].get(k) for k in keys}))
    print('reproduction byte-identical:', summary['reproduction']['byte_identical'], f'| {time.time() - t0:.0f}s')
print('exp_4 proxy fits cached:', P.fit_count(), '| + 1 reproduction check')"""),
    ],
}

selected = sys.argv[1:] or list(NOTEBOOKS)
for name, cells in NOTEBOOKS.items():
    if name not in selected:
        continue
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "exp4", "display_name": "exp_4 (.venv-torch)", "language": "python"}
    notebook.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
                      for kind, src in cells]
    nbf.write(notebook, ROOT / name)
    print("wrote", name, len(cells), "cells")
