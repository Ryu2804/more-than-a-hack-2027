"""Write the exp_3 notebooks and their kernel spec (never reads executed outputs).

Pass notebook names to rebuild only those, so rebuilding one never wipes another's outputs.
"""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP3_PYTHON", sys.executable)
KERNEL = ROOT / "kernels" / "exp3"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_3 (.venv-torch)",
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
if not (ROOT / 'exp3_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_3')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)
import exp3_common as C
import exp1_common as C1
"""
SETUP = IMPORTS + """protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""
SETUP_FREEZE = IMPORTS + """if not C.PROTOCOL.exists():
    C.freeze_prereg()
    print('PREREG frozen now (before any exp_3 fit)')
protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""

INTRO = """# exp_3 — train on the competition's possession definition (pre-registered in `PREREG.md`)

**Question.** exp_1/exp_2 learn IDSSE's provider possession field (4.58 possessions/min, including
touch-level spells). CHALLENGE.md asks for *controlled* possession: a touch, a deflection or a won aerial
duel does not change possession. Does SetSeq trained, calibrated and decoded on a controlled-possession
relabel (`L_ctrl`) beat the provider-trained ensemble re-decoded on `L_ctrl` (H1)? And how do the
conclusions move under other label definitions (S1) and with more ball coverage (S2)?

**`L_ctrl`.** A team run is a controlled possession if it starts the half or follows a dead ball, lasts
≥ 2 s, or (if the event sync passes its pre-registered check) contains a controlled on-ball event;
otherwise it joins the previous team. The event sync failed its check, so `L_ctrl` = rules (a) + (b),
which equals `L_track(2 s)` (PREREG §2.3).

**What is new vs exp_0/exp_1/exp_2/rasya/ryu.**

| component | exp_3 | closest prior work |
|---|---|---|
| training / selection labels | `L_ctrl` (controlled possession, CHALLENGE.md) | exp_0–exp_2: provider `BallPossession`; ryu: event-derived possessions |
| label sensitivity (S1) | every arm against provider, `L_track(1/2/3 s)`, `L_ctrl`, re-selected per label set | none |
| coverage sensitivity (S2) | clean ball re-revealed on 12 % of hidden frames, saved weights | none |
| fusion | exp_0's geometric acceptance + a replay guard | exp_0: acceptance only |
"""

NOTEBOOKS = {
    "00_labels_protocol.ipynb": [
        ("markdown", INTRO),
        ("code", SETUP_FREEZE + "import exp3_protocol as P, exp3_diagnostics as G, exp3_labels as L"),
        ("markdown", "## 1. Label integrity\nRelabelings may only move team runs: D frames and frames outside the halves are unchanged. Under the fallback, `L_ctrl` must equal `L_track(2 s)` on every match."),
        ("code", """integrity = pd.DataFrame(P.label_integrity())
C1.write_json(C.RESULTS / 'label_integrity.json', integrity.to_dict(orient='records'))
print(integrity.to_string(index=False))
assert integrity.drop(columns='match').all().all()"""),
        ("markdown", "## 2. Event sync (IDSSE open event data)\nKick-off sync, then label-free snap to the acting player's closest approach to the ball. Acceptance (PREREG §2.3): pooled shot residual against the provider's `CalculatedFrame`, median |r| ≤ 5 and p90 |r| ≤ 12 frames."),
        ("code", """sync_rows, pooled = G.sync_report()
C1.write_json(C.RESULTS / 'sync_report.json', {'per_match': sync_rows, 'pooled': pooled, 'events_used': P.EVENTS_USED})
print(pd.DataFrame(sync_rows).round(3).to_string(index=False))
print('pooled:', json.dumps(pooled, indent=1))
assert P.EVENTS_USED == pooled['snapped']['accepted'], 'EVENTS_USED must follow the pre-registered sync rule'"""),
        ("markdown", "## 3. Label-only statistics (before any model is scored on `L_ctrl`)\nPossessions per minute of half time, mean possession duration, share of graded frames changed against the provider labels, agreement with `L_ctrl`."),
        ("code", """labels = pd.DataFrame(P.label_table())
labels.to_csv(C.RESULTS / 'label_stats.csv', index=False)
cols = ['possessions', 'possessions_per_min', 'mean_possession_s', 'frames_changed_share', 'agreement_with_ctrl', 'dead_share']
print(labels.pivot(index='match', columns='label_set', values='possessions_per_min').round(2).to_string())
pooled_rates = labels.groupby('label_set')[cols].mean()
pooled_rates['possessions_total'] = labels.groupby('label_set').possessions.sum()
print(); print(pooled_rates.round(4).to_string())
C1.write_json(C.RESULTS / 'label_stats_summary.json', pooled_rates.reset_index().to_dict(orient='records'))"""),
        ("markdown", "## 4. Re-derivation of the planning findings (cached `exp2_aug`, `target_like`, exp_2's own fold configs)\nNo fit and no re-selection. Oracles replace dead/alive or team with the provider truth."),
        ("code", """oracles = pd.DataFrame(G.oracle_rows())
print(oracles.round(4).to_string(index=False)); print('mean', oracles.drop(columns='match').mean().round(4).to_dict())
misses = G.miss_rows()
miss = G.miss_summary(misses)
print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in miss.items()}, indent=1))
scored = pd.DataFrame(G.label_score_rows())
table = scored.groupby('label_set')[['spell_f1', 'macro_f1', 'count_error_ratio']].mean()
print(table.round(4).to_string())
C1.write_json(C.RESULTS / 'planning_rederivation.json', {
    'oracles': oracles.to_dict(orient='records'), 'oracle_means': oracles.drop(columns='match').mean().to_dict(),
    'misses': miss, 'exp2_aug_vs_label_sets': table.reset_index().to_dict(orient='records')})"""),
        ("markdown", "## 5. S2 coverage view tokens\nClean ball re-revealed on one contiguous block of 12 % of each hidden-ball run of the `target_like` view (seeded, label-free)."),
        ("code", """t0 = time.time()
coverage = pd.DataFrame(P.build_coverage_tokens())
coverage.to_csv(C.RESULTS / 'coverage_stats.csv', index=False)
print(coverage.round(4).to_string(index=False))
print('mean', coverage.drop(columns='match').mean().round(4).to_dict(), f'| {time.time() - t0:.0f}s')
print('exp_3 fits so far:', P.exp3_fit_count(), '| fingerprints:', {a: P.fingerprint(a) for a in P.MODEL_ARMS})
print('exp_2 caches read in place:', P.EXP2_FINGERPRINTS)"""),
    ],
    "01_reference_on_ctrl.ipynb": [
        ("markdown", "# 01 — Zero-fit arms on `L_ctrl`\n\nCached exp_2 predictions (`exp1_ref`, `exp2_aug`, and the 3-seed ensemble `exp3_ref_ens3`) are re-selected (temperature + decoder, nested) and scored against `L_ctrl` on both views. The rule arms (baseline, rasya C06/C08) and ryu's shipped pipeline are re-run and scored against every label set. Nothing is fitted."),
        ("code", SETUP + "import exp3_protocol as P"),
        ("code", """cache = {}
for variant in ('exp3_ref_ens3', 'exp2_aug', 'exp1_ref'):
    for view in C.VIEWS:
        t0 = time.time()
        result = P.evaluate_variant(variant, view, C.EVAL_LABEL, cache)
        C1.write_json(C.RESULTS / f'variant_{variant}__{view}__{C.EVAL_LABEL}.json', result)
        frame = pd.DataFrame(result['rows'])
        print(f'== {variant:14s} {view:12s} L_ctrl mean spellF1 {frame.spell_f1.mean():.4f}  max|count err| {frame.count_error_ratio.abs().max():.3f}  ({time.time() - t0:.0f}s)')
        print('   final config:', {k: round(v, 3) for k, v in result['final_config'].items()})"""),
        ("code", """t0 = time.time()
rows = []
for view in C.VIEWS:
    rows += P.score_reference(P.reference_predictions(view), view, ('ctrl', 'provider', 'track_1', 'track_3'))
reference = pd.DataFrame(rows)
reference.to_csv(C.RESULTS / 'reference_rows.csv', index=False)
print(reference.groupby(['label_set', 'view', 'variant'])[['spell_f1', 'macro_f1', 'count_error_ratio']].mean().round(4).to_string())
print(f'{time.time() - t0:.0f}s')"""),
    ],
    "02_fits.ipynb": [
        ("markdown", "# 02 — `exp3_ctrl` fits (3 seeds × 28 nested fits)\n\nexp2_aug recipe (round 0 original + rounds 1–2 `target_aug`, mirror, `TrainConfig()`) with `L_ctrl` training targets. Each fit predicts both views of its held-out matches and saves its weights. Cached by fingerprint (includes the `L_ctrl` code hash); progress in `results/fit_log.txt`."),
        ("code", SETUP + "import exp3_protocol as P, exp1_net as N\nprint('device', N.DEVICE, '| config', N.TrainConfig().as_dict())"),
        ("code", """log_path = C.RESULTS / 'fit_log.txt'
def log(message, **_):
    print(message, flush=True)
    with log_path.open('a') as handle:
        handle.write(message + '\\n')
t0 = time.time()
for arm in P.MODEL_ARMS:
    start = time.time()
    P.ensure_predictions(arm, log=log)
    log(f'ARM DONE {arm} {time.time() - start:.0f}s')
log(f'ALL FITS DONE {time.time() - t0:.0f}s, exp_3 fits cached {P.exp3_fit_count()}')"""),
    ],
    "03_evaluation.ipynb": [
        ("markdown", "# 03 — Nested evaluation: every variant × view × label set\n\nFor each label set (`L_ctrl` first; then S1: provider, `L_track(1 s)`, `L_track(3 s)`; `L_track(2 s)` ≡ `L_ctrl`) temperature and decoder are re-selected on the six inner validation matches against that label set, and the held-out match is scored once against it. Results already written by notebook 01 (same code path) are reused. Evaluation never fits: a missing prediction is an error."),
        ("code", SETUP + "import exp3_protocol as P, exp3_report as R"),
        ("code", """cache = {}
for label_set in R.S1_LABELS:
    for variant in P.VARIANTS:
        for view in C.VIEWS:
            path = C.RESULTS / f'variant_{variant}__{view}__{label_set}.json'
            if path.exists():
                result = C1.read_json(path); tag = 'cached'
            else:
                t0 = time.time()
                result = P.evaluate_variant(variant, view, label_set, cache)
                C1.write_json(path, result); tag = f'{time.time() - t0:.0f}s'
            frame = pd.DataFrame(result['rows'])
            print(f'{label_set:8s} {view:12s} {variant:15s} spellF1 {frame.spell_f1.mean():.4f}  max|cnt| {frame.count_error_ratio.abs().max():.3f}  gamma {sorted(set(frame.gamma))}  ({tag})', flush=True)"""),
        ("markdown", "## Per-match detail on `L_ctrl` (held-out matches)"),
        ("code", """keep = ['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'count_error_ratio', 'temperature', 'lambda_team', 'lambda_dead', 'gamma']
for variant in ('exp3_ctrl_ens3', 'exp3_ref_ens3'):
    for view in C.VIEWS:
        frame = pd.DataFrame(R.variant_rows(variant, view, 'ctrl'))
        print(f'== {variant} / {view}'); print(frame[keep].round(3).to_string(index=False))"""),
        ("markdown", "## Diagnostic: argmax (no decoder), `L_ctrl`"),
        ("code", """argmax = pd.concat([pd.DataFrame(P.argmax_rows(v, view, 'ctrl', cache))
                    for v in ('exp3_ctrl_ens3', 'exp3_ref_ens3') for view in C.VIEWS], ignore_index=True)
argmax.to_csv(C.RESULTS / 'argmax_rows.csv', index=False)
print(argmax.groupby(['variant', 'view'])[['spell_f1', 'macro_f1', 'count_error_ratio']].mean().round(4).to_string())"""),
        ("markdown", "## S2: coverage view (clean ball re-revealed on 12 % of hidden frames), saved weights, fold configs from `target_like`"),
        ("code", """t0 = time.time()
rows = []
for variant in ('exp3_ctrl_ens3', 'exp3_ref_ens3', 'exp3_ctrl', 'exp2_aug'):
    for label_set in ('ctrl', 'provider'):
        selections = C1.read_json(C.RESULTS / f'variant_{variant}__target_like__{label_set}.json')['selections']
        rows += P.evaluate_coverage(variant, label_set, selections)
coverage = pd.DataFrame(rows)
coverage.to_csv(C.RESULTS / 'coverage_rows.csv', index=False)
print(coverage.groupby(['label_set', 'variant'])[['spell_f1', 'macro_f1', 'count_error_ratio']].mean().round(4).to_string())
print(f'{time.time() - t0:.0f}s')"""),
    ],
    "04_comparison.ipynb": [
        ("markdown", "# 04 — Comparison and pre-registered decisions\n\nPaired match bootstrap (1000 draws, seed 20261108, 95 %). Primary: `L_ctrl`, `target_like`. Everything else is report-only (PREREG §6)."),
        ("code", SETUP + "import exp3_protocol as P, exp3_report as R\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"),
        ("code", """parts = [pd.read_csv(C.RESULTS / 'reference_rows.csv'), pd.read_csv(C.RESULTS / 'argmax_rows.csv')]
for label_set in R.S1_LABELS:
    for variant in P.VARIANTS:
        for view in C.VIEWS:
            parts.append(pd.DataFrame(R.variant_rows(variant, view, label_set)))
rows = pd.concat(parts, ignore_index=True)
rows.to_csv(C.RESULTS / 'per_match_results.csv', index=False)
table = R.summary_table(rows)
table.to_csv(C.RESULTS / 'comparison_table.csv', index=False)
cols = ['variant', 'mean_spell_f1', 'min_spell_f1', 'mean_macro_f1', 'mean_abs_count_error', 'max_abs_count_error',
        'delta_vs_ref_ens3', 'ci95_low_vs_ref', 'ci95_high_vs_ref', 'wins_vs_ref', 'delta_vs_exp2_aug', 'ci95_low_vs_exp2', 'ci95_high_vs_exp2', 'count_gate']
for label_set in R.S1_LABELS:
    for view in C.VIEWS:
        part = table[(table.label_set == label_set) & (table.view == view)]
        print(f'== {label_set} / {view}'); print(R.as_markdown(part, cols)); print()"""),
        ("markdown", "## exp_0 quality gates on `L_ctrl` (vs the same view's baseline)"),
        ("code", """gate_rows = R.gate_rows(rows)
print(pd.DataFrame([{'view': g['view'], 'variant': g['variant'], **g['checks'], 'pass': g['pass'],
                     'worst_count_error': round(g['worst_count_error'], 3), 'worst_match': g['worst_count_match']}
                    for g in gate_rows]).to_string(index=False))"""),
        ("markdown", "## Pre-registered decision (H1) and the adoption tree"),
        ("code", """decision = R.decide(rows)
C1.write_json(C.RESULTS / 'decision.json', decision)
h1 = decision['rules']['H1']
print(f"H1 {'PASSED' if h1['passed'] else 'FAILED'}: exp3_ctrl_ens3 - exp3_ref_ens3 (L_ctrl, target_like) = {h1['mean_delta']:+.4f} [{h1['ci95'][0]:+.4f}, {h1['ci95'][1]:+.4f}], wins {h1['wins']}/7")
print('candidate:', decision['candidate_variant'], '| final config:', {k: round(v, 3) for k, v in decision['candidate_final_config'].items()})
ro = decision['report_only']
for item in [ro['H1_original_view']] + ro['ctrl_ens3_vs_exp2_aug'] + ro['ref_ens3_vs_exp2_aug'] + ro['S1_ctrl_ens3_vs_ref_ens3']:
    print(f"  {item['label_set']:8s} {item['view']:12s} {item['a']:15s} - {item['b']:14s} {item['mean_delta']:+.4f} [{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}] wins {item['wins']}")
print('seed spread (L_ctrl):', {k: round(v, 4) for k, v in ro['seed_spread'].items()})"""),
        ("markdown", "## S2: coverage − target_like (paired by match)"),
        ("code", """coverage = pd.read_csv(C.RESULTS / 'coverage_rows.csv')
shift = R.coverage_shift(rows, coverage)
print(pd.DataFrame(shift).round(4).to_string(index=False))
report = {'prereg_sha256': protocol['prereg_sha256'], 'bootstrap': 'paired match bootstrap, 1000 draws, seed 20261108, 95% percentile',
          'table': R.records(table), 'gates_ctrl': gate_rows, 'decision': decision, 'coverage_shift': shift,
          'coverage_stats': pd.read_csv(C.RESULTS / 'coverage_stats.csv').to_dict(orient='records')}
C1.write_json(C.RESULTS / 'proxy_report.json', report)"""),
        ("markdown", "## Figure (proxy only)"),
        ("code", """order = ['baseline', 'ryu_final', 'exp1_ref', 'exp2_aug', 'exp3_ref_ens3', 'exp3_ctrl', 'exp3_ctrl_ens3']
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for label_set, marker in zip(R.S1_LABELS, ('o', 's', '^', 'v')):
    part = table[(table.label_set == label_set) & (table.view == 'target_like')].set_index('variant').reindex(order)
    axes[0].plot(order, part.mean_spell_f1, marker=marker, label=label_set)
    axes[1].plot(order, part.mean_count_error, marker=marker, label=label_set)
axes[0].set_ylabel('mean spellF1, target_like (7 held-out matches)'); axes[0].set_title('spellF1 by arm and evaluation label set')
axes[1].axhline(0, color='0.5', lw=0.8); axes[1].set_ylabel('mean count error (pred / true - 1)'); axes[1].set_title('count bias')
for ax in axes:
    ax.tick_params(axis='x', rotation=35); ax.legend(title='labels')
fig.tight_layout(); fig.savefig(C.RESULTS / 'fig_exp3_labels.png', dpi=130)
print('saved', C.RESULTS / 'fig_exp3_labels.png')"""),
    ],
    "05_target.ipynb": [
        ("markdown", "# 05 — BAR-ATH: fusion, plausibility, candidate (aggregates only)\n\nFull-data members (exp_3: 3 new fits; exp_2: cached), broadcast fusion with the pinned exp_0 rule and the replay guard, label-free plausibility aggregates (never a choice), the adoption tree's candidate CSV (official validator) and its byte-identical rebuild from independent fits. No public-window check."),
        ("code", SETUP + "import exp3_protocol as P, exp3_target as E\nassert C.public_check_spent(), 'exp_1 public check marker missing'"),
        ("code", """t0 = time.time()
match, items, fusion = E.inputs()
C1.write_json(C.RESULTS / 'target_fusion.json', fusion)
print(json.dumps({k: v for k, v in fusion.items() if k != 'halves'}, indent=1))
for half in fusion['halves']:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in half.items()})
print(f'{time.time() - t0:.0f}s')"""),
        ("code", """t0 = time.time()
models = {arm: E.full_model(arm) for members in E.MEMBERS.values() for arm in members}
plausibility = E.plausibility(match, items, models)
C1.write_json(C.RESULTS / 'target_plausibility.json', plausibility)
keys = ('dead_share', 'possessions_per_min', 'mean_possession_s', 'possession_count', 'dead_share_minus_prior')
print(pd.DataFrame({k: {m: v.get(m) for m in keys} for k, v in plausibility.items() if isinstance(v, dict)}).T.round(3).to_string())
print('prior dead share', plausibility['published_dead_share_prior'], '| ryu possessions/min', plausibility['ryu_possessions_per_min'])
print(f'{time.time() - t0:.0f}s')"""),
        ("code", """t0 = time.time()
summary = E.build_candidate(match, items, models, fusion)
summary['reproduction'] = E.reproduce(match, items, fusion)
C1.write_json(C.RESULTS / 'target_candidate.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'share_graded'}, indent=1))
print(f'{time.time() - t0:.0f}s')"""),
    ],
}

NOTEBOOKS["06_touch.ipynb"] = [
    ("markdown", "# 06 — S3 (stretch, report only): touch features\n\nFour sub-step ball-contact features per 5 fps step (per-team min player–ball distance over the 5 frames behind the step, peak change of ball velocity, team of the nearest player at that peak; PREREG §5.1) appended to exp_1's global vector. One seed (20261108), `exp3_ctrl` recipe and `L_ctrl` targets, 28 nested fits. Compared with `exp3_ctrl` seed 0. Never changes the candidate."),
    ("code", SETUP + "import exp3_protocol as P, exp3_touch as S, exp3_report as R, exp2_report as R2\nassert all(len(list(P.prediction_folder(a).glob('*.npz'))) == 28 for a in P.MODEL_ARMS), 'primary fits must finish first'"),
    ("code", """t0 = time.time()
features = pd.DataFrame(S.build_feature_cache())
features.to_csv(C.RESULTS / 's3_feature_stats.csv', index=False)
print(features.groupby('kind').mean(numeric_only=True).round(4).to_string(), f'| {time.time() - t0:.0f}s')"""),
    ("code", """log_path = C.RESULTS / 'fit_log.txt'
def log(message, **_):
    print(message, flush=True)
    with log_path.open('a') as handle:
        handle.write(message + '\\n')
t0 = time.time()
predictions = S.ensure_predictions(log=log)
log(f'ARM DONE {S.ARM} {time.time() - t0:.0f}s, fingerprint {S.fingerprint()}')"""),
    ("code", """rows, deltas = [], []
for label_set in ('ctrl', 'provider'):
    for view in C.VIEWS:
        result = P.evaluate_predictions(predictions, S.ARM, view, label_set)
        C1.write_json(C.RESULTS / f'variant_{S.ARM}__{view}__{label_set}.json', result)
        both = pd.DataFrame(result['rows'] + R.variant_rows('exp3_ctrl', view, label_set))
        d = R2.delta(both, view, S.ARM, 'exp3_ctrl')
        deltas.append({'label_set': label_set, **d})
        rows += result['rows']
        print(f"{label_set:8s} {view:12s} touch {pd.DataFrame(result['rows']).spell_f1.mean():.4f} | delta vs exp3_ctrl {d['mean_delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] wins {d['wins']}")
C1.write_json(C.RESULTS / 's3_touch.json', {'arm': S.ARM, 'fingerprint': S.fingerprint(), 'deltas_vs_exp3_ctrl_seed0': deltas,
                                            'rows': rows, 'report_only': True})"""),
]

NOTEBOOKS["07_hedge_provider.ipynb"] = [
    ("markdown", "# 07 — Post-hoc hedge (NOT pre-registered): provider-selected decode of the same ensemble\n\nThe adopted candidate (`exp3_ref_ens3`, notebook 05) is decoded for `L_ctrl`. Because BAR-ATH's ground-truth definition is unknown (REPORT §7), this notebook writes a second file from the *same* cached exp_2 ensemble, decoded with the config selected on the 7 `target_like` out-of-fold predictions against **provider** labels. It does not change the pre-registered candidate. Aggregates only; validator; byte-identical rebuild from 3 independent fits; no public check."),
    ("code", SETUP + "import exp3_target as E\nassert C.public_check_spent(), 'exp_1 public check marker missing'"),
    ("code", """t0 = time.time()
match, items, fusion = E.inputs()
models = {arm: E.full_model(arm) for arm in E.MEMBERS[E.HEDGE_VARIANT]}
summary = E.build_hedge(match, items, models, fusion)
summary['reproduction'] = E.reproduce_hedge(match, items, fusion)
C1.write_json(C.RESULTS / 'target_hedge_provider.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'share_graded'}, indent=1))
print(f'{time.time() - t0:.0f}s')"""),
]

selected = sys.argv[1:] or list(NOTEBOOKS)
for name, cells in NOTEBOOKS.items():
    if name not in selected:
        continue
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "exp3", "display_name": "exp_3 (.venv-torch)", "language": "python"}
    notebook.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
                      for kind, src in cells]
    nbf.write(notebook, ROOT / name)
    print("wrote", name, len(cells), "cells")
