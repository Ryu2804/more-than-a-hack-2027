"""Write the exp_2 notebooks and their kernel spec (never reads executed outputs).

Pass notebook names to rebuild only those, so rebuilding one never wipes another's outputs.
"""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP2_PYTHON", sys.executable)
KERNEL = ROOT / "kernels" / "exp2"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_2 (.venv-torch)",
    "language": "python",
    "env": {
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(ROOT / "cache/numba"),
        "MPLCONFIGDIR": str(ROOT / "cache/matplotlib"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "8",
    },
}, indent=2))

SETUP = """from pathlib import Path
import os, sys, json, time
ROOT = Path.cwd()
if not (ROOT / 'exp2_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_2')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
import exp2_common as C
protocol = C.verify_prereg()
print('PREREG verified', protocol['prereg_sha256'][:16], '| frozen', protocol['frozen_at_utc'])
"""

INTRO = """# exp_2 — transfer-robust SetSeq (pre-registered in `PREREG.md`)

**Question.** exp_1's SetSeq wins on the in-domain proxy (0.656 spellF1), but IDSSE parks the ball off
the pitch during dead time (off-pitch on 26 % of visible frames vs 4.2 % in BAR-ATH). Does the model
lean on a dead-ball cue that BAR-ATH does not have, and does *target-like observation augmentation*
fix that without losing in-domain accuracy?

**What is new vs exp_0 / exp_1 / rasya / ryu**

| component | exp_2 | closest prior work |
|---|---|---|
| evaluation | second, label-free **`target_like` view** of the same held-out matches (off-pitch ball hidden after 1.48 s; ball hidden on long slow-play runs) applied to *every* arm | exp_0/exp_1: one in-domain view; ryu: other sources |
| training | **`target_aug`**: two of three corruption rounds lose off-pitch tails (random length) and, in half of the long dead spells, the ball | exp_1: corruption resampling + mirror only |
| decoder (H2) | **explicit-duration HSMM** with entropy-centred log-normal priors from the training folds' labels | exp_1: constant / boundary Viterbi; ryu: rate-calibrated λ |
| ensemble (H3) | 3 training seeds, probabilities averaged | ryu: 3 seeds (TCN) |

**Protocol.** exp_1's frozen judge and nested leave-one-match-out (21 pair + 7 outer fits per arm);
each fit predicts both views with the same weights. Decisions follow `PREREG.md` (hash checked in every
notebook). BAR-ATH is touched only in notebook 05, aggregates only; the public window is never opened.
"""

NOTEBOOKS = {
    "00_protocol.ipynb": [
        ("markdown", INTRO),
        ("code", SETUP + "import exp2_protocol as P, exp2_views as V\nimport exp1_common as C1"),
        ("markdown", "## 1. Frozen judge, proxy integrity, and what the view changes\nThe round-0 copies must equal the frozen degrader; the `target_like` view may only remove ball observations."),
        ("code", """integrity = pd.DataFrame(P.view_integrity())
C1.write_json(C.RESULTS / 'view_integrity.json', integrity.to_dict(orient='records'))
print(integrity.to_string(index=False))
assert integrity.drop(columns='match').all().all()"""),
        ("markdown", "## 2. Token caches: `target_like` (held-out view) and `target_aug` (training rounds 1-2)\nOnly IDSSE aggregates. `outside_of_visible` should fall from about 0.26 to about 0.04; labels are used here only to describe the views."),
        ("code", """t0 = time.time()
stats = pd.DataFrame(P.build_token_caches())
stats.to_csv(C.RESULTS / 'view_stats.csv', index=False)
cols = ['ball_visible', 'outside_of_visible', 'visible_on_dead', 'visible_on_alive', 'outside_of_visible_dead']
print(stats[['match', 'kind', 'round'] + cols].round(3).to_string(index=False))
print('\\nmean by kind'); print(stats.groupby('kind')[cols].mean().round(3).to_string())
print(f'built in {time.time() - t0:.0f}s')"""),
        ("markdown", "## 3. Sanity checks on a real IDSSE match\n* `target_like` ignores labels (shuffled labels give the same ball);\n* both transforms commute with the home/away + 180° mirror (frames that differ are counted; float rounding at the touchline is the only expected source);\n* `target_aug` is reproducible from its seed."),
        ("code", """from dataclasses import replace
name = C.MATCH_IDS[0]
match = C1.load_degraded(name, 0)
shuffled = replace(match, labels=np.random.default_rng(1).permutation(match.labels))
same = lambda a, b: np.array_equal(a.ball, b.ball, equal_nan=True)
print('target_like label independence:', same(V.target_like(match), V.target_like(shuffled)))
for label, apply in (('target_like', V.target_like),
                     ('target_aug', lambda m: V.target_aug(m, np.random.default_rng(V.aug_seed(name, 1))))):
    a, b = apply(V.mirror_match(match)), V.mirror_match(apply(match))
    differ = ~((np.isnan(a.ball) & np.isnan(b.ball)) | np.isclose(a.ball, b.ball)).all(axis=1)
    print(f'{label} mirror commutation: frames that differ = {int(differ.sum())} of {len(differ)}')
r1 = V.target_aug(C1.load_degraded(name, 1), np.random.default_rng(V.aug_seed(name, 1)))
r2 = V.target_aug(C1.load_degraded(name, 1), np.random.default_rng(V.aug_seed(name, 1)))
print('target_aug reproducible:', same(r1, r2))
print('exp_2 fits cached so far:', P.cached_fit_count(), '| fingerprints:', {a: P.fingerprint(a) for a in P.MODEL_ARMS})"""),
    ],
    "01_reference_arms.ipynb": [
        ("markdown", "# 01 — Reference arms on both views\n\n`baseline`, rasya C06/C08 (rasya's frozen constants) and ryu's shipped TCN pipeline (zero-shot, all of ryu's constants) are re-run on the `original` and the `target_like` view of the same seven held-out matches. Nothing is fitted."),
        ("code", SETUP + "import exp2_protocol as P"),
        ("code", """t0 = time.time()
rows = pd.concat([pd.DataFrame(P.evaluate_reference(view)) for view in C.VIEWS], ignore_index=True)
rows.to_csv(C.RESULTS / 'reference_rows.csv', index=False)
for view in C.VIEWS:
    part = rows[rows.view == view]
    print(f'== {view}')
    print(part.pivot(index='match', columns='variant', values='spell_f1').round(4).to_string())
    print(part.groupby('variant')[['spell_f1', 'macro_f1', 'n_true', 'n_pred', 'count_error_ratio']].mean().round(4).to_string())
print(f'{time.time() - t0:.0f}s')"""),
    ],
    "02_fits.ipynb": [
        ("markdown", "# 02 — SetSeq fits (4 arms × 28 nested fits)\n\n`exp1_ref` re-fits exp_1's SetSeq (exp_1 saved no weights); `exp2_aug` and its two extra seeds use the `target_aug` training recipe. Each fit predicts both views of its held-out matches and saves its weights. Fits are cached by code + config fingerprint; progress is appended to `results/fit_log.txt`."),
        ("code", SETUP + "import exp2_protocol as P, exp1_net as N\nimport exp1_common as C1\nprint('device', N.DEVICE, '| config', N.TrainConfig().as_dict())"),
        ("code", """log_path = C.RESULTS / 'fit_log.txt'
def log(message, **_):
    print(message, flush=True)
    with log_path.open('a') as handle:
        handle.write(message + '\\n')
t0 = time.time()
predictions = {}
for arm in P.MODEL_ARMS:
    start = time.time()
    predictions[arm] = P.ensure_predictions(arm, log=log)
    log(f'ARM DONE {arm} {time.time() - start:.0f}s')
log(f'ALL FITS DONE {time.time() - t0:.0f}s, cached fits {P.cached_fit_count()}')"""),
        ("markdown", "## Reproduction of exp_1\nThe `exp1_ref` re-fit uses exp_1's code, data and seed. It should reproduce exp_1's cached original-view outer predictions; any difference is reported, and `exp1_ref` (the re-fit) is used for both views so deltas stay paired."),
        ("code", """check = P.reproduction_check(predictions['exp1_ref'])
C1.write_json(C.RESULTS / 'reproduction_exp1_ref.json', check)
print(json.dumps(check, indent=1))"""),
    ],
    "03_evaluation.ipynb": [
        ("markdown", "# 03 — Nested evaluation on both views\n\nEvery variant is evaluated separately on each view with exp_1's nesting: temperature and decoder settings are chosen on the six inner validation matches (pair models) of that view, the held-out match is scored once. Final configs (for BAR-ATH) are selected on the 7 out-of-fold predictions."),
        ("code", SETUP + "import exp2_protocol as P\nimport exp1_common as C1"),
        ("code", """cache = {}
for variant in P.VARIANTS:
    for view in C.VIEWS:
        t0 = time.time()
        result = P.evaluate_variant(variant, view, cache)
        C1.write_json(C.RESULTS / f'variant_{variant}__{view}.json', result)
        frame = pd.DataFrame(result['rows'])
        print(f'== {variant:15s} {view:12s} mean spellF1 {frame.spell_f1.mean():.4f}  max|count err| {frame.count_error_ratio.abs().max():.3f}  ({time.time() - t0:.0f}s)')
        print('   final config:', {k: round(v, 3) for k, v in result['final_config'].items()})"""),
        ("markdown", "## Per-match detail (held-out matches)"),
        ("code", """keep = ['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'count_error_ratio']
for variant in ('exp1_ref', 'exp2_aug', 'exp2_aug_hsmm', 'exp2_aug_ens3'):
    for view in C.VIEWS:
        frame = pd.DataFrame(C1.read_json(C.RESULTS / f'variant_{variant}__{view}.json')['rows'])
        extra = [c for c in ('gamma', 'weight', 'lambda_team', 'lambda_dead', 'temperature') if c in frame]
        print(f'== {variant} / {view}'); print(frame[keep + extra].round(3).to_string(index=False))"""),
        ("markdown", "## Diagnostic: argmax (no decoder)"),
        ("code", """argmax = pd.concat([pd.DataFrame(P.evaluate_argmax(source, view, cache))
                    for source in ('exp1_ref', 'exp2_aug') for view in C.VIEWS], ignore_index=True)
argmax.to_csv(C.RESULTS / 'argmax_rows.csv', index=False)
print(argmax.groupby(['variant', 'view'])[['spell_f1', 'macro_f1', 'count_error_ratio']].mean().round(4).to_string())"""),
    ],
    "04_comparison.ipynb": [
        ("markdown", "# 04 — Comparison and pre-registered decisions\n\nPaired match bootstrap (1000 draws, seed 20261108, 95 %). With seven matches the intervals are wide: read decisions, not third decimals."),
        ("code", SETUP + "import exp2_protocol as P, exp2_report as R\nimport exp1_common as C1\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"),
        ("code", """parts = [pd.read_csv(C.RESULTS / 'reference_rows.csv'), pd.read_csv(C.RESULTS / 'argmax_rows.csv')]
for variant in P.VARIANTS:
    for view in C.VIEWS:
        parts.append(pd.DataFrame(C1.read_json(C.RESULTS / f'variant_{variant}__{view}.json')['rows']))
rows = pd.concat(parts, ignore_index=True)
R.assert_proxy_only(rows)
rows.to_csv(C.RESULTS / 'per_match_results.csv', index=False)
table = R.summary_table(rows)
table.to_csv(C.RESULTS / 'comparison_table.csv', index=False)
cols = ['variant', 'mean_spell_f1', 'mean_macro_f1', 'mean_abs_count_error', 'max_abs_count_error', 'delta_vs_exp1_ref', 'ci95_low', 'ci95_high', 'wins_vs_exp1_ref', 'count_gate']
for view in C.VIEWS:
    print(f'== {view}'); print(R.as_markdown(table[table.view == view], cols))"""),
        ("markdown", "## Robustness: target_like minus original, per arm (paired by match)"),
        ("code", """shift = pd.DataFrame(R.view_shift(rows))
print(shift.round(4).to_string(index=False))"""),
        ("markdown", "## Quality gates (exp_0) per view"),
        ("code", """gate_rows = [R.gates(rows, view, v) for view in C.VIEWS for v in R.LABELS
             if v in set(rows[rows.view == view].variant) and v != 'baseline']
print(pd.DataFrame([{'view': g['view'], 'variant': g['variant'], **g['checks'], 'pass': g['pass'],
                     'worst_count_error': round(g['worst_count_error'], 3), 'worst_match': g['worst_count_match']}
                    for g in gate_rows]).to_string(index=False))"""),
        ("markdown", "## Pre-registered decisions (PREREG §6)"),
        ("code", """decision = R.decide(rows)
C1.write_json(C.RESULTS / 'decision.json', decision)
report = {'prereg_sha256': protocol['prereg_sha256'], 'bootstrap': 'paired match bootstrap, 1000 draws, seed 20261108, 95% percentile',
          'table': table.to_dict(orient='records'), 'deltas_vs_exp1_ref': R.deltas_vs_reference(rows),
          'view_shift': R.view_shift(rows), 'gates': gate_rows, 'decision': decision,
          'reproduction_exp1_ref': C1.read_json(C.RESULTS / 'reproduction_exp1_ref.json')}
C1.write_json(C.RESULTS / 'proxy_report.json', report)
for name, rule in decision['rules'].items():
    shown = {k: rule[k] for k in ('mean_delta', 'ci95', 'wins', 'seed_spread', 'max_abs_count_error') if k in rule}
    print(f"{name:34s} {'PASSED' if rule['passed'] else 'FAILED':7s} {rule['rule']:45s} {shown}")
print('supported:', decision['supported'], '| adopt exp_2 candidate:', decision['adopt_exp2_candidate'])"""),
        ("markdown", "## Figure (proxy only)"),
        ("code", """order = ['baseline', 'rasya_c06', 'rasya_c08', 'ryu_final', 'exp1_ref', 'exp2_aug', 'exp2_aug_const', 'exp2_aug_hsmm', 'exp2_aug_ens3']
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for view, marker in zip(C.VIEWS, ('o', 's')):
    part = table[table.view == view].set_index('variant').reindex(order)
    axes[0].plot(order, part.mean_spell_f1, marker=marker, label=view)
    axes[1].plot(order, part.max_abs_count_error, marker=marker, label=view)
axes[0].set_ylabel('mean spellF1 (7 held-out matches)'); axes[0].set_title('spellF1 by arm and view')
axes[1].axhline(0.3, color='0.5', ls='--', label='±30% gate'); axes[1].set_ylabel('max |count error|')
axes[1].set_title('worst-match count error')
for ax in axes:
    ax.tick_params(axis='x', rotation=35); ax.legend()
fig.tight_layout(); fig.savefig(C.RESULTS / 'fig_exp2_views.png', dpi=130)
print('saved', C.RESULTS / 'fig_exp2_views.png')"""),
    ],
    "05_target.ipynb": [
        ("markdown", "# 05 — BAR-ATH: label-free plausibility and (only if adopted) the candidate\n\nAggregates only. Full-data SetSeq fits (all 7 proxy matches, `target_aug` recipe, seeds 0-2) are decoded with the final configs selected on the `target_like` out-of-fold predictions. The plausibility numbers are reported, never used for a choice. The candidate CSV is written only if `results/decision.json` adopts it, and must rebuild byte-identically from an independent fit. No public-window check."),
        ("code", SETUP + "import exp2_protocol as P, exp2_target as E\nimport exp1_common as C1\nassert C.public_check_spent(), 'exp_1 public check marker missing'"),
        ("code", """t0 = time.time()
decision = C1.read_json(C.RESULTS / 'decision.json')
match, item = E.load_target()
models = {arm: E.full_model(arm) for arm in P.ENSEMBLE}
plausibility = E.plausibility(match, item, models)
C1.write_json(C.RESULTS / 'target_plausibility.json', plausibility)
keys = ('dead_share', 'possessions_per_min', 'mean_possession_s', 'dead_share_minus_prior', 'possession_count')
print(pd.DataFrame({k: {m: v[m] for m in keys} for k, v in plausibility.items() if isinstance(v, dict)}).T.round(3).to_string())
print('published dead-share prior', plausibility['published_dead_share_prior'], '| ball coverage', round(plausibility['ball_coverage_graded'], 3))
print(f'{time.time() - t0:.0f}s')"""),
        ("code", """if decision['adopt_exp2_candidate']:
    summary = E.build_candidate(match, item, models['exp2_aug'])
    summary['reproduction'] = E.reproduce(match, item)
else:
    summary = {'adopted': False, 'reason': 'PREREG adoption rule (H1) not met; exp_1 candidate stays',
               'rules': {k: v['passed'] for k, v in decision['rules'].items()}}
C1.write_json(C.RESULTS / 'target_candidate.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'share_graded'}, indent=1))"""),
    ],
}

selected = sys.argv[1:] or list(NOTEBOOKS)
for name, cells in NOTEBOOKS.items():
    if name not in selected:
        continue
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "exp2", "display_name": "exp_2 (.venv-torch)", "language": "python"}
    notebook.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
                      for kind, src in cells]
    nbf.write(notebook, ROOT / name)
    print("wrote", name, len(cells), "cells")
