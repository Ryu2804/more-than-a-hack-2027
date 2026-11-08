"""Write the exp_1 notebooks and their kernel spec (never reads executed outputs)."""

from pathlib import Path
import json
import os
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("EXP1_PYTHON", sys.executable)  # kernel uses the interpreter running this
KERNEL = ROOT / "kernels" / "exp1"
KERNEL.mkdir(parents=True, exist_ok=True)
(KERNEL / "kernel.json").write_text(json.dumps({
    "argv": [PYTHON, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "exp_1 (.venv-torch)",
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
if not (ROOT / 'exp1_common.py').exists():
    raise RuntimeError('Run with working directory exp_jeremy/exp_1')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', 30)
import exp1_common as C
"""

INTRO = """# exp_1 — SetSeq-BD: set-sequence network with boundary-aware decoding

**Question.** Can a model that reads the *raw, identity-free player set* through time recover
possession across long ball gaps better than (a) hand-crafted rules and (b) hand-crafted features
with a gradient-boosted classifier — on the same frozen, degraded proxy?

**What is new here (not done in exp_0, exp_rasya or exp_ryu):**

| component | exp_1 | closest prior work in the team |
|---|---|---|
| input | DeepSets encoder over every visible player token (no identities, no hand features) | exp_0: 88 hand features; ryu: 23 hand team-shape/IFAB features; rasya: nearest-player rules |
| temporal model | 2-layer BiGRU over 60 s windows with a 3-class state head | ryu: dilated TCN (dead + team heads); exp_0: per-frame GBM; rasya: rule hold/carry |
| decoder | **boundary-aware Viterbi**: a learned boundary head sets a time-varying switch cost | exp_0 / ryu / rasya: one constant switch penalty (ryu: calibrated to a possession rate) |
| training data | **corruption resampling** of IDSSE (provider frame labels): each training match re-degraded with 2 extra seeds + home/away mirror | exp_0: one frozen corruption; ryu: Metrica + SkillCorner with event-derived labels; rasya: clean IDSSE |

ryu's work lives on the `ryu` git branch (the `exp_ryu/` working tree only holds the baseline); its shipped
model is run zero-shot on the proxy as the `ryu_final` reference arm in notebook 01.

**Protocol.** Seven IDSSE matches degraded by exp_0's frozen degrader (snapshot in `judge/`, SHA-256
checked). Nested leave-one-match-out: decoder settings for held-out match *h* are chosen on the
other six using models that never saw *h* or the validation match. Paired bootstrap over matches.
The reference arms (baseline, rasya C06/C08, exp_0 LightGBM) are re-run on the *identical* data.

**Data hygiene.** BAR-ATH is only touched in notebook 04, locally, with aggregate outputs only.
The public ground truth is opened once at the very end, after the predictor is frozen.
"""

NOTEBOOKS = {
    "00_data_protocol.ipynb": [
        ("markdown", INTRO),
        ("code", SETUP),
        ("markdown", "## 1. Freeze the judge\nThe degrader and feature code are verbatim copies of exp_0 at the time of the snapshot. The copied degraded matches must be reproduced bit-for-bit by applying the frozen degrader to the clean matches."),
        ("code", """judge = C.verify_judge()
proxy = C.verify_frozen_proxy()
C.write_json(C.RESULTS / 'judge_freeze.json', {'hashes': judge, 'frozen_proxy': proxy})
print(json.dumps(judge, indent=1))
print(pd.DataFrame(proxy).to_string(index=False))"""),
        ("markdown", "## 2. Protocol, frozen before any model is fitted"),
        ("code", """import exp1_net as N, exp1_decode as D, exp1_protocol as P
protocol = {
    'matches': C.MATCH_IDS,
    'evaluation_corruption_seed': 'BASE_SEED + index (exp_0 frozen)',
    'augmentation_rounds': C.AUG_ROUNDS,
    'augmentation_seed': 'BASE_SEED + 1000 * round + index (training matches only)',
    'model_fps': 5,
    'setseq_train_config': N.TrainConfig().as_dict(),
    'decoder_lambdas': D.LAMBDAS,
    'decoder_gammas': D.GAMMAS,
    'selection_objective': 'mean spellF1 - 0.15 * mean excess |count error| above 0.30 (exp_0)',
    'selection_scope': 'inner validation = six non-held-out matches, predicted by pair models trained without h and v',
    'bootstrap': '1000 paired match draws, seed 20261108, 95% percentile',
    'quality_gates': C.QUALITY_GATES,
    'model_variants': {k: list(map(str, v)) for k, v in P.DECODER_VARIANTS.items()},
    'rule_arms': P.RULE_ARMS,
}
path = C.RESULTS / 'protocol.json'
if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol, default=str)):
    raise RuntimeError('Protocol changed after freezing; diagnose before rerunning')
C.write_json(path, protocol)
print('protocol sha256', C.sha256(path))"""),
        ("markdown", "## 3. Token cache: frozen corruption (round 0) + two augmentation rounds per match\nOnly aggregate statistics are shown. Augmentation rounds should look like round 0 statistically."),
        ("code", """t0 = time.time()
summary = pd.DataFrame(P.build_token_cache())
summary.to_csv(C.RESULTS / 'data_summary.csv', index=False)
cols = ['match', 'round', 'ball_visible', 'gap_mean_seconds', 'mean_objects', 'tokens_per_step', 'boundary_rate', 'label_H', 'label_A', 'label_D']
print(summary[cols].round(3).to_string(index=False))
print('\\nmean by round'); print(summary.groupby('round')[cols[2:]].mean().round(3).to_string())
print(f'built in {time.time() - t0:.0f}s')"""),
        ("markdown", "## 4. Sanity checks\n* the mirror augmentation is an involution and swaps only H/A;\n* the set encoder is invariant to token order;\n* tokens do not depend on labels (shuffling labels leaves them unchanged)."),
        ("code", """import torch, exp1_tokens as T
from dataclasses import replace
item = P.load_tokens(C.MATCH_IDS[0])
tok, glo, st = item['tokens'][:300].astype('float32'), item['glob'][:300].astype('float32'), item['state'][:300]
t2, g2, s2 = T.mirror(*T.mirror(tok, glo, st))
print('mirror involution:', np.array_equal(t2, tok) and np.array_equal(g2, glo) and np.array_equal(s2, st))
print('mirror keeps D:', bool(np.all((T.mirror(tok, glo, st)[2] == 2) == (st == 2))))
torch.manual_seed(0)
model = N.SetSeq(64, 64).to(N.DEVICE).eval()
perm = np.random.default_rng(0).permutation(T.MAX_TOKENS)
args = lambda t, m: (torch.tensor(t)[None].to(N.DEVICE), torch.tensor(m)[None].to(N.DEVICE), torch.tensor(glo)[None].to(N.DEVICE))
with torch.no_grad():
    a = model(*args(tok, item['mask'][:300]))[0]
    b = model(*args(tok[:, perm], item['mask'][:300][:, perm]))[0]
print('token-order invariance, max |diff|:', float((a - b).abs().max()))
match = C.load_degraded(C.MATCH_IDS[0])
shuffled = replace(match, labels=np.random.default_rng(1).permutation(match.labels))
x1, x2 = T.build_tokens(match), T.build_tokens(shuffled)
print('label independence:', np.array_equal(x1.tokens, x2.tokens) and np.array_equal(x1.glob, x2.glob))
print('trainable parameters:', sum(p.numel() for p in model.parameters()))"""),
        ("markdown", "## 5. exp_0 feature cache for the LightGBM reference arm"),
        ("code", """t0 = time.time()
P.build_gbm_features()
import exp1_arms as A
print({n: A.gbm_features(n)['x'].shape for n in C.MATCH_IDS}, f'{time.time() - t0:.0f}s')"""),
    ],
    "01_reference_arms.ipynb": [
        ("markdown", "# 01 — Reference arms on the identical frozen proxy\n\n* `baseline`: organisers' starter (the `exp_ryu/` working tree).\n* `ryu_final`: ryu's committed final pipeline (`origin/ryu` bd90a9c, `015_final_pipeline/run.py` + `weights_new`: TCN ensemble trained by ryu on Metrica + SkillCorner) applied zero-shot with all of ryu's constants; λ is calibrated, as ryu does, to the training matches' possession rate.\n* `rasya_c06`, `rasya_c08`: rasya's best deployable pipelines from `exp_rasya/.../012_spellf1` with the parameters rasya froze (touch radius 0.8 m, minlen 30, merge 150; dead cue out 15 frames or slow <2 m/s for 25 / 400 frames). No re-tuning here.\n* `gbm_const`: exp_0's feature family `deadball` + LightGBM (frozen snapshot), decoded with the exp_1 constant-penalty Viterbi selected inside training folds."),
        ("code", SETUP + "import exp1_protocol as P"),
        ("markdown", "## Rule arms"),
        ("code", """rule_rows = pd.DataFrame(P.evaluate_rules())
rule_rows.to_csv(C.RESULTS / 'rule_rows.csv', index=False)
print(rule_rows.pivot(index='match', columns='variant', values='spell_f1').round(4).to_string())
print(rule_rows.groupby('variant')[['spell_f1', 'macro_f1', 'n_pred', 'n_true']].mean().round(4).to_string())"""),
        ("markdown", "## ryu's shipped TCN pipeline, zero-shot"),
        ("code", """import warnings
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    ryu_rows = pd.DataFrame(P.evaluate_ryu())
ryu_rows.to_csv(C.RESULTS / 'ryu_rows.csv', index=False)
print(ryu_rows[['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'ryu_lambda', 'ryu_target_possessions']].round(4).to_string(index=False))
print('mean spellF1', round(ryu_rows.spell_f1.mean(), 4), f'{time.time() - t0:.0f}s')"""),
        ("markdown", "## exp_0 LightGBM arm (21 pair fits + 7 outer fits)"),
        ("code", """t0 = time.time()
gbm_predictions = P.ensure_predictions('gbm')
result = P.evaluate_variant('gbm_const', gbm_predictions)
C.write_json(C.RESULTS / 'variant_gbm_const.json', result)
print(pd.DataFrame(result['rows'])[['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'lambda_team', 'lambda_dead']].round(4).to_string(index=False))
print(f'{time.time() - t0:.0f}s')"""),
    ],
    "02_setseq_nested.ipynb": [
        ("markdown", "# 02 — SetSeq training and nested evaluation\n\nOne SetSeq model per unordered pair of left-out matches (21) and per outer fold (7). Each sees the frozen corruption plus two re-corruptions of its training matches, with random home/away mirroring. Decoder variants share the *same* predictions, so decoder ablations are paired exactly.\n\nThen the augmentation ablation (`setseq_noaug`): same network, frozen corruption only, no mirroring."),
        ("code", SETUP + "import exp1_protocol as P, exp1_net as N\nprint('device', N.DEVICE, '| config', N.TrainConfig().as_dict())"),
        ("markdown", "## Fits (cached by code + config fingerprint)"),
        ("code", """t0 = time.time()
setseq_predictions = P.ensure_predictions('setseq')
print(f'total {time.time() - t0:.0f}s')"""),
        ("markdown", "## Decoder variants on identical predictions"),
        ("code", """for variant in ('setseq_bd', 'setseq_const', 'setseq_argmax'):
    t0 = time.time()
    result = P.evaluate_variant(variant, setseq_predictions)
    C.write_json(C.RESULTS / f'variant_{variant}.json', result)
    frame = pd.DataFrame(result['rows'])
    print(f'== {variant}  mean spellF1 {frame.spell_f1.mean():.4f}  ({time.time() - t0:.0f}s)')
    keep = [c for c in ['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'gamma', 'lambda_team', 'lambda_dead', 'temperature'] if c in frame]
    print(frame[keep].round(4).to_string(index=False))
    if result['final_config']:
        print('final config (selected on 7 out-of-fold predictions, used for BAR-ATH):', result['final_config'])"""),
        ("markdown", "## Boundary head diagnostic (outer folds)\nHow well does p(boundary) rank true possession boundaries on held-out matches?"),
        ("code", """from sklearn.metrics import roc_auc_score, average_precision_score
rows = []
for name in C.MATCH_IDS:
    _, boundary = setseq_predictions[f'outer_{name}'][name]
    target = P.load_tokens(name)['boundary']
    rows.append({'match': name, 'auc': roc_auc_score(target, boundary), 'avg_precision': average_precision_score(target, boundary), 'base_rate': target.mean()})
print(pd.DataFrame(rows).round(4).to_string(index=False))"""),
        ("markdown", "## Augmentation ablation"),
        ("code", """t0 = time.time()
noaug_predictions = P.ensure_predictions('setseq_noaug')
result = P.evaluate_variant('setseq_noaug_bd', noaug_predictions)
C.write_json(C.RESULTS / 'variant_setseq_noaug_bd.json', result)
frame = pd.DataFrame(result['rows'])
print(f'setseq_noaug_bd mean spellF1 {frame.spell_f1.mean():.4f} ({time.time() - t0:.0f}s)')
print(frame[['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'gamma']].round(4).to_string(index=False))"""),
    ],
    "03_comparison.ipynb": [
        ("markdown", "# 03 — Comparison on the identical proxy\n\nAll arms are scored by the official metric on the same seven degraded matches. Δ is paired by match; intervals are 95% paired bootstrap over matches (1000 draws). With seven matches, intervals are wide: read *decisions*, not third decimals."),
        ("code", SETUP + "import exp1_report as R, exp1_protocol as P\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"),
        ("markdown", "## Post-hoc diagnostic (added after the frozen run)\nThe frozen selection chose γ = 0 (constant penalty) in every fold, so `setseq_bd` ≡ `setseq_const`. To measure what the boundary-aware decoder does when it *is* used, γ is forced into {0.5, 1, 2}; everything else stays nested. This arm is a diagnostic, never a candidate."),
        ("code", """forced = P.evaluate_variant('setseq_bd_forced', P.ensure_predictions('setseq'))
C.write_json(C.RESULTS / 'variant_setseq_bd_forced.json', forced)
print(pd.DataFrame(forced['rows'])[['match', 'spell_f1', 'n_true', 'n_pred', 'gamma', 'lambda_team', 'lambda_dead']].round(4).to_string(index=False))"""),
        ("code", """parts = [pd.read_csv(C.RESULTS / 'rule_rows.csv'), pd.read_csv(C.RESULTS / 'ryu_rows.csv')]
for variant in ('gbm_const', 'setseq_argmax', 'setseq_const', 'setseq_noaug_bd', 'setseq_bd', 'setseq_bd_forced'):
    parts.append(pd.DataFrame(C.read_json(C.RESULTS / f'variant_{variant}.json')['rows']))
rows = pd.concat(parts, ignore_index=True)
report = R.build_report(rows)
table = pd.DataFrame(report['table'])
print(R.as_markdown(table))"""),
        ("markdown", "## Per-match spellF1 and possession counts"),
        ("code", """print(R.per_match_matrix(rows).round(4).to_string())
counts = R.per_match_matrix(rows, 'n_pred')
counts.insert(0, 'n_true', R.per_match_matrix(rows, 'n_true')['baseline'])
print(); print(counts.to_string())"""),
        ("markdown", "## Pairwise: main arm vs every other arm, and ablations"),
        ("code", """print(pd.DataFrame(report['pairwise_main_vs'])[['a', 'b', 'mean_delta', 'ci95', 'wins', 'decision']].to_string(index=False))
print(); print(pd.DataFrame(report['ablations'])[['ablation', 'a', 'b', 'mean_delta', 'ci95', 'wins', 'decision']].to_string(index=False))
print(); print('quality gates (exp_0 protocol):', json.dumps(report['gates'], indent=1))"""),
        ("markdown", "## Figures (proxy only)"),
        ("code", """matrix = R.per_match_matrix(rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = list(matrix.mean().sort_values().index)
for name in C.MATCH_IDS:
    axes[0].plot(order, matrix.loc[name, order], marker='o', alpha=0.45, lw=1)
axes[0].plot(order, matrix[order].mean(), color='black', marker='D', lw=2.5, label='mean')
axes[0].set_ylabel('spellF1 (held-out match)'); axes[0].set_title('Per-match spellF1 by arm (lines = matches)')
axes[0].tick_params(axis='x', rotation=35); axes[0].legend()
ratio = R.per_match_matrix(rows, 'count_error_ratio')[order]
axes[1].axhspan(-0.3, 0.3, color='0.9', label='+-30% gate')
for name in C.MATCH_IDS:
    axes[1].plot(order, ratio.loc[name], marker='o', alpha=0.45, lw=1)
axes[1].axhline(0, color='black', lw=1); axes[1].set_ylabel('predicted / true possessions - 1')
axes[1].set_title('Over/under-segmentation'); axes[1].tick_params(axis='x', rotation=35); axes[1].legend()
fig.tight_layout(); fig.savefig(C.RESULTS / 'fig_proxy_comparison.png', dpi=130)
print('saved', C.RESULTS / 'fig_proxy_comparison.png')"""),
    ],
    "04_target_inference.ipynb": [
        ("markdown", "# 04 — BAR-ATH inference (aggregates only)\n\nThe final SetSeq model is trained on all seven proxy matches (+ augmentation) and decoded with the configuration selected on the seven out-of-fold predictions (`variant_setseq_bd.json → final_config`). Nothing per-frame is printed. A second independent fit must reproduce the CSV byte-for-byte. Last, the public 5-minute window is opened **once** as a correctness check; it is recorded, never used to choose anything, and the predictor cannot be rebuilt afterwards."),
        ("code", SETUP + "import exp1_target as X"),
        ("code", """t0 = time.time()
summary = X.build_candidate()
C.write_json(C.RESULTS / 'target_summary.json', summary)
print(json.dumps(summary, indent=1)); print(f'{time.time() - t0:.0f}s')"""),
        ("code", """print(json.dumps(X.reproduce(), indent=1))"""),
        ("markdown", "## Single-use public-window sanity check (after freeze; 17 possessions, wide interval)"),
        ("code", """print(json.dumps(X.public_sanity_once(), indent=1))"""),
    ],
    "05_followup_centred_bd.ipynb": [
        ("markdown", "# 05 — Pre-registered follow-up: centred boundary-aware decoding\n\nSee `PREREG_followup.md` (hash frozen in `results/followup_protocol.json` before this ran). Switch cost becomes `max(0, λ − γ·(log b_t − log mean b))`, so a likely boundary *lowers* the cost; the λ grid is widened to {1/16 … 16}. Same cached SetSeq predictions, same nested protocol. Motivated by exp_1's outer results, so not blind."),
        ("code", SETUP + "import exp1_followup as F"),
        ("code", """protocol = F.verify_prereg()
print('pre-registration verified', protocol['prereg_sha256'][:16], protocol['arms'])"""),
        ("code", """for variant in ('setseq_const_wide', 'setseq_cbd_wide'):
    t0 = time.time()
    result = F.evaluate(variant)
    C.write_json(C.RESULTS / f'variant_{variant}.json', result)
    frame = pd.DataFrame(result['rows'])
    print(f'== {variant}  mean spellF1 {frame.spell_f1.mean():.4f}  ({time.time() - t0:.0f}s)')
    print(frame[['match', 'spell_f1', 'macro_f1', 'n_true', 'n_pred', 'gamma', 'lambda_team', 'lambda_dead']].round(4).to_string(index=False))
    print('final config:', result['final_config'])"""),
        ("markdown", "## Pre-registered decisions"),
        ("code", """rows = pd.concat([pd.read_csv(C.RESULTS / 'rule_rows.csv')] + [
    pd.DataFrame(C.read_json(C.RESULTS / f'variant_{v}.json')['rows'])
    for v in ('setseq_const', 'setseq_bd', 'setseq_const_wide', 'setseq_cbd_wide')], ignore_index=True)
decision = F.decide(rows)
C.write_json(C.RESULTS / 'followup_decision.json', decision)
print(json.dumps(decision, indent=1))"""),
    ],
    "06_followup_target.ipynb": [
        ("markdown", "# 06 — BAR-ATH follow-up candidate (only if rule 3 adopted it)\n\nAggregates only. No public-window check: it was single-use and is spent. Byte-identical rebuild required."),
        ("code", SETUP + "import exp1_followup as F"),
        ("code", """first = F.build_candidate()
second = F.build_candidate(F.REPRODUCED)
summary = {**first, 'rebuild_sha256': second['sha256'], 'byte_identical': first['sha256'] == second['sha256']}
C.write_json(C.RESULTS / 'followup_target_summary.json', summary)
print(json.dumps(summary, indent=1))"""),
    ],
}

import sys  # noqa: E402

selected = sys.argv[1:] or list(NOTEBOOKS)  # rebuild only the named notebooks (keeps others' outputs)
for name, cells in NOTEBOOKS.items():
    if name not in selected:
        continue
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "exp1", "display_name": "exp_1 (.venv-torch)", "language": "python"}
    notebook.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nbf.write(notebook, ROOT / name)
    print("wrote", name, len(cells), "cells")
