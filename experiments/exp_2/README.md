# exp_2 — transfer-robust SetSeq

Pre-registered follow-up to exp_1 (`PREREG.md`, SHA-256 in `results/protocol.json`). Results and
decisions are in `REPORT.md`. exp_1 is reused by import (`../exp_1` on `sys.path`), never copied.

## Layout

| path | what |
|---|---|
| `PREREG.md` | hypotheses H1–H3, views, arms, grids, decision rules (frozen before any fit) |
| `DATA_SOURCES.md` | licence gate for extra data (Metrica: no licence → skipped; SkillCorner: not used, label mismatch) |
| `00_protocol.ipynb` | PREREG + judge checks, `target_like` / `target_aug` token caches, view statistics, sanity checks |
| `01_reference_arms.ipynb` | baseline, rasya C06/C08, ryu (zero-shot) on both views |
| `02_fits.ipynb` | 4 SetSeq arms × 28 nested fits (weights saved), exp_1 reproduction check |
| `03_evaluation.ipynb` | nested evaluation of every variant on both views, argmax diagnostic |
| `04_comparison.ipynb` | tables, paired deltas vs `exp1_ref`, gates, pre-registered decisions, figure |
| `05_target.ipynb` | BAR-ATH: label-free plausibility aggregates; candidate CSV only if adopted |
| `exp2_views.py` | `target_like` (label-free stress view) and `target_aug` (training augmentation) |
| `exp2_hsmm.py` | explicit-duration (semi-Markov) decoder, entropy-centred log-normal duration priors |
| `exp2_protocol.py` | token caches, model arms, cached fits by fingerprint, ensembles, nested evaluation |
| `exp2_report.py` / `exp2_target.py` | tables + PREREG decisions / BAR-ATH aggregates and candidate |
| `exp2_common.py` | paths, PREREG freeze/verify (hash + constants) |
| `tests/test_exp2.py` | synthetic tests: label independence, mirror commutation, augmentation, HSMM |
| `results/` | JSON/CSV aggregates (proxy per match; BAR-ATH aggregates only) and the figure |
| `cache/`, `predictions/`, `kernels/` | gitignored |

## Reproduce

Data setup is exp_1's (`../exp_1/README.md`): the 14 IDSSE proxy files and exp_1's token cache in
`../exp_1/cache/`, ryu's reference in `../exp_1/reference/ryu/`. exp_2 reads them in place.

```bash
cd exp_jeremy/exp_2
PY=/home/jeremy-mboe/Documents/Kuliah/Sem3/Machine_Learning/.venv-torch/bin/python
$PY build_notebooks.py            # pass names to rebuild only some notebooks
for nb in 00_protocol 01_reference_arms 02_fits 03_evaluation 04_comparison 05_target; do
  $PY run_notebook.py $nb.ipynb
done
$PY -m pytest -q -p no:cacheprovider tests/
```

`results/protocol.json` is written once by `exp2_common.freeze_prereg()` and refuses to be rewritten;
every notebook calls `verify_prereg()` (PREREG hash, judge hashes, constants read from the code).

## Hygiene

BAR-ATH is read only in notebook 05's kernel; cells print aggregates only and failures are summarised
in one line. The public ground truth is never opened (exp_1 spent the single check). Do not commit
`cache/`, `predictions/` or `kernels/`.
