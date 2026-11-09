# exp_3 — train on the competition's possession definition

Pre-registered follow-up to exp_2 (`PREREG.md`, SHA-256 in `results/protocol.json`). Results and
decisions are in `REPORT.md`. exp_1 and exp_2 are reused by import (`../exp_2` on `sys.path`, which
adds `../exp_1`). They are never copied or edited. exp_2's cached predictions and full-data models
are read in place and fingerprint-checked.

## Layout

| path | what |
|---|---|
| `PREREG.md` | what was seen before freezing, `L_ctrl`, arms, grids, H1 rule, adoption tree, fusion rule, S1–S3 |
| `DATA_SOURCES.md` | IDSSE event and match-information files (Figshare, CC BY 4.0) and the event schema used |
| `00_labels_protocol.ipynb` | freezes PREREG; label integrity; event sync report; label-only statistics; re-derivation of the planning findings; S2 coverage tokens |
| `01_reference_on_ctrl.ipynb` | zero-fit arms (cached `exp1_ref`, `exp2_aug`, `exp3_ref_ens3`) on `L_ctrl`; rule arms and ryu against every label set |
| `02_fits.ipynb` | `exp3_ctrl` × 3 seeds × 28 nested fits (`L_ctrl` targets, exp2_aug recipe) |
| `03_evaluation.ipynb` | every variant × view × label set (nested re-selection per label set), argmax diagnostic, S2 |
| `04_comparison.ipynb` | tables, paired deltas vs `exp3_ref_ens3` and `exp2_aug`, gates, H1 decision, S2 shift, figure |
| `05_target.ipynb` | BAR-ATH: fusion (+ replay guard), plausibility aggregates, candidate CSV, byte-identical rebuild |
| `06_touch.ipynb` | S3 (stretch): touch features, 28 fits, vs `exp3_ctrl` seed 0 |
| `07_hedge_provider.ipynb` | post-hoc hedge (not pre-registered): same ensemble, provider-selected decoder, second validated CSV + rebuild |
| `exp3_labels.py` | `L_track(τ)`, `L_ctrl`, event parsing, kick-off sync, snap refinement, sync checks |
| `exp3_protocol.py` | label sets → 5 fps targets, `exp3_ctrl` fits, cached exp_2 loader (never fits), nested evaluation, S2 |
| `exp3_coverage.py` | S2 view: clean ball re-revealed on 12 % of each hidden-ball run (label-free) |
| `exp3_fusion.py` + `judge/` | pinned verbatim exp_0 `fusion_candidate` (SHA-256 in `judge/fusion_pin.json`) + replay guard |
| `exp3_report.py` / `exp3_target.py` / `exp3_diagnostics.py` / `exp3_touch.py` | tables + decision / BAR-ATH / planning re-derivation / S3 |
| `exp3_common.py` | paths, PREREG freeze / verify (PREREG hash, judge, fusion pin, constants read from code) |
| `tests/test_exp3.py` | synthetic tests: `L_ctrl` (a)–(c), merge direction, halves, D untouched; `L_track`; event sync and snap; replay guard; coverage view; S3 mirror |
| `results/` | JSON / CSV aggregates (IDSSE per match; BAR-ATH aggregates only) and the figure |
| `cache/`, `predictions/`, `kernels/`, `external/*.xml` | gitignored |

## Reproduce

The data setup is exp_1's (`../exp_1/README.md`). exp_2's caches must exist (`../exp_2/cache`). The
IDSSE event files are fetched with the command in `DATA_SOURCES.md`.

```bash
cd exp_jeremy/exp_3
PY=/home/jeremy-mboe/Documents/Kuliah/Sem3/Machine_Learning/.venv-torch/bin/python
$PY build_notebooks.py            # pass names to rebuild only some notebooks
for nb in 00_labels_protocol 01_reference_on_ctrl 02_fits 03_evaluation 04_comparison 05_target 06_touch 07_hedge_provider; do
  $PY run_notebook.py $nb.ipynb
done
$PY -m pytest -q -p no:cacheprovider tests/
```

`results/protocol.json` is written once, by notebook 00 before any fit, and refuses to be rewritten.
Every notebook calls `verify_prereg()`. Notebook 03 reuses result files already written by the same
evaluation function (notebook 01, and a helper run of the cached arms while the fits were running).

## Hygiene

- BAR-ATH is read only in notebook 05's kernel; its cells print aggregates only, and failures are
  summarised in one line.
- The public ground truth is never opened (exp_1 spent the single check).
- Do not commit `cache/`, `predictions/`, `kernels/` or the raw event XML.
