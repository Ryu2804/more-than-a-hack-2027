# exp_1 — SetSeq-BD (set-sequence network + boundary-aware decoding)

New experiment for frame-level possession, built to be compared like-for-like with exp_0,
exp_rasya and ryu on one frozen proxy. Results and the comparison are in `REPORT.md`.

## Layout

| path | what |
|---|---|
| `00_data_protocol.ipynb` | freeze judge snapshot + protocol, build token cache (3 corruption rounds), sanity checks |
| `01_reference_arms.ipynb` | baseline, ryu's shipped TCN (zero-shot), rasya C06/C08 rules, exp_0 LightGBM (nested) |
| `02_setseq_nested.ipynb` | 28 SetSeq fits (21 pair + 7 outer), decoder variants, boundary-head AUC, augmentation ablation |
| `03_comparison.ipynb` | paired bootstrap table, pairwise deltas, ablations, gates, figure |
| `04_target_inference.ipynb` | BAR-ATH CSV (aggregates only), byte-identical rebuild, single-use public sanity check |
| `05_followup_centred_bd.ipynb` | pre-registered follow-up (`PREREG_followup.md`): centred boundary decoder, wide λ grid |
| `06_followup_target.ipynb` | BAR-ATH follow-up CSV — runs only if rule 3 adopts it (it did not; not executed) |
| `exp1_followup.py` | centred decoder, nested evaluation, pre-registered decision rules |
| `exp1_common.py` | paths, frozen hashes, judge import, scoring/bootstrap wrappers |
| `exp1_tokens.py` | identity-free player tokens + global features at 5 fps, mirror augmentation |
| `exp1_net.py` | SetSeq model (DeepSets pooling + BiGRU, state + boundary heads), deterministic training |
| `exp1_decode.py` | boundary-aware Viterbi, temperature fit, train-only grid selection |
| `exp1_protocol.py` | nested pair/outer fits with cached predictions, rule + ryu arm evaluation |
| `exp1_arms.py` / `exp1_ryu.py` | reference arms (baseline, rasya, exp_0 GBM) / ryu adapter |
| `exp1_report.py` / `exp1_target.py` | report tables + gates / BAR-ATH inference |
| `judge/` | verbatim snapshot of exp_0 `experiment_data.py` + `experiment_model.py` (SHA-256 pinned) |
| `reference/ryu/` | ryu's code + `weights_new` extracted from `origin/ryu` (commit in `SOURCE_COMMIT`) |
| `results/` | JSON/CSV outputs and the figure (proxy aggregates; BAR-ATH aggregates only) |
| `predictions/` | private BAR-ATH submission CSV (never commit) |
| `cache/` | proxy copies, token cache, model predictions (never commit) |

## Data setup (nothing below is committed; see `.gitignore`)

* **Proxy (IDSSE, CC BY 4.0):** put the 14 files `{J03WMX,J03WN1,J03WOH,J03WOY,J03WPY,J03WQQ,J03WR9}_{clean,degraded}.npz`
  in `cache/proxy/`. Locally they are in `exp_jeremy/exp_1/cache/proxy/` (copied from exp_0's cache,
  produced by exp_0's `download_idsse.py` + degrader). Notebook 00 checks them bit-exact against the
  frozen degrader in `judge/`.
* **ryu reference arm:** `git fetch origin ryu && bash fetch_ryu_reference.sh`.
* **BAR-ATH (notebook 04 only):** the code looks for `data/tactical/BAR-ATH_tactical_atd.txt` in this
  repo, its parents, or `$MTAH_REPO`.

## Reproduce

Interpreter: the shared `.venv-torch` (torch CUDA, numpy 2.x, lightgbm, numba, nbclient).

```bash
cd exp_jeremy/exp_1
PY=/home/jeremy-mboe/Documents/Kuliah/Sem3/Machine_Learning/.venv-torch/bin/python
$PY build_notebooks.py            # (re)writes notebooks + kernel spec; pass names to rebuild only some
for nb in 00_data_protocol 01_reference_arms 02_setseq_nested 03_comparison 04_target_inference; do
  $PY run_notebook.py $nb.ipynb
done
$PY -m pytest -q -p no:cacheprovider tests/
```

Notebook 04 refuses to rebuild the candidate once `results/public_check_started.json` exists
(the public window is single-use by design).

## Hygiene

* BAR-ATH is read only inside notebook 04's kernel; cells print aggregates only and failures are
  summarised in one line by `run_notebook.py`.
* The public ground truth is opened once, after the predictor and decoder are frozen.
* exp_0 is still being edited elsewhere, so exp_1 pins its own judge snapshot and proxy copies.
