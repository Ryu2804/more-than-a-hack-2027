# Technical report (IEEE, Typst)

`report.typ` is the team technical report (built copy at the repository root: `TECHNICAL_REPORT.pdf`) for the locked method `farras_wave5_short_gru25_dead45`.
Numbers are loaded from `../final_method.json` and `report_numbers.json`; nothing is typed by hand
except the EDA statistics in Table I, which are outputs of `experiments/E00_eda/E00_eda.ipynb`.

```bash
python docs/report/make_figures.py      # figures/*.pdf + report_numbers.json (re-scores the frozen CSV with the official scorer)
typst compile --root docs docs/report/report.typ docs/report/report.pdf
```

`make_figures.py` reads the frozen CSV from release commit `7254990` via `git show`, so the
`farras` history must be fetched. It needs only matplotlib, numpy and pandas; no match data.
The IEEE layout comes from the Typst Universe package `@preview/charged-ieee:0.1.4`.
