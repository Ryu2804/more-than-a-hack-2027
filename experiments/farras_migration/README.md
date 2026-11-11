# Farras migration (pending)

Farras wave2 operates on ryu cache + Metrica sample. To migrate to common benchmark:
- Replace Metrica training with IDSSE degraded inputs.
- Port HGB + Viterbi to per-match prediction.
- Run common benchmark runner.

Status: scaffold only; requires data conversion and training.
