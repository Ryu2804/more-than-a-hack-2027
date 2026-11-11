# Pre-registration Template (Common Benchmark)

## 1. Hypothesis
State the expected effect and direction.

## 2. Data and split
- Dataset: IDSSE
- DEV matches:
- CONFIRM matches (not used for tuning):
- Degradation version (hash):

## 3. Method
- Model/decoder:
- Parameters to tune:
- Selection objective:

## 4. Gates (pre-run)
- Pre-gate: PREREG hash recorded before execution.
- Post-gate: results JSON + metrics complete + no NaN.

## 5. Success criteria
- spellF1 improvement with 95% CI excluding 0.
- Majority wins (>= 4/5 or majority).
- Possession count error within ±30% (or justified).
- No regression on confirm set.

## 6. Hygiene
- Public window used only as sanity veto.
- No hand labels on BAR-ATH.
- No competition data uploaded to cloud LLM.
- Byte-identical rebuild and hash recorded.