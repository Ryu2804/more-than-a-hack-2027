#!/usr/bin/env bash
# Extract ryu's final pipeline (code + weights_new) read-only from the pinned ryu-branch commit
# into reference/ryu/, which exp1_ryu.py imports. Run inside a clone that has fetched `ryu`:
#   git fetch origin ryu && bash fetch_ryu_reference.sh
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=bd90a9c2439b4ed957e716c39801d74bbae6ec19
DEST=reference/ryu
mkdir -p "$DEST/weights_new"
for f in experiments/features.py experiments/common.py experiments/other_games.py \
         experiments/015_final_pipeline/decoder.py experiments/015_final_pipeline/train_model.py \
         experiments/015_final_pipeline/run.py experiments/015_final_pipeline/NOTES.md; do
  git show "$COMMIT:$f" > "$DEST/$(basename "$f")"
done
for f in meta.npz tcn_seed0.pt tcn_seed1.pt tcn_seed2.pt training_report.json; do
  git show "$COMMIT:experiments/015_final_pipeline/weights_new/$f" > "$DEST/weights_new/$f"
done
echo "$COMMIT" > "$DEST/SOURCE_COMMIT"
echo "ryu reference extracted to $DEST"
