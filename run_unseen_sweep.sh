#!/usr/bin/env bash
# Re-run the CVEfixes window sweep on the truly-unseen subset only
# (case-insensitive dedup against training set).
# Mirrors the original tab:external-comparison: seed 42, transformer + conv1d,
# sym + causal, B = 5/10/20/40.

set -u
mkdir -p logs
OUT=logs/cvefixes_unseen_window_sweep.txt
: > "$OUT"

source venv/bin/activate

for ARCH in transformer conv1d; do
  for AGG in before only_before; do
    for B in 5 10 20 40; do
      echo "================================================================================" | tee -a "$OUT"
      echo "MODEL=$ARCH | AGG=$AGG | BACKS=$B" | tee -a "$OUT"
      echo "================================================================================" | tee -a "$OUT"
      python evaluate_cvefixes.py -a "$AGG" --model "$ARCH" --backs "$B" \
          --meta --enriched --exclude-train-overlap --seed 42 2>&1 \
        | grep -E "(Accuracy|Precision|Recall|F1 Score|Confusion|Excluded|Remaining|Test samples|Test repos|Total samples|^---|FATAL|ERROR|Loading)" \
        | tee -a "$OUT"
      echo "" | tee -a "$OUT"
    done
  done
done

echo "================================================================================" | tee -a "$OUT"
echo "DONE: results in $OUT" | tee -a "$OUT"
