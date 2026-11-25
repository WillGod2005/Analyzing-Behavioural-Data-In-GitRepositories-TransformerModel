#!/usr/bin/env bash
# Re-run the CVEfixes window sweep on the truly-unseen subset only
# (case-insensitive dedup against training set).
# Mirrors the original tab:external-comparison: seed 42, transformer + conv1d,
# sym + causal, B = 5/10/20/40.
#
# LEAKAGE-FREE version (rerun 2026-11): threshold = tau_val from train.py's pooled
# internal validation (results/preds_<exp>_<model>.npz), not tuned on external labels.
# Uses the miniconda python3 (TF 2.21, same as training) instead of venv (TF 2.20).

set -u
OUTDIR=rerun_logs/external
mkdir -p "$OUTDIR"
OUT="$OUTDIR/cvefixes_unseen_window_sweep_leakage_free.txt"
: > "$OUT"

for ARCH in transformer conv1d; do
  for AGG in before only_before; do
    if [ "$AGG" = "before" ]; then AGGNAME=before_cve; WIN=sym; else AGGNAME=only_before; WIN=causal; fi
    for B in 5 10 20 40; do
      NPZ="results/preds_Aggregate.${AGGNAME}_R1_B${B}_meta_${ARCH}.npz"
      RAW="$OUTDIR/unseen_${ARCH}_${WIN}_b${B}_s42.log"
      echo "================================================================================" | tee -a "$OUT"
      echo "MODEL=$ARCH | AGG=$AGG | BACKS=$B" | tee -a "$OUT"
      echo "================================================================================" | tee -a "$OUT"
      if [ ! -f "$NPZ" ]; then
        echo "FATAL: missing $NPZ (run train.py for this config first)" | tee -a "$OUT"
        continue
      fi
      python3 -u evaluate_cvefixes.py -a "$AGG" --model "$ARCH" --backs "$B" \
          --meta --enriched --exclude-train-overlap --seed 42 \
          --tau-from-npz "$NPZ" --save-preds "$OUTDIR/extpreds_unseen_${ARCH}_${WIN}_b${B}_s42.npz" \
          > "$RAW" 2>&1
      grep -E "(LEAKAGE-FREE|THRESHOLD|Accuracy|Precision|Recall|F1 Score|ROC-AUC|Avg Precision|Confusion|Excluded|Remaining|Test samples|Test repos|Total samples|^---|FATAL|ERROR|Loading|SUMMARY)" "$RAW" \
        | tee -a "$OUT"
      echo "" | tee -a "$OUT"
    done
  done
done

echo "================================================================================" | tee -a "$OUT"
echo "DONE: results in $OUT" | tee -a "$OUT"
