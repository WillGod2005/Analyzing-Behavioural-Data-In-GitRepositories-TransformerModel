#!/bin/bash
# Run all window configurations on CVEfixes enriched data (seed 42 only)
# 4 window sizes x 2 aggregations x 2 models = 16 runs
#
# LEAKAGE-FREE version (rerun 2026-11): each config is scored at tau_val, the
# threshold selected on train.py's pooled internal validation predictions
# (results/preds_<exp>_<model>.npz). Nothing is tuned on the external labels.
# Requires the seed-42 train.py run for every (model, agg, B) to exist first.
# The original March output (cvefixes_window_results.txt) is left untouched.

OUTDIR=rerun_logs/external
mkdir -p "$OUTDIR"
LOG="$OUTDIR/cvefixes_window_results_leakage_free.txt"
echo "================================================================================" > "$LOG"
echo "CVEfixes Window Configuration Sweep (LEAKAGE-FREE: threshold = tau_val)" >> "$LOG"
echo "Date: $(date)" >> "$LOG"
echo "Seed: 42 (single trained model per config)" >> "$LOG"
echo "Data: Enriched CVEfixes (212 repos, ~2404 samples)" >> "$LOG"
echo "================================================================================" >> "$LOG"
echo "" >> "$LOG"

for MODEL in transformer conv1d; do
  for AGG in before only_before; do
    if [ "$AGG" = "before" ]; then AGGNAME=before_cve; WIN=sym; else AGGNAME=only_before; WIN=causal; fi
    for BACKS in 5 10 20 40; do
      NPZ="results/preds_Aggregate.${AGGNAME}_R1_B${BACKS}_meta_${MODEL}.npz"
      RAW="$OUTDIR/enriched_${MODEL}_${WIN}_b${BACKS}_s42.log"
      echo "[$(date +%H:%M:%S)] --- Running: $MODEL | $AGG | B=$BACKS (tau from $NPZ) -> $RAW ---"
      echo "================================================================================" >> "$LOG"
      echo "MODEL=$MODEL | AGG=$AGG | BACKS=$BACKS" >> "$LOG"
      echo "================================================================================" >> "$LOG"
      if [ ! -f "$NPZ" ]; then
        echo "FATAL: missing $NPZ (run train.py for this config first)" | tee -a "$LOG"
        continue
      fi
      python3 -u evaluate_cvefixes.py -a "$AGG" --model "$MODEL" --backs "$BACKS" --meta --seed 42 --enriched \
          --tau-from-npz "$NPZ" --save-preds "$OUTDIR/extpreds_enriched_${MODEL}_${WIN}_b${BACKS}_s42.npz" \
          > "$RAW" 2>&1
      grep -v "PerformanceWarning\|frame.insert\|newframe\|cur_repo\|tensorflow/core\|^ *$\|INFO " "$RAW" >> "$LOG"
      echo "" >> "$LOG"
      echo "[$(date +%H:%M:%S)] --- Done: $MODEL | $AGG | B=$BACKS ---"
    done
  done
done

echo "================================================================================" >> "$LOG"
echo "DONE: $(date)" >> "$LOG"
echo "All 16 configurations complete."
