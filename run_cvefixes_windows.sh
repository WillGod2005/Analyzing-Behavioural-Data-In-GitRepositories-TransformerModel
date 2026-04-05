#!/bin/bash
# Run all window configurations on CVEfixes enriched data (seed 42 only)
# 4 window sizes x 2 aggregations x 2 models = 16 runs

LOG="cvefixes_window_results.txt"
echo "================================================================================" > "$LOG"
echo "CVEfixes Window Configuration Sweep" >> "$LOG"
echo "Date: $(date)" >> "$LOG"
echo "Seed: 42 (single trained model per config)" >> "$LOG"
echo "Data: Enriched CVEfixes (212 repos, ~2404 samples)" >> "$LOG"
echo "================================================================================" >> "$LOG"
echo "" >> "$LOG"

for MODEL in transformer conv1d; do
  for AGG in before only_before; do
    for BACKS in 5 10 20 40; do
      echo "--- Running: $MODEL | $AGG | B=$BACKS ---"
      echo "================================================================================" >> "$LOG"
      echo "MODEL=$MODEL | AGG=$AGG | BACKS=$BACKS" >> "$LOG"
      echo "================================================================================" >> "$LOG"
      python -u evaluate_cvefixes.py -a "$AGG" --model "$MODEL" --backs "$BACKS" --meta --seed 42 --enriched 2>&1 | \
        tee -a /tmp/cvefixes_raw_output.txt | \
        grep -v "PerformanceWarning\|frame.insert\|newframe\|cur_repo\|tensorflow/core\|^ *$\|INFO " >> "$LOG"
      echo "" >> "$LOG"
      echo "--- Done: $MODEL | $AGG | B=$BACKS ---"
    done
  done
done

echo "================================================================================" >> "$LOG"
echo "DONE: $(date)" >> "$LOG"
echo "All 16 configurations complete."
