#!/bin/bash
# Extended comparison: Transformer vs Conv1D for 3 additional dataset seeds
# Seeds: 7, 256, 999 (to complement existing 42, 123)
# Window sizes: 5, 10, 20, 40
# Aggregation: before (symmetric), only_before (causal)
#
# Results appended to comparison_results_extra.txt

set -e

RESULTS_FILE="comparison_results_extra.txt"
echo "========================================" >> "$RESULTS_FILE"
echo "EXTRA SEEDS COMPARISON: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

SEEDS="7 256 999"
WINDOWS="5 10 20 40"
DATA_LOC="data_collection"
CACHE_LOC="ready_data"
EPOCHS=50
BATCH=32
KFOLDS=10

for SEED in $SEEDS; do
    for B in $WINDOWS; do
        for AGGR in "before" "only_before"; do
            if [ "$AGGR" = "before" ]; then
                POOLING="center"
                AGGR_LABEL="symmetric"
            else
                POOLING="last"
                AGGR_LABEL="causal"
            fi

            LOG_PREFIX="/tmp/comparison_${AGGR}_b${B}_s${SEED}"

            echo ""
            echo "======================================================"
            echo "  $AGGR_LABEL | B=$B | SEED=$SEED"
            echo "======================================================"

            # 1) Transformer
            echo "[$(date)] transformer | $AGGR | B=$B | seed=$SEED"
            ./venv/bin/python3 train.py \
                -a "$AGGR" \
                --model transformer \
                --pooling "$POOLING" \
                -b "$B" -r 1 -e "$EPOCHS" --batch "$BATCH" -k "$KFOLDS" \
                --metadata \
                --data-location "$DATA_LOC" \
                --cache-location "$CACHE_LOC" \
                --cache \
                --seed "$SEED" \
                2>&1 | tee "${LOG_PREFIX}_transformer.log"

            echo "" >> "$RESULTS_FILE"
            echo "--- transformer | $AGGR | B=$B | seed=$SEED ---" >> "$RESULTS_FILE"
            grep -E "CRITICAL.*(Best val|Single best|Ensemble)" "${LOG_PREFIX}_transformer.log" >> "$RESULTS_FILE" || true
            grep -E "CRITICAL (F1|Acc)" "${LOG_PREFIX}_transformer.log" | tail -4 >> "$RESULTS_FILE" || true

            # 2) Conv1D (reuses cached dataset)
            echo "[$(date)] conv1d | $AGGR | B=$B | seed=$SEED"
            ./venv/bin/python3 train.py \
                -a "$AGGR" \
                --model conv1d \
                -b "$B" -r 1 -e "$EPOCHS" --batch "$BATCH" -k "$KFOLDS" \
                --metadata \
                --data-location "$DATA_LOC" \
                --cache-location "$CACHE_LOC" \
                --cache \
                --seed "$SEED" \
                2>&1 | tee "${LOG_PREFIX}_conv1d.log"

            echo "" >> "$RESULTS_FILE"
            echo "--- conv1d | $AGGR | B=$B | seed=$SEED ---" >> "$RESULTS_FILE"
            grep -E "CRITICAL.*(Best val|Single best|Ensemble)" "${LOG_PREFIX}_conv1d.log" >> "$RESULTS_FILE" || true
            grep -E "CRITICAL (F1|Acc)" "${LOG_PREFIX}_conv1d.log" | tail -4 >> "$RESULTS_FILE" || true
        done
    done
done

echo "" >> "$RESULTS_FILE"
echo "EXTRA SEEDS COMPARISON COMPLETE: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo ""
echo "All results saved to $RESULTS_FILE"
