#!/bin/bash
# Quick progress monitor for cursor_style v4 runs
CURSOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "cursor_style v4 Progress Monitor — $(date)"
echo "=================================================="

# Active SLURM jobs
echo ""
echo "Active jobs:"
squeue -u "${SLURM_USER:-$USER}" -o "%.10i %.2t %.10M %j %R" 2>/dev/null | grep -v "^$" | head -20

# Full pipeline progress
PIPE_DIR="$CURSOR_DIR/results/full_pipeline"
if [ -d "$PIPE_DIR" ]; then
    PIPE_DESIGNS=$(find "$PIPE_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
    PIPE_SVAS=$(find "$PIPE_DIR" -name "sva_assertion.sv" 2>/dev/null | wc -l)
    echo ""
    echo "Full Pipeline: $PIPE_DESIGNS designs, $PIPE_SVAS SVA files"
fi

# Full baseline progress
BASE_DIR="$CURSOR_DIR/results/full_baseline"
if [ -d "$BASE_DIR" ]; then
    BASE_DESIGNS=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
    BASE_SVAS=$(find "$BASE_DIR" -name "sva_assertion.sv" 2>/dev/null | wc -l)
    echo "Full Baseline: $BASE_DESIGNS designs, $BASE_SVAS SVA files"
fi

# Ablation progress
for VARIANT in no_verify no_jg_tools no_rag no_tools; do
    ABL_DIR="$CURSOR_DIR/results/ablation_${VARIANT}"
    if [ -d "$ABL_DIR" ]; then
        ABL_SVAS=$(find "$ABL_DIR" -name "sva_assertion.sv" 2>/dev/null | wc -l)
        echo "Ablation ($VARIANT): $ABL_SVAS SVA files"
    fi
done

# Evaluation status
for DIR in "$PIPE_DIR" "$BASE_DIR"; do
    SUMMARY="$DIR/evaluation_summary.json"
    if [ -f "$SUMMARY" ]; then
        echo ""
        echo "Evaluation: $(basename $DIR)"
        cat "$SUMMARY"
    fi
done

echo ""
echo "=================================================="
