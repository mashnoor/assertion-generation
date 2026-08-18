#!/bin/bash
#SBATCH --job-name=fv_v4_mm_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00

# ===========================================================================
# Evaluation for multi-model runs (no GPU needed — SSH to the JasperGold host for JG)
# Usage: MODEL=llama3.1:70b sbatch slurm_v4_multimodel_eval.sh
#   or:  sbatch --dependency=afterok:<array_job_id> slurm_v4_multimodel_eval.sh
# ===========================================================================

MODEL="${MODEL:?ERROR: MODEL env var must be set (e.g. MODEL=llama3.1:70b)}"
MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')

echo "================================================================"
echo "Eval Job $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Model: $MODEL ($MODEL_TAG)"
echo "Start: $(date)"
echo "================================================================"

module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_DIR="$BASE_DIR"
DESIGNS_CSV="$BASE_DIR/designs.csv"

PIPELINE_DIR="${CURSOR_DIR}/results/${MODEL_TAG}_pipeline"
BASELINE_DIR="${CURSOR_DIR}/results/${MODEL_TAG}_baseline"
EXPORT_DIR="${CURSOR_DIR}/results/paper_results"

mkdir -p "$EXPORT_DIR"

# ---------------------------------------------------------------------------
# 1. Evaluate PIPELINE results
# ---------------------------------------------------------------------------
if [ -d "$PIPELINE_DIR" ]; then
    echo ""
    echo "================================================================"
    echo "[EVAL] Pipeline: $PIPELINE_DIR"
    echo "================================================================"

    python -u evaluate_v4.py \
        --debug_dir "$PIPELINE_DIR" \
        --designs_csv "$DESIGNS_CSV" \
        --output "$PIPELINE_DIR/evaluation_results.csv" \
        --vacuity

    echo "--- ${MODEL_TAG} Pipeline Summary ---"
    cat "$PIPELINE_DIR/evaluation_summary.json" 2>/dev/null
    cp "$PIPELINE_DIR/evaluation_summary.json" \
       "$EXPORT_DIR/${MODEL_TAG}_pipeline_eval_summary.json" 2>/dev/null
    cp "$PIPELINE_DIR/evaluation_results.csv" \
       "$EXPORT_DIR/${MODEL_TAG}_pipeline_eval_results.csv" 2>/dev/null
else
    echo "[WARN] Pipeline dir not found: $PIPELINE_DIR"
fi

# ---------------------------------------------------------------------------
# 2. Evaluate BASELINE results
# ---------------------------------------------------------------------------
if [ -d "$BASELINE_DIR" ]; then
    echo ""
    echo "================================================================"
    echo "[EVAL] Baseline: $BASELINE_DIR"
    echo "================================================================"

    python -u evaluate_v4.py \
        --debug_dir "$BASELINE_DIR" \
        --designs_csv "$DESIGNS_CSV" \
        --output "$BASELINE_DIR/evaluation_results.csv" \
        --vacuity

    echo "--- ${MODEL_TAG} Baseline Summary ---"
    cat "$BASELINE_DIR/evaluation_summary.json" 2>/dev/null
    cp "$BASELINE_DIR/evaluation_summary.json" \
       "$EXPORT_DIR/${MODEL_TAG}_baseline_eval_summary.json" 2>/dev/null
    cp "$BASELINE_DIR/evaluation_results.csv" \
       "$EXPORT_DIR/${MODEL_TAG}_baseline_eval_results.csv" 2>/dev/null
else
    echo "[WARN] Baseline dir not found: $BASELINE_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Print comparison
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "COMPARISON: $MODEL ($MODEL_TAG)"
echo "================================================================"
echo ""
echo "Pipeline:"
cat "$PIPELINE_DIR/evaluation_summary.json" 2>/dev/null
echo ""
echo "Baseline:"
cat "$BASELINE_DIR/evaluation_summary.json" 2>/dev/null
echo ""
echo "================================================================"
echo "Done: $(date)"
echo "Results copied to: $EXPORT_DIR/"
echo "================================================================"
