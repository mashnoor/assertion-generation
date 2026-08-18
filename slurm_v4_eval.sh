#!/bin/bash
#SBATCH --job-name=fv_v4_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00

# ===========================================================================
# Evaluation + Results Export (no GPU needed — just SSH to the JasperGold host for JG)
# Run after array job completes:
#   sbatch --dependency=afterok:<array_job_id> slurm_v4_eval.sh
# ===========================================================================

echo "================================================================"
echo "Eval Job $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
echo "================================================================"

module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_DIR="$BASE_DIR"
DESIGNS_CSV="$BASE_DIR/designs.csv"

PIPELINE_DIR="${CURSOR_DIR}/results/full_pipeline"
BASELINE_DIR="${CURSOR_DIR}/results/full_baseline"
EXPORT_DIR="${CURSOR_DIR}/results/paper_results"

mkdir -p "$EXPORT_DIR"

# ---------------------------------------------------------------------------
# 1. Evaluate PIPELINE results
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[EVAL] Evaluating pipeline results: $PIPELINE_DIR"
echo "================================================================"

python -u evaluate_v4.py \
    --debug_dir "$PIPELINE_DIR" \
    --designs_csv "$DESIGNS_CSV" \
    --output "$PIPELINE_DIR/evaluation_results.csv" \
    --vacuity

echo "--- Pipeline Summary ---"
cat "$PIPELINE_DIR/evaluation_summary.json" 2>/dev/null

# ---------------------------------------------------------------------------
# 2. Evaluate BASELINE results
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[EVAL] Evaluating baseline results: $BASELINE_DIR"
echo "================================================================"

python -u evaluate_v4.py \
    --debug_dir "$BASELINE_DIR" \
    --designs_csv "$DESIGNS_CSV" \
    --output "$BASELINE_DIR/evaluation_results.csv" \
    --vacuity

echo "--- Baseline Summary ---"
cat "$BASELINE_DIR/evaluation_summary.json" 2>/dev/null

# ---------------------------------------------------------------------------
# 3. Evaluate ABLATION results (if they exist)
# ---------------------------------------------------------------------------
for ABLATION in no_verify no_jg_tools no_rag no_tools; do
    ABLATION_DIR="${CURSOR_DIR}/results/ablation_${ABLATION}"
    if [ -d "$ABLATION_DIR" ]; then
        echo ""
        echo "================================================================"
        echo "[EVAL] Evaluating ablation: $ABLATION"
        echo "================================================================"
        python -u evaluate_v4.py \
            --debug_dir "$ABLATION_DIR" \
            --designs_csv "$DESIGNS_CSV" \
            --output "$ABLATION_DIR/evaluation_results.csv" \
            --vacuity
        echo "--- $ABLATION Summary ---"
        cat "$ABLATION_DIR/evaluation_summary.json" 2>/dev/null
    fi
done

# ---------------------------------------------------------------------------
# 4. Export comprehensive results for paper
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[EXPORT] Generating paper results"
echo "================================================================"

# Build the command dynamically based on which directories exist
DIRS=""
LABELS=""

if [ -d "$PIPELINE_DIR" ]; then
    DIRS="$DIRS $PIPELINE_DIR"
    LABELS="$LABELS Pipeline"
fi
if [ -d "$BASELINE_DIR" ]; then
    DIRS="$DIRS $BASELINE_DIR"
    LABELS="$LABELS Baseline"
fi
for ABLATION in no_verify no_jg_tools no_rag no_tools; do
    ABLATION_DIR="${CURSOR_DIR}/results/ablation_${ABLATION}"
    if [ -d "$ABLATION_DIR" ]; then
        DIRS="$DIRS $ABLATION_DIR"
        LABELS="$LABELS Ablation_${ABLATION}"
    fi
done

if [ -n "$DIRS" ]; then
    python -u export_results.py \
        --results_dirs $DIRS \
        --labels $LABELS \
        --designs_csv "$DESIGNS_CSV" \
        --output_dir "$EXPORT_DIR"
fi

# ---------------------------------------------------------------------------
# 5. Copy all summary files into export dir
# ---------------------------------------------------------------------------
echo ""
echo "Copying evaluation summaries to $EXPORT_DIR..."
cp "$PIPELINE_DIR/evaluation_summary.json" "$EXPORT_DIR/pipeline_eval_summary.json" 2>/dev/null
cp "$PIPELINE_DIR/evaluation_results.csv" "$EXPORT_DIR/pipeline_eval_results.csv" 2>/dev/null
cp "$PIPELINE_DIR/run_metadata.json" "$EXPORT_DIR/pipeline_run_metadata.json" 2>/dev/null
cp "$BASELINE_DIR/evaluation_summary.json" "$EXPORT_DIR/baseline_eval_summary.json" 2>/dev/null
cp "$BASELINE_DIR/evaluation_results.csv" "$EXPORT_DIR/baseline_eval_results.csv" 2>/dev/null

for ABLATION in no_verify no_jg_tools no_rag no_tools; do
    ABLATION_DIR="${CURSOR_DIR}/results/ablation_${ABLATION}"
    if [ -d "$ABLATION_DIR" ]; then
        cp "$ABLATION_DIR/evaluation_summary.json" "$EXPORT_DIR/ablation_${ABLATION}_eval_summary.json" 2>/dev/null
        cp "$ABLATION_DIR/evaluation_results.csv" "$EXPORT_DIR/ablation_${ABLATION}_eval_results.csv" 2>/dev/null
    fi
done

echo ""
echo "================================================================"
echo "FINAL COMPARISON"
echo "================================================================"
echo ""
echo "Pipeline:"
cat "$PIPELINE_DIR/evaluation_summary.json" 2>/dev/null
echo ""
echo "Baseline:"
cat "$BASELINE_DIR/evaluation_summary.json" 2>/dev/null
echo ""
for ABLATION in no_verify no_jg_tools no_rag no_tools; do
    ABLATION_DIR="${CURSOR_DIR}/results/ablation_${ABLATION}"
    if [ -d "$ABLATION_DIR" ]; then
        echo "Ablation ($ABLATION):"
        cat "$ABLATION_DIR/evaluation_summary.json" 2>/dev/null
        echo ""
    fi
done

echo "================================================================"
echo "Done: $(date)"
echo "Paper results: $EXPORT_DIR/"
echo "================================================================"
