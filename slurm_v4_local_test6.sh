#!/bin/bash
#SBATCH --job-name=fv_v4_local6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# ===========================================================================
# Local model test: qwen3.5:35b + qwen3-embedding on 6 designs
# Runs cursor_style pipeline + baseline + evaluation
# ===========================================================================

MODEL="${MODEL:-qwen3.5:35b}"
DESIGN_IDS="ns_2-w_128-opd_2-0 ns_10-w_128-opd_3-2 ni_4_nn_4_ne_4_wd_32_opd_2_0 ni_4_nn_4_ne_4_wd_32_opd_3_0 ni_4_nn_8_ne_16_wd_32_opd_5_0 ni_16_nn_4_ne_8_wd_32_opd_5_0"

echo "================================================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Model   : $MODEL"
echo "Designs : $DESIGN_IDS"
echo "Start   : $(date)"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------
module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_DIR="$BASE_DIR"
DESIGNS_CSV="$CURSOR_DIR/test_6_designs.csv"
SPECS_CSV="$BASE_DIR/assertion_specs.csv"

# ---------------------------------------------------------------------------
# 2. Start Ollama
# ---------------------------------------------------------------------------
OLLAMA_PORT=11434
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
~/bin/ollama serve &
OLLAMA_PID=$!

OLLAMA_READY=0
for i in $(seq 1 120); do
    if curl -s --max-time 2 "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        echo "[ollama] Ready after ${i}s."
        OLLAMA_READY=1
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "[ollama] ERROR: Not ready after 120s. Aborting."
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Pull models
# ---------------------------------------------------------------------------
echo "[ollama] Pulling ${MODEL}..."
~/bin/ollama pull "${MODEL}"

echo "[ollama] Pulling qwen3-embedding:latest..."
~/bin/ollama pull qwen3-embedding:latest

# ---------------------------------------------------------------------------
# 4. Environment variables
# ---------------------------------------------------------------------------
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export DB_PATH="${CURSOR_DIR}/chroma_db_cursor_local"

echo "[env] OLLAMA_BASE_URL         = $OLLAMA_BASE_URL"
echo "[env] OLLAMA_EMBEDDING_MODEL  = $OLLAMA_EMBEDDING_MODEL"
echo "[env] DB_PATH                 = $DB_PATH"

# ---------------------------------------------------------------------------
# 5. Results directories
# ---------------------------------------------------------------------------
PIPELINE_DIR="${CURSOR_DIR}/results/local_test6_pipeline_${SLURM_JOB_ID}"
BASELINE_DIR="${CURSOR_DIR}/results/local_test6_baseline_${SLURM_JOB_ID}"
mkdir -p "$PIPELINE_DIR" "$BASELINE_DIR"

# ---------------------------------------------------------------------------
# 6. Run cursor_style PIPELINE on 6 designs (3 specs each)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[PIPELINE] Starting cursor_style pipeline (6 designs, 3 specs each)"
echo "================================================================"

python -u pipeline_v4.py \
    --provider ollama \
    --model "${MODEL}" \
    --designs_csv "$DESIGNS_CSV" \
    --specs_csv "$SPECS_CSV" \
    --db_path "$DB_PATH" \
    --debug_dir "$PIPELINE_DIR" \
    --spec_limit 3 \
    --design_ids $DESIGN_IDS

PIPELINE_EXIT=$?
echo "[PIPELINE] Exit code: $PIPELINE_EXIT"

# ---------------------------------------------------------------------------
# 7. Run BASELINE on 6 designs (3 specs each)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[BASELINE] Starting baseline (6 designs, 3 specs each)"
echo "================================================================"

python -u baseline_v4.py \
    --designs_csv "$DESIGNS_CSV" \
    --specs_csv "$SPECS_CSV" \
    --debug_dir "$BASELINE_DIR" \
    --model "${MODEL}" \
    --spec_limit 3 \
    --design_ids $DESIGN_IDS

BASELINE_EXIT=$?
echo "[BASELINE] Exit code: $BASELINE_EXIT"

# ---------------------------------------------------------------------------
# 8. Evaluate PIPELINE
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[EVAL] Evaluating pipeline results"
echo "================================================================"

python -u evaluate_v4.py \
    --debug_dir "$PIPELINE_DIR" \
    --designs_csv "$DESIGNS_CSV" \
    --output "$PIPELINE_DIR/evaluation_results.csv" \
    --vacuity

echo ""
echo "--- Pipeline Summary ---"
cat "$PIPELINE_DIR/evaluation_summary.json" 2>/dev/null || echo "(no summary)"

# ---------------------------------------------------------------------------
# 9. Evaluate BASELINE
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[EVAL] Evaluating baseline results"
echo "================================================================"

python -u evaluate_v4.py \
    --debug_dir "$BASELINE_DIR" \
    --designs_csv "$DESIGNS_CSV" \
    --output "$BASELINE_DIR/evaluation_results.csv" \
    --vacuity

echo ""
echo "--- Baseline Summary ---"
cat "$BASELINE_DIR/evaluation_summary.json" 2>/dev/null || echo "(no summary)"

# ---------------------------------------------------------------------------
# 10. Side-by-side comparison
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "COMPARISON: Pipeline vs Baseline (local ${MODEL})"
echo "================================================================"
echo ""
echo "Pipeline results:"
cat "$PIPELINE_DIR/evaluation_summary.json" 2>/dev/null
echo ""
echo "Baseline results:"
cat "$BASELINE_DIR/evaluation_summary.json" 2>/dev/null
echo ""

# ---------------------------------------------------------------------------
# 11. Cleanup
# ---------------------------------------------------------------------------
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null

echo "================================================================"
echo "Done: $(date)"
echo "Pipeline dir : $PIPELINE_DIR"
echo "Baseline dir : $BASELINE_DIR"
echo "Pipeline exit: $PIPELINE_EXIT"
echo "Baseline exit: $BASELINE_EXIT"
echo "================================================================"

[ "$PIPELINE_EXIT" -ne 0 ] && exit "$PIPELINE_EXIT"
[ "$BASELINE_EXIT" -ne 0 ] && exit "$BASELINE_EXIT"
exit 0
