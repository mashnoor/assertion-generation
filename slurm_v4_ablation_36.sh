#!/bin/bash
#SBATCH --job-name=fv_ablat36
#SBATCH --array=0-3
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ===========================================================================
# Ablation study on 36 designs (6 original + 30 new), 3 specs each = 108 specs
#
# Array task mapping:
#   0 = no_verify
#   1 = no_jg_tools
#   2 = no_rag
#   3 = no_tools
#
# Existing 6-design results will be skipped via --resume.
# ===========================================================================

set -euo pipefail

MODEL="${MODEL:-qwen3.5:35b}"
MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Map task ID to ablation variant
VARIANTS=(no_verify no_jg_tools no_rag no_tools)
VARIANT="${VARIANTS[$TASK_ID]}"

# All 36 design IDs: 6 original + 30 new (15 pipeline + 15 fsm)
DESIGN_IDS="ns_2-w_128-opd_2-0 ns_10-w_128-opd_3-2 ni_4_nn_4_ne_4_wd_32_opd_2_0 ni_4_nn_4_ne_4_wd_32_opd_3_0 ni_4_nn_8_ne_16_wd_32_opd_5_0 ni_16_nn_4_ne_8_wd_32_opd_5_0 ns_10-w_128-opd_3-1 ns_10-w_128-opd_5-5 ns_2-w_128-opd_2-4 ns_2-w_128-opd_2-5 ns_2-w_128-opd_4-0 ns_2-w_128-opd_4-2 ns_2-w_128-opd_4-3 ns_2-w_128-opd_5-0 ns_5-w_128-opd_2-4 ns_5-w_128-opd_2-5 ns_5-w_128-opd_3-2 ns_5-w_128-opd_4-0 ns_50-w_128-opd_2-5 ns_50-w_128-opd_3-5 ns_50-w_128-opd_4-4 ni_16_nn_16_ne_16_wd_32_opd_3_0 ni_16_nn_16_ne_32_wd_32_opd_5_0 ni_16_nn_16_ne_64_wd_32_opd_3_0 ni_16_nn_16_ne_64_wd_32_opd_5_0 ni_16_nn_4_ne_12_wd_32_opd_3_0 ni_16_nn_4_ne_16_wd_32_opd_3_0 ni_16_nn_8_ne_16_wd_32_opd_2_0 ni_16_nn_8_ne_24_wd_32_opd_3_0 ni_16_nn_8_ne_24_wd_32_opd_5_0 ni_16_nn_8_ne_32_wd_32_opd_5_0 ni_4_nn_16_ne_16_wd_32_opd_2_0 ni_4_nn_16_ne_32_wd_32_opd_4_0 ni_4_nn_4_ne_8_wd_32_opd_3_0 ni_4_nn_8_ne_32_wd_32_opd_2_0 ni_4_nn_8_ne_32_wd_32_opd_5_0"

echo "================================================================"
echo "Ablation Study — Variant: $VARIANT (task $TASK_ID)"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Model: $MODEL"
echo "Designs: 36 (6 original + 30 new)"
echo "Start: $(date)"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4
export PYTHONUNBUFFERED=1

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_DIR="$BASE_DIR"
DESIGNS_CSV="$BASE_DIR/designs.csv"
SPECS_CSV="$BASE_DIR/assertion_specs.csv"

# ---------------------------------------------------------------------------
# 2. Start Ollama
# ---------------------------------------------------------------------------
OLLAMA_PORT=$((11434 + TASK_ID))
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE=-1
~/bin/ollama serve &
OLLAMA_PID=$!

OLLAMA_READY=0
for i in $(seq 1 180); do
    if curl -s --max-time 2 "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        echo "[ollama] Ready after ${i}s"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "[ollama] ERROR: Not ready. Aborting."
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

~/bin/ollama pull "${MODEL}" 2>&1 | tail -1
~/bin/ollama pull qwen3-embedding:latest 2>&1 | tail -1

export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export DB_PATH="${CURSOR_DIR}/chroma_db_ablation36_${MODEL_TAG}_task_${TASK_ID}"

# ---------------------------------------------------------------------------
# 3. Run ablation variant
# ---------------------------------------------------------------------------
RESULT_DIR="${CURSOR_DIR}/results/${MODEL_TAG}_ablation_${VARIANT}"
mkdir -p "$RESULT_DIR"

echo ""
echo "================================================================"
echo "[ABLATION] Running variant: $VARIANT on 36 designs"
echo "================================================================"

python -u pipeline_v4.py \
    --provider ollama \
    --model "${MODEL}" \
    --designs_csv "$DESIGNS_CSV" \
    --specs_csv "$SPECS_CSV" \
    --db_path "$DB_PATH" \
    --debug_dir "$RESULT_DIR" \
    --spec_limit 3 \
    --design_ids $DESIGN_IDS \
    --ablation "$VARIANT" \
    --resume

EXIT_CODE=$?
echo "[ABLATION] $VARIANT exit: $EXIT_CODE"

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
echo "[EVAL] Evaluating $VARIANT..."
python -u evaluate_v4.py \
    --debug_dir "$RESULT_DIR" \
    --designs_csv "$DESIGNS_CSV" \
    --output "$RESULT_DIR/evaluation_results.csv" \
    --vacuity

echo "--- $VARIANT Summary ---"
cat "$RESULT_DIR/evaluation_summary.json" 2>/dev/null
echo ""

# ---------------------------------------------------------------------------
# 5. Cleanup
# ---------------------------------------------------------------------------
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null

echo "================================================================"
echo "Done: $(date)"
echo "Results: $RESULT_DIR/"
echo "================================================================"
