#!/bin/bash
#SBATCH --job-name=assertionbench
#SBATCH --output=results/slurm_ab_%A_%a.out
#SBATCH --error=results/slurm_ab_%A_%a.err
#SBATCH --array=0-15
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

###############################################################################
# AssertionBench experiment: baseline + pipeline on ~122 designs
# 16 tasks × 8 designs/task = 128 slots (covers all 122 designs)
#
# Usage:
#   sbatch slurm_ab.sh                           # default: qwen3.5:35b
#   MODEL="mistral-small3.1:24b" sbatch slurm_ab.sh
#
# After completion, run eval:
#   sbatch --dependency=afterok:<JOB_ID> slurm_ab_eval.sh
###############################################################################

set -euo pipefail

# ---- Config ----
MODEL="${MODEL:-qwen3.5:35b}"
MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')
DESIGNS_PER_TASK=8
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB_DIR="${PROJECT_DIR}/for_assertion_bench"
BASELINE_DIR="${AB_DIR}/results/${MODEL_TAG}_baseline"
PIPELINE_DIR="${AB_DIR}/results/${MODEL_TAG}_pipeline"
DB_PATH="${AB_DIR}/chroma_db_ab_task_${TASK_ID}"

# ---- Module + venv ----
module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"

# ---- Ollama setup ----
OLLAMA_PORT=$((11434 + TASK_ID))
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

echo "[$(date)] Task ${TASK_ID}: starting ollama on port ${OLLAMA_PORT}"
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE=-1
~/bin/ollama serve &
OLLAMA_PID=$!

# Wait for ollama to be ready
OLLAMA_READY=0
for i in $(seq 1 180); do
    if curl -s --max-time 2 "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "[$(date)] Ollama ready after ${i}s"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "[$(date)] ERROR: Ollama not ready after 180s. Aborting."
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

# Pull models
echo "[$(date)] Pulling model ${MODEL}..."
~/bin/ollama pull "${MODEL}" 2>&1 | tail -1
echo "[$(date)] Pulling embedding model..."
~/bin/ollama pull "qwen3-embedding:latest" 2>&1 | tail -1

# ---- Compute offset/limit for this task ----
OFFSET=$((TASK_ID * DESIGNS_PER_TASK))

echo "[$(date)] Task ${TASK_ID}: offset=${OFFSET} limit=${DESIGNS_PER_TASK}"
echo "[$(date)] Model: ${MODEL} (tag: ${MODEL_TAG})"

# ---- Create output dirs ----
mkdir -p "${BASELINE_DIR}" "${PIPELINE_DIR}"

# ---- Run baseline ----
echo "[$(date)] ===== BASELINE ====="
cd "${AB_DIR}"
python baseline_ab.py \
    --debug_dir "${BASELINE_DIR}" \
    --provider ollama \
    --model "${MODEL}" \
    --offset "${OFFSET}" \
    --limit "${DESIGNS_PER_TASK}" \
    --design_timeout 600 \
    --resume \
    2>&1 || echo "[$(date)] Baseline finished with errors"

# ---- Run pipeline ----
echo "[$(date)] ===== PIPELINE ====="
python pipeline_ab.py \
    --debug_dir "${PIPELINE_DIR}" \
    --db_path "${DB_PATH}" \
    --provider ollama \
    --model "${MODEL}" \
    --offset "${OFFSET}" \
    --limit "${DESIGNS_PER_TASK}" \
    --design_timeout 900 \
    --max_verify 3 \
    --resume \
    2>&1 || echo "[$(date)] Pipeline finished with errors"

# ---- Cleanup ----
echo "[$(date)] Stopping ollama..."
kill ${OLLAMA_PID} 2>/dev/null || true
wait ${OLLAMA_PID} 2>/dev/null || true

echo "[$(date)] Task ${TASK_ID} complete."
