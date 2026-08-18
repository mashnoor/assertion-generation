#!/bin/bash
#SBATCH --job-name=ab_hf
#SBATCH --output=results/slurm_ab_hf_%A_%a.out
#SBATCH --error=results/slurm_ab_hf_%A_%a.err
#SBATCH --array=0-15
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

###############################################################################
# AssertionBench experiment with HuggingFace transformers local inference
#
# Model is loaded directly via transformers (no server needed).
# Ollama runs only for embeddings (qwen3-embedding).
# 16 tasks x 8 designs = 128 slots (covers all 122 designs)
#
# Usage:
#   HF_MODEL="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" sbatch slurm_ab_hf.sh
#
# After completion, run eval:
#   MODEL_TAG="..." sbatch --dependency=afterok:<JOB_ID> slurm_ab_eval.sh
###############################################################################

set -euo pipefail

# ---- Config ----
HF_MODEL="${HF_MODEL:-Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled}"
MODEL_TAG="${MODEL_TAG:-$(echo "$HF_MODEL" | sed 's/[:.\/]/_/g')}"
DESIGNS_PER_TASK=8
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Ollama port (for embeddings only)
OLLAMA_PORT=$((11434 + TASK_ID))

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB_DIR="${PROJECT_DIR}/for_assertion_bench"
BASELINE_DIR="${AB_DIR}/results/${MODEL_TAG}_baseline"
PIPELINE_DIR="${AB_DIR}/results/${MODEL_TAG}_pipeline"
DB_PATH="${AB_DIR}/chroma_db_ab_task_${TASK_ID}"

# ---- Module + venv ----
module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.4.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"

# ---- Environment ----
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

# Verify GPU
echo "[$(date)] Task ${TASK_ID}: GPU info:"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

# ---- Start ollama for embeddings ----
echo "[$(date)] Starting ollama on port ${OLLAMA_PORT} (embeddings only)..."
OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" ~/bin/ollama serve &
OLLAMA_PID=$!

for i in $(seq 1 60); do
    if curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "[$(date)] Ollama ready after ${i}s"
        break
    fi
    sleep 3
done

echo "[$(date)] Pulling embedding model..."
~/bin/ollama pull "qwen3-embedding:latest" 2>&1 | tail -1

# ---- Compute offset/limit ----
OFFSET=$((TASK_ID * DESIGNS_PER_TASK))

echo "[$(date)] Task ${TASK_ID}: offset=${OFFSET} limit=${DESIGNS_PER_TASK}"
echo "[$(date)] HF Model: ${HF_MODEL}"
echo "[$(date)] Model tag: ${MODEL_TAG}"

# ---- Create output dirs ----
mkdir -p "${BASELINE_DIR}" "${PIPELINE_DIR}"

# ---- Run baseline ----
echo "[$(date)] ===== BASELINE ====="
cd "${AB_DIR}"
python baseline_ab.py \
    --debug_dir "${BASELINE_DIR}" \
    --provider hf \
    --model "${HF_MODEL}" \
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
    --provider hf \
    --model "${HF_MODEL}" \
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
