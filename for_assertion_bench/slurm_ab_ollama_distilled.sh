#!/bin/bash
#SBATCH --job-name=ab_distilled
#SBATCH --output=results/slurm_ab_distilled_%A_%a.out
#SBATCH --error=results/slurm_ab_distilled_%A_%a.err
#SBATCH --array=0-15
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

###############################################################################
# AssertionBench: Jackrong distilled model via ollama (GGUF Q4_K_M, 16GB)
#
# Uses ollama for BOTH the LLM and embeddings — proven fast path.
# Model: kwangsuklee/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF
# 16 tasks x 8 designs = 128 slots (covers all 122 designs)
#
# Usage:
#   sbatch slurm_ab_ollama_distilled.sh
###############################################################################

set -euo pipefail

# ---- Config ----
LLM_MODEL="kwangsuklee/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:latest"
MODEL_TAG="Qwen3_5-27B-Claude-Distilled"
DESIGNS_PER_TASK=8
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Unique ollama port per task
OLLAMA_PORT=$((11434 + TASK_ID))

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB_DIR="${PROJECT_DIR}/for_assertion_bench"
BASELINE_DIR="${AB_DIR}/results/${MODEL_TAG}_baseline"
PIPELINE_DIR="${AB_DIR}/results/${MODEL_TAG}_pipeline"
DB_PATH="${AB_DIR}/chroma_db_ab_task_${TASK_ID}"

# ---- Module + venv ----
module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"

# ---- Environment ----
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

# Verify GPU
echo "[$(date)] Task ${TASK_ID}: GPU info:"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

# ---- Start ollama ----
echo "[$(date)] Starting ollama on port ${OLLAMA_PORT}..."
OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" ~/bin/ollama serve &
OLLAMA_PID=$!

for i in $(seq 1 60); do
    if curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "[$(date)] Ollama ready after ${i}s"
        break
    fi
    sleep 3
done

# Pull models (fast if already cached)
echo "[$(date)] Pulling LLM model..."
~/bin/ollama pull "${LLM_MODEL}" 2>&1 | tail -1
echo "[$(date)] Pulling embedding model..."
~/bin/ollama pull "qwen3-embedding:latest" 2>&1 | tail -1

# ---- Compute offset/limit ----
OFFSET=$((TASK_ID * DESIGNS_PER_TASK))

echo "[$(date)] Task ${TASK_ID}: offset=${OFFSET} limit=${DESIGNS_PER_TASK}"
echo "[$(date)] LLM Model: ${LLM_MODEL}"
echo "[$(date)] Model tag: ${MODEL_TAG}"

# ---- Create output dirs ----
mkdir -p "${BASELINE_DIR}" "${PIPELINE_DIR}"

# ---- Run baseline ----
echo "[$(date)] ===== BASELINE ====="
cd "${AB_DIR}"
python baseline_ab.py \
    --debug_dir "${BASELINE_DIR}" \
    --provider ollama \
    --model "${LLM_MODEL}" \
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
    --model "${LLM_MODEL}" \
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
