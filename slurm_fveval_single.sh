#!/bin/bash
#SBATCH --job-name=fveval_d2sva
#SBATCH --output=results/slurm_fveval_%A.out
#SBATCH --error=results/slurm_fveval_%A.err
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00

# Parameters passed via env vars:
#   TASK_ID   0-11  (0-5 = pipeline, 6-11 = fsm)
#   MODEL     (default: qwen3.5:35b)
#   NUM_TRIALS (default: 5)

set -euo pipefail

MODEL="${MODEL:-qwen3.5:35b}"
MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')
NUM_TRIALS="${NUM_TRIALS:-5}"
DESIGNS_PER_TASK=16
TASK_ID="${TASK_ID:?TASK_ID must be set}"

if [ "$TASK_ID" -lt 6 ]; then
    DATASET="design2sva_pipeline"
    LOCAL_TASK=$TASK_ID
else
    DATASET="design2sva_fsm"
    LOCAL_TASK=$((TASK_ID - 6))
fi

OFFSET=$((LOCAL_TASK * DESIGNS_PER_TASK))
OLLAMA_PORT=$((11434 + TASK_ID))

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_DIR="$PROJECT_DIR"
DATASET_CSV="$PROJECT_DIR/FVEval/data_design2sva/data/${DATASET}.csv"
OUTPUT_DIR="${CURSOR_DIR}/results/fveval_${MODEL_TAG}_${DATASET}"
DB_PATH="${OUTPUT_DIR}/chroma_task_${TASK_ID}"

echo "================================================================"
echo "FVEval Design2SVA: Task ${TASK_ID} (${DATASET})"
echo "  Offset  : ${OFFSET}  Limit: ${DESIGNS_PER_TASK}"
echo "  Model   : ${MODEL}"
echo "  Trials  : ${NUM_TRIALS}"
echo "  Port    : ${OLLAMA_PORT}"
echo "  Start   : $(date)"
echo "================================================================"

module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4
export PYTHONUNBUFFERED=1

export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OLLAMA_KEEP_ALIVE=-1

~/bin/ollama serve &
OLLAMA_PID=$!

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

~/bin/ollama pull "${MODEL}" 2>&1 | tail -1
~/bin/ollama pull "qwen3-embedding:latest" 2>&1 | tail -1

mkdir -p "${OUTPUT_DIR}"
cd "${CURSOR_DIR}"

python -u run_fveval_design2sva.py \
    --dataset_csv "${DATASET_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --provider ollama \
    --model "${MODEL}" \
    --num_trials "${NUM_TRIALS}" \
    --offset "${OFFSET}" \
    --limit "${DESIGNS_PER_TASK}" \
    --db_path "${DB_PATH}" \
    --max_verify 3 \
    --design_timeout 600 \
    --resume

EXIT_CODE=$?

kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true

echo "================================================================"
echo "Task ${TASK_ID} done: $(date) | Exit: ${EXIT_CODE}"
echo "================================================================"
