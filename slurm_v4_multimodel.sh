#!/bin/bash
#SBATCH --job-name=fv_v4_mm
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-31

# ===========================================================================
# Multi-model 192-design run: cursor_style pipeline + baseline
# Usage: MODEL=llama3.1:70b sbatch slurm_v4_multimodel.sh
# ===========================================================================

MODEL="${MODEL:?ERROR: MODEL env var must be set (e.g. MODEL=llama3.1:70b)}"
CHUNK_SIZE=6
TOTAL=192

OFFSET=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
LIMIT=$CHUNK_SIZE
if [ $(( OFFSET + LIMIT )) -gt $TOTAL ]; then
    LIMIT=$(( TOTAL - OFFSET ))
fi

# Derive a filesystem-safe model tag: llama3.1:70b -> llama31_70b
MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')

echo "================================================================"
echo "Array Job ${SLURM_ARRAY_JOB_ID} Task ${SLURM_ARRAY_TASK_ID}"
echo "Node     : $SLURMD_NODENAME"
echo "Model    : $MODEL"
echo "Model Tag: $MODEL_TAG"
echo "Offset   : $OFFSET  Limit: $LIMIT"
echo "Start    : $(date)"
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
# 2. Start Ollama (unique port per task)
# ---------------------------------------------------------------------------
OLLAMA_PORT=$(( 11434 + SLURM_ARRAY_TASK_ID ))
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE=-1
~/bin/ollama serve &
OLLAMA_PID=$!

OLLAMA_READY=0
for i in $(seq 1 180); do
    if curl -s --max-time 2 "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        echo "[ollama] Ready after ${i}s (port $OLLAMA_PORT)"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "[ollama] ERROR: Not ready after 180s. Aborting."
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

~/bin/ollama pull "${MODEL}" 2>&1 | tail -1
~/bin/ollama pull qwen3-embedding:latest 2>&1 | tail -1

# ---------------------------------------------------------------------------
# 3. Environment variables
# ---------------------------------------------------------------------------
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"

# Per-task ChromaDB (avoids concurrent writes)
export DB_PATH="${CURSOR_DIR}/chroma_db_${MODEL_TAG}_task_${SLURM_ARRAY_TASK_ID}"

# Model-specific results directories
PIPELINE_DIR="${CURSOR_DIR}/results/${MODEL_TAG}_pipeline"
BASELINE_DIR="${CURSOR_DIR}/results/${MODEL_TAG}_baseline"
mkdir -p "$PIPELINE_DIR" "$BASELINE_DIR"

echo "[env] OLLAMA_BASE_URL        = $OLLAMA_BASE_URL"
echo "[env] OLLAMA_EMBEDDING_MODEL = $OLLAMA_EMBEDDING_MODEL"
echo "[env] DB_PATH                = $DB_PATH"
echo "[env] PIPELINE_DIR           = $PIPELINE_DIR"
echo "[env] BASELINE_DIR           = $BASELINE_DIR"

# ---------------------------------------------------------------------------
# 4. Run cursor_style PIPELINE on this chunk
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[PIPELINE] Task $SLURM_ARRAY_TASK_ID: offset=$OFFSET limit=$LIMIT model=$MODEL"
echo "================================================================"

python -u pipeline_v4.py \
    --provider ollama \
    --model "${MODEL}" \
    --designs_csv "$DESIGNS_CSV" \
    --specs_csv "$SPECS_CSV" \
    --db_path "$DB_PATH" \
    --debug_dir "$PIPELINE_DIR" \
    --spec_limit 3 \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --resume

PIPELINE_EXIT=$?
echo "[PIPELINE] Task $SLURM_ARRAY_TASK_ID exit: $PIPELINE_EXIT"

# ---------------------------------------------------------------------------
# 5. Run BASELINE on this chunk
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[BASELINE] Task $SLURM_ARRAY_TASK_ID: offset=$OFFSET limit=$LIMIT model=$MODEL"
echo "================================================================"

python -u baseline_v4.py \
    --designs_csv "$DESIGNS_CSV" \
    --specs_csv "$SPECS_CSV" \
    --debug_dir "$BASELINE_DIR" \
    --model "${MODEL}" \
    --spec_limit 3 \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --resume

BASELINE_EXIT=$?
echo "[BASELINE] Task $SLURM_ARRAY_TASK_ID exit: $BASELINE_EXIT"

# ---------------------------------------------------------------------------
# 6. Cleanup
# ---------------------------------------------------------------------------
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null

echo "================================================================"
echo "Task $SLURM_ARRAY_TASK_ID done: $(date)"
echo "Model: $MODEL ($MODEL_TAG)"
echo "Pipeline exit: $PIPELINE_EXIT"
echo "Baseline exit: $BASELINE_EXIT"
echo "================================================================"
