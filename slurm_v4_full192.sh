#!/bin/bash
#SBATCH --job-name=fv_v4_full
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-31

# ===========================================================================
# Full 192-design run: cursor_style pipeline + baseline
# Array job: 32 tasks × 6 designs each = 192 designs
# ===========================================================================

MODEL="${MODEL:-qwen3.5:35b}"
CHUNK_SIZE=6
TOTAL=192

OFFSET=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
LIMIT=$CHUNK_SIZE
if [ $(( OFFSET + LIMIT )) -gt $TOTAL ]; then
    LIMIT=$(( TOTAL - OFFSET ))
fi

echo "================================================================"
echo "Array Job ${SLURM_ARRAY_JOB_ID} Task ${SLURM_ARRAY_TASK_ID}"
echo "Node     : $SLURMD_NODENAME"
echo "Model    : $MODEL"
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
export DB_PATH="${CURSOR_DIR}/chroma_db_cursor_task_${SLURM_ARRAY_TASK_ID}"

# Shared results directories
PIPELINE_DIR="${CURSOR_DIR}/results/full_pipeline"
BASELINE_DIR="${CURSOR_DIR}/results/full_baseline"
mkdir -p "$PIPELINE_DIR" "$BASELINE_DIR"

echo "[env] OLLAMA_BASE_URL        = $OLLAMA_BASE_URL"
echo "[env] OLLAMA_EMBEDDING_MODEL = $OLLAMA_EMBEDDING_MODEL"
echo "[env] DB_PATH                = $DB_PATH"

# ---------------------------------------------------------------------------
# 4. Run cursor_style PIPELINE on this chunk
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[PIPELINE] Task $SLURM_ARRAY_TASK_ID: offset=$OFFSET limit=$LIMIT"
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
echo "[BASELINE] Task $SLURM_ARRAY_TASK_ID: offset=$OFFSET limit=$LIMIT"
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
# 6. Task 0: Wait for all tasks then run evaluation + export
# ---------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo ""
    echo "[task0] Waiting for SLURM array to finish for evaluation..."
    echo "[task0] Evaluation will be run by a separate slurm_v4_eval.sh job"
fi

# ---------------------------------------------------------------------------
# 7. Cleanup
# ---------------------------------------------------------------------------
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null

echo "================================================================"
echo "Task $SLURM_ARRAY_TASK_ID done: $(date)"
echo "Pipeline exit: $PIPELINE_EXIT"
echo "Baseline exit: $BASELINE_EXIT"
echo "================================================================"
