#!/bin/bash
#SBATCH --job-name=fv_v4
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ---------------------------------------------------------------------------
# MODEL can be overridden at submission time:
#   MODEL=qwen3.5:35b sbatch slurm_v4.sh
# ---------------------------------------------------------------------------
MODEL="${MODEL:-qwen3.5:35b}"

echo "================================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Model         : $MODEL"
echo "Start time    : $(date)"
echo "Working dir   : $(pwd)"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. Load Python module and activate venv
# ---------------------------------------------------------------------------
module load python/python-3.11.4-gcc-12.2.0

source "${VENV_PATH:-$HOME/venv}/bin/activate"

# Repo root (directory containing this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 2. Thread limits (prevents numpy / ChromaDB Rust binding crashes on HPC)
# ---------------------------------------------------------------------------
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

# ---------------------------------------------------------------------------
# 3. Start Ollama on port 11434
# ---------------------------------------------------------------------------
OLLAMA_PORT=11434
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"

echo "[ollama] Starting ollama serve on port ${OLLAMA_PORT}..."
~/bin/ollama serve &
OLLAMA_PID=$!
echo "[ollama] PID = $OLLAMA_PID"

# ---------------------------------------------------------------------------
# 4. Wait for Ollama to be ready (poll curl for up to 120 seconds)
# ---------------------------------------------------------------------------
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
    echo "[ollama] ERROR: Ollama did not become ready within 120s. Aborting."
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Pull required models
# ---------------------------------------------------------------------------
echo "[ollama] Pulling ${MODEL}..."
~/bin/ollama pull "${MODEL}"

echo "[ollama] Pulling qwen3-embedding:latest..."
~/bin/ollama pull qwen3-embedding:latest

# ---------------------------------------------------------------------------
# 6. Export environment variables for pipeline
# ---------------------------------------------------------------------------
export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"

# Fresh ChromaDB for cursor_style — separate from old pipeline_v3 collection
export DB_PATH="$REPO_DIR/chroma_db_cursor"

echo "[env] OLLAMA_BASE_URL   = $OLLAMA_BASE_URL"
echo "[env] OLLAMA_EMBEDDING_MODEL = $OLLAMA_EMBEDDING_MODEL"
echo "[env] DB_PATH           = $DB_PATH"

# ---------------------------------------------------------------------------
# 7. Create results dir if it does not exist
# ---------------------------------------------------------------------------
RESULTS_DIR="$REPO_DIR/results/v4_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# 8. Run pipeline_v4.py
#    Uses big_designs.csv + big_specs.csv from new_plan/
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[pipeline_v4] Starting pipeline run"
echo "================================================================"

python pipeline_v4.py \
    --provider ollama \
    --model "${MODEL}" \
    --designs_csv $REPO_DIR/designs.csv \
    --specs_csv $REPO_DIR/assertion_specs.csv \
    --db_path "${DB_PATH}" \
    --debug_dir "${RESULTS_DIR}" \
    --resume

PIPELINE_EXIT=$?
echo "[pipeline_v4] Exit code: $PIPELINE_EXIT"

# ---------------------------------------------------------------------------
# 9. Run evaluate_v4.py --vacuity
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[evaluate_v4] Starting evaluation"
echo "================================================================"

python evaluate_v4.py \
    --debug_dir "${RESULTS_DIR}" \
    --designs_csv $REPO_DIR/designs.csv \
    --output "${RESULTS_DIR}/evaluation_results.csv" \
    --vacuity

EVAL_EXIT=$?
echo "[evaluate_v4] Exit code: $EVAL_EXIT"

# ---------------------------------------------------------------------------
# 10. Print summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[summary] evaluation_summary.json"
echo "================================================================"
SUMMARY="${RESULTS_DIR}/evaluation_summary.json"
if [ -f "$SUMMARY" ]; then
    cat "$SUMMARY"
else
    echo "(evaluation_summary.json not found — check eval logs)"
fi

# ---------------------------------------------------------------------------
# 11. Kill Ollama
# ---------------------------------------------------------------------------
echo ""
echo "[ollama] Stopping ollama (PID $OLLAMA_PID)..."
kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true
echo "[ollama] Stopped."

echo ""
echo "================================================================"
echo "Job finished at: $(date)"
echo "Pipeline exit  : $PIPELINE_EXIT"
echo "Eval exit      : $EVAL_EXIT"
echo "Results dir    : $RESULTS_DIR"
echo "================================================================"

# Propagate non-zero exit from pipeline or eval
[ "$PIPELINE_EXIT" -ne 0 ] && exit "$PIPELINE_EXIT"
[ "$EVAL_EXIT" -ne 0 ]     && exit "$EVAL_EXIT"
exit 0
