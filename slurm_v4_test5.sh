#!/bin/bash
#SBATCH --job-name=fv_v4_test5
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

MODEL="${MODEL:-qwen3.5:35b}"
echo "================================================================"
echo "Job ID : $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Model: $MODEL"
echo "Start  : $(date)"
echo "================================================================"

module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"

# Repo root (directory containing this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

OLLAMA_PORT=11434
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
~/bin/ollama serve &
OLLAMA_PID=$!

for i in $(seq 1 120); do
    curl -s --max-time 2 "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1 && echo "[ollama] Ready after ${i}s." && break
    sleep 1
done

~/bin/ollama pull "${MODEL}"
~/bin/ollama pull qwen3-embedding:latest

export OLLAMA_EMBEDDING_MODEL="qwen3-embedding:latest"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export DB_PATH="$REPO_DIR/chroma_db_cursor"

RESULTS_DIR="$REPO_DIR/results/v4_test5_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "[pipeline_v4] Running on 5 smallest designs"
echo "================================================================"

python -u pipeline_v4.py \
    --provider ollama \
    --model "${MODEL}" \
    --designs_csv $REPO_DIR/designs.csv \
    --specs_csv $REPO_DIR/assertion_specs.csv \
    --db_path "${DB_PATH}" \
    --debug_dir "${RESULTS_DIR}" \
    --limit 5

PIPELINE_EXIT=$?
echo "[pipeline_v4] Exit: $PIPELINE_EXIT"

echo "================================================================"
echo "[evaluate_v4] Evaluating"
echo "================================================================"

python -u evaluate_v4.py \
    --debug_dir "${RESULTS_DIR}" \
    --designs_csv $REPO_DIR/designs.csv \
    --output "${RESULTS_DIR}/evaluation_results.csv" \
    --vacuity

cat "${RESULTS_DIR}/evaluation_summary.json" 2>/dev/null || echo "(no summary)"

kill "$OLLAMA_PID" 2>/dev/null; wait "$OLLAMA_PID" 2>/dev/null
echo "================================================================"
echo "Done: $(date)"
echo "================================================================"
