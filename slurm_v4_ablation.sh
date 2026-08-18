#!/bin/bash
#SBATCH --job-name=fv_v4_ablat
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# ===========================================================================
# Ablation study on 6 test designs (3 specs each = 18 specs per variant)
#
# Variants:
#   1. full        — All tools + verification (reference, already done)
#   2. no_verify   — Context gathering + generation, NO JG verification loop
#   3. no_jg_tools — Only ChromaDB/RAG tools, no JasperGold tools
#   4. no_rag      — Only JG tools, no ChromaDB semantic search
#   5. no_tools    — No tool calls, just enhanced prompt + generation
#   6. baseline    — Raw RTL + spec → LLM (already done, reference)
# ===========================================================================

MODEL="${MODEL:-qwen3.5:35b}"
DESIGN_IDS="ns_2-w_128-opd_2-0 ns_10-w_128-opd_3-2 ni_4_nn_4_ne_4_wd_32_opd_2_0 ni_4_nn_4_ne_4_wd_32_opd_3_0 ni_4_nn_8_ne_16_wd_32_opd_5_0 ni_16_nn_4_ne_8_wd_32_opd_5_0"

echo "================================================================"
echo "Ablation Study"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Model: $MODEL"
echo "Designs: $DESIGN_IDS"
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
DESIGNS_CSV="$CURSOR_DIR/test_6_designs.csv"
SPECS_CSV="$BASE_DIR/assertion_specs.csv"

# ---------------------------------------------------------------------------
# 2. Start Ollama
# ---------------------------------------------------------------------------
OLLAMA_PORT=11434
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
export DB_PATH="${CURSOR_DIR}/chroma_db_cursor_ablation"

# ---------------------------------------------------------------------------
# 3. Run ablation variants
# ---------------------------------------------------------------------------
ABLATION_VARIANTS="no_verify no_jg_tools no_rag no_tools"

for VARIANT in $ABLATION_VARIANTS; do
    RESULT_DIR="${CURSOR_DIR}/results/ablation_${VARIANT}"
    mkdir -p "$RESULT_DIR"

    echo ""
    echo "================================================================"
    echo "[ABLATION] Variant: $VARIANT"
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
        --ablation "$VARIANT"

    EXIT_CODE=$?
    echo "[ABLATION] $VARIANT exit: $EXIT_CODE"

    # Run evaluation
    echo "[EVAL] Evaluating $VARIANT..."
    python -u evaluate_v4.py \
        --debug_dir "$RESULT_DIR" \
        --designs_csv "$DESIGNS_CSV" \
        --output "$RESULT_DIR/evaluation_results.csv" \
        --vacuity

    echo "--- $VARIANT Summary ---"
    cat "$RESULT_DIR/evaluation_summary.json" 2>/dev/null
    echo ""
done

# ---------------------------------------------------------------------------
# 4. Export ablation comparison
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[EXPORT] Ablation comparison"
echo "================================================================"

# Include full pipeline + baseline from previous runs if available
EXPORT_DIR="${CURSOR_DIR}/results/ablation_comparison"
mkdir -p "$EXPORT_DIR"

DIRS=""
LABELS=""

# Reference: full pipeline from local test
FULL_DIR="${CURSOR_DIR}/results/local_test6_pipeline_568228"
if [ -d "$FULL_DIR" ]; then
    DIRS="$DIRS $FULL_DIR"
    LABELS="$LABELS Full_Pipeline"
fi

# Reference: baseline from local test
BASE_DIR_REF="${CURSOR_DIR}/results/local_test6_baseline_568228"
if [ -d "$BASE_DIR_REF" ]; then
    DIRS="$DIRS $BASE_DIR_REF"
    LABELS="$LABELS Baseline"
fi

# Ablation variants
for VARIANT in $ABLATION_VARIANTS; do
    RESULT_DIR="${CURSOR_DIR}/results/ablation_${VARIANT}"
    if [ -d "$RESULT_DIR" ]; then
        DIRS="$DIRS $RESULT_DIR"
        LABELS="$LABELS Ablation_${VARIANT}"
    fi
done

python -u export_results.py \
    --results_dirs $DIRS \
    --labels $LABELS \
    --designs_csv "$DESIGNS_CSV" \
    --output_dir "$EXPORT_DIR"

# ---------------------------------------------------------------------------
# 5. Print final comparison
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "ABLATION STUDY RESULTS"
echo "================================================================"
echo ""

echo "Full Pipeline (reference):"
cat "$FULL_DIR/evaluation_summary.json" 2>/dev/null
echo ""
echo "Baseline (reference):"
cat "$BASE_DIR_REF/evaluation_summary.json" 2>/dev/null
echo ""
for VARIANT in $ABLATION_VARIANTS; do
    echo "Ablation ($VARIANT):"
    cat "${CURSOR_DIR}/results/ablation_${VARIANT}/evaluation_summary.json" 2>/dev/null
    echo ""
done

# ---------------------------------------------------------------------------
# 6. Cleanup
# ---------------------------------------------------------------------------
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null

echo "================================================================"
echo "Done: $(date)"
echo "Ablation results: $EXPORT_DIR/"
echo "================================================================"
