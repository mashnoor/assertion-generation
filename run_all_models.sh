#!/bin/bash
# ===========================================================================
# Submit pipeline + baseline + eval for all open-source models
# Usage: ./run_all_models.sh
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODELS=("mixtral:8x7b" "llama3.1:70b" "llama3.1:8b")

echo "================================================================"
echo "Submitting experiments for ${#MODELS[@]} models"
echo "Models: ${MODELS[*]}"
echo "================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')

    echo "--- Submitting: $MODEL ($MODEL_TAG) ---"

    # Submit the array job (32 tasks × 6 designs = 192 designs)
    ARRAY_JOB_ID=$(MODEL="$MODEL" sbatch --parsable slurm_v4_multimodel.sh)
    echo "  Array job: $ARRAY_JOB_ID"

    # Submit eval job chained after array completes
    EVAL_JOB_ID=$(MODEL="$MODEL" sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} slurm_v4_multimodel_eval.sh)
    echo "  Eval  job: $EVAL_JOB_ID (depends on $ARRAY_JOB_ID)"

    echo ""
done

echo "================================================================"
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in: results/<model_tag>_{pipeline,baseline}/"
echo "Eval summaries copied to: results/paper_results/"
echo "================================================================"
