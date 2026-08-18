#!/bin/bash
# Submit 12 separate FVEval Design2SVA jobs (6 pipeline + 6 FSM)
# Usage: ./submit_fveval.sh [MODEL] [NUM_TRIALS]

MODEL="${1:-qwen3.5:35b}"
NUM_TRIALS="${2:-5}"

echo "Submitting 12 jobs: MODEL=${MODEL}, NUM_TRIALS=${NUM_TRIALS}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_fveval_single.sh"

for TASK_ID in $(seq 0 11); do
    JOB_ID=$(TASK_ID=$TASK_ID MODEL=$MODEL NUM_TRIALS=$NUM_TRIALS sbatch --export=ALL "$SLURM_SCRIPT" | awk '{print $NF}')
    echo "  Task ${TASK_ID}: job ${JOB_ID}"
done

echo "Done."
