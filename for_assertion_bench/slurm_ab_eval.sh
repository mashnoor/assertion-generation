#!/bin/bash
#SBATCH --job-name=ab_eval
#SBATCH --output=results/slurm_ab_eval_%j.out
#SBATCH --error=results/slurm_ab_eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

###############################################################################
# Evaluate AssertionBench results (no GPU needed, just JG via SSH)
#
# Usage:
#   MODEL="qwen3.5:35b" sbatch slurm_ab_eval.sh
#   MODEL="qwen3.5:35b" sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm_ab_eval.sh
###############################################################################

set -euo pipefail

MODEL="${MODEL:-qwen3.5:35b}"
MODEL_TAG=$(echo "$MODEL" | sed 's/[:.\/]/_/g')

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB_DIR="${PROJECT_DIR}/for_assertion_bench"

module load python/python-3.11.4-gcc-12.2.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

cd "${AB_DIR}"

echo "[$(date)] ===== Evaluating BASELINE ====="
python evaluate_ab.py \
    --debug_dir "results/${MODEL_TAG}_baseline" \
    --output "results/${MODEL_TAG}_baseline_eval.csv" \
    --resume \
    2>&1

echo ""
echo "[$(date)] ===== Evaluating PIPELINE ====="
python evaluate_ab.py \
    --debug_dir "results/${MODEL_TAG}_pipeline" \
    --output "results/${MODEL_TAG}_pipeline_eval.csv" \
    --resume \
    2>&1

echo ""
echo "[$(date)] ===== COMPARISON ====="
python3 -c "
import json, os
for method in ['baseline', 'pipeline']:
    path = f'results/${MODEL_TAG}_{method}/evaluation_summary.json'
    if os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        print(f'{method.upper():10s}: syntax={s[\"avg_syntax\"]:.3f}  '
              f'pass_rate={s[\"avg_pass_rate\"]:.3f}  '
              f'proven={s[\"total_proven\"]}/{s[\"total_assertions\"]}')
    else:
        print(f'{method.upper():10s}: (no results)')
"

echo ""
echo "[$(date)] Evaluation complete."
