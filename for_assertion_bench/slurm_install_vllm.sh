#!/bin/bash
#SBATCH --job-name=install_vllm
#SBATCH --output=results/slurm_install_vllm_%j.out
#SBATCH --error=results/slurm_install_vllm_%j.err
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.4.0
source "${VENV_PATH:-$HOME/venv}/bin/activate"

echo "[$(date)] Installing vllm with uv..."
echo "Python: $(python3 --version)"
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(nvcc --version | tail -1)"

uv pip install vllm 2>&1

echo ""
echo "[$(date)] Verifying installation..."
python3 -c "import vllm; print(f'vllm {vllm.__version__} installed successfully')"

echo "[$(date)] Done."
