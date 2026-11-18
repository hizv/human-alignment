#!/bin/bash
#SBATCH --job-name=navon-align
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/navon/slurm-%j.out
# Uncomment and adjust if your cluster requires a specific account/GPU type:
##SBATCH --account=your_account

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/navon

export PATH="/storage/ice1/4/5/nparikh44/.local/bin:$PATH"
export UV_CACHE_DIR="/storage/ice1/4/5/nparikh44/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/storage/ice1/4/5/nparikh44/.local/share/uv/python"
export UV_TOOL_DIR="/storage/ice1/4/5/nparikh44/.local/share/uv/tools"
export UV_TOOL_BIN_DIR="/storage/ice1/4/5/nparikh44/.local/bin"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
nvidia-smi || true

if [ ! -f outputs/navon/manifest.json ]; then
    .venv/bin/python -m human_alignment.navon generate
fi

.venv/bin/python -m human_alignment.navon embed
.venv/bin/python -m human_alignment.navon analyze
