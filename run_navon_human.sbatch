#!/bin/bash
#SBATCH --job-name=navon-human
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/navon_human/slurm-%j.out
##SBATCH --account=your_account

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Same cluster defaults as run_navon.sbatch; environment overrides are supported.
NAVON_UV_ROOT="${NAVON_UV_ROOT:-/storage/ice1/4/5/nparikh44}"
export PATH="$NAVON_UV_ROOT/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NAVON_UV_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$NAVON_UV_ROOT/.local/share/uv/python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$NAVON_UV_ROOT/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$NAVON_UV_ROOT/.local/bin}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

: "${NAVON_HUMAN_MANIFEST:?Set NAVON_HUMAN_MANIFEST to the curated human-response JSON manifest}"
NAVON_HUMAN_FAMILY="${NAVON_HUMAN_FAMILY:-dino}"
NAVON_HUMAN_REPRESENTATION="${NAVON_HUMAN_REPRESENTATION:-pre_logits}"
NAVON_HUMAN_EMBEDDINGS="${NAVON_HUMAN_EMBEDDINGS:-outputs/navon_human/${NAVON_HUMAN_FAMILY}_base_${NAVON_HUMAN_REPRESENTATION}.npz}"
NAVON_HUMAN_OUT="${NAVON_HUMAN_OUT:-outputs/navon_human/run-${SLURM_JOB_ID:-local}}"

# Install separately with uv sync --locked before submitting this job.
# --no-sync keeps dependency installation out of the inference job.
run=(uv run --locked --no-sync python -m human_alignment.navon_human)
common=(--manifest "$NAVON_HUMAN_MANIFEST" --family "$NAVON_HUMAN_FAMILY"
        --representation "$NAVON_HUMAN_REPRESENTATION" --embeddings "$NAVON_HUMAN_EMBEDDINGS")
"${run[@]}" validate "${common[@]}"
echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
nvidia-smi || true
if [[ ! -f "$NAVON_HUMAN_EMBEDDINGS" ]]; then
    "${run[@]}" embed "${common[@]}" --models "${NAVON_MODELS:-models}" \
        --batch-size "${NAVON_HUMAN_BATCH_SIZE:-32}"
fi
"${run[@]}" train "${common[@]}" --out "$NAVON_HUMAN_OUT" \
    --epochs "${NAVON_HUMAN_EPOCHS:-500}" --seeds 0 1 2
