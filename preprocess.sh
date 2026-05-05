#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch preprocess.slurm SUBJECT EMU_ID [REGION ...]"
  echo "Example: sbatch preprocess.slurm YFB 44 HPC"
  exit 1
fi

SUBJECT="$1"
EMU_ID="$2"
shift 2

REGION_ARGS=()
if [ "$#" -gt 0 ]; then
  REGION_ARGS=(--regions "$@")
fi

mkdir -p logs

echo "Starting preprocess job"
echo "Subject: ${SUBJECT}"
echo "EMU ID: ${EMU_ID}"
echo "Regions: ${*:-HPC}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"

# Replace this block with your cluster's environment setup.
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate spectral-subspace
fi

# Override these if the cluster sees different mount points than your laptop.
export SPECTRAL_SUBSPACE_PREFIX="${SPECTRAL_SUBSPACE_PREFIX:-${HOME}/hungyun-elias/data}"
export SPECTRAL_SUBSPACE_DATADIR="${SPECTRAL_SUBSPACE_DATADIR:-/path/to/stitched/EMU-18112}"

python preprocess.py --subject "${SUBJECT}" --emu-id "${EMU_ID}" "${REGION_ARGS[@]}"
