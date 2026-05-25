#!/bin/bash
#SBATCH --job-name=simulate_sh
#SBATCH --output=logs/simulate_sh-%j.out
#SBATCH --error=logs/simulate_sh-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs
CACHE_DIR="${SLURM_SUBMIT_DIR:-$PWD}/.cache"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

N_SEEDS="${1:-10}"
SIM_DURATION_S="${2:-600}"
OUTPUT_SUBDIR="${3:-}"

echo "Starting simulate_sh job"
echo "Seeds: ${N_SEEDS}"
echo "Duration (s): ${SIM_DURATION_S}"
echo "Output subdir: ${OUTPUT_SUBDIR}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"

uv run python -u simulate_sh.py \
  --n-seeds "${N_SEEDS}" \
  --sim-duration-s "${SIM_DURATION_S}" \
  --output-subdir "${OUTPUT_SUBDIR}"
