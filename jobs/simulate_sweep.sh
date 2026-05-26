#!/bin/bash
#SBATCH --job-name=simulate_sweep
#SBATCH --output=logs/simulate_sweep-%j.out
#SBATCH --error=logs/simulate_sweep-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
mkdir -p logs

CACHE_DIR="${REPO_DIR}/.cache"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

N_SEEDS="${1:-10}"
SIM_DURATION_S="${2:-600}"
OUTPUT_SUBDIR="${3:-}"

echo "Starting simulate_sweep job"
echo "Seeds: ${N_SEEDS}"
echo "Duration (s): ${SIM_DURATION_S}"
echo "Output subdir: ${OUTPUT_SUBDIR}"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Repo dir: ${REPO_DIR}"

uv run python -u "${REPO_DIR}/simulate_sweep.py" \
  --n-seeds "${N_SEEDS}" \
  --sim-duration-s "${SIM_DURATION_S}" \
  --output-subdir "${OUTPUT_SUBDIR}"
