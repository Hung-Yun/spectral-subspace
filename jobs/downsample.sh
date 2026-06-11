#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --output=logs/downsample-%j.out
#SBATCH --error=logs/downsample-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
mkdir -p logs

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch jobs/downsample.sh SUBJECT EMU_ID [REGION ...]"
  echo "Example: sbatch jobs/downsample.sh YFB 44 HPC"
  exit 1
fi

SUBJECT="$1"
EMU_ID="$2"
shift 2

REGION_ARGS=()
EXTRA_ARGS=()
REGIONS=()

for arg in "$@"; do
  case "$arg" in
    --no-match-sessions)
      EXTRA_ARGS+=(--no-match-sessions)
      ;;
    *)
      REGIONS+=("$arg")
      ;;
  esac
done

if [ "${#REGIONS[@]}" -gt 0 ]; then
  REGION_ARGS=(--regions "${REGIONS[@]}")
fi

echo "Starting downsample job"
echo "Subject: ${SUBJECT}"
echo "EMU ID: ${EMU_ID}"
echo "Regions: ${REGIONS[*]:-HPC}"
echo "No match sessions: $([ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo yes || echo no)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Repo dir: ${REPO_DIR}"

uv run python "${REPO_DIR}/downsample.py" \
  --subject "${SUBJECT}" \
  --emu-id "${EMU_ID}" \
  "${REGION_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
