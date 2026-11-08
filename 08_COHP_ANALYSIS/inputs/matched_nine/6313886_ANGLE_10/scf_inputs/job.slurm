#!/bin/bash
#SBATCH --job-name=C06__ANGLE_10
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -p debug
#SBATCH -w node03
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# MEMORY_REQUEST_HOLD: certify peak and scheduler memory before use
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?}"
: "${COHP_AUTHOR_APPROVAL_FILE:?No_author_approval_file}"
python3 -B run_lobster_guarded.py
