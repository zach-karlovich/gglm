#!/bin/bash
# Full corpus collect on Rivanna -> /scratch, then parse + chunk.
# Network I/O, no GPU, so this runs on a standard (CPU) node.
#
# Build the venv on a login node first:  module load uv && uv sync
# Submit:   sbatch scripts/collect.slurm
# Monitor:  squeue --me     tail -f logs/gglm-collect-<jobid>.out
# Cancel:   scancel <jobid>
#
#SBATCH --job-name=gglm-collect
#SBATCH -A ds5002
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=8000
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddf4me@virginia.edu
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load uv
export GGLM_DATA="/scratch/$USER/gglm"
mkdir -p "$GGLM_DATA" logs

# NTRS deep (stays on-topic), OSTI moderate (its full-text search is noisy).
bash scripts/collect.sh 200 50
uv run --no-sync python scripts/parse_all.py
uv run --no-sync python -m gglm.chunk

echo "corpus at $GGLM_DATA"
du -sh "$GGLM_DATA"/raw "$GGLM_DATA"/chunks/chunks.jsonl
