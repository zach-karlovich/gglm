#!/bin/bash
# Week 4 eval: demo prompt + refusal gate on the custom split -> eval/week4.
# Calibrates the threshold itself and logs the table; pass a number to
# override:  sbatch scripts/week4.slurm [0.62]
# Monitor:  squeue --me     tail -f logs/gglm-week4-<jobid>.out
#
#SBATCH --job-name=gglm-week4
#SBATCH -A ds5002
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddf4me@virginia.edu
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load uv
export GGLM_DATA="/scratch/$USER/gglm"
export HF_HOME="/scratch/$USER/models/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

uv run --no-sync python scripts/week4_eval.py ${1:+--threshold "$1"}
