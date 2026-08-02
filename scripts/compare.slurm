#!/bin/bash
# Comparison-model eval: three lm_eval tasks + the custom split -> eval/compare/<slug>.
#   sbatch scripts/compare.slurm meta-llama/Llama-3.1-8B-Instruct
# Monitor:  squeue --me     tail -f logs/gglm-compare-<jobid>.out
#
#SBATCH --job-name=gglm-compare
#SBATCH -A ds5002
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
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

# both comparison models are gated; die before burning a gpu allocation
[ -s "$HF_HOME/token" ] || { echo "no HF token at $HF_HOME/token" >&2; exit 1; }

uv run --no-sync python scripts/compare_eval.py --model "$1" "${@:2}"
