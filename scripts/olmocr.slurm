#!/bin/bash
# Run olmOCR over the corpus PDFs that need re-OCR -> markdown in $GGLM_DATA/ocr.
# Requires the olmocr env from scripts/setup_olmocr.sh and a parsed catalog
# (so kinds are populated). GPU job.
#
# Submit:   sbatch scripts/olmocr.slurm
# Monitor:  squeue --me     tail -f logs/gglm-olmocr-<jobid>.out
#
#SBATCH --job-name=gglm-olmocr
#SBATCH -A ds5002
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48000
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddf4me@virginia.edu
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load uv
module load poppler 2>/dev/null || true          # see setup_olmocr.sh if missing
export GGLM_DATA="/scratch/$USER/gglm"
OLMO="/scratch/$USER/olmocr-env"
mkdir -p "$GGLM_DATA/ocr" logs

# Build the re-OCR queue from the catalog (degraded/scanned/mixed with a PDF).
mapfile -t QUEUE < <(uv run --no-sync python scripts/olmocr_queue.py)
echo "re-OCR queue: ${#QUEUE[@]} PDFs"
[[ ${#QUEUE[@]} -eq 0 ]] && { echo "nothing to OCR"; exit 0; }

"$OLMO/bin/olmocr" "$GGLM_DATA/ocr" --markdown --pdfs "${QUEUE[@]}"

echo "markdown written under $GGLM_DATA/ocr/markdown/"
