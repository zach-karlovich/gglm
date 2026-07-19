#!/usr/bin/env bash
# setup_olmocr.sh — build the olmOCR environment on Rivanna. Run once, on a
# login node (needs internet; installs a large GPU stack incl. torch + vllm).
#
# olmOCR (allenai/olmOCR-2-7B-1025-FP8, a 7B VLM) re-OCRs the corpus PDFs that
# parse.py flagged 'digital-degraded' / 'scanned' / 'mixed'. It is CUDA-only,
# needs an NVIDIA GPU with >=12 GB VRAM, and pins its own torch/vllm — so it
# gets its OWN venv in /scratch and never touches gglm's .venv. No Docker.
set -euo pipefail

command -v module &>/dev/null && module load uv || true

OLMO_DIR="/scratch/$USER/olmocr-env"
export PYTHONNOUSERSITE=1

echo "==> olmOCR venv: $OLMO_DIR"
uv venv --python 3.11 "$OLMO_DIR"

# GPU wheels from PyTorch's CUDA 12.8 index (matches olmOCR's install docs).
uv pip install -p "$OLMO_DIR" \
  "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# poppler (pdftoppm) is a runtime dependency olmOCR shells out to for page
# rendering. There's no root here, so try a module; fall back to conda-forge.
module load poppler 2>/dev/null || true
if command -v pdftoppm &>/dev/null; then
  echo "==> poppler: $(command -v pdftoppm)"
else
  echo "!!  poppler (pdftoppm) NOT on PATH."
  echo "    Try:  module spider poppler   (then module load the version it lists)"
  echo "    Or:   conda create -y -p /scratch/$USER/poppler -c conda-forge poppler"
  echo "          then prepend /scratch/$USER/poppler/bin to PATH in olmocr.slurm"
fi

"$OLMO_DIR/bin/python" -c "import olmocr, sys; print('olmocr OK', getattr(olmocr,'__version__','?'))"
echo "==> Done. Run olmOCR over the queue with: sbatch scripts/olmocr.slurm"
