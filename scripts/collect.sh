#!/usr/bin/env bash
# Collect the gglm corpus: every query against NTRS and OSTI, PDFs included.
# usage: bash scripts/collect.sh [max-per-query]
# On Rivanna: export GGLM_DATA=/scratch/$USER/gglm first.
set -euo pipefail
cd "$(dirname "$0")/.."

MAX="${1:-200}"

QUERIES=(
    "two-stage light gas gun"
    "single-stage gas gun"
    "light gas gun"
    "sabot design"
    "sabot separation"
    "gas gun pump tube"
    "launch tube erosion"
    "hypervelocity impact"
    "impact flash"
    "ballistic range instrumentation"
    "photonic Doppler velocimetry"
    "flyer plate impact"
    "Whipple shield"
)

for q in "${QUERIES[@]}"; do
    echo "=== ntrs: $q"
    uv run python -m gglm.sources.ntrs "$q" "$MAX" --download
    echo "=== osti: $q"
    uv run python -m gglm.sources.osti "$q" "$MAX" --download
done

uv run python -m gglm.catalog
