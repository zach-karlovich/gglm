#!/usr/bin/env bash
# Collect the gglm corpus: every query against NTRS and OSTI, PDFs included.
# usage: bash scripts/collect.sh [ntrs-max] [osti-max]
#   one arg  -> same depth for both (back-compat)
#   two args -> NTRS deep, OSTI shallower. OSTI's full-text search matches very
#               broadly (tens of thousands of loose hits), so cap it lower to
#               keep off-topic docs out of the corpus.
# On Rivanna: export GGLM_DATA=/scratch/$USER/gglm first.
set -euo pipefail
cd "$(dirname "$0")/.."

NTRS_MAX="${1:-200}"
OSTI_MAX="${2:-$NTRS_MAX}"

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
    uv run python -m gglm.sources.ntrs "$q" "$NTRS_MAX" --download
    echo "=== osti: $q"
    uv run python -m gglm.sources.osti "$q" "$OSTI_MAX" --download
done

uv run python -m gglm.catalog
