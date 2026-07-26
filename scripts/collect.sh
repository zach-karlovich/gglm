#!/usr/bin/env bash
# Collect the gglm corpus: every query against NTRS, OSTI, and DTIC.
# usage: bash scripts/collect.sh [ntrs-max] [osti-max] [dtic-max]
#   one arg  -> same depth for all (back-compat)
#   OSTI's full-text search matches very broadly (tens of thousands of loose
#   hits), so cap it lower to keep off-topic docs out of the corpus. DTIC goes
#   through the archive.org mirror; its phrase search stays on topic, so it
#   takes the NTRS depth by default.
# On Rivanna: export GGLM_DATA=/scratch/$USER/gglm first.
set -euo pipefail
cd "$(dirname "$0")/.."

NTRS_MAX="${1:-200}"
OSTI_MAX="${2:-$NTRS_MAX}"
DTIC_MAX="${3:-$NTRS_MAX}"

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

# One source failing (an API hiccup, a CDN 500) must not abandon the rest of
# the corpus: log it and carry on. The catalog records what actually landed.
collect() {
    local source="$1" query="$2" max="$3"
    echo "=== $source: $query"
    uv run python -m "gglm.sources.$source" "$query" "$max" --download \
        || echo "!!! $source failed on '$query', continuing"
}

for q in "${QUERIES[@]}"; do
    collect ntrs "$q" "$NTRS_MAX"
    collect osti "$q" "$OSTI_MAX"
    collect dtic "$q" "$DTIC_MAX"
done

uv run python -m gglm.catalog
