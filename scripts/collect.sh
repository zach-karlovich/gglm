#!/usr/bin/env bash
# Collect the gglm corpus: every query against NTRS, OSTI, and DTIC.
# usage: bash scripts/collect.sh [ntrs-max] [osti-max] [dtic-max]
#   The defaults are the depths the corpus was built with. OSTI's full-text
#   search matches very broadly (tens of thousands of loose hits), so it
#   defaults lower to keep off-topic docs out. DTIC goes through the
#   archive.org mirror; its phrase search stays on topic, so it takes the
#   NTRS depth.
# On a cluster: export GGLM_DATA=/scratch/$USER/gglm first.
set -euo pipefail
cd "$(dirname "$0")/.."

NTRS_MAX="${1:-200}"
OSTI_MAX="${2:-50}"
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
    # phenomena the first thirteen missed. Narrow phrases on purpose:
    # "composite materials" or "failure modes" match half of OSTI. No
    # flash x-ray query, that phrase belongs to solar flares and EMP
    # testing too; those papers arrive under debris cloud instead.
    "debris cloud"
    "spall fracture"
    "ballistic limit equation"
    "crater scaling"
    "shock equation of state"
    "composite overwrapped pressure vessel"
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
