# gglm

Gas Gun Language Model. A RAG assistant for light gas gun (LGG) and
hypervelocity impact (HVI) scientific literature. UVA DS5002 final project.

## Setup

```
uv sync
uv run pytest
```

Each document is labeled `digital`, `digital-degraded`, `scanned`, or `mixed`
based on text coverage and OCR quality. `digital-degraded` has a text layer but
letter-spaced legacy OCR — ingestible now, queue for re-OCR (olmOCR, on
Rivanna). `scanned` documents need OCR from scratch.

Test fixtures: Mock & Holt (1976), NSWC/DL TR-3473; Rynearson & Rand (1972),
TEES-9075-CR-72-02 — US government reports, public domain.

## Data layout

Bulk data (raw PDFs, parsed pages, chunks) lives under `$GGLM_DATA`, default
`data/`. On Rivanna point it at scratch; the catalog stays in the repo since
scratch purges idle files:

```
export GGLM_DATA=/scratch/$USER/gglm
bash scripts/collect.sh            # NTRS + OSTI queries, download PDFs
uv run python scripts/parse_all.py # page JSONL + kind distribution
uv run python -m gglm.chunk        # retrieval-ready chunks
```

`data/catalog.jsonl` (provenance, licenses, citations) is always repo-local.
