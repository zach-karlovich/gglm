# gglm

Gas Gun Language Model — a RAG assistant for light gas gun (LGG) and
hypervelocity impact (HVI) literature. UVA DS5002 final project.

The domain is narrow and the stakes are lopsided: a wrong firing parameter is
worse than no answer. gglm answers only from documents it can cite, and refuses
when the corpus doesn't cover the question.

## Setup

```
uv sync
uv run pytest
```

## Pipeline

Bulk data lives under `$GGLM_DATA` (default `data/`); on Rivanna, point it at
scratch. The catalog stays repo-local so a scratch purge can't take it.

```
export GGLM_DATA=/scratch/$USER/gglm
bash scripts/collect.sh 200 50     # download PDFs — NTRS deep, OSTI shallow
uv run python scripts/parse_all.py # page text + kind classification
uv run python -m gglm.chunk        # 300-word windows -> chunks.jsonl
```

## Sources

Public, keyless, distribution-unlimited only: NTRS (NASA) and OSTI (DOE). Every
document is tracked in `data/catalog.jsonl` — an append-only provenance log
(title, authors, citation URL, license, sha256, kind) that doubles as the
license audit. Chunks carry their catalog key, so answers cite real sources by
number, not chunk indices. Raw PDFs aren't redistributed; the citation URLs make
the corpus reconstructible.

## Notes

- **OCR** — parse labels each PDF `digital` / `digital-degraded` / `scanned` /
  `mixed`; degraded and scanned docs queue for re-OCR with olmOCR on Rivanna.
- **Retrieval** — hybrid BM25 + dense (RRF): BM25 catches report numbers and
  acronyms, dense catches paraphrase.
- **Corpus** — NTRS deep, OSTI shallow; OSTI's full-text search is noisy at depth.
- **Refusal** — below a calibrated retrieval score, gglm declines rather than guess.

## Test fixtures

Mock & Holt (1976), NSWC/DL TR-3473 (`digital`); Rynearson & Rand (1972),
TEES-9075-CR-72-02 (`digital-degraded`) — public-domain US government reports.
