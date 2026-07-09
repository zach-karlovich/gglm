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
