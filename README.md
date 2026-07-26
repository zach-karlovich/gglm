# gglm

Gas Gun Language Model: a RAG assistant for light gas gun (LGG) and
hypervelocity impact (HVI) literature. UVA DS5002 final project.

gglm answers only from documents it can cite and refuses when the corpus
doesn't cover the question. A wrong firing parameter is worse than no answer.

## Setup

```
uv python install 3.12
uv sync              # base pipeline
uv sync --group rag  # retrieval + evaluation stack (GPU-sized)
uv run pytest
```

The venv has to be built on a uv-managed interpreter. Triton JIT-compiles a
CUDA helper against the interpreter's headers the first time a kernel runs,
and a system Python without its dev headers fails there at the first
`generate` call, deep inside a traceback that looks like a torch problem.

## Pipeline

Bulk data lives under `$GGLM_DATA` (default `data/`); on Rivanna, point it at
scratch. The catalog stays repo-local so a scratch purge can't take it.

```
export GGLM_DATA=/scratch/$USER/gglm
bash scripts/collect.sh 200 50     # download PDFs, NTRS deep, OSTI shallow
uv run python scripts/parse_all.py # page text + kind classification
uv run python -m gglm.chunk        # 300-word windows -> chunks.jsonl
uv run python -m gglm.index        # embed chunks -> index/*.npy
```

Parse labels each PDF `digital` / `digital-degraded` / `scanned` / `mixed`;
chunking skips `scanned` docs, which have no usable text layer.

## Retrieval

Hybrid BM25 + dense, fused with reciprocal rank fusion and deduplicated to one
chunk per document. BM25 catches report numbers and acronyms; dense catches
paraphrase. `retrieve.COMBOS` names the arms compared in Check-in 4: bge-small
and Qwen3-Embedding under cosine and L2, bm25 alone, and the hybrid.
Generation is Qwen2.5-7B answering from numbered sources it has to cite.

## Sources

NASA NTRS, DOE OSTI, and DTIC. Public keyless APIs, distribution-unlimited
documents only. NTRS is crawled deep and stays on topic; OSTI shallow, since
its full-text search gets noisy at depth. DTIC blocks scripted clients, so
its approved-for-public-release reports come through the Internet Archive
mirror (`collection:dticarchive`); catalog entries keep both the mirror URL
and the canonical DTIC citation page.

Every download gets a line in `data/catalog.jsonl`, an append-only provenance
log (title, authors, citation URL, license, sha256, kind) that doubles as the
license audit. Chunks carry their catalog key, so answers cite documents, not
chunk indices. Raw PDFs are not redistributed; the citation URLs make the
corpus reconstructible.

## Evaluation

Two test sets live in `data/eval/`: ten hand-written LGG/HVI pairs from
Check-in 3 (`manual_pairs.jsonl`) and a synthetic split of ~100 QA pairs
written by Qwen2.5-14B from sampled chunks (`rag_test.jsonl`).

Benchmarks run pre- and post-RAG: `squadv2`, `gsm8k_cot`, and `arc_challenge`
through lm_eval, plus the synthetic split answered with and without retrieval.
Scoring is deterministic: token F1 and substring hit for answers, support
recall and hit@5 for retrieval. Every run writes paired results/samples JSONs
to `$GGLM_DATA/eval/checkin4/`; render the captured metrics as markdown with

```
uv run python -m gglm.eval.report
```

Planned: a refusal threshold calibrated on retrieval scores, so gglm declines
rather than guesses when support is weak.

## Test fixtures

Mock & Holt (1976), NSWC/DL TR-3473 (`digital`); Rynearson & Rand (1972),
TEES-9075-CR-72-02 (`digital-degraded`). Public-domain US government reports
the parser tests run against.

## Declaration of AI

I used Claude for scaffolding and coding assistance throughout this repo, as
declared in my course check-ins. The corpus design, source selection,
relevance-filter audits, model choices, and the analysis are mine.
