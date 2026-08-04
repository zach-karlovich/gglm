# gglm | Gas Gun Language Model

A RAG assistant for light gas gun (LGG) and hypervelocity impact (HVI)
literature. gglm answers only from documents it can cite, and it refuses when
the corpus doesn't cover the question instead of making something up.

**Hugging Face:** [jzkarlovich/gglm](https://huggingface.co/jzkarlovich/gglm)
(the model card, with full evaluation results and corpus provenance)

## Setup

```bash
uv python install 3.12
uv sync              # base pipeline
uv sync --group rag  # retrieval + evaluation stack (GPU-sized)
uv run pytest
```

## Pipeline

Bulk data lives under `$GGLM_DATA` (default `data/`). The collect, parse, and
chunk stages run on any machine; embedding and generation want a GPU. On a
shared cluster, point `$GGLM_DATA` at scratch storage and keep the catalog in
the repo so a scratch purge can't take the provenance record.

```bash
bash scripts/collect.sh            # download PDFs, NTRS deep, OSTI shallow
uv run python scripts/parse_all.py # page text + kind classification
uv run python -m gglm.chunk        # 300-word windows -> chunks.jsonl
uv run python -m gglm.index        # embed chunks -> index/*.npy
```

Parse labels each PDF `digital` / `digital-degraded` / `scanned` / `mixed`;
chunking skips `scanned` docs, which have no usable text layer.

## Retrieval

Hybrid [BM25](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
+ dense, fused with
[reciprocal rank fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) and
deduplicated to one chunk per document. BM25 catches report numbers and
acronyms; dense catches paraphrase. `retrieve.COMBOS` names the compared arms:
[bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5) and
[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) under
cosine and L2, bm25 alone, and the hybrid. Generation is
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
answering from numbered sources it has to cite.

## Asking

Ask from the command line. Answers are short, cite sources by number, and end
with the source list, pages and links included:

```console
$ uv run python -m gglm.ask "What is the approximate maximum velocity of a two-stage light gas gun?"
The approximate maximum velocity of a two-stage light gas gun is 8.0 km/sec [1, 3].

Sources:
  [1] Concept definition study for an extremely large aerophysics range facility, pp. 9-9  (https://ntrs.nasa.gov/citations/19930013798)
  [2] Response of Materials to Impulsive Loading, pp. 66-67  (https://archive.org/details/DTIC_AD0783315)
  [3] New higher-order Godunov code for modelling performance of two-stage light gas guns, pp. 6-6  (https://ntrs.nasa.gov/citations/19960008802)
  [4] Preliminary Assessment of the Use of Heavy Gases in Two-Stage Light Gas Guns, pp. 1-2  (https://ntrs.nasa.gov/citations/20180007479)
  [5] Results of Two-Stage Light-Gas Gun Development Efforts and Hypervelocity Impact Tests of Advanced Thermal Protection Materials, pp. 6-6  (https://ntrs.nasa.gov/citations/19980236871)
```

If the best retrieved chunk's cosine falls below 0.55, gglm declines before
generation and shows the nearest sources instead (`--no-gate` overrides):

```console
$ uv run python -m gglm.ask "What is the best recipe for vanilla ice cream?"
The corpus doesn't support an answer to this. (best chunk score 0.35, threshold 0.55)
Nearest sources, for reference:
  [1] A Low Altitude Meteorological Data Base.  (https://archive.org/details/DTIC_ADA039063)
  [2] Microstructure and Dynamic Failure Properties of Freeze-Cast Materials for Thermobaric Warhead Cases  (https://archive.org/details/DTIC_ADA574034)
  [3] High pressure cosmochemistry of major planetary interiors: Laboratory studies of the water-rich region of the system ammonia-water  (https://ntrs.nasa.gov/citations/19870013963)
  [4] Space Station Planetology Experiments (SSPEX)  (https://ntrs.nasa.gov/citations/19860017664)
  [5] Reports of Planetary Geology and Geophysics Program, 1984  (https://ntrs.nasa.gov/citations/19850015163)
```

## Sources

[NASA NTRS](https://ntrs.nasa.gov), [DOE OSTI](https://www.osti.gov), and
[DTIC](https://discover.dtic.mil). Public keyless APIs, distribution-unlimited
documents only. NTRS is crawled deep and stays on topic; OSTI shallow, since
its full-text search gets noisy at depth. DTIC blocks scripted clients, so
its approved-for-public-release reports come through the
[Internet Archive mirror](https://archive.org/details/dticarchive)
(`collection:dticarchive`); catalog entries keep both the mirror URL and the
canonical DTIC citation page.

Every download gets a line in `data/catalog.jsonl`, an append-only provenance
log (title, authors, citation URL, license, sha256, kind) that doubles as the
license audit. Chunks carry their catalog key, so answers cite documents, not
chunk indices. Raw PDFs are not redistributed; the citation URLs make the
corpus reconstructible.

## Evaluation

Two test sets live in `data/eval/`. `manual_pairs.jsonl` holds ten hand-written
LGG/HVI pairs from the early literature review, used for the retrieval
comparison and gate calibration. `rag_test.jsonl` is the synthetic test split:
91 QA pairs written by
[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) from
sampled chunks, filtered, then hand-audited across three passes.

The model is measured with and without retrieval on the synthetic split, and
on three [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness)
benchmarks ([squadv2](https://huggingface.co/datasets/rajpurkar/squad_v2),
[gsm8k_cot](https://huggingface.co/datasets/openai/gsm8k),
[arc_challenge](https://huggingface.co/datasets/allenai/ai2_arc)). Scoring
is deterministic: token F1 and substring hit for answers, support recall and
hit@5 for retrieval. Every run writes paired results/samples JSONs under
`$GGLM_DATA/eval/`, and

```bash
uv run python -m gglm.eval.report
```

renders the captured metrics as markdown tables.

The refusal gate is a threshold on the best retrieved chunk's cosine,
calibrated at 0.55: in-domain dev questions score 0.568-0.820, off-domain
probes 0.279-0.524. Checked against the full test split: 0 of 91 answerable
questions refused, 10 of 10 off-domain probes refused. The calibration table
and gate report live in `data/eval/`.

## Declaration of AI

I used Claude for scaffolding and coding assistance throughout this repo. The
corpus design, source selection, relevance-filter audits, model choices, and
the analysis are mine.
