---
license: apache-2.0
thumbnail: https://huggingface.co/jzkarlovich/gglm/resolve/main/assets/social-card.png
language:
- en
base_model: Qwen/Qwen2.5-7B-Instruct
pipeline_tag: text-generation
tags:
- rag
- question-answering
- aerospace
- hypervelocity-impact
- retrieval
metrics:
- f1
- exact_match
model-index:
- name: gglm
  results:
  - task:
      type: question-answering
      name: Corpus-grounded QA (LGG/HVI)
    dataset:
      name: gglm synthetic test split (91 pairs, hand-audited)
      type: gglm-rag-test
    metrics:
    - type: f1
      name: Token F1 (with retrieval)
      value: 0.275
    - type: substring_hit
      name: Substring hit (with retrieval)
      value: 0.275
    - type: hit_at_5
      name: Retrieval hit@5
      value: 0.758
    source:
      name: Evaluation evidence in this repo (data/eval)
      url: https://huggingface.co/jzkarlovich/gglm/tree/main/data/eval
---

# gglm — Gas Gun Language Model

gglm is a retrieval-augmented QA assistant for light gas gun (LGG) and hypervelocity
impact (HVI) literature. It is a RAG project built on stock
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): the model stays
frozen, this repository ships no trained weights, and everything gglm knows lives in a
corpus of public government research reports that it retrieves from and cites
([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)). RAG was the right shape for
this problem because it keeps answers checkable against real reports and keeps the
compute small. Nothing is trained, retrieval is a plain numpy index, and a frozen 7B
runs comfortably on a single GPU or, quantized, on a laptop.

**GitHub:** [zach-karlovich/gglm](https://github.com/zach-karlovich/gglm)

## Introduction

Light gas gun and hypervelocity impact research lives in decades of government
technical reports. The details that matter, like sabot design, impact flash, and
launch parameters and velocities, are scattered across [NASA NTRS](https://ntrs.nasa.gov),
[DOE OSTI](https://www.osti.gov), and [DTIC](https://discover.dtic.mil) documents
going back to the 1950s. A stock LLM asked about this material will produce fluent
prose with made-up specifics, a well-documented failure mode
([Ji et al., 2023](https://arxiv.org/abs/2202.03629)), and it will not tell you when
it doesn't know: on [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2),
where half the questions are deliberately unanswerable
([Rajpurkar et al., 2018](https://arxiv.org/abs/1806.03822)), the base model abstained
zero times in roughly 6,000 chances (NoAns exact = 0.0). gglm was designed around avoiding
exactly that. It answers only from retrieved report chunks that it cites by number,
and it refuses when the corpus doesn't support an answer. On a 91-question test set
written from the corpus, retrieval plus a demonstration prompt raised token F1 from
0.087 to 0.275 (substring hit 0.011 to 0.275), and the calibrated refusal gate
declined 0 of 91 answerable questions while refusing 10 of 10 off-domain ones.

## Data

The pipeline data is a corpus of public, distribution-unlimited government research
reports: 1,796 documents collected from [NTRS](https://ntrs.nasa.gov),
[OSTI](https://www.osti.gov), and DTIC (via its approved-for-public-release
[Internet Archive mirror](https://archive.org/details/dticarchive), since DTIC blocks
scripted clients) using 19 LGG/HVI search queries. A phrase-based relevance filter
kept 1,099 on-topic documents, which parse and chunk into 51,699 chunks of 300 words
with 50-word overlap. Each chunk carries its source document's catalog key, so answers
cite reports rather than chunk indices, and every download is a line in an append-only
provenance catalog (`data/catalog.jsonl`: title, authors, year, license flag, citation
URL). Raw PDFs are not redistributed, but the catalog URLs make the corpus
reconstructible. The evaluation data was constructed separately. Ten hand-written QA
pairs from the early literature review (`data/eval/manual_pairs.jsonl`) served
for retrieval comparison and gate calibration, never for the headline numbers. The
test split (`data/eval/rag_test.jsonl`) is 91 synthetic QA pairs written by the larger
[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) from sampled
chunks, so the 7B isn't grading its own work, then run through automatic filters that
rejected 38% of candidates and a three-pass hand audit that cut the surviving 100
pairs to 91.

## Methodology

The generator was picked by comparing three Qwen2.5 sizes (1.5B, 7B, 14B) with size
as the only variable. The 7B kept the details that matter while compressing toward short
answers as demonstrations were added; the 14B stayed too elaborate to justify its
compute, and the 1.5B, tempting because it would run on any laptop, over-compressed
and dropped specifics. The model then stays frozen. Of the methods covered in class,
model editing (ROME/MEMIT) was rejected because facts written into weights can't be
cited or audited, and LoRA was held in reserve and never needed.

Retrieval is dense-only:
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) embeds chunks
and questions (512-token cap), and cosine similarity over plain numpy arrays returns
the top k=5 chunks, deduplicated to one per document. Dense-only was a measured choice
rather than an assumption: in a five-arm comparison (two embedders under cosine and
L2, BM25 ([Robertson & Zaragoza, 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)),
and a reciprocal-rank-fusion hybrid ([Cormack et al., 2009](https://dl.acm.org/doi/10.1145/1571941.1572114))),
BM25 was the weakest arm at 0.335 support recall, and fusing it in dragged the hybrid
below dense alone (0.444 vs 0.490). The RAG prompt adds two worked examples in the
few-shot style of [Brown et al., 2020](https://arxiv.org/abs/2005.14165), a
one-sentence cited answer and an explicit refusal, which fixed the model's verbosity
and raised token F1 by 44% with retrieval unchanged.

Refusal is a deterministic gate rather than an instruction. If the best retrieved
chunk's cosine falls below 0.55, gglm declines before generation. The threshold was
calibrated on in-domain dev questions (scores 0.568 to 0.820) against off-domain
probes (0.279 to 0.524), which it splits with margin on both sides.

## Evaluation

Results on three benchmarks plus the 91-pair custom test split, for gglm, its base
model, and two comparison models of similar size. Benchmarks run through
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (greedy,
seed 42, batch 8); custom-split scoring is deterministic token F1 and substring hit.

| Model | squadv2 exact / F1 | squadv2 NoAns exact | gsm8k_cot (strict) | arc_challenge (acc_norm) | custom-91 token F1 | custom-91 substring hit |
|---|---|---|---|---|---|---|
| **gglm** (Qwen2.5-7B + RAG + gate) | 9.91 / 21.07 † | 0.0 † | 0.704 † | 0.551 † | **0.275** | **0.275** |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (base, alone) | 9.91 / 21.07 | 0.0 | 0.704 | 0.551 | 0.087 | 0.011 |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (alone) | 22.71 / 30.49 | 0.0 | 0.762 | 0.560 | 0.083 | 0.000 |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (alone) | 20.72 / 31.78 | 0.03 | 0.500 | 0.603 | 0.098 | 0.000 |

† gglm changes no weights, so its benchmark rows equal the base model's by
construction (greedy decoding, fixed seed); the carried-over files state this and a
rerun flag exists. Retrieval hit@5 on the custom split is 0.758.

The benchmarks were chosen to measure what gglm cares about.
[squadv2](https://huggingface.co/datasets/rajpurkar/squad_v2) directly measures
abstention through its unanswerable half, and the result is the motivating failure:
NoAns exact is 0.0 for Qwen and Llama and 0.03 for Mistral, two abstentions in
roughly 5,900 chances. [gsm8k_cot](https://huggingface.co/datasets/openai/gsm8k)
([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) is a canary that the pipeline
doesn't degrade general reasoning, and
[arc_challenge](https://huggingface.co/datasets/allenai/ai2_arc)
([Clark et al., 2018](https://arxiv.org/abs/1803.05457)) covers closed-book science
QA, the nearest standard task to the domain. The comparison models are the
obvious same-size instruct alternatives from different model families, chosen so the
custom-split gap shows the domain knowledge is missing from this class of model
rather than from Qwen specifically. No stock model got past a token F1 of 0.1 on the
corpus-grounded questions, neither comparison model produced a single exact answer
string (substring hit 0.000), and the same frozen Qwen with retrieval reaches 0.275.

## Usage and Intended Uses

gglm is a literature reference for LGG/HVI researchers: ask a technical question, get
a one-sentence answer with numbered citations into real, pullable reports, or a
refusal when the corpus can't support one. It is not a firing-procedure authority
(see [Limitations](#limitations)). There are no weights to download from this repo.
The generator loads straight from
[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) via
[Transformers](https://huggingface.co/docs/transformers), and the pipeline's front
door is one command. Recommended settings are the calibrated defaults: greedy
decoding, k=5 retrieved chunks, refusal threshold 0.55.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# the generator gglm uses, stock and frozen - bf16, greedy decoding
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="auto"
)
model.eval()
```

With the repo cloned, the chunk index built (`python -m gglm.index`), and the
retrieval stack installed (`uv sync --group rag`), the full pipeline is:

```bash
python -m gglm.ask "What gas typically drives the second stage?"
```

## Prompt Format

The RAG prompt is a system instruction, two worked demonstrations (a one-sentence
cited answer and a refusal), and a user turn holding the numbered sources above the
question, rendered through the model's own chat template.

```text
system: You are a light gas gun and hypervelocity impact assistant. Answer the
        question concisely using only the numbered sources. Cite the sources you
        use by number. If they do not contain the answer, say so.

user:      Sources:
           [1] Two-stage light gas guns: ... The first stage can be driven by
           powder or gas, and the second stage is typically gas.
           Question: What can drive the first stage?
assistant: Powder or gas [1].

user:      Sources:
           [1] Sabot design: A sabot carries the projectile through the launch
           tube and is designed to separate from it after the projectile exits...
           Question: What alloy is the rupture diaphragm made of?
assistant: The sources do not contain this.

user:      Sources:
           [1] {title}: {chunk text}
           ...
           [5] {title}: {chunk text}
           Question: {your question}
```

## Expected Output Format

Answers are short, cite sources by number, and end with a printed source list
carrying pages and links; broad questions may run a few sentences. Refusals come in
two shapes: the in-prompt refusal when sources were retrieved but don't contain the
answer, and the gated refusal, printed before generation, when the best chunk scores
below threshold. The answer and gated refusal below are captured output; the
in-prompt refusal shows the trained format.

```console
$ python -m gglm.ask "What is a light gas gun and what are its uses?"
A light gas gun is a device used to accelerate projectiles to high velocities. It
operates by using a low molecular weight gas as a propellant. These guns are
utilized for testing projectiles or targets under high-velocity conditions. They
have been employed in various applications including meteoroid impact studies,
high-velocity flight testing, and small-scale aerodynamic tests [2][4].

Sources:
  [1] PROCEEDINGS OF THE SIXTH SYMPOSIUM ON HYPERVELOCITY IMPACT, 30 APRIL-2 May 1963, CLEVELAND, OHIO, VOLUME 1, pp. 106-123  (https://archive.org/details/DTIC_AD0423798)
  [2] An Efficient and Effective Light Gas Gun Design for Millimeter-scale Hypervelocity Testing, pp. 1-8  (https://ntrs.nasa.gov/citations/20220013697)
  [3] Development of a high-velocity free-flight launcher : the Ames light-gas gun, pp. 6-7  (https://ntrs.nasa.gov/citations/19930093745)
  [4] Optimization of a two stage light gas gun, pp. 14-16  (https://ntrs.nasa.gov/citations/19730003723)
  [5] Design of 50MM Powder to Air to Light Gas Gun Converter, pp. 6-7  (https://archive.org/details/DTIC_ADA485255)

$ python -m gglm.ask "What alloy is the rupture diaphragm made of?"
The sources do not contain this.

$ python -m gglm.ask "Who won the 2026 World Cup?"
The corpus doesn't support an answer to this. (best chunk score 0.35, threshold 0.55)
Nearest sources, for reference:
  [1] Lunar Impact Flash Locations from NASA's Lunar Impact Monitoring Program  (https://ntrs.nasa.gov/citations/20150021386)
  [2] Titanium Response to Simulated Nuclear Cloud Particle Environments. Volume II. Test Data.  (https://archive.org/details/DTIC_ADA055886)
  [3] An Exploration of the Equation of State Dependence of Core-collapse Supernova Explosion Outcomes and Signatures  (https://www.osti.gov/biblio/3374191)
  [4] HYPERVELOCITY IMPACT OF RODS ON FINITE TARGETS  (https://archive.org/details/DTIC_AD0392113)
  [5] Secondary impact hazard assessment  (https://ntrs.nasa.gov/citations/19940010010)
```

## Limitations

The headline numbers should be read with a few things in mind:

- The test set is synthetic. It was written by
  [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), a sibling
  of the generator, so family bias is possible even with the size gap, and it was
  machine-generated and hand-audited rather than expert-written.
- The retrieval-arm comparison rests on ten hand-written prompts. Its ordering is
  suggestive, not conclusive.
- The refusal threshold was calibrated on small probe sets (10 in-domain, 10
  off-domain), though it validated cleanly on all 91 test questions.
- The oldest literature is under-represented: 55 scanned documents with no usable
  text layer are skipped entirely, and degraded 1960s OCR survives only partially.
- [PyMuPDF](https://pymupdf.readthedocs.io) extraction loses equations and tables, so
  a quantitative question whose answer lives in a table can retrieve the right report
  and still fail.

Finally, and by design: gglm is a literature reference, not a firing-procedure
authority. It points to documents for a researcher to pull and read. It does not
validate launcher configurations, and a retrieved sentence stripped of its report's
context is not an operating instruction.

## Corpus Provenance

The full collected corpus, per `data/catalog.jsonl` (one line per document: title,
authors, year, license flag, citation URL):

| Source | Documents | Years | License |
|---|---|---|---|
| [NASA NTRS](https://ntrs.nasa.gov) | 716 | 1955–2026 | explicit flags: GOV_PUBLIC_USE_PERMITTED 393 · PUBLIC_USE_PERMITTED 264 · MAY_INCLUDE_COPYRIGHT_MATERIAL 49 · GOV_PERMITTED 10 |
| [DOE OSTI](https://www.osti.gov) | 690 | 2013–2026 | no per-record flag; OSTI serves full text only for publicly released documents |
| [DTIC](https://discover.dtic.mil) (via [Internet Archive mirror](https://archive.org/details/dticarchive)) | 363 | 1960–2018 | approved-for-public-release collection |

The 1,769 documents span 1955 to 2026, and 146 of them come from the 1960s, the
golden age of light-gas-gun development. Raw PDFs are not redistributed; every entry
is keyed to a public archive, so the citation URLs make the corpus reconstructible by
anyone.

## FAQ

**Why RAG instead of fine-tuning?** Users need to verify answers against real
reports, and fine-tuned facts can't be cited or audited. The failure analysis also
showed the model lacks facts, not skill: given the right passage, it answers well.

**Why a retrieval-score threshold instead of letting the model refuse?** Measured on
squadv2, the model refused zero times in about 6,000 unanswerable questions. Model
self-refusal is a hope, not a mechanism. A deterministic threshold is calibratable,
auditable, and can't be persuaded.

**Token F1 of 0.275 sounds low. Is this actually good?** The honest comparison is
0.087 without retrieval, and hit@5 shows the right document is found 76% of the time.
Token F1 punishes any wording not in the gold answer, and the logged samples include
answers that are right, cited, and still score low. The residual gap is extraction,
not search.

**Why is the hybrid retriever not used?** It lost. BM25 was the weakest arm on this
corpus and rank fusion averaged that weakness in. Hybrid-beats-dense was an early
assumption; it was measured and dropped.

## Declaration of AI

I used Claude for scaffolding and coding assistance throughout this repo. The corpus
design, source selection, relevance-filter audits, model choices, and the analysis
are mine.
