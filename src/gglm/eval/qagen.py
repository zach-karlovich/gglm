"""Generate a synthetic QA test split from the chunk corpus.
One question/answer pair per sampled chunk -> data/eval/rag_test.jsonl."""

import json
import os
import random
import re
import unicodedata
from pathlib import Path

from gglm import index
from gglm.eval.score import content_words

QAGEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # stronger than the 7B under test, so it isn't grading itself


def eval_dir():
    """data/eval in the repo, or under the CWD when gglm is installed as a copy."""
    # like catalog.py: eval sets live with the repo, so a scratch purge can't take them
    override = os.environ.get("GGLM_EVAL")
    if override:
        return Path(override)
    beside_source = Path(__file__).resolve().parents[3] / "data" / "eval"
    return beside_source if beside_source.is_dir() else Path("data/eval")


EVAL_DIR = eval_dir()

PROMPT = (
    "Below is a passage from a technical report about light gas guns or "
    "hypervelocity impact.\n\nPassage:\n{passage}\n\n"
    "Write one factual question about the physics, hardware, or measurements "
    "described here, and its answer.\n"
    "Rules:\n"
    "- The question must stand alone. A researcher who has never seen this "
    "passage must be able to understand what is being asked, so name the "
    "specific apparatus, material, or experiment instead of writing \"the "
    "text\", \"the document\", \"the given equation\", or \"Table 2\".\n"
    "- The answer must be short, factual, and stated in the passage.\n"
    "- Skip symbols, file names, equation fragments, and document trivia.\n"
    'Respond with JSON only: {{"question": "...", "answer": "..."}}'
)

MIN_WORDS = 150 # skip stub chunks
PER_DOC = 2 # don't let one report dominate the split
GROUNDED = 0.6 # answer content words that must appear in the chunk
DUP_JACCARD = 0.8 # near-duplicate question cutoff
MIN_CHUNK_TERMS = 3  # the passage itself has to be about the domain
MIN_QUESTION_WORDS = 6 # "What velocity?" is not a test question
MIN_NOVELTY = 0.25  # an answer that only echoes the question tests nothing
MAX_ECHO = 0.65     # ... and one that repeats most of it is a restatement

# questions that only make sense with the passage in hand, or that ask about
# the bibliography rather than the physics
META = re.compile(
    r"\b(text|passage|document|excerpt|article|section|chapter|paragraph"
    r"|mentioned|aforementioned|given|above|below|following"
    r"|figure|figs?|table|equation|eqs?"
    r"|title|titled|published|cited|references)\b",
    re.I,
)
FILENAME = re.compile(r"^\S+\.\w{2,4}$")
OCR_PUNCT = str.maketrans({"−": "-", "–": "-", "—": "-"})  # minus and dashes NFKC won't fold


def clean_text(s):
    """NFKC plus the punctuation OCR text tends to carry."""
    return unicodedata.normalize("NFKC", s).translate(OCR_PUNCT)


def sample_chunks(chunks, n=150, seed=42):
    """Seed-fixed sample of substantial, on-topic chunks, at most PER_DOC each."""
    from gglm import relevance

    rng = random.Random(seed)
    # the chunk itself must be on topic - an Apollo report passes the document
    # filter, then donates a paragraph about lava flows
    eligible = [
        c for c in chunks
        if len(c["text"].split()) >= MIN_WORDS
        and len(relevance.hits(c["text"])) >= MIN_CHUNK_TERMS
    ]
    rng.shuffle(eligible)
    picked, per_doc = [], {}
    for c in eligible:
        if per_doc.get(c["key"], 0) >= PER_DOC:
            continue
        per_doc[c["key"]] = per_doc.get(c["key"], 0) + 1
        picked.append(c)
        if len(picked) == n:
            break
    return picked


def readable(s):
    """Fraction of characters that are plain text, to catch mangled equations."""
    if not s:
        return 0.0
    return sum(c.isascii() and (c.isalnum() or c.isspace()) for c in s) / len(s)


def _words(q, a):
    from gglm.relevance import singularize

    return content_words(singularize(q)), content_words(singularize(a))


def novelty(q, a):
    """Share of the answer's content words the question hasn't already given."""
    q_words, a_words = _words(q, a)
    return len(a_words - q_words) / len(a_words) if a_words else 0.0


def echo(q, a):
    """Share of the question's content words the answer just repeats back."""
    q_words, a_words = _words(q, a)
    return len(q_words & a_words) / len(q_words) if q_words else 1.0


def usable(q, a):
    """Reject meta-questions and answers that are symbols, filenames, or echoes."""
    if len(q.split()) < MIN_QUESTION_WORDS or META.search(q):
        return False
    if len(a) < 3 or not any(c.isalnum() for c in a):
        return False
    if FILENAME.match(a) or novelty(q, a) < MIN_NOVELTY or echo(q, a) > MAX_ECHO:
        return False
    return readable(q) > 0.9 and readable(a) > 0.8


def parse_pair(text):
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
    q, a = clean_text(q), clean_text(a) # OCR text carries fi-ligatures and unicode minus signs
    if not q.endswith("?") or not a or not usable(q, a):
        return None
    return q, a


def grounded(answer, chunk_text):
    words = content_words(answer)
    if not words:
        return False
    return len(words & content_words(chunk_text)) / len(words) >= GROUNDED


def near_duplicate(question, seen_questions):
    q = content_words(question)
    for prev in seen_questions:
        if q and len(q & prev) / len(q | prev) >= DUP_JACCARD:
            return True
    return False


def generate_testset(n_keep=100, n_sample=400, seed=42, out_path=None):  # oversample, the filters reject a lot
    from gglm.generate import chat, load_generator

    out_path = Path(out_path) if out_path else EVAL_DIR / "rag_test.jsonl"
    if out_path.exists():
        print(f"{out_path} exists, not regenerating")
        return out_path

    chunks = index.load_chunks()
    sampled = sample_chunks(chunks, n=n_sample, seed=seed)
    model, tok = load_generator(QAGEN_MODEL)

    kept, seen = [], []
    n_dropped = 0
    for c in sampled:
        messages = [{"role": "user", "content": PROMPT.format(passage=c["text"])}]
        raw = chat(model, tok, messages, max_new_tokens=200)
        pair = parse_pair(raw)
        if pair is None or not grounded(pair[1], c["text"]) or near_duplicate(pair[0], seen):
            n_dropped += 1
            continue
        q, a = pair
        seen.append(content_words(q))
        kept.append(
            {
                "qid": f"synth-{len(kept):03d}",
                "question": q,
                "answer": a,
                "chunk_key": c["key"],
                "chunk_i": c["i"],
                "title": c["title"],
                "url": c["url"],
            }
        )
        if len(kept) == n_keep:
            break

    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"{len(kept)} QA pairs ({n_dropped} dropped by filters) -> {out_path}")
    return out_path


if __name__ == "__main__":
    generate_testset()
