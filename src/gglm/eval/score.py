"""Deterministic scoring for open-ended answers. SQuAD-style token F1 plus a
substring hit; no LLM judge, so the same inputs always score the same."""

import re
import string
from collections import Counter

# strip "[2]" and "source" before scoring - only the RAG arm emits them,
# so counting them would penalize exactly the arm we want to measure
CITATION = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]|\bsources?\b", re.I)

ARTICLES = {"a", "an", "the"}
STOPWORDS = ARTICLES | {
    "is", "are", "was", "were", "be", "of", "in", "to", "and", "or",
    "for", "with", "that", "it", "its", "on", "by", "as", "at", "from",
}
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s):
    words = CITATION.sub(" ", s).lower().translate(_PUNCT).split()
    return " ".join(w for w in words if w not in ARTICLES)


def content_words(s):
    return {w for w in s.lower().translate(_PUNCT).split() if w not in STOPWORDS}


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def substring_hit(pred, gold):
    return float(normalize(gold) in normalize(pred))


def support_recall(gold, chunks):
    """Fraction of gold content words found in the best single retrieved chunk."""
    g = content_words(gold)
    if not g or not chunks:
        return 0.0
    return max(len(g & content_words(c["text"])) / len(g) for c in chunks)


def hit_at_k(retrieved, chunk_key):
    """Did the document the answer came from make the retrieved set?"""
    return float(any(c["key"] == chunk_key for c in retrieved))
