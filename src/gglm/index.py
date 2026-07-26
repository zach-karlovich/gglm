"""Embed chunks and search them. Embeddings persist as .npy under DATA/index,
row-aligned with chunks.jsonl; search is a brute-force matmul."""

import json
import os
import re
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("GGLM_DATA", "data"))
CHUNKS_PATH = DATA / "chunks" / "chunks.jsonl"
INDEX_DIR = DATA / "index"

METRICS = ("cosine", "dot", "l2")
MAX_SEQ_LEN = 512  # chunks are 300 words; degraded OCR tokenizes letter-by-letter and blows way past this


def load_chunks(path=None):
    path = Path(path) if path else CHUNKS_PATH
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def slug(model_name):
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def tok(s):
    return s.lower().split()


def build_dense(model_name, chunks=None, out_dir=None, batch_size=32):  # 32 is safe on a 40GB card
    """Embed every chunk with model_name, save raw vectors so one matrix serves all metrics."""
    from sentence_transformers import SentenceTransformer

    chunks = chunks if chunks is not None else load_chunks()
    model = SentenceTransformer(model_name)
    # same cap for every embedder, so no arm sees more text than another
    model.max_seq_length = min(model.max_seq_length or MAX_SEQ_LEN, MAX_SEQ_LEN)
    emb = model.encode(
        [c["text"] for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    out_dir = Path(out_dir) if out_dir else INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug(model_name)}.npy"
    np.save(out, np.asarray(emb, dtype=np.float32))
    print(f"{emb.shape[0]} x {emb.shape[1]} embeddings -> {out}")

    # the next embedder needs the card to itself
    del model
    free_gpu()
    return out


def free_gpu():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DenseIndex:
    """Brute-force search over a persisted embedding matrix."""

    def __init__(self, model_name, metric="cosine", index_dir=None):
        from sentence_transformers import SentenceTransformer

        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.metric = metric
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = min(
            self.model.max_seq_length or MAX_SEQ_LEN, MAX_SEQ_LEN
        )  # match build_dense
        path = Path(index_dir or INDEX_DIR) / f"{slug(model_name)}.npy"
        if not path.exists():  # the usual cause is a shell without GGLM_DATA exported
            raise FileNotFoundError(
                f"no index at {path.resolve()} - build it with 'python -m gglm.index' "
                f"or export GGLM_DATA to point at the data root"
            )
        self.emb = np.load(path)
        norms = np.linalg.norm(self.emb, axis=1)
        self._unit = self.emb / np.maximum(norms, 1e-12)[:, None]
        self._sqnorm = norms**2

    def _encode_query(self, query):
        # Qwen3-Embedding defines a query prompt; bge does not
        prompts = getattr(self.model, "prompts", None) or {}
        if "query" in prompts:
            q = self.model.encode([query], prompt_name="query")[0]
        else:
            q = self.model.encode([query])[0]
        return q.astype(np.float32)

    def search(self, query, n):
        q = self._encode_query(query)
        if self.metric == "cosine":
            scores = self._unit @ (q / max(np.linalg.norm(q), 1e-12))
        elif self.metric == "dot":
            scores = self.emb @ q
        else:  # l2: 2e.q - |e|^2 orders by ascending distance
            scores = 2 * (self.emb @ q) - self._sqnorm
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order]


class BM25Index:
    """Exact-term matching for acronyms, report numbers, material codes."""

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tok(c["text"]) for c in chunks])

    def search(self, query, n):
        scores = self.bm25.get_scores(tok(query))
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order]


if __name__ == "__main__":
    import sys

    from gglm.retrieve import EMBEDDERS

    for name in sys.argv[1:] or EMBEDDERS:
        build_dense(name)
