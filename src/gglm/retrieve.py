"""Retrieval over the chunk index: search arms fused with reciprocal rank
fusion, one chunk per document. COMBOS names the compared arms."""

from gglm import index

BGE_SMALL = "BAAI/bge-small-en-v1.5"
QWEN_EMB = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDERS = (BGE_SMALL, QWEN_EMB)

# bge and Qwen3 embeddings come out normalized, so cosine and L2 rank the
# same; bm25-only and hybrid vary the mechanism instead
COMBOS = {
    "bge-small-cosine": {"model": BGE_SMALL, "metric": "cosine", "bm25": False},
    "bge-small-l2": {"model": BGE_SMALL, "metric": "l2", "bm25": False},
    "qwen-emb-cosine": {"model": QWEN_EMB, "metric": "cosine", "bm25": False},
    "bm25-only": {"model": None, "metric": "bm25", "bm25": True},
    "hybrid": {"model": BGE_SMALL, "metric": "cosine", "bm25": True},
}


class Retriever:
    """Rank-fuse search arms and return the top chunks, one per document."""

    def __init__(self, chunks, arms, k=5, n=30, rrf_k=60):
        self.chunks = chunks
        self.arms = list(arms)
        self.k = k # chunks returned
        self.n = n # candidates per arm before fusion
        self.rrf_k = rrf_k # 60 from the RRF paper

    def retrieve(self, query):
        if len(self.arms) == 1:
            order = [i for i, _ in self.arms[0].search(query, self.n)]
        else:
            votes = {}
            for arm in self.arms:
                for rank, (i, _) in enumerate(arm.search(query, self.n)):
                    votes[i] = votes.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
            order = sorted(votes, key=votes.get, reverse=True)
        seen, top = set(), []
        for i in order:
            key = self.chunks[i]["key"]
            if key in seen:
                continue
            seen.add(key)
            top.append(self.chunks[i])
            if len(top) == self.k:
                break
        return top


def build(combo, chunks=None, k=5):
    """Build the named COMBOS retriever over the chunk corpus."""
    spec = COMBOS[combo]
    if chunks is None:
        chunks = index.load_chunks()
    arms = []
    if spec["model"]:
        arms.append(index.DenseIndex(spec["model"], spec["metric"]))
    if spec["bm25"]:
        arms.append(index.BM25Index(chunks))
    return Retriever(chunks, arms, k=k)
