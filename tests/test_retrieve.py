"""Tests for rank fusion and dedup, using stub search arms. Run with: uv run pytest"""

from gglm.retrieve import COMBOS, Retriever

CHUNKS = [
    {"key": "doc-a", "i": 0, "text": "alpha"},
    {"key": "doc-a", "i": 1, "text": "beta"},
    {"key": "doc-b", "i": 0, "text": "gamma"},
    {"key": "doc-c", "i": 0, "text": "delta"},
]


class StubArm:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, n):
        return [(i, 1.0) for i in self.ranking[:n]]


def test_single_arm_keeps_its_order():
    r = Retriever(CHUNKS, [StubArm([3, 2, 0])], k=3)
    assert [c["key"] for c in r.retrieve("q")] == ["doc-c", "doc-b", "doc-a"]


def test_one_chunk_per_document():
    r = Retriever(CHUNKS, [StubArm([0, 1, 2, 3])], k=4)
    keys = [c["key"] for c in r.retrieve("q")]
    assert keys == ["doc-a", "doc-b", "doc-c"]  # chunk 1 (also doc-a) deduped


def test_rrf_prefers_agreement():
    # chunk 2 appears in both arms; chunks 0 and 3 in only one arm each
    arms = [StubArm([0, 2]), StubArm([3, 2])]
    r = Retriever(CHUNKS, arms, k=1)
    assert r.retrieve("q")[0]["key"] == "doc-b"


def test_combos_registry_shape():
    assert len(COMBOS) >= 3
    for spec in COMBOS.values():
        assert {"model", "metric", "bm25"} <= spec.keys()
