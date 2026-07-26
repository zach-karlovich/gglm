"""Tests for the ask front door. Run with: uv run pytest"""

from gglm.ask import gate, support_score
from gglm.generate import DEMOS, build_messages


def test_gate_refuses_below_threshold():
    assert not gate(0.3, threshold=0.6)
    assert gate(0.7, threshold=0.6)
    assert gate(0.6, threshold=0.6)  # boundary answers


class StubDense:
    def search(self, query, n):
        return [(0, 0.42)]


class StubRetriever:
    arms = [StubDense()]


def test_support_score_reads_the_dense_arm():
    assert support_score(StubRetriever(), "q") == 0.42


def test_rag_prompt_carries_the_demos():
    contexts = [{"title": "T", "text": "some chunk text"}]
    msgs = build_messages("What gas?", contexts)
    # system + two demo exchanges + the real question
    assert len(msgs) == 1 + 2 * len(DEMOS) + 1
    assert msgs[0]["role"] == "system"
    assert msgs[2]["content"] == DEMOS[0][2]  # cited one-liner, verbatim
    assert msgs[4]["content"] == "The sources do not contain this."
    assert msgs[-1]["role"] == "user"
    assert "What gas?" in msgs[-1]["content"]


def test_bare_prompt_has_no_demos():
    # the pre-RAG baseline stays a plain question
    assert len(build_messages("What gas?")) == 2
