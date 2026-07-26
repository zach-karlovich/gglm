"""Tests for the deterministic scorers. Run with: uv run pytest"""

from gglm.eval.score import (
    hit_at_k,
    normalize,
    substring_hit,
    support_recall,
    token_f1,
)


def test_normalize_drops_punctuation_and_articles():
    assert normalize("The piston, ruptures a diaphragm!") == "piston ruptures diaphragm"


def test_token_f1_exact_match():
    assert token_f1("powder or gas", "Powder or gas.") == 1.0


def test_token_f1_partial_overlap():
    score = token_f1("hydrogen is used", "hydrogen or helium")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    assert token_f1("completely unrelated words", "powder or gas") == 0.0


def test_token_f1_empty_pred():
    assert token_f1("", "powder or gas") == 0.0


def test_substring_hit():
    assert substring_hit("It is usually hydrogen or helium.", "hydrogen or helium") == 1.0
    assert substring_hit("It is nitrogen.", "hydrogen or helium") == 0.0


def test_support_recall_finds_best_chunk():
    chunks = [
        {"text": "totally off topic content"},
        {"text": "the sabot separates from the projectile after the muzzle"},
    ]
    assert support_recall("sabot separates after muzzle", chunks) == 1.0
    assert support_recall("sabot separates after muzzle", []) == 0.0


def test_hit_at_k():
    retrieved = [{"key": "ntrs:123"}, {"key": "osti:456"}]
    assert hit_at_k(retrieved, "osti:456") == 1.0
    assert hit_at_k(retrieved, "ntrs:999") == 0.0
