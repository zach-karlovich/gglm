"""Tests for test-split generation filters. Run with: uv run pytest"""

from gglm.eval.qagen import parse_pair, sample_chunks, usable

GOOD = [
    ("What was the impact velocity for Plate No. 10?", "24,550 FPS"),
    ("What is the optimal number of metallic bumpers for the station?", "At most four."),
    ("What was the primary NDE technique used to find COPV impact damage?", "Visual inspection"),
]

# every one of these survived the first generation run
BAD = [
    ("What symbol is used frequently in the text?", "o"),
    ("What language was used for the program mentioned in the text?", "MATLAB"),
    ("What does eo equal according to the given equations?", "e = 1 n sum"),
    ("What is the file name for the damage information of the tests?", "EHMLIY.WKI"),
    ("What is the duration of the exposure time mentioned in the text?", "5 ns"),
]


def test_usable_accepts_standalone_questions():
    for q, a in GOOD:
        assert usable(q, a), q


def test_usable_rejects_meta_and_junk():
    for q, a in BAD:
        assert not usable(q, a), q


def test_short_questions_are_rejected():
    assert not usable("What velocity?", "6 km/s")


def test_bibliographic_questions_are_rejected():
    assert not usable("What is the title of the book by H.J. Melosh?", "Impact Cratering")
    assert not usable("Which report was published in 1989 by the lab?", "TR-3473")


def test_figure_abbreviations_are_rejected():
    assert not usable(
        "Which material's Hugoniot is compared in Extended Data Fig. 1?", "vitreous carbon"
    )


def test_answers_that_echo_the_question_are_rejected():
    assert not usable(
        "What is the muzzle velocity of the two-stage gun at maximum charge?",
        "The muzzle velocity of the two-stage gun at maximum charge is the muzzle velocity.",
    )


def test_mangled_equations_are_rejected():
    assert not usable("What is the internal energy of the mixture?", "𝑒𝑜= 1 𝑛 ∑︁ 𝜖𝑖𝑘")


def test_parse_pair_applies_the_filters():
    meta = '{"question": "What is described in the text?", "answer": "A gun"}'
    assert parse_pair(meta) is None
    good = '{"question": "What gas fills the pump tube on this gun?", "answer": "Hydrogen"}'
    assert parse_pair(good) == ("What gas fills the pump tube on this gun?", "Hydrogen")


def test_sampling_requires_domain_dense_chunks():
    chunks = [
        {"key": "a", "i": 0, "text": "lava flows traced from their vents " * 40},
        {"key": "b", "i": 0,
         "text": "the light gas gun fired a sabot and the hypervelocity impact "
                 "crater was measured " * 20},
    ]
    picked = sample_chunks(chunks, n=5)
    assert [c["key"] for c in picked] == ["b"]
