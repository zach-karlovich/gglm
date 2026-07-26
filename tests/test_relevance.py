"""Tests for the off-topic filter. Run with: uv run pytest"""

from gglm.relevance import hits, is_topical, singularize

ON_TOPIC = [
    "Optimization of a two stage light gas gun",
    "A New Technique for Achieving Impact Velocities Greater Than 10 km/s",
    "Concept definition study for an extremely large aerophysics range",
    "Hypervelocity Impacts into Stainless-Steel Tubes",
    "Shock compression of crystalline TeO2 to the high pressure regime",
]

OFF_TOPIC = [
    "Tree drought physiology: critical research questions",
    "Dynamic regulation of water potential in Juniperus osteosperma",
    "Mycorrhiza-induced mycocypins of Laccaria bicolor",
    "Flash flourishing of Northern Hemisphere vegetation",
    "Towards advanced polarized electron sources",
]


def test_on_topic_titles_are_kept():
    for title in ON_TOPIC:
        assert is_topical(title)[0], title


def test_off_topic_titles_are_dropped():
    for title in OFF_TOPIC:
        assert not is_topical(title)[0], title


def test_plurals_match_singular_terms():
    assert "impact velocity" in hits("measured impact velocities")
    assert "impact crater" in hits("the impact craters were")
    assert "gas gun" in hits("two-stage light gas guns")


def test_singularize_leaves_short_words_alone():
    # "gas" must not become "ga"
    assert singularize("gas guns") == "gas gun"


def test_bare_words_do_not_match():
    # "flash" and "impact" alone are why the terms are phrases
    assert hits("flash drought over the plains") == set()
    assert hits("the economic impact of policy") == set()


def test_body_text_rescues_a_bland_title():
    bland = "Report Number 4417, Volume II"
    assert not is_topical(bland)[0]
    body = "The light gas gun was fired. Sabot separation was clean. Impact crater depth measured."
    assert is_topical(bland, body)[0]
