"""Decide whether a document is actually about LGG/HVI. OSTI's full-text
search returns tree physiology for a sabot query, so don't trust the search."""

import re

# phrases, not bare words: "impact" alone matches climate impact and
# "flash" matches flash drought - both actually showed up in the corpus

DOMAIN_TERMS = (
    # launchers
    "light gas gun", "light-gas gun", "gas gun", "two-stage gun", "powder gun",
    "launch tube", "pump tube", "sabot", "obturator", "gun barrel", "muzzle",
    "railgun", "rail gun", "electromagnetic launcher", "explosive launcher",
    "ballistic range", "free flight range", "aeroballistic", "aerophysics range",
    "gun launch", "launcher", "hypervelocity gun", "hypervelocity range",
    # facility names carry the domain when a title says nothing else
    "aedc", "arnold engineering", "range g",
    # impact
    "hypervelocity", "impact crater", "cratering", "penetrator",
    "penetration mechanics", "terminal ballistics", "ballistic limit",
    "whipple shield", "bumper shield", "orbital debris", "micrometeoroid",
    "micrometeorite", "meteoroid", "spacecraft shielding", "impact damage",
    "impact testing",
    "impact experiment", "oblique impact", "impact velocity",
    # shock physics
    "shock compression", "shock wave", "shock-induced", "shock induced",
    "hugoniot", "equation of state",
    # "spall" alone is bearing fatigue too, so qualify it
    "flyer plate", "impactor", "spallation", "spall fracture", "spall strength",
    "spall damage", "shock loading", "dynamic compression",
    "shaped charge", "shock physics", "ramp compression",
    # diagnostics
    "photonic doppler", "velocimetry", "visar", "impact flash", "framing camera",
    "flash x-ray", "flash radiography", "streak camera", "photogate",
    "crater scaling", "penetration equation", "composite overwrapped",
    # projectiles and debris
    "projectile", "ejecta", "fragmentation", "debris cloud",
)

TITLE_HITS = 1  # a domain phrase in the title is enough
TEXT_HITS = 3   # otherwise the body has to keep coming back to the domain


def singularize(text):
    """Crude plural stripping, enough to match phrases across word forms."""
    words = []
    for word in re.findall(r"[a-z0-9-]+", (text or "").lower()):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def hits(text):
    """Distinct domain phrases appearing in text."""
    normalized = singularize(text)
    return {term for term in DOMAIN_TERMS if singularize(term) in normalized}


def is_topical(title, text=""):
    """(keep?, matched phrases). Title decides when it can; body breaks ties."""
    title_hits = hits(title)
    if len(title_hits) >= TITLE_HITS:
        return True, title_hits
    text_hits = hits(text)
    return len(text_hits) >= TEXT_HITS, text_hits
