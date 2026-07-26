"""Pick the refusal threshold for ask.py.
usage: uv run python scripts/calibrate_refusal.py [--combo qwen-emb-cosine]

Scores the best-chunk cosine for questions the corpus can answer (the ten
check-in 3 dev pairs) against questions it can't (off-domain probes below).
Needs the built index, so this runs on Rivanna. Prints a markdown table for
the model card and a suggested cutoff.
"""

import argparse
import json

from gglm import index, retrieve
from gglm.eval.qagen import eval_dir

# clearly outside the corpus - the check-in 3 "ice cream" test, expanded
OFF_DOMAIN = [
    "What is the best recipe for vanilla ice cream?",
    "Who won the 2022 World Cup final?",
    "How do I file a Virginia state tax extension?",
    "What are the common side effects of ibuprofen?",
    "When was the Eiffel Tower completed?",
    "How do I train a puppy not to bite?",
    "What is a good beginner sourdough starter ratio?",
    "How many strings does a cello have?",
    "What time zone is Anchorage, Alaska in?",
    "Which planets have retrograde rotation?",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combo", default="qwen-emb-cosine")
    args = ap.parse_args()

    spec = retrieve.COMBOS[args.combo]
    dense = index.DenseIndex(spec["model"], spec["metric"])

    with open(eval_dir() / "manual_pairs.jsonl", encoding="utf-8") as f:
        in_domain = [json.loads(l)["question"] for l in f]

    def top(question):
        return dense.search(question, 1)[0][1]

    print(f"| question | best chunk score | corpus can answer |")
    print("|---|---|---|")
    on = sorted((top(q), q) for q in in_domain)
    off = sorted((top(q), q) for q in OFF_DOMAIN)
    for score, q in on:
        print(f"| {q[:60]} | {score:.3f} | yes |")
    for score, q in off:
        print(f"| {q[:60]} | {score:.3f} | no |")

    lo, hi = off[-1][0], on[0][0]  # highest off-domain vs lowest in-domain
    print()
    if hi > lo:
        cut = (lo + hi) / 2
        print(f"clean separation: off-domain tops out at {lo:.3f}, in-domain "
              f"starts at {hi:.3f}. Suggested REFUSAL_THRESHOLD = {cut:.2f}")
    else:
        print(f"no clean cut: off-domain reaches {lo:.3f} but in-domain starts "
              f"at {hi:.3f}. Pick the tradeoff by eye from the table.")


if __name__ == "__main__":
    main()
