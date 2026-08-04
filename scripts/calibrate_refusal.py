"""Pick the refusal threshold for ask.py.
usage: uv run python scripts/calibrate_refusal.py [--combo qwen-emb-cosine]

Scores the best-chunk cosine for questions the corpus can answer (the ten
hand-written dev pairs) against questions it can't (off-domain probes below).
Needs the built index and really wants a GPU - the 0.6B embedder crawls on
a login node. week4_eval.py calls suggest() itself, so the usual path is
just sbatch scripts/week4.slurm and read the table in the log.
"""

import argparse
import json

from gglm import index, retrieve
from gglm.eval.qagen import eval_dir

# clearly outside the corpus - the original "ice cream" test, expanded
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


def suggest(combo="qwen-emb-cosine"):
    """Score both question sets. Returns (rows, suggested cutoff or None)."""
    spec = retrieve.COMBOS[combo]
    dense = index.DenseIndex(spec["model"], spec["metric"])

    with open(eval_dir() / "manual_pairs.jsonl", encoding="utf-8") as f:
        in_domain = [json.loads(l)["question"] for l in f]

    def top(question):
        return dense.search(question, 1)[0][1]

    on = sorted((top(q), q) for q in in_domain)
    off = sorted((top(q), q) for q in OFF_DOMAIN)
    rows = [(q, score, "yes") for score, q in on] + [(q, score, "no") for score, q in off]

    lo, hi = off[-1][0], on[0][0]  # highest off-domain vs lowest in-domain
    cut = (lo + hi) / 2 if hi > lo else None
    return rows, cut


def print_table(rows, cut):
    print("| question | best chunk score | corpus can answer |")
    print("|---|---|---|")
    for q, score, answerable in rows:
        print(f"| {q[:60]} | {score:.3f} | {answerable} |")
    print()
    if cut is not None:
        print(f"clean separation. Suggested REFUSAL_THRESHOLD = {cut:.2f}")
    else:
        print("no clean cut between the sets. Pick the tradeoff by eye.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combo", default="qwen-emb-cosine")
    args = ap.parse_args()
    print_table(*suggest(args.combo))


if __name__ == "__main__":
    main()
