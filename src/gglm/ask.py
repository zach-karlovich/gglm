"""Ask gglm a question from the command line.
usage: python -m gglm.ask "What gas drives the second stage?" [--combo NAME] [--no-gate]"""

import argparse

from gglm import index, retrieve

# cosine of the best retrieved chunk below which we refuse to answer.
# Calibrated 2026-07-26 on the full index: dev questions score 0.568-0.820,
# off-domain probes 0.279-0.524, so 0.55 splits them with margin both ways.
# Gate check: 0/91 answerable refused, 10/10 off-domain refused.
REFUSAL_THRESHOLD = 0.55


def gate(support, threshold=REFUSAL_THRESHOLD):
    """True when the corpus supports answering at all."""
    return support >= threshold


def support_score(retriever, question):
    """Cosine of the best chunk from the dense arm."""
    dense = retriever.arms[0]  # build() puts the dense index first
    return dense.search(question, 1)[0][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    ap.add_argument("--combo", default="qwen-emb-cosine", choices=list(retrieve.COMBOS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-gate", action="store_true", help="answer even without support")
    args = ap.parse_args()
    question = " ".join(args.question)

    if retrieve.COMBOS[args.combo]["model"] is None:
        ap.error("the refusal gate needs a dense arm; pick a combo with an embedder")

    chunks = index.load_chunks()
    retriever = retrieve.build(args.combo, chunks=chunks, k=args.k)
    support = support_score(retriever, question)
    hits = retriever.retrieve(question)

    if not gate(support) and not args.no_gate:
        print(f"The corpus doesn't support an answer to this. (best chunk score "
              f"{support:.2f}, threshold {REFUSAL_THRESHOLD})")
        print("Nearest sources, for reference:")
        for n, c in enumerate(hits, 1):
            print(f"  [{n}] {c['title']}  ({c['url']})")
        return

    from gglm.generate import answer, load_generator

    model, tok = load_generator()
    print(answer(model, tok, question, hits))
    print("\nSources:")
    for n, c in enumerate(hits, 1):
        pages = f", pp. {c['pages'][0]}-{c['pages'][1]}" if c.get("pages") else ""
        print(f"  [{n}] {c['title']}{pages}  ({c['url']})")


if __name__ == "__main__":
    main()
