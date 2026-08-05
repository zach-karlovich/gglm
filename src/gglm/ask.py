"""Ask gglm a question from the command line; no question starts a REPL.
usage: python -m gglm.ask ["What gas drives the second stage?"] [--combo NAME] [--model ID] [--no-gate]"""

import argparse
import os
import re
import sys

from gglm import index, retrieve

# cosine of the best retrieved chunk below which we refuse to answer.
# Calibrated 2026-07-26 on the full index: dev questions score 0.568-0.820,
# off-domain probes 0.279-0.524, so 0.55 splits them with margin both ways.
# Gate check: 0/91 answerable refused, 10/10 off-domain refused.
REFUSAL_THRESHOLD = 0.55


def paint(code, s):
    """ANSI-color s for terminals; plain when piped or NO_COLOR is set."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def gate(support, threshold=REFUSAL_THRESHOLD):
    """True when the corpus supports answering at all."""
    return support >= threshold


def support_score(retriever, question):
    """Cosine of the best chunk from the dense arm."""
    dense = retriever.arms[0]  # build() puts the dense index first
    return dense.search(question, 1)[0][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--combo", default="qwen-emb-cosine", choices=list(retrieve.COMBOS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=None,
                    help="generator model id, for cards the 7B default won't fit")
    ap.add_argument("--no-gate", action="store_true", help="answer even without support")
    args = ap.parse_args()

    if retrieve.COMBOS[args.combo]["model"] is None:
        ap.error("the refusal gate needs a dense arm; pick a combo with an embedder")

    if not args.question:
        print(f"{paint('1;36', 'Embedder:')} {retrieve.COMBOS[args.combo]['model']}")
    chunks = index.load_chunks()
    retriever = retrieve.build(args.combo, chunks=chunks, k=args.k)
    gen = None  # single-shot loads it lazily, after the gate

    def ask(question, label=False):
        nonlocal gen
        support = support_score(retriever, question)
        if not gate(support) and not args.no_gate:
            print(paint("33", f"The corpus doesn't support an answer to this. (best "
                              f"chunk score {support:.2f}, threshold {REFUSAL_THRESHOLD})"))
            return
        hits = retriever.retrieve(question)

        from gglm.generate import GENERATOR, answer, load_generator

        if gen is None:
            gen = load_generator(args.model or GENERATOR)
        model, tok = gen
        text = answer(model, tok, question, hits)
        print(f"{paint('1;32', 'Answer:')} {text}" if label else text)
        # uncited sources drop out; an answer citing nothing keeps the full list
        cited = {int(n) for grp in re.findall(r"\[([\d,\s]+)\]", text)
                 for n in grp.split(",") if n.strip().isdigit()}
        print(paint("2", "\nSources:"))
        for n, c in enumerate(hits, 1):
            if cited and n not in cited:
                continue
            pages = f", pp. {c['pages'][0]}-{c['pages'][1]}" if c.get("pages") else ""
            print(paint("2", f"  [{n}] {c['title']}{pages}  ({c['url']})"))

    if args.question:
        ask(" ".join(args.question))
        return

    import readline  # noqa: F401  arrow-key editing and history in input()

    # \001/\002 tell readline the escapes are zero-width
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    prompt = "\001\x1b[1;36m\002gglm>\001\x1b[0m\002 " if color else "gglm> "

    # topical questions are the expected case, so pay the generator load up
    # front instead of on the first answer
    from gglm.generate import GENERATOR, load_generator

    model_id = args.model or GENERATOR
    print(f"{paint('1;36', 'Generator:')} {model_id}")
    gen = load_generator(model_id)

    while True:  # REPL: retriever and generator stay loaded between questions
        try:
            print()
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question:
            print()
            ask(question, label=True)
            index.free_gpu()


if __name__ == "__main__":
    main()
