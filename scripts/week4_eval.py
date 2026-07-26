"""Re-run the custom split with the demo prompt, and measure the refusal gate.
usage: uv run python scripts/week4_eval.py [--combo qwen-emb-cosine] [--threshold 0.60]

Outputs go to GGLM_DATA/eval/week4/, never near the frozen check-in 4
results. Reports answer scores against the check-in baseline plus refusal
rates: on the 91 test questions (should stay near zero, they are all
answerable) and on the off-domain probes (should be near 100 percent).
"""

import argparse
import json
import os
from pathlib import Path

from gglm import index, retrieve
from gglm.ask import REFUSAL_THRESHOLD, gate, support_score
from gglm.eval import bench
from gglm.eval.qagen import eval_dir
from calibrate_refusal import OFF_DOMAIN

OUT = Path(os.environ.get("GGLM_DATA", "data")) / "eval" / "week4"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combo", default="qwen-emb-cosine")
    ap.add_argument("--threshold", type=float, default=REFUSAL_THRESHOLD)
    args = ap.parse_args()

    from gglm.generate import load_generator

    chunks = index.load_chunks()
    retriever = retrieve.build(args.combo, chunks=chunks)
    testset = eval_dir() / "rag_test.jsonl"

    # gate first, it needs no generator
    with open(testset, encoding="utf-8") as f:
        questions = [json.loads(l)["question"] for l in f]
    refused_in = [q for q in questions if not gate(support_score(retriever, q), args.threshold)]
    refused_out = [q for q in OFF_DOMAIN if not gate(support_score(retriever, q), args.threshold)]
    gate_report = {
        "threshold": args.threshold,
        "false_refusals": f"{len(refused_in)}/{len(questions)}",
        "true_refusals": f"{len(refused_out)}/{len(OFF_DOMAIN)}",
        "false_refusal_questions": refused_in,
    }
    bench.dump(gate_report, OUT / "gate_report.json")
    print(f"gate at {args.threshold}: refused {len(refused_in)}/{len(questions)} answerable, "
          f"{len(refused_out)}/{len(OFF_DOMAIN)} off-domain")

    # then the answers with the demo prompt
    model, tok = load_generator()
    bench.run_custom(model, tok, retriever, testset, OUT, "week4_rag")


if __name__ == "__main__":
    main()
