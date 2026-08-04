"""Benchmarks and the custom split for a comparison model.
usage: uv run python scripts/compare_eval.py --model mistralai/Mistral-7B-Instruct-v0.3 [--rag] [--limit 2]

Outputs go to GGLM_DATA/eval/compare/<slug>/, never near the frozen
earlier results.
"""

import argparse
import os
from pathlib import Path

from gglm.eval import bench
from gglm.eval.qagen import eval_dir
from gglm.generate import load_generator


def slug(model_id):
    return model_id.split("/")[-1].lower().replace(".", "-") # Mistral-7B-Instruct-v0.3 -> mistral-7b-instruct-v0-3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rag", action="store_true",
                    help="also run the custom split through the retriever")
    ap.add_argument("--combo", default="qwen-emb-cosine") # the winning combo
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    tag = slug(args.model)
    out = Path(os.environ.get("GGLM_DATA", "data")) / "eval" / "compare" / tag
    testset = eval_dir() / "rag_test.jsonl"

    model, tok = load_generator(args.model)
    bench.run_custom(model, tok, None, testset, out, tag, limit=args.limit) # fast, surfaces template trouble first
    if args.rag:
        from gglm import index, retrieve

        retriever = retrieve.build(args.combo, chunks=index.load_chunks())
        bench.run_custom(model, tok, retriever, testset, out, f"{tag}_rag", limit=args.limit)
        del retriever # run_lm_eval frees the gpu itself
    bench.run_lm_eval(model, tok, out, tag, limit=args.limit) # longest last, checkpoints per task


if __name__ == "__main__":
    main()
