"""Print the results JSONs from a run as markdown tables for the write-up.
usage: python -m gglm.eval.report [--dir DIR]"""

import argparse
import json
import os
from pathlib import Path

RUN_DIR = Path(os.environ.get("GGLM_DATA", "data")) / "eval" / "checkin4"


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def custom_table(run_dir):
    rows = [r for tag in ("pre_rag", "post_rag")
            if (r := load(run_dir / f"{tag}_custom_results.json"))]
    if not rows:
        return
    print("### Custom RAG test split\n")
    print("| arm | n | token F1 | substring hit | hit@5 |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['tag']} | {r['n']} | {fmt(r['token_f1'])} "
              f"| {fmt(r['substring_hit'])} | {fmt(r.get('hit_at_k', ''))} |")


def combo_table(run_dir):
    combos = load(run_dir / "combo_results.json")
    if not combos:
        return
    print("\n### Embedding x metric arms, 10 manual prompts\n")
    print("| combo | support recall | token F1 |")
    print("|---|---|---|")
    for name, r in combos.items():
        print(f"| {name} | {fmt(r['support_recall'])} | {fmt(r['token_f1'])} |")


def benchmark_table(run_dir):
    for tag in ("pre_rag", "post_rag"):
        results = load(run_dir / f"{tag}_benchmark_results.json")
        if not results:
            continue
        print(f"\n### lm_eval benchmarks, {tag}\n")
        print("| task | metric | value |")
        print("|---|---|---|")
        for task, metrics in results.items():
            if task == "note":
                continue
            for key, val in metrics.items():
                if "stderr" in key or not isinstance(val, float):
                    continue
                print(f"| {task} | {key.removesuffix(',none')} | {fmt(val)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=RUN_DIR, type=Path)
    args = ap.parse_args()
    if not any(args.dir.glob("*_results.json")):  # silence here just looks like a hang
        print(f"no results JSONs in {args.dir.resolve()} - point --dir at a run's output")
        return
    custom_table(args.dir)
    combo_table(args.dir)
    benchmark_table(args.dir)


if __name__ == "__main__":
    main()
