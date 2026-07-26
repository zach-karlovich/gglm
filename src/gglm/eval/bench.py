"""Benchmark runners: lm_eval tasks and the custom RAG test split. Results
and logged samples go to separate JSON files, per the assignment."""

import json
from pathlib import Path
from statistics import mean

TASKS = ("squadv2", "gsm8k_cot", "arc_challenge") # squadv2 + gsm8k from HW6, arc for science QA
BATCH_SIZE = 8 # fixed - "auto" probes short samples then OOMs on real squadv2 contexts on a 40GB card


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_testset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_lm_eval(model, tok, out_dir, tag, tasks=TASKS, batch_size=BATCH_SIZE, limit=None):
    """Evaluate each task, rewriting the JSONs after every task so a timeout
    only loses the task in flight. limit is for smoke tests only."""
    from lm_eval import evaluator

    from gglm.index import free_gpu

    free_gpu()
    out_dir = Path(out_dir)
    all_results, all_samples = {}, {}
    for task in tasks:
        res = evaluator.simple_evaluate(
            model="hf",
            model_args={"pretrained": model, "dtype": "bfloat16", "tokenizer": tok},
            tasks=[task],
            log_samples=True,
            batch_size=batch_size,
            random_seed=42,
            limit=limit,
        )
        all_results.update(res["results"])
        all_samples.update(res["samples"])
        dump(all_results, out_dir / f"{tag}_benchmark_results.json")
        dump(all_samples, out_dir / f"{tag}_benchmark_samples.json")
        print(f"{tag} {task}: {res['results'][task]}")
    return all_results


def run_custom(model, tok, retriever, testset_path, out_dir, tag, limit=None):
    """Answer every test question. No retriever = the pre-RAG baseline."""
    from gglm.generate import answer
    from gglm.eval import score

    out_dir = Path(out_dir)
    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]

    samples = []
    for i, r in enumerate(rows):
        contexts = retriever.retrieve(r["question"]) if retriever else None
        pred = answer(model, tok, r["question"], contexts)
        rec = {
            "qid": r["qid"],
            "question": r["question"],
            "gold": r["answer"],
            "prediction": pred,
            "token_f1": score.token_f1(pred, r["answer"]),
            "substring_hit": score.substring_hit(pred, r["answer"]),
        }
        if retriever:
            rec["retrieved"] = [
                {"key": c["key"], "i": c["i"], "title": c["title"]} for c in contexts
            ]
            if "chunk_key" in r:
                rec["hit_at_k"] = score.hit_at_k(contexts, r["chunk_key"])
        samples.append(rec)
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            dump(samples, out_dir / f"{tag}_custom_samples.json")
            print(f"{tag} custom split: {i + 1}/{len(rows)}")

    results = {
        "tag": tag,
        "n": len(samples),
        "retrieval": bool(retriever),
        "token_f1": mean(s["token_f1"] for s in samples),
        "substring_hit": mean(s["substring_hit"] for s in samples),
    }
    hitks = [s["hit_at_k"] for s in samples if "hit_at_k" in s]
    if hitks:
        results["hit_at_k"] = mean(hitks)
    dump(results, out_dir / f"{tag}_custom_results.json")
    print(f"{tag} custom split: {results}")
    return results


def run_combos(model, tok, manual_path, out_dir, k=5):
    """Every COMBOS arm answers the manual prompts with the same generator."""
    from gglm import index, retrieve
    from gglm.generate import answer
    from gglm.eval import score

    out_dir = Path(out_dir)
    pairs = load_testset(manual_path)
    chunks = index.load_chunks()

    results, samples = {}, []
    for combo in retrieve.COMBOS:
        retriever = retrieve.build(combo, chunks=chunks, k=k)
        f1s, supports = [], []
        for r in pairs:
            contexts = retriever.retrieve(r["question"])
            pred = answer(model, tok, r["question"], contexts)
            f1 = score.token_f1(pred, r["answer"])
            support = score.support_recall(r["answer"], contexts)
            f1s.append(f1)
            supports.append(support)
            samples.append(
                {
                    "combo": combo,
                    "qid": r["qid"],
                    "question": r["question"],
                    "gold": r["answer"],
                    "prediction": pred,
                    "token_f1": f1,
                    "support_recall": support,
                    "retrieved": [
                        {"key": c["key"], "i": c["i"], "title": c["title"]}
                        for c in contexts
                    ],
                }
            )
        results[combo] = {
            "token_f1": mean(f1s),
            "support_recall": mean(supports),
            "n": len(pairs),
        }
        dump(results, out_dir / "combo_results.json")
        dump(samples, out_dir / "combo_samples.json")
        print(f"combo {combo}: {results[combo]}")
        del retriever
        index.free_gpu()  # each arm loads its own embedder
    return results
