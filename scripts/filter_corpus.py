"""Flag off-topic documents in the catalog so chunking can skip them.
usage: uv run python scripts/filter_corpus.py [--apply]

Writes a `topical` field back to the catalog. Nothing is deleted, so a
rejected document keeps its provenance and the call can be revisited.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from gglm import catalog, relevance

DATA = Path(os.environ.get("GGLM_DATA", "data"))
SAMPLE_PAGES = 12  # enough to get past cover pages and tables of contents
READABLE_KINDS = {"digital"}
MIN_SAMPLE_WORDS = 200


def text_sample(key, pages=SAMPLE_PAGES):
    """First few pages of a parsed document, or '' if it isn't parsed."""
    source, doc_id = key.split(":", 1)
    path = DATA / "parsed" / source / f"{doc_id}.jsonl"
    if not path.exists():
        return ""
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= pages:
                break
            out.append(json.loads(line).get("text", ""))
    return " ".join(out)


def verdict(key, entry):
    """(keep?, matched terms, unreadable?)."""
    title = entry.get("title", "")
    sample = text_sample(key)
    # scanned docs parse to garbage, so judge them on title alone and keep
    # when inconclusive - they're the oldest reports, a miss costs the most
    readable = (
        entry.get("kind") in READABLE_KINDS
        and len(sample.split()) >= MIN_SAMPLE_WORDS
    )
    keep, matched = relevance.is_topical(title, sample if readable else "")
    if not keep and not readable:
        return True, matched, True
    return keep, matched, False


def main(apply=False):
    entries = catalog.load()
    kept, dropped, unreadable = [], [], 0
    for key, entry in entries.items():
        keep, matched, unread = verdict(key, entry)
        unreadable += unread
        (kept if keep else dropped).append((key, entry, matched))

    print(f"{len(kept)} topical, {len(dropped)} off-topic, {len(entries)} total")
    print(f"{unreadable} kept unjudged: too little readable text to score\n")

    by_source = Counter(e.get("source") for _, e, _ in dropped)
    print("| source | dropped |")
    print("|---|---|")
    for source, n in by_source.most_common():
        print(f"| {source} | {n} |")

    # sample per source: the drops are mostly OSTI, which would otherwise
    # hide a bad call in the smaller, higher-quality NTRS and DTIC sets
    for source in by_source:
        print(f"\nwould drop from {source}:")
        for key, entry, _ in [d for d in dropped if d[1].get("source") == source][:12]:
            print(f"  {key:<22} {entry.get('title', '')[:64]}")

    if not apply:
        print("\ndry run; rerun with --apply to write the topical flag")
        return

    changed = 0
    for group, value in ((kept, True), (dropped, False)):
        for key, entry, _ in group:
            if entry.get("topical") is not value:
                catalog.update(key, topical=value)
                changed += 1
    print(f"\n{changed} entries updated -> {catalog.CATALOG}")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
