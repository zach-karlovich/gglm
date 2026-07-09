"""Chunk parsed page text into retrieval-ready chunks.

Reads page JSONL from DATA/parsed/{source}/{id}.jsonl, joins catalog metadata,
writes one chunk per line to DATA/chunks/chunks.jsonl. Chunks are word windows
that may span pages; the page range rides along as metadata. Documents with a
duplicate sha256 (same work published in more than one place) are skipped.
"""

import json
import os
from pathlib import Path

from gglm import catalog

DATA = Path(os.environ.get("GGLM_DATA", "data"))
CHUNK_WORDS = 300
OVERLAP = 50
SKIP_KINDS = {"scanned"}


def chunk_pages(pages, size=CHUNK_WORDS, overlap=OVERLAP):
    """Word windows over concatenated pages. Yields (text, first_page, last_page)."""
    words = []
    for p in pages:
        words += [(w, p["page_number"]) for w in p["text"].split()]
    i = 0
    while i < len(words):
        window = words[i:i + size]
        yield " ".join(w for w, _ in window), window[0][1], window[-1][1]
        if i + size >= len(words):
            return
        i += size - overlap


def chunk_doc(parsed_path, entry):
    """Chunk one parsed document into dicts carrying its citation metadata."""
    with open(parsed_path, encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    return [
        {
            "key": entry["key"],
            "title": entry["title"],
            "year": entry.get("year"),
            "url": entry["citation_url"],
            "pages": [first, last],
            "i": i,
            "text": text,
        }
        for i, (text, first, last) in enumerate(chunk_pages(pages))
    ]


def chunk_all(parsed_dir=DATA / "parsed", out_path=DATA / "chunks" / "chunks.jsonl"):
    """Chunk every parsed document with a catalog entry. Dedup by sha256."""
    entries = catalog.load()
    seen = set()
    n_docs = n_chunks = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for path in sorted(Path(parsed_dir).glob("*/*.jsonl")):
            key = f"{path.parent.name}:{path.stem}"
            entry = entries.get(key)
            if entry is None or entry.get("kind") in SKIP_KINDS:
                continue
            if entry.get("sha256") in seen:
                continue
            seen.add(entry.get("sha256"))
            chunks = chunk_doc(path, entry)
            for c in chunks:
                f.write(json.dumps(c) + "\n")
            n_docs += 1
            n_chunks += len(chunks)
    print(f"{n_chunks} chunks from {n_docs} documents -> {out_path}")
    return out_path


if __name__ == "__main__":
    chunk_all()
