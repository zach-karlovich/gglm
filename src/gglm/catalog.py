"""Append-only document catalog: data/catalog.jsonl, one JSON line per entry.

This file is the single source of truth for what is in the corpus, where it
came from, and how to cite it. Entries are keyed (e.g. "ntrs:19730003723").
Updates are appended as new lines with the same key; the latest line wins.
Nothing is ever overwritten, so the file doubles as an audit log.
"""

import json
from pathlib import Path

CATALOG = Path("data/catalog.jsonl")


def load(path=CATALOG):
    """Return {key: entry}. Later lines override earlier ones."""
    path = Path(path)
    if not path.exists():
        return {}
    entries = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["key"]] = entry
    return entries


def append(entry, path=CATALOG):
    """Add one entry as a new line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def update(key, path=CATALOG, **fields):
    """Append an updated copy of an existing entry with new fields merged in."""
    entries = load(path)
    if key not in entries:
        raise KeyError(f"not in catalog: {key}")
    entry = entries[key] | fields
    append(entry, path)
    return entry


if __name__ == "__main__":
    # usage: python -m gglm.catalog   (prints a summary)
    entries = load()
    print(f"{len(entries)} documents in {CATALOG}")
    for e in entries.values():
        downloaded = "pdf" if e.get("pdf") else "---"
        print(f"{e['key']:<22} {e.get('year') or '----':4}  {downloaded:3}  {e['title'][:60]}")
