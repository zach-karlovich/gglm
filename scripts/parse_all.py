"""Parse every downloaded PDF and record its kind in the catalog.

usage: uv run python scripts/parse_all.py
Reads DATA/raw/{source}/{id}.pdf, writes DATA/parsed/{source}/{id}.jsonl,
prints the kind distribution. Already-parsed documents are skipped.
"""

import os
from collections import Counter
from pathlib import Path

from gglm import catalog
from gglm.parse.pdf import PdfParser

DATA = Path(os.environ.get("GGLM_DATA", "data"))


def main():
    entries = catalog.load()
    parser = PdfParser()
    kinds = Counter()
    for pdf in sorted((DATA / "raw").glob("*/*.pdf")):
        key = f"{pdf.parent.name}:{pdf.stem}"
        if key not in entries:
            print(f"skip, not in catalog: {pdf}")
            continue
        out = DATA / "parsed" / pdf.parent.name / f"{pdf.stem}.jsonl"
        if out.exists() and entries[key].get("kind"):
            kinds[entries[key]["kind"]] += 1
            continue
        try:
            result = parser.parse(pdf)
        except Exception as e:
            print(f"parse failed: {key}: {e}")
            continue
        parser.save_jsonl(result, out)
        catalog.update(key, kind=result["kind"], n_pages=result["n_pages"], sha256=result["sha256"])
        kinds[result["kind"]] += 1
        print(f"{key:<22} {result['kind']:<17} {result['n_pages']:4} pages")

    print("\nkind distribution:")
    for kind, n in kinds.most_common():
        print(f"  {kind:<17} {n}")


if __name__ == "__main__":
    main()
