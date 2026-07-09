"""Parse a PDF into page-level text. Output: JSONL, one line per page."""

import hashlib
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

MIN_CHARS_PER_PAGE = 100
MIN_TOKENS = 50
DEGRADED_RATIO = 0.2


class PdfParser:
    """Extract page-level text from a PDF."""

    def parse(self, pdf_path):
        """Return {source, sha256, n_pages, kind, pages}."""
        pdf_path = Path(pdf_path)
        pages = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                pages.append(
                    {"page_number": i + 1, "text": text, "char_count": len(text)}
                )
        return {
            "source": pdf_path.name,
            "sha256": file_sha256(pdf_path),
            "n_pages": len(pages),
            "kind": classify(pages),
            "pages": pages,
        }

    def save_jsonl(self, result, out_path):
        """Write one JSON object per page; doc fields repeated on each line."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc_fields = {k: result[k] for k in ("source", "sha256", "kind")}
        with out_path.open("w", encoding="utf-8") as f:
            for page in result["pages"]:
                f.write(json.dumps(doc_fields | page) + "\n")
        return out_path


def single_letter_ratio(text):
    """Clean prose ~0.05; letter-spaced legacy OCR 0.5+. Calibrated: Mock&Holt 0.060, Rynearson 0.688."""
    tokens = re.findall(r"[A-Za-z]+", text)
    if len(tokens) < MIN_TOKENS:
        return None
    return sum(1 for t in tokens if len(t) == 1) / len(tokens)


def classify(pages):
    """'digital', 'digital-degraded', 'scanned', or 'mixed' by text coverage and OCR quality."""
    if not pages:
        return "scanned"
    with_text = sum(1 for p in pages if p["char_count"] >= MIN_CHARS_PER_PAGE)
    coverage = with_text / len(pages)
    if coverage >= 0.7:
        ratios = [r for p in pages if (r := single_letter_ratio(p["text"])) is not None]
        if ratios and sorted(ratios)[len(ratios) // 2] > DEGRADED_RATIO:
            return "digital-degraded"
        return "digital"
    if coverage <= 0.1:
        return "scanned"
    return "mixed"


def file_sha256(path):
    """SHA-256 of the file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m gglm.parse.pdf <file.pdf> [out.jsonl]")
        sys.exit(1)

    pdf_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf_file.with_suffix(".jsonl")

    parser = PdfParser()
    result = parser.parse(pdf_file)
    parser.save_jsonl(result, out_file)
    print(f"{result['source']}: {result['n_pages']} pages, kind={result['kind']}")
    print(f"wrote {out_file}")