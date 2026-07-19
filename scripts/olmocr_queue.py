"""Print PDF paths whose catalog kind needs re-OCR, one per line.

usage: uv run python scripts/olmocr_queue.py
'digital-degraded' (letter-spaced legacy OCR) and 'scanned' (no text layer)
always qualify; 'mixed' (partial text layer) is included too. Feed the output
to olmOCR: olmocr $GGLM_DATA/ocr --markdown --pdfs $(scripts/olmocr_queue.py)
"""

from gglm import catalog

OCR_KINDS = {"digital-degraded", "scanned", "mixed"}


def main():
    for entry in catalog.load().values():
        if entry.get("kind") in OCR_KINDS and entry.get("pdf"):
            print(entry["pdf"])


if __name__ == "__main__":
    main()
