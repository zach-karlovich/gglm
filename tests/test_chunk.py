"""Tests for chunking. Run with: uv run pytest"""

from pathlib import Path

from gglm.chunk import CHUNK_WORDS, OVERLAP, chunk_pages
from gglm.parse.pdf import PdfParser

TEST_PDF = Path(__file__).parent / "documents" / "mock_holt_1976_nswc_gas_gun.pdf"


def test_short_text_is_one_chunk():
    pages = [{"page_number": 1, "text": "a short page"}]
    assert list(chunk_pages(pages)) == [("a short page", 1, 1)]


def test_windows_overlap():
    pages = [{"page_number": 1, "text": " ".join(f"w{i}" for i in range(700))}]
    chunks = list(chunk_pages(pages))
    assert len(chunks) == 3
    assert len(chunks[0][0].split()) == CHUNK_WORDS
    # the last OVERLAP words of one chunk open the next
    assert chunks[0][0].split()[-OVERLAP:] == chunks[1][0].split()[:OVERLAP]


def test_chunks_span_pages():
    pages = PdfParser().parse(TEST_PDF)["pages"]
    chunks = list(chunk_pages(pages))
    assert any(first != last for _, first, last in chunks)
    assert all(first <= last for _, first, last in chunks)


def test_empty_pages_yield_nothing():
    assert list(chunk_pages([{"page_number": 1, "text": ""}])) == []
