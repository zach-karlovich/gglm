"""Tests for the PDF parser:

Mock & Holt (1976), NSWC/DL TR-3473 

Run with:  uv run pytest
"""

from pathlib import Path

from gglm.parse.pdf import PdfParser

TEST_PDF = Path(__file__).parent / "documents" / "mock_holt_1976_nswc_gas_gun.pdf"
THESIS_PDF = Path(__file__).parent / "documents" / "rynearson_rand_1972_tees_gas_gun_thesis.pdf"


def test_parse_basics():
    result = PdfParser().parse(TEST_PDF)
    assert result["n_pages"] == 64
    assert result["source"] == TEST_PDF.name
    # DTIC embedded an OCR text layer in this scan, so it counts as digital
    assert result["kind"] == "digital"


def test_text_is_really_there():
    result = PdfParser().parse(TEST_PDF)
    intro = result["pages"][8]["text"]  # page 9 = Chapter I, Introduction
    assert "gas gun" in intro.lower()


def test_jsonl_has_one_line_per_page(tmp_path):
    parser = PdfParser()
    result = parser.parse(TEST_PDF)
    out = parser.save_jsonl(result, tmp_path / "out.jsonl")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result["n_pages"]


def test_thesis_is_digital_degraded():
    result = PdfParser().parse(THESIS_PDF)
    assert result["n_pages"] == 104
    assert result["kind"] == "digital-degraded"
