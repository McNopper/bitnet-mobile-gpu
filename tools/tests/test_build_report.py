# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

"""Tests for tools/build_report.py (structural lint and PDF verification)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_report import lint_structural, verify_pdf

MINIMAL_A4_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595.28 841.89]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)


def test_structural_clean_text_passes(tmp_path: Path) -> None:
    ok = "\\documentclass{article}\n\\begin{document}\nhi\n\\end{document}\n"
    assert lint_structural(ok, tmp_path) == []


def test_structural_rejects_bom_and_non_ascii(tmp_path: Path) -> None:
    findings = lint_structural("\ufeff\\documentclass{article}", tmp_path)
    assert any("byte-order mark" in f for f in findings)
    findings = lint_structural("plain text: \u00a7 and \u00d7 here", tmp_path)
    assert any("non-ASCII" in f for f in findings)


def test_structural_rejects_unbalanced_environment(tmp_path: Path) -> None:
    text = "\\begin{document}\nno end\n"
    findings = lint_structural(text, tmp_path)
    assert any("environment document" in f for f in findings)


def test_structural_rejects_external_include_and_absolute_path(tmp_path: Path) -> None:
    findings = lint_structural("\\input{other.tex}", tmp_path)
    assert any("external include" in f for f in findings)
    findings = lint_structural("see C:/Users/me/file.tex", tmp_path)
    assert any("absolute local path" in f for f in findings)


def test_structural_rejects_missing_includegraphics(tmp_path: Path) -> None:
    findings = lint_structural("\\includegraphics{figures/nope.png}", tmp_path)
    assert any("includegraphics target missing" in f for f in findings)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "ok.png").write_bytes(b"\x89PNG")
    findings = lint_structural("\\includegraphics{figures/ok.png}", tmp_path)
    assert findings == []


def test_verify_pdf_reads_a4_and_page_count(tmp_path: Path) -> None:
    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(MINIMAL_A4_PDF)
    pages, size = verify_pdf(pdf)
    assert pages == 1
    assert "210.0 x 297.0" in size


def test_verify_pdf_rejects_letter_size(tmp_path: Path) -> None:
    pdf = tmp_path / "t.pdf"
    pdf.write_bytes(MINIMAL_A4_PDF.replace(b"595.28 841.89", b"612.00 792.00"))
    try:
        verify_pdf(pdf)
    except SystemExit as exc:
        assert "not A4" in str(exc)
    else:
        raise AssertionError("letter size must fail")
