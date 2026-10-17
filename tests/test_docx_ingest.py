import io

import pytest

docx = pytest.importorskip("docx")

from papertrail.ingest.docx import load_docx  # noqa: E402


def _build() -> bytes:
    d = docx.Document()
    d.add_heading("Token-Depth Vulnerability", level=2)
    p = d.add_paragraph("Models are vulnerable due to shallow alignment.")
    p.add_run("65").font.superscript = True
    d.add_paragraph("A second body paragraph.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_headings_and_superscripts_excluded_from_text():
    text, section_map, _, err = load_docx(_build())
    assert err == ""
    assert "Token-Depth" not in text
    assert "alignment.65" not in text
    assert "shallow alignment." in text
    assert section_map[0] == "Token-Depth Vulnerability"
