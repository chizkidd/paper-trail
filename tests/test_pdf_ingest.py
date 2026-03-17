"""Tests for papertrail.ingest.pdf (file size validation, scanned detection)."""
import pytest

from papertrail import config
from papertrail.ingest.pdf import load_pdf, looks_scanned


class TestLooksScanned:
    def test_empty_string_is_scanned(self):
        assert looks_scanned("") is True

    def test_short_string_is_scanned(self):
        assert looks_scanned("abc") is True

    def test_sufficient_text_not_scanned(self):
        assert looks_scanned("a" * 100) is False

    def test_threshold_boundary(self):
        threshold = config.PDF_SCANNED_THRESHOLD
        assert looks_scanned("a" * (threshold - 1)) is True
        assert looks_scanned("a" * threshold) is False


class TestLoadPdfSizeLimit:
    def test_oversized_pdf_rejected(self):
        # Create a fake byte string larger than the limit
        oversized = b"x" * (config.PDF_MAX_BYTES + 1)
        text, section_map, page_map, err = load_pdf(oversized)
        assert text == ""
        assert "size limit" in err.lower() or "mb" in err.lower()
        assert section_map == {}
        assert page_map == {}

    def test_exactly_at_limit_not_rejected_by_size_check(self):
        # A byte string exactly at the limit should pass the size check
        # (it will fail later because it's not a valid PDF, but not due to size)
        at_limit = b"x" * config.PDF_MAX_BYTES
        _, _, _, err = load_pdf(at_limit)
        # Error should NOT be about size limit
        assert "size limit" not in err.lower()

    def test_invalid_pdf_bytes_returns_error(self):
        text, section_map, page_map, err = load_pdf(b"not a pdf")
        assert text == ""
        assert err != ""
