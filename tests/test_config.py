"""Sanity checks for the centralised config module."""
from papertrail import config


def test_hybrid_alpha_in_range():
    assert 0.0 < config.HYBRID_ALPHA < 1.0


def test_chunk_size_positive():
    assert config.CHUNK_SIZE > 0


def test_chunk_overlap_less_than_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_pdf_max_bytes_positive():
    assert config.PDF_MAX_BYTES > 0


def test_hf_timeout_positive():
    assert config.HF_TIMEOUT > 0


def test_ollama_timeout_positive():
    assert config.OLLAMA_TIMEOUT > 0


def test_hf_api_url_contains_model():
    assert config.HF_MODEL in config.HF_API_URL


def test_tfidf_max_features_positive():
    assert config.TFIDF_MAX_FEATURES > 0
