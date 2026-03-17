"""Tests for the safe (pickle-free) cache serialization in papertrail.qa."""
import json
import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from papertrail.cache import save_vectorizer as _save_vectorizer, load_vectorizer as _load_vectorizer


@pytest.fixture()
def fitted_vectorizer():
    """A vectorizer fitted on a small corpus."""
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how vexingly quick daft zebras jump",
    ]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100, stop_words="english", sublinear_tf=True)
    vec.fit_transform(corpus)
    return vec, corpus


class TestVectorizerSerialization:
    def test_round_trip_vocabulary(self, fitted_vectorizer):
        vec, _ = fitted_vectorizer
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            _save_vectorizer(vec, path)
            loaded = _load_vectorizer(path)
            assert loaded.vocabulary_ == vec.vocabulary_
        finally:
            os.unlink(path)

    def test_round_trip_idf(self, fitted_vectorizer):
        vec, _ = fitted_vectorizer
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            _save_vectorizer(vec, path)
            loaded = _load_vectorizer(path)
            np.testing.assert_allclose(loaded._tfidf.idf_, vec.idf_, rtol=1e-6)
        finally:
            os.unlink(path)

    def test_transform_output_matches(self, fitted_vectorizer):
        vec, corpus = fitted_vectorizer
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            _save_vectorizer(vec, path)
            loaded = _load_vectorizer(path)
            original = vec.transform(corpus)
            restored = loaded.transform(corpus)
            # Outputs should be nearly identical
            diff = (original - restored)
            assert abs(diff).max() < 1e-6
        finally:
            os.unlink(path)

    def test_saved_file_is_valid_json(self, fitted_vectorizer):
        vec, _ = fitted_vectorizer
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            _save_vectorizer(vec, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "vocabulary" in data
            assert "idf" in data
            assert isinstance(data["vocabulary"], dict)
            assert isinstance(data["idf"], list)
        finally:
            os.unlink(path)

    def test_no_pickle_in_saved_file(self, fitted_vectorizer):
        """Saved cache file must be plain JSON, not pickle bytes."""
        vec, _ = fitted_vectorizer
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            _save_vectorizer(vec, path)
            with open(path, "rb") as f:
                header = f.read(2)
            # Pickle files start with 0x80 followed by protocol byte.
            assert header[:1] != b"\x80", "File appears to be pickle, not JSON"
        finally:
            os.unlink(path)
