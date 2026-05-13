"""
Safe (pickle-free) persistence for DocRetriever.

Cache format (one set of files per document, keyed by content hash):
  <hash>_meta.json       — chunks, sections, pages (plain JSON)
  <hash>_embeddings.npy  — dense embedding matrix (numpy binary)
  <hash>_matrix.npz      — sparse TF-IDF matrix (scipy binary)
  <hash>_vocab.json      — TF-IDF vocabulary + IDF weights (plain JSON)

None of these formats can execute arbitrary code, unlike pickle.

Writes are atomic: data lands in *.tmp files first, then os.replace() moves
them into their final names so a crash mid-write never leaves a partial cache.
"""
import hashlib
import json
import logging
import os
from typing import Optional

import numpy as np

from papertrail import config

logger = logging.getLogger(__name__)


# ── Vectorizer serialization ───────────────────────────────────────────────────

def save_vectorizer(vec, path: str) -> None:
    """Serialize a fitted TF-IDF vectorizer to JSON — no pickle."""
    data = {
        "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def load_vectorizer(path: str):
    """Reconstruct a TF-IDF vectorizer from its JSON state — no pickle."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idf = np.array(data["idf"], dtype=np.float64)
    n   = len(idf)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=config.TFIDF_MAX_FEATURES,
        stop_words="english",
        sublinear_tf=True,
    )
    vec.vocabulary_ = data["vocabulary"]

    # Rebuild TfidfTransformer internals so vec.transform() works.
    tfidf = TfidfTransformer(sublinear_tf=True)
    tfidf.idf_ = idf
    tfidf._idf_diag = sp.diags(idf, offsets=0, shape=(n, n), format="csr", dtype=np.float64)
    vec._tfidf = tfidf
    return vec


# ── Retriever persistence ──────────────────────────────────────────────────────

def cache_base(source_name: str, text_hash: str) -> str:
    key = hashlib.md5(f"{source_name}:{text_hash}".encode()).hexdigest()
    return os.path.join(config.CACHE_DIR, key)


def save_retriever(retriever, source_name: str, text_hash: str) -> None:
    """Persist retriever state atomically — JSON, .npy, .npz (no pickle).

    Writes to *.tmp files first, then renames them into place so a crash
    mid-write never leaves a partial/corrupt cache.
    """
    tmp_files: list[str] = []
    try:
        import scipy.sparse as sp

        os.makedirs(config.CACHE_DIR, exist_ok=True)
        base = cache_base(source_name, text_hash)

        # Temporary file paths (written first, then renamed atomically).
        tmp_meta  = f"{base}_meta.json.tmp"
        tmp_emb   = f"{base}_embeddings.npy.tmp"
        tmp_mat   = f"{base}_matrix.tmp.npz"   # scipy appends .npz only when missing
        tmp_vocab = f"{base}_vocab.json.tmp"
        tmp_files = [tmp_meta, tmp_emb, tmp_mat, tmp_vocab]

        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks":         retriever.chunks,
                    "chunk_sections": retriever.chunk_sections,
                    "chunk_pages":    retriever.chunk_pages,
                },
                f,
            )

        np.save(tmp_emb, retriever.embeddings)
        sp.save_npz(tmp_mat, retriever.matrix)
        save_vectorizer(retriever.vectorizer, tmp_vocab)

        # Atomic rename — POSIX guarantees os.replace() is atomic.
        os.replace(tmp_meta,  f"{base}_meta.json")
        os.replace(tmp_emb,   f"{base}_embeddings.npy")
        os.replace(tmp_mat,   f"{base}_matrix.npz")
        os.replace(tmp_vocab, f"{base}_vocab.json")

    except Exception as exc:
        logger.warning("Cache save failed (non-fatal): %s", exc)
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_retriever(source_name: str, text_hash: str) -> Optional[object]:
    """Restore retriever from safe cache files. Returns None on any failure."""
    try:
        import scipy.sparse as sp

        base = cache_base(source_name, text_hash)
        required = [
            f"{base}_meta.json",
            f"{base}_embeddings.npy",
            f"{base}_matrix.npz",
            f"{base}_vocab.json",
        ]
        if not all(os.path.exists(p) for p in required):
            return None

        with open(f"{base}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Import here to avoid circular imports (DocRetriever → embedder → streamlit)
        from papertrail.retrieval.retriever import DocRetriever
        from papertrail.retrieval.embedder import load_embedder

        r = DocRetriever.__new__(DocRetriever)
        r.chunks         = meta["chunks"]
        r.chunk_sections = meta["chunk_sections"]
        r.chunk_pages    = meta["chunk_pages"]
        r.embeddings     = np.load(f"{base}_embeddings.npy")
        r.matrix         = sp.load_npz(f"{base}_matrix.npz")
        r.vectorizer     = load_vectorizer(f"{base}_vocab.json")
        r.embedder       = load_embedder()
        return r

    except Exception as exc:
        logger.warning("Cache load failed, will re-index: %s", exc)
        return None
