import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from papertrail import config
from papertrail.retrieval.embedder import load_embedder

logger = logging.getLogger(__name__)


class DocRetriever:
    def __init__(
        self,
        chunks: List[str],
        chunk_sections: Optional[List[str]] = None,
        chunk_pages:    Optional[List[Optional[int]]] = None,
        embed_chunks:   Optional[List[str]] = None,
    ):
        self.chunks         = chunks
        self.chunk_sections = chunk_sections or [""] * len(chunks)
        self.chunk_pages    = chunk_pages    or [None] * len(chunks)
        texts               = embed_chunks if embed_chunks is not None else chunks
        self.embedder       = load_embedder()
        self.embeddings     = self.embedder.encode(texts, normalize_embeddings=True)
        self.vectorizer     = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=config.TFIDF_MAX_FEATURES,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(
        self,
        question: str,
        top_k: int = 60,
        alpha: float = config.HYBRID_ALPHA,
    ) -> List[Tuple[str, float, int]]:
        q_emb  = self.embedder.encode([question], normalize_embeddings=True)
        dense  = np.dot(self.embeddings, q_emb[0])
        sparse = cosine_similarity(self.vectorizer.transform([question]), self.matrix).flatten()

        def norm01(x):
            x = np.asarray(x, dtype=float)
            mn, mx = float(x.min()), float(x.max())
            return np.zeros_like(x) if mx - mn < 1e-12 else (x - mn) / (mx - mn)

        scores  = alpha * norm01(dense) + (1.0 - alpha) * norm01(sparse)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i]), int(i)) for i in top_idx]

    def score_sentences(self, question: str, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        q_vec = self.vectorizer.transform([question])
        s_mat = self.vectorizer.transform(sentences)
        return [float(x) for x in cosine_similarity(q_vec, s_mat).flatten()]

    def get_attribution(self, chunk_idx: int) -> str:
        section = self.chunk_sections[chunk_idx] if chunk_idx < len(self.chunk_sections) else ""
        page    = self.chunk_pages[chunk_idx]    if chunk_idx < len(self.chunk_pages)    else None
        parts   = []
        if section:
            parts.append(f"§ {section}")
        if page is not None:
            parts.append(f"p. {page}")
        return "  ·  ".join(parts)
