import logging
from typing import List, Optional, Tuple

import streamlit as st

from papertrail import config

logger = logging.getLogger(__name__)

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    _SUPPORT = True
except ImportError:
    _SUPPORT = False


@st.cache_resource(show_spinner="Loading reranker (first run only)...")
def load_cross_encoder() -> Optional[object]:
    if not _SUPPORT:
        return None
    try:
        return CrossEncoder(config.RERANKER_MODEL)
    except Exception as exc:
        logger.warning("Failed to load cross-encoder reranker: %s", exc)
        return None


def rerank(
    question:   str,
    candidates: List[Tuple[str, float, int]],
    top_k:      int = 5,
) -> List[Tuple[str, float, int]]:
    ce = load_cross_encoder()
    if ce is None:
        if candidates:
            logger.debug("Reranker unavailable — returning top-%d by retrieval score.", top_k)
        return candidates[:top_k]

    if not candidates:
        return []

    pairs = [(question, c[0]) for c in candidates]
    try:
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
    except Exception as exc:
        logger.warning("Reranking failed, falling back to retrieval order: %s", exc)
        return candidates[:top_k]
