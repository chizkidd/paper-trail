from typing import List, Optional, Tuple
import streamlit as st

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    _SUPPORT = True
except Exception:
    _SUPPORT = False


@st.cache_resource(show_spinner="Loading reranker (first run only)...")
def load_cross_encoder() -> Optional[object]:
    if not _SUPPORT: return None
    try:    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except: return None


def rerank(
    question:   str,
    candidates: List[Tuple[str,float,int]],
    top_k:      int = 5,
) -> List[Tuple[str,float,int]]:
    ce = load_cross_encoder()
    if ce is None or not candidates:
        return candidates[:top_k]
    try:
        scores = ce.predict([(question, c[0]) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
    except Exception:
        return candidates[:top_k]
