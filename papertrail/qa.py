"""
Orchestrator. app.py calls answer_question() and build_knowledge_base().
Multi-document: session state holds a list of indexed documents.
Persistence: each retriever is serialized to disk on creation and reloaded on startup.
See papertrail/cache.py for the safe (pickle-free) serialization format.
"""
import hashlib
import html
import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

from papertrail import cache as doc_cache
from papertrail.ingest.chunking import prepare_chunks
from papertrail.retrieval.retriever import DocRetriever
from papertrail.retrieval.rerank import rerank
from papertrail.retrieval.mmr import (
    extractive_answer_from_results,
    focused_supporting_from_indices,
    focused_supporting_fallback,
)
from papertrail.retrieval.evidence import build_evidence_pack, format_extractive_answer
from papertrail.llm.hf import generate_answer_hf
from papertrail.llm.ollama import generate_answer_ollama
from papertrail.utils.text import as_text, match_label, normalize_ws, remove_display_latex_anywhere

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Multi-document helpers ─────────────────────────────────────────────────────

def get_documents() -> List[Dict]:
    """Return the list of indexed documents from session state."""
    return st.session_state.get("documents", [])


def get_active_retriever() -> Optional[DocRetriever]:
    docs = get_documents()
    idx  = st.session_state.get("active_doc_idx", 0)
    if not docs or idx >= len(docs):
        return None
    return docs[idx]["retriever"]


def get_active_doc() -> Optional[Dict]:
    docs = get_documents()
    idx  = st.session_state.get("active_doc_idx", 0)
    return docs[idx] if docs and idx < len(docs) else None


# ── Knowledge base ─────────────────────────────────────────────────────────────

def build_knowledge_base(
    text: str,
    source_name: str,
    section_map=None,
    page_map=None,
) -> str:
    """
    Index text and append to the documents list.
    Tries to load from disk cache first. Returns "" on success, error string on failure.
    """
    raw, sections, pages, embed = prepare_chunks(text, section_map, page_map)
    if len(raw) < 2:
        return "Not enough text to index. Try a longer document."

    th = _text_hash(text)

    # Try cache first
    retriever = doc_cache.load_retriever(source_name, th)

    if retriever is None:
        try:
            retriever = DocRetriever(raw, sections, pages, embed)
            doc_cache.save_retriever(retriever, source_name, th)
        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            return f"Indexing error: {exc}"

    if "documents" not in st.session_state:
        st.session_state.documents = []

    # Replace existing entry rather than duplicate
    for i, doc in enumerate(st.session_state.documents):
        if doc["source_name"] == source_name:
            st.session_state.documents[i] = {
                "source_name": source_name,
                "chunk_count": len(raw),
                "retriever":   retriever,
                "messages":    [],
                "text_hash":   th,
            }
            st.session_state.active_doc_idx = i
            return ""

    st.session_state.documents.append({
        "source_name": source_name,
        "chunk_count": len(raw),
        "retriever":   retriever,
        "messages":    [],
        "text_hash":   th,
    })
    st.session_state.active_doc_idx = len(st.session_state.documents) - 1
    return ""


def remove_document(idx: int) -> None:
    docs = st.session_state.get("documents", [])
    if 0 <= idx < len(docs):
        docs.pop(idx)
        st.session_state.documents = docs
        active = st.session_state.get("active_doc_idx", 0)
        if active >= len(docs):
            st.session_state.active_doc_idx = max(0, len(docs) - 1)


# ── Answer pipeline ────────────────────────────────────────────────────────────

def answer_question(retriever: DocRetriever, question: str) -> tuple:
    """
    Full pipeline: hybrid retrieval → rerank → MMR extraction → optional generation.
    Returns (answer_html, answer_text, score, extras, attribution_html).
    """
    n_chunks   = len(retriever.chunks)
    dynamic_k  = min(40, max(20, n_chunks // 5))
    candidates = retriever.query(question, top_k=dynamic_k)
    reranked   = rerank(question, candidates, top_k=5)

    if not reranked or reranked[0][1] < 0.01:
        msg = "Nothing relevant found. Try rephrasing using keywords from the document."
        return f"<p>{msg}</p>", msg, 0.0, [], ""

    best_chunk, best_score, best_idx = reranked[0]
    label, css_class = match_label(best_score)
    attr_text        = retriever.get_attribution(best_idx)

    results_for_mmr = [(c, s) for c, s, _ in reranked]
    answer_sents, (sup_ci, sup_si), chunks_sents = extractive_answer_from_results(
        retriever, question, results_for_mmr
    )

    evidence_pack = build_evidence_pack(
        retriever, reranked, question,
        max_chunks=4, max_sents_per_chunk=2, max_chars=1400,
    )
    mode = st.session_state.get("answer_mode", "Hugging Face (best effort)")

    generated, is_generated = "", False
    if mode == "Local (Ollama)":
        generated, is_generated = generate_answer_ollama(question, evidence_pack)
    elif mode == "Hugging Face (best effort)":
        generated, is_generated = generate_answer_hf(question, evidence_pack)

    if is_generated:
        answer_text  = remove_display_latex_anywhere(as_text(generated))
        source_label = "generated answer"
    else:
        answer_text  = remove_display_latex_anywhere(format_extractive_answer(answer_sents, attr_text))
        source_label = "grounded answer"

    if sup_si is not None:
        focused = focused_supporting_from_indices(chunks_sents, sup_ci, sup_si)
    else:
        focused = focused_supporting_fallback(best_chunk, answer_sents)
    focused = remove_display_latex_anywhere(as_text(focused))

    attribution_html = (
        f'<div class="attr-bar"><span class="attr-pill">{html.escape(attr_text)}</span></div>'
        if attr_text else ""
    )

    passages = [html.escape(f"{source_label.capitalize()}: {answer_text}")]
    if focused and normalize_ws(focused) != normalize_ws(answer_text):
        passages.append(html.escape(f"Supporting passage: {focused}"))

    answer_body = "</p><p>".join(passages)
    answer_html = (
        f'<div class="msg-label">{source_label}</div>'
        f"<p>{answer_body}</p>"
        f'<span class="match-pill {css_class}">{label}</span>'
    )

    extras, seen = [], set()
    for chunk, score, _ in reranked[1:]:
        if score < 0.01:
            continue
        key = normalize_ws(chunk[:200]).lower()
        if key in seen:
            continue
        seen.add(key)
        extras.append((chunk, score))

    return answer_html, answer_text, best_score, extras, attribution_html
