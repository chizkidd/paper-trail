"""
Orchestrator. app.py calls answer_question() and build_knowledge_base().
Multi-document: callers manage the documents list and active_doc_idx; this
module is pure Python with no Streamlit dependency.
Persistence: each retriever is serialized to disk on creation and reloaded on startup.
See papertrail/cache.py for the safe (pickle-free) serialization format.
"""
import hashlib
import html
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

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
from papertrail.utils.text import (
    as_text, is_noise_sentence, match_label,
    normalize_ws, remove_display_latex_anywhere, split_sentences,
)

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _top_sents(embedder, chunk: str, q_emb, max_sents: int = 3) -> str:
    """Return the top-N most question-relevant sentences from a chunk, in reading order."""
    sents = [s for s in split_sentences(chunk) if not is_noise_sentence(s)]
    if not sents:
        return ""
    if len(sents) <= max_sents:
        return " ".join(sents)
    s_emb = embedder.encode(sents, normalize_embeddings=True)
    scores = np.dot(s_emb, q_emb[0])
    top_idx = sorted(np.argsort(scores)[::-1][:max_sents].tolist())
    return " ".join(sents[i] for i in top_idx)


# ── Multi-document helpers ─────────────────────────────────────────────────────

def get_documents(documents: List[Dict]) -> List[Dict]:
    """Return the documents list passed in by the caller."""
    return documents


def get_active_retriever(
    documents: List[Dict],
    active_doc_idx: int = 0,
) -> Optional[DocRetriever]:
    if not documents or active_doc_idx >= len(documents):
        return None
    return documents[active_doc_idx]["retriever"]


def get_active_doc(
    documents: List[Dict],
    active_doc_idx: int = 0,
) -> Optional[Dict]:
    return documents[active_doc_idx] if documents and active_doc_idx < len(documents) else None


# ── Knowledge base ─────────────────────────────────────────────────────────────

def build_knowledge_base(
    text: str,
    source_name: str,
    documents: List[Dict],
    section_map=None,
    page_map=None,
) -> Tuple[str, List[Dict], int]:
    """
    Index text and return an updated documents list.
    Tries to load from disk cache first.
    Returns (error_str, updated_docs, new_active_idx).
    On success error_str is ""; on failure it contains the error message.
    """
    raw, sections, pages, embed = prepare_chunks(text, section_map, page_map)
    if len(raw) < 2:
        return "Not enough text to index. Try a longer document.", documents, 0

    th = _text_hash(text)
    retriever = doc_cache.load_retriever(source_name, th)

    if retriever is None:
        try:
            retriever = DocRetriever(raw, sections, pages, embed)
            doc_cache.save_retriever(retriever, source_name, th)
        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            return f"Indexing error: {exc}", documents, 0

    new_entry = {
        "source_name": source_name,
        "chunk_count": len(raw),
        "retriever":   retriever,
        "messages":    [],
        "text_hash":   th,
    }
    docs = list(documents)
    for i, doc in enumerate(docs):
        if doc["source_name"] == source_name:
            docs[i] = new_entry
            return "", docs, i
    docs.append(new_entry)
    return "", docs, len(docs) - 1


def remove_document(
    idx: int,
    documents: List[Dict],
    active_doc_idx: int = 0,
) -> Tuple[List[Dict], int]:
    """Remove document at idx and return (updated_docs, new_active_idx)."""
    docs = list(documents)
    if 0 <= idx < len(docs):
        docs.pop(idx)
    new_idx = min(active_doc_idx, max(0, len(docs) - 1))
    return docs, new_idx


def export_conversation_markdown(doc: dict) -> str:
    """Serialize a document's Q&A history to a Markdown string."""
    source = doc.get("source_name", "Document")
    messages = doc.get("messages", [])
    q_count = sum(1 for m in messages if m.get("role") == "user")
    lines = [
        f"# Papertrail Export: {source}",
        "",
        f"*{q_count} question{'s' if q_count != 1 else ''}*",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            lines.append(f"**Q:** {msg.get('content', '').strip()}")
            lines.append("")
        elif role == "assistant":
            lines.append(f"**A:** {msg.get('answer_text', '').strip()}")
            if msg.get("attr_text"):
                lines.append(f"*Source: {msg['attr_text']}*")
            lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


# ── Answer pipeline ────────────────────────────────────────────────────────────

def answer_question(
    retriever: DocRetriever,
    question: str,
    mode: str = "Hugging Face (best effort)",
) -> tuple:
    """
    Full pipeline: hybrid retrieval → rerank → MMR extraction → optional generation.
    Returns (answer_html, answer_text, score, extras, attribution_html).
    """
    n_chunks   = len(retriever.chunks)
    dynamic_k  = min(40, max(20, n_chunks // 5))
    q_emb      = retriever.embedder.encode([question], normalize_embeddings=True)
    candidates = retriever.query(question, top_k=dynamic_k, q_emb=q_emb)
    reranked   = rerank(question, candidates, top_k=5)

    if not reranked or reranked[0][1] < 0.01:
        msg = "Nothing relevant found. Try rephrasing using keywords from the document."
        return f"<p>{msg}</p>", msg, 0.0, [], ""

    best_chunk, best_score, best_idx = reranked[0]
    label, css_class = match_label(best_score)
    attr_text        = retriever.get_attribution(best_idx)

    results_for_mmr = [(c, s) for c, s, _ in reranked]
    answer_sents, (sup_ci, sup_si), chunks_sents = extractive_answer_from_results(
        retriever, question, results_for_mmr, q_emb=q_emb
    )

    evidence_pack = build_evidence_pack(
        retriever, reranked, question,
        max_chunks=4, max_sents_per_chunk=2, max_chars=1400,
        q_emb=q_emb,
    )
    # mode is passed in by the caller (app.py reads it from session state)

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

    passages = [html.escape(f"{source_label.capitalize()}: {answer_text}")]
    if focused and normalize_ws(focused) != normalize_ws(answer_text):
        passages.append(html.escape(f"Supporting passage: {focused}"))

    answer_body = "</p><p>".join(passages)
    answer_html = (
        f'<div class="msg-label">{html.escape(source_label)}</div>'
        f"<p>{answer_body}</p>"
        f'<span class="match-pill {html.escape(css_class)}">{html.escape(label)}</span>'
    )

    extras, seen = [], set()
    for chunk, score, _ in reranked[1:]:
        if score < 0.01:
            continue
        key = normalize_ws(chunk[:200]).lower()
        if key in seen:
            continue
        seen.add(key)
        snippet = remove_display_latex_anywhere(
            _top_sents(retriever.embedder, chunk, q_emb)
        )
        if snippet:
            extras.append((snippet, score))

    # Return plain attr_text — callers build attribution HTML at render time
    # so it is never stored as raw HTML in session state.
    return answer_html, answer_text, best_score, extras, attr_text
