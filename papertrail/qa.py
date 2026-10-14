"""
Orchestrator. app.py calls answer_question(), answer_question_stream(),
answer_across_docs(), and build_knowledge_base().
Multi-document: callers manage the documents list and active_doc_idx; this
module is pure Python with no Streamlit dependency.
Persistence: each retriever is serialized to disk on creation and reloaded on startup.
See papertrail/cache.py for the safe (pickle-free) serialization format.
"""
import hashlib
import logging
from typing import Dict, Iterator, List, Optional, Tuple

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
from papertrail.llm.hf import generate_answer_hf, generate_answer_hf_stream
from papertrail.llm.ollama import generate_answer_ollama, generate_answer_ollama_stream
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
    Returns (answer_text, focused, source_label, css_class, match_lbl, score, extras, attr_text).
    All values are plain Python — no HTML.
    """
    n_chunks   = len(retriever.chunks)
    dynamic_k  = min(40, max(20, n_chunks // 5))
    q_emb      = retriever.embedder.encode([question], normalize_embeddings=True)
    candidates = retriever.query(question, top_k=dynamic_k, q_emb=q_emb)
    reranked   = rerank(question, candidates, top_k=5)

    if not reranked or reranked[0][1] < 0.01:
        msg = "Nothing relevant found. Try rephrasing using keywords from the document."
        return msg, "", "no result", "", "", 0.0, [], ""

    best_chunk, best_score, best_idx = reranked[0]
    match_lbl, css_class = match_label(best_score)
    attr_text             = retriever.get_attribution(best_idx)

    results_for_mmr = [(c, s) for c, s, _ in reranked]
    answer_sents, (sup_ci, sup_si), chunks_sents = extractive_answer_from_results(
        retriever, question, results_for_mmr, q_emb=q_emb
    )

    evidence_pack = build_evidence_pack(
        retriever, reranked, question,
        max_chunks=4, max_sents_per_chunk=2, max_chars=3500,
        q_emb=q_emb,
    )

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

    return answer_text, focused, source_label, css_class, match_lbl, best_score, extras, attr_text


def answer_question_stream(
    retriever: DocRetriever,
    question: str,
    mode: str = "Hugging Face (best effort)",
) -> Tuple[dict, Iterator[str]]:
    """
    Fast retrieval phase (blocking ~100ms) + lazy LLM token stream.
    Returns (metadata_dict, token_iterator).
    For Structured mode the iterator yields the extractive answer in one chunk.
    For Ollama/HF the iterator yields live tokens as they arrive.
    """
    n_chunks   = len(retriever.chunks)
    dynamic_k  = min(40, max(20, n_chunks // 5))
    q_emb      = retriever.embedder.encode([question], normalize_embeddings=True)
    candidates = retriever.query(question, top_k=dynamic_k, q_emb=q_emb)
    reranked   = rerank(question, candidates, top_k=5)

    _empty: dict = {
        "attr_text": "", "extras": [], "score": 0.0,
        "css_class": "", "match_lbl": "", "focused": "",
        "source_label": "no result", "extractive_fallback": "",
    }
    if not reranked or reranked[0][1] < 0.01:
        msg = "Nothing relevant found. Try rephrasing using keywords from the document."
        return _empty, iter([msg])

    best_chunk, best_score, best_idx = reranked[0]
    match_lbl, css_class = match_label(best_score)
    attr_text             = retriever.get_attribution(best_idx)

    results_for_mmr = [(c, s) for c, s, _ in reranked]
    answer_sents, (sup_ci, sup_si), chunks_sents = extractive_answer_from_results(
        retriever, question, results_for_mmr, q_emb=q_emb
    )

    evidence_pack = build_evidence_pack(
        retriever, reranked, question,
        max_chunks=4, max_sents_per_chunk=2, max_chars=3500,
        q_emb=q_emb,
    )

    extractive = remove_display_latex_anywhere(
        format_extractive_answer(answer_sents, attr_text)
    )

    if sup_si is not None:
        focused = focused_supporting_from_indices(chunks_sents, sup_ci, sup_si)
    else:
        focused = focused_supporting_fallback(best_chunk, answer_sents)
    focused = remove_display_latex_anywhere(as_text(focused))

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

    if mode == "Local (Ollama)":
        token_stream: Iterator[str] = generate_answer_ollama_stream(question, evidence_pack)
        source_label = "generated answer"
    elif mode == "Hugging Face (best effort)":
        token_stream = generate_answer_hf_stream(question, evidence_pack)
        source_label = "generated answer"
    else:
        token_stream = iter([extractive])
        source_label = "grounded answer"

    metadata: dict = {
        "attr_text":           attr_text,
        "extras":              extras,
        "score":               best_score,
        "css_class":           css_class,
        "match_lbl":           match_lbl,
        "focused":             focused,
        "source_label":        source_label,
        "extractive_fallback": extractive,
    }
    return metadata, token_stream


def answer_across_docs(
    documents: List[Dict],
    question: str,
    mode: str = "Hugging Face (best effort)",
) -> tuple:
    """
    Cross-document search: queries all loaded documents and merges by reranker score.
    Returns same 8-tuple as answer_question.
    """
    if not documents:
        msg = "No documents loaded."
        return msg, "", "no result", "", "", 0.0, [], ""

    embedder = documents[0]["retriever"].embedder
    q_emb    = embedder.encode([question], normalize_embeddings=True)

    # Collect candidates from all retrievers, tagged with their source
    all_candidates: List[tuple] = []
    for doc in documents:
        retriever = doc["retriever"]
        n_chunks  = len(retriever.chunks)
        dynamic_k = min(40, max(20, n_chunks // 5))
        for chunk, score, chunk_idx in retriever.query(question, top_k=dynamic_k, q_emb=q_emb):
            all_candidates.append((chunk, score, chunk_idx, retriever, doc["source_name"]))

    if not all_candidates:
        msg = "Nothing relevant found across any loaded document."
        return msg, "", "no result", "", "", 0.0, [], ""

    # Cross-encoder reranks the merged pool (inter-doc comparable joint scores)
    rerank_input = [(chunk, score, idx) for chunk, score, idx, _, _ in all_candidates]
    reranked     = rerank(question, rerank_input, top_k=5)

    if not reranked or reranked[0][1] < 0.01:
        msg = "Nothing relevant found. Try rephrasing using keywords from the documents."
        return msg, "", "no result", "", "", 0.0, [], ""

    # Map chunk text → (retriever, chunk_idx, doc_name); first occurrence wins on collision
    chunk_to_src: Dict[str, tuple] = {}
    for chunk, _, chunk_idx, retriever, doc_name in all_candidates:
        if chunk not in chunk_to_src:
            chunk_to_src[chunk] = (retriever, chunk_idx, doc_name)

    best_chunk, best_score, _ = reranked[0]
    best_retriever, best_chunk_idx, best_doc_name = chunk_to_src.get(
        best_chunk, (documents[0]["retriever"], 0, documents[0]["source_name"])
    )

    match_lbl, css_class = match_label(best_score)
    base_attr  = best_retriever.get_attribution(best_chunk_idx)
    attr_text  = f"{best_doc_name}: {base_attr}" if base_attr else best_doc_name

    results_for_mmr = [(c, s) for c, s, _ in reranked]
    answer_sents, (sup_ci, sup_si), chunks_sents = extractive_answer_from_results(
        best_retriever, question, results_for_mmr, q_emb=q_emb
    )

    evidence_pack = build_evidence_pack(
        best_retriever, reranked, question,
        max_chunks=4, max_sents_per_chunk=2, max_chars=3500,
        q_emb=q_emb,
    )

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

    extras, seen = [], set()
    for chunk, score, _ in reranked[1:]:
        if score < 0.01:
            continue
        key = normalize_ws(chunk[:200]).lower()
        if key in seen:
            continue
        seen.add(key)
        src_retriever = chunk_to_src.get(chunk, (best_retriever, 0, ""))[0]
        snippet = remove_display_latex_anywhere(
            _top_sents(src_retriever.embedder, chunk, q_emb)
        )
        if snippet:
            extras.append((snippet, score))

    return answer_text, focused, source_label, css_class, match_lbl, best_score, extras, attr_text
