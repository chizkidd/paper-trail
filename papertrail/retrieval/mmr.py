from typing import Dict, List, Tuple
import numpy as np
from papertrail.retrieval.retriever import DocRetriever
from papertrail.utils.text import (
    is_noise_sentence, remove_display_latex_anywhere,
    split_sentences, strip_leading_display_latex,
)


def build_sentence_pool(results, max_chunks: int = 3):
    chunks_sents, pool_sents, meta = [], [], []
    for ci, (chunk, _) in enumerate(results[:max_chunks]):
        sents = split_sentences(chunk)
        chunks_sents.append(sents)
        for si, s in enumerate(sents):
            pool_sents.append(s)
            meta.append((ci, si))
    return pool_sents, meta, chunks_sents


def select_pool_sentences_mmr(
    retriever:  DocRetriever,
    question:   str,
    pool_sents: List[str],
    k:          int   = 3,
    min_rel:    float = 0.2,
    lambda_div: float = 0.72,
    max_chars:  int   = 650,
) -> List[int]:
    cleaned, idx_map = [], []
    for i, s in enumerate(pool_sents):
        if is_noise_sentence(s):
            continue
        s2 = strip_leading_display_latex(s)
        if s2:
            cleaned.append(s2)
            idx_map.append(i)
    if not cleaned:
        return []

    q_emb = retriever.embedder.encode([question], normalize_embeddings=True)
    s_emb = retriever.embedder.encode(cleaned,    normalize_embeddings=True)
    rel   = np.dot(s_emb, q_emb[0])
    sim   = np.dot(s_emb, s_emb.T)

    candidates = [i for i, r in enumerate(rel) if r >= min_rel]
    if not candidates:
        return []

    selected, total_chars = [], 0
    while candidates and len(selected) < k and total_chars < max_chars:
        best_i, best_score = None, -1e9
        for i in candidates:
            red = float(np.max(sim[i, selected])) if selected else 0.0
            mmr = lambda_div * float(rel[i]) - (1.0 - lambda_div) * red
            if mmr > best_score:
                best_score, best_i = mmr, i
        if best_i is None:
            break

        sent_len = len(cleaned[best_i])
        if selected and (total_chars + 1 + sent_len) > max_chars:
            break

        selected.append(best_i)
        total_chars += sent_len + 1
        candidates.remove(best_i)

    return [idx_map[i] for i in sorted(set(selected))]


def extractive_answer_from_results(
    retriever: DocRetriever, question: str, results
) -> Tuple[List[str], Tuple, List]:
    pool_sents, meta, chunks_sents = build_sentence_pool(results, max_chunks=3)
    chosen = select_pool_sentences_mmr(retriever, question, pool_sents)

    if not chosen:
        for s in split_sentences(results[0][0]):
            if not is_noise_sentence(s):
                s2 = strip_leading_display_latex(s)
                return ([s2] if s2 else []), (0, None), chunks_sents
        return [], (0, None), chunks_sents

    picked = [strip_leading_display_latex(pool_sents[i]) for i in chosen]
    picked = [x for x in picked if x]

    chunk_votes: Dict[int,List[int]] = {}
    for i in chosen:
        ci, si = meta[i]
        chunk_votes.setdefault(ci, []).append(si)
    best_ci = max(chunk_votes.items(), key=lambda kv: len(kv[1]))[0]
    return picked, (best_ci, chunk_votes[best_ci]), chunks_sents


def focused_supporting_from_indices(
    chunks_sents:        List[List[str]],
    chosen_chunk_index:  int,
    chosen_sent_indices: List[int],
) -> str:
    sents = chunks_sents[chosen_chunk_index] if chosen_chunk_index < len(chunks_sents) else []
    if not sents:
        return ""
    keep = set()
    for i in chosen_sent_indices:
        for j in range(max(0, i-1), min(len(sents), i+2)):
            keep.add(j)
    out = " ".join(sents[i] for i in sorted(keep)).strip()
    return remove_display_latex_anywhere(strip_leading_display_latex(out))


def focused_supporting_fallback(best_chunk: str, answer_sents: List[str]) -> str:
    if answer_sents:
        return " ".join(answer_sents).strip()
    sents = []
    for s in split_sentences(best_chunk):
        if is_noise_sentence(s):
            continue
        s2 = strip_leading_display_latex(s)
        if s2:
            sents.append(s2)
        if len(sents) >= 3:
            break
    return " ".join(sents).strip()
