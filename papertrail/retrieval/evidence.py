import re
from typing import List, Tuple
import numpy as np
from papertrail.retrieval.retriever import DocRetriever
from papertrail.utils.text import (
    as_text, is_noise_sentence, remove_display_latex_anywhere, split_sentences,
)


def build_evidence_pack(
    retriever:          DocRetriever,
    reranked:           List[Tuple[str,float,int]],
    question:           str,
    max_chunks:         int = 4,
    max_sents_per_chunk:int = 2,
    max_chars:          int = 1400,
    q_emb=None,
) -> str:
    lines: List[str] = []
    total = 0
    if q_emb is None:
        q_emb = retriever.embedder.encode([question], normalize_embeddings=True)

    for chunk, score, idx in reranked[:max_chunks]:
        if score < 0.01:
            continue
        attr  = retriever.get_attribution(idx) or f"Chunk: {idx}"
        sents = [s for s in split_sentences(chunk) if not is_noise_sentence(s)]
        if not sents:
            continue

        s_emb = retriever.embedder.encode(sents, normalize_embeddings=True)
        rel = np.dot(s_emb, q_emb[0])
        order = np.argsort(rel)[::-1][:max_sents_per_chunk]

        for j in order:
            sent = remove_display_latex_anywhere(as_text(sents[int(j)]))
            if not sent:
                continue
            line = f"[{attr}] {sent}"
            if total + len(line) + 1 > max_chars:
                return "\n".join(lines).strip()
            lines.append(line)
            total += len(line) + 1
    return "\n".join(lines).strip()


_FOOTNOTE_RE = re.compile(r"(?<=[A-Za-z][.!?,;:\"”’)])\d{1,3}(?=\s|$)")
_MAX_ANSWER_SENTS = 2
_MAX_ANSWER_CHARS = 400


def strip_footnote_markers(text: str) -> str:
    """Remove bare footnote numbers glued to punctuation, e.g. 'depths.65 This' -> 'depths. This'."""
    return re.sub(r"\s{2,}", " ", _FOOTNOTE_RE.sub("", text)).strip()


def format_extractive_answer(answer_sents: List[str], attr_text: str = "") -> str:
    """Short verbatim answer: at most 2 sentences / ~400 chars. Source is shown by the UI, not inlined."""
    kept: List[str] = []
    total = 0
    for s in (answer_sents or [])[:_MAX_ANSWER_SENTS]:
        s = strip_footnote_markers(as_text(s))
        if not s:
            continue
        if kept and total + len(s) > _MAX_ANSWER_CHARS:
            break
        kept.append(s)
        total += len(s)
    core = remove_display_latex_anywhere(" ".join(kept).strip())
    return core or "I cannot determine this from the document."
