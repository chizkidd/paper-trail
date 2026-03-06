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
) -> str:
    lines: List[str] = []
    total = 0
    q_emb = retriever.embedder.encode([question], normalize_embeddings=True)
    for chunk, score, idx in reranked[:max_chunks]:
        if score < 0.01: continue
        attr  = retriever.get_attribution(idx) or f"Chunk: {idx}"
        sents = [s for s in split_sentences(chunk) if not is_noise_sentence(s)]
        if not sents: continue
        s_emb = retriever.embedder.encode(sents, normalize_embeddings=True)
        order = np.argsort(np.dot(s_emb, q_emb[0]))[::-1][:max_sents_per_chunk]
        for j in order:
            sent = remove_display_latex_anywhere(as_text(sents[int(j)]))
            if not sent: continue
            line = f"[{attr}] {sent}"
            if total + len(line) + 1 > max_chars:
                return "\n".join(lines).strip()
            lines.append(line)
            total += len(line) + 1
    return "\n".join(lines).strip()


def format_extractive_answer(answer_sents: List[str], attr_text: str) -> str:
    core = remove_display_latex_anywhere(
        " ".join(as_text(s) for s in (answer_sents or []) if s).strip()
    )
    if not core: core = "I cannot determine this from the document."
    return f"{core}\n\nEvidence: ({attr_text})" if attr_text else core
