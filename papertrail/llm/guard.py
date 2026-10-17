"""
Faithfulness guard for AI-written answers.

An answer is accepted only if every sentence is supported by the evidence:
similar in meaning to some evidence sentence (embedding cosine) AND built
mostly from words the evidence contains. Anything else is rejected and the
caller falls back to the verbatim document quote.
"""
import re
from typing import List

import numpy as np

from papertrail.utils.text import split_sentences

MIN_COSINE  = 0.60   # best-match similarity to an evidence sentence
MIN_OVERLAP = 0.60   # share of the sentence's 4+ letter words found in the evidence
REFUSAL     = "i cannot determine this from the document"

_TAG_RE  = re.compile(r"\[[^\]]{1,120}\]|\((?:§|Chunk:)[^)]{1,120}\)")
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")


def evidence_sentences(evidence_pack: str) -> List[str]:
    """Strip the leading '[attribution] ' tag from each evidence-pack line."""
    out = []
    for line in (evidence_pack or "").splitlines():
        line = re.sub(r"^\[[^\]]*\]\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def clean_generated(text: str) -> str:
    """Remove any citation/bracket tags the model emitted despite instructions."""
    return re.sub(r"\s{2,}", " ", _TAG_RE.sub("", text or "")).strip()


def is_supported(answer: str, evidence_pack: str, embedder) -> bool:
    answer = clean_generated(answer)
    if not answer or REFUSAL in answer.lower():
        return False
    ev_sents = evidence_sentences(evidence_pack)
    ans_sents = split_sentences(answer)
    if not ev_sents or not ans_sents:
        return False

    ev_words = set(w.lower() for w in _WORD_RE.findall(" ".join(ev_sents)))
    ev_emb  = embedder.encode(ev_sents,  normalize_embeddings=True)
    ans_emb = embedder.encode(ans_sents, normalize_embeddings=True)
    best = np.max(np.dot(ans_emb, ev_emb.T), axis=1)

    for sent, cos in zip(ans_sents, best):
        words = [w.lower() for w in _WORD_RE.findall(sent)]
        overlap = sum(1 for w in words if w in ev_words) / len(words) if words else 1.0
        if cos < MIN_COSINE or overlap < MIN_OVERLAP:
            return False
    return True
