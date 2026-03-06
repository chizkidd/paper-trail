from typing import Dict, List, Optional, Tuple
from papertrail.utils.text import dedup_paragraphs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 30) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    for para in paragraphs:
        words = para.split()
        if current and len(current) + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] + words
        else:
            current.extend(words)
        while len(current) > int(chunk_size * 1.5):
            chunks.append(" ".join(current[:chunk_size]))
            current = current[chunk_size - overlap:]
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) >= 20]


def _resolve(m: Optional[Dict], i: int):
    if not m: return None
    val = None
    for k in sorted(m.keys()):
        if k <= i: val = m[k]
    return val


def prepare_chunks(
    text: str,
    section_map: Optional[Dict[int,str]] = None,
    page_map:    Optional[Dict[int,int]] = None,
) -> Tuple[List[str], List[str], List, List[str]]:
    """Return (raw_chunks, chunk_sections, chunk_pages, embed_chunks)."""
    raw      = chunk_text(dedup_paragraphs(text))
    sections = [_resolve(section_map, i) or "" for i in range(len(raw))]
    pages    = [_resolve(page_map,    i)     for i in range(len(raw))]
    embed    = [f"{s}. {c}" if s else c for c, s in zip(raw, sections)]
    return raw, sections, pages, embed
