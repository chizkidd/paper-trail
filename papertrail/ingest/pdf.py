import re
import unicodedata
from typing import Dict, List, Tuple

_HEADING_RE = re.compile(r"^(?:\d+[\d\.]*\s+)?[A-Z][^\n]{0,80}$")


def _normalize(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").replace("\u00A0", " ")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    out = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            out.append("")
            continue
        line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
        line = re.sub(r"([A-Za-z])(\d)",  r"\1 \2", line)
        line = re.sub(r"(\d)([A-Za-z])",  r"\1 \2", line)
        out.append(re.sub(r"\s+", " ", line).strip())
    return "\n".join(out).strip()

def _extract_blocks(page) -> str:
    blocks = sorted(page.get_text("blocks") or [], key=lambda b: (round(b[1],1), round(b[0],1)))
    return "\n".join((b[4] or "").strip() for b in blocks if (b[4] or "").strip())

def looks_scanned(t: str, min_chars: int = 40) -> bool:
    return len((t or "").strip()) < min_chars

def load_pdf(file_bytes: bytes) -> Tuple[str, Dict[int,str], Dict[int,int], str]:
    """Return (text, section_map, page_map, err). PyMuPDF with block fallback."""
    try:
        import fitz
    except Exception:
        try:
            import pymupdf as fitz  # type: ignore
        except Exception:
            return "", {}, {}, "PyMuPDF not installed. Add 'pymupdf' to requirements.txt."
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: List[str] = []
        section_map: Dict[int,str] = {}
        page_map:    Dict[int,int] = {}
        para_idx = 0
        for page_num in range(1, doc.page_count + 1):
            page = doc.load_page(page_num - 1)
            raw  = _normalize(page.get_text("text") or "")
            if looks_scanned(raw):
                alt = _normalize(_extract_blocks(page))
                if len(alt) > len(raw): 
                    raw = alt
            if looks_scanned(raw): 
                raw = ""
            page_map[para_idx] = page_num
            for line in raw.splitlines():
                line = line.strip()
                if not line: continue
                if _HEADING_RE.match(line) and not line.endswith(".") and len(line) < 80:
                    section_map[para_idx] = line
                para_idx += 1
            pages_text.append(raw)
        text = "\n\n".join(p for p in pages_text if p).strip()
        if not text:
            return "", {}, {}, "PDF parsed but no usable text found. It may be scanned/image-only."
        return text, section_map, page_map, ""
    except Exception as e:
        return "", {}, {}, f"PDF parse error: {e}"
