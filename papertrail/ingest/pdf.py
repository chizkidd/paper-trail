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
    blocks = sorted(page.get_text("blocks") or [], key=lambda b: (round(b[1], 1), round(b[0], 1)))
    return "\n".join((b[4] or "").strip() for b in blocks if (b[4] or "").strip())


def looks_scanned(t: str, min_chars: int = 40) -> bool:
    return len((t or "").strip()) < min_chars


def _ocr_page(page) -> str:
    """OCR a single PyMuPDF page via pytesseract. Returns empty string if unavailable."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def load_pdf(file_bytes: bytes) -> Tuple[str, Dict[int, str], Dict[int, int], str]:
    """
    Return (text, section_map, page_map, err).
    Extraction order per page:
      1. PyMuPDF text stream
      2. PyMuPDF block-level fallback
      3. Tesseract OCR (if pytesseract + Pillow installed)
    """
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
        section_map: Dict[int, str] = {}
        page_map:    Dict[int, int] = {}
        para_idx = 0
        ocr_used = False

        for page_num in range(1, doc.page_count + 1):
            page = doc.load_page(page_num - 1)

            # Attempt 1: standard text stream
            raw = _normalize(page.get_text("text") or "")

            # Attempt 2: block-level extraction
            if looks_scanned(raw):
                alt = _normalize(_extract_blocks(page))
                if len(alt) > len(raw):
                    raw = alt

            # Attempt 3: OCR fallback
            if looks_scanned(raw):
                ocr_text = _ocr_page(page)
                if ocr_text.strip():
                    raw = _normalize(ocr_text)
                    ocr_used = True
                else:
                    raw = ""

            page_map[para_idx] = page_num
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if _HEADING_RE.match(line) and not line.endswith(".") and len(line) < 80:
                    section_map[para_idx] = line
                para_idx += 1
            pages_text.append(raw)

        text = "\n\n".join(p for p in pages_text if p).strip()
        if not text:
            return "", {}, {}, (
                "PDF parsed but no usable text found. "
                "It appears to be scanned/image-only and OCR is not available. "
                "Install pytesseract and Pillow to enable OCR support."
            )

        note = " (OCR used for some pages)" if ocr_used else ""
        return text, section_map, page_map, note

    except Exception as e:
        return "", {}, {}, f"PDF parse error: {e}"