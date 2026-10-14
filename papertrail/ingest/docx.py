import logging
import re
from typing import Dict, Tuple

from papertrail import config

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^[A-Z0-9][\w\s,\-:]{2,79}$")


def load_docx(file_bytes: bytes) -> Tuple[str, Dict[int, str], Dict[int, int], str]:
    """
    Extract text from a .docx file.
    Returns (text, section_map, page_map, err).
    section_map: paragraph_index → heading text (detected via Heading styles).
    page_map: all None — DOCX has no reliable page numbers; attribution shows section only.
    err: "" on success, human-readable message on failure.
    """
    if len(file_bytes) > config.PDF_MAX_BYTES:
        limit_mb = config.PDF_MAX_BYTES // (1024 * 1024)
        return "", {}, {}, f"File too large (limit {limit_mb} MB)."

    try:
        import docx as _docx  # python-docx
    except ImportError:
        return "", {}, {}, "python-docx not installed. Add it to requirements.txt."

    try:
        import io
        doc = _docx.Document(io.BytesIO(file_bytes))
    except Exception as exc:
        return "", {}, {}, f"Could not open DOCX file: {exc}"

    section_map: Dict[int, str] = {}
    page_map:    Dict[int, int] = {}
    para_texts = []
    para_idx   = 0
    last_heading = ""

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue

        style_name = (para.style.name or "") if para.style else ""
        is_heading = style_name.startswith("Heading") or (
            _HEADING_RE.match(text) and not text.endswith(".") and len(text) < 80
        )

        if is_heading:
            last_heading = text
            section_map[para_idx] = text
        elif last_heading:
            section_map[para_idx] = last_heading

        para_texts.append(text)
        para_idx += 1

    if not para_texts:
        return "", {}, {}, "No readable text found in the DOCX file."

    return "\n\n".join(para_texts), section_map, page_map, ""
