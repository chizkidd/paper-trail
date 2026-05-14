import re
from typing import List, Tuple

_SENT_SPLIT_RE  = re.compile(r"(?<=[.!?])\s+")
_LATEX_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*\\\]\s*$")
_LATEX_LEAD_RE  = re.compile(r"^\s*\\\[[\s\S]*?\\\]\s*")
_LATEX_ANY_RE   = re.compile(r"\\\[[\s\S]*?\\\]")
_LABEL_RE       = re.compile(r"^\s*[A-Za-z][A-Za-z\s\-]{0,40}:\s+")
_SECTION_RE     = re.compile(
    r"""\b\d+(?:\.\d+)+\s+[^\n]{1,80}?(?=\n|\s{2,}|\s*[:\-]\s|\.(?:\s|$)|$)""",
    re.VERBOSE,
)
# Unicode math ranges: operators (∀–⋿), letterlike (℀–⅏), misc A (⟀–⟯), misc B (⦀–⧿), supplemental (⨀–⫿)
_UNICODE_MATH_RE = re.compile(r"[∀-⋿℀-⅏⟀-⟯⦀-⧿⨀-⫿]")


def as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list,tuple)):
        return " ".join(str(i) for i in x if i is not None)
    return str(x)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", as_text(s) or "").strip()

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://","https://")):
        url = "https://" + url
    return url

def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()] if text else []

def match_label(score: float) -> Tuple[str, str]:
    if score >= 0.25:
        return "strong match",  "match-high"
    if score >= 0.08:
        return "partial match", "match-medium"
    return "weak match",   "match-low"

def is_noise_sentence(s: str) -> bool:
    n = normalize_ws(s)
    return not n or len(n) < 35 or bool(_LATEX_BLOCK_RE.match(n)) or _is_math_heavy(n)

def strip_leading_display_latex(s: str) -> str:
    return _LATEX_LEAD_RE.sub("", s or "", count=1).strip()

def remove_display_latex_anywhere(s: str) -> str:
    return normalize_ws(_LATEX_ANY_RE.sub("", s or ""))

def _is_math_heavy(line: str) -> bool:
    """True when line contains Unicode math symbols — the reliable signal for mangled PDF equations."""
    return bool(_UNICODE_MATH_RE.search(line))


def clean_raw_passage(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    text = _LATEX_ANY_RE.sub("", text)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out = []
    for p in paras:
        if _is_math_heavy(p):
            continue
        p = _LABEL_RE.sub("", p)
        p = _SECTION_RE.sub("", p)
        p = normalize_ws(p)
        if p:
            out.append(p)
    result = "\n\n".join(out).strip()
    if max_chars and len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "…"
    return result

def dedup_paragraphs(text: str) -> str:
    if not text:
        return ""
    seen, out = set(), []
    for p in (ln.strip() for ln in text.splitlines() if ln.strip()):
        key = normalize_ws(p).lower()
        if len(key) >= 40:
            if key in seen:
                continue
            seen.add(key)
        out.append(p)
    return "\n".join(out).strip()
