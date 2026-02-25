import html
import io
import re
import time
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
    URL_SUPPORT = True
except ImportError:
    URL_SUPPORT = False

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Papertrail",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink:    #1a1a2e;
    --paper:  #f5f0e8;
    --accent: #c0392b;
    --muted:  #7a7060;
    --border: #d4cfc4;
    --green:  #155724;
    --green-bg: #d4edda;
    --yellow: #856404;
    --yellow-bg: #fff3cd;
    --red:    #721c24;
    --red-bg: #f8d7da;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--paper);
    color: var(--ink);
}
.stApp { background: var(--paper); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: var(--ink); border-right: none; }
section[data-testid="stSidebar"] * { color: var(--paper) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px dashed rgba(255,255,255,0.35) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
    color: rgba(245,240,232,0.85) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 6px !important;
    color: var(--paper) !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: opacity 0.15s;
}
section[data-testid="stSidebar"] .stButton button:hover { opacity: 0.85; }

/* ── Header ── */
.doc-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ink);
    margin-bottom: 0.15rem;
}
.doc-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Source badge ── */
.source-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    margin-top: 0.5rem;
}
.source-badge {
    background: var(--ink);
    color: var(--paper) !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.05em;
    max-width: 500px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.chunk-count {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
}

/* ── Chat messages ── */
.msg-user {
    background: var(--ink);
    color: var(--paper);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1.1rem;
    margin: 0.75rem 0 0.75rem clamp(20px, 10%, 80px);
    font-size: 0.95rem;
    line-height: 1.5;
    word-break: break-word;
}
.msg-bot {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem clamp(20px, 10%, 80px) 0.25rem 0;
    font-size: 0.95rem;
    line-height: 1.65;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    word-break: break-word;
}
.msg-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Attribution bar ── */
.attribution-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 0.3rem clamp(20px, 10%, 80px) 0.75rem 0;
    padding: 0;
}
.attr-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink);
    background: #e8e3d8;
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 3px;
    letter-spacing: 0.03em;
}
.attr-page {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    padding: 2px 0;
}
.match-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.match-high   { background: var(--green-bg);  color: var(--green); }
.match-medium { background: var(--yellow-bg); color: var(--yellow); }
.match-low    { background: var(--red-bg);    color: var(--red); }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1.1rem;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    background: white !important;
    transition: border-color 0.15s;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class Chunk:
    """A text chunk with source attribution metadata."""
    __slots__ = ("text", "section", "page", "source_type")

    def __init__(self, text: str, section: str = "", page: Optional[int] = None, source_type: str = ""):
        self.text = text
        self.section = section          # nearest heading before this chunk
        self.page = page                # PDF page number (1-indexed), None for URL/paste
        self.source_type = source_type  # "pdf" | "url" | "paste"

    def attribution_label(self) -> str:
        """Human-readable source label for the attribution bar."""
        parts = []
        if self.section:
            parts.append(f"§ {self.section}")
        if self.page is not None:
            parts.append(f"page {self.page}")
        return "  ·  ".join(parts) if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LATEX_DISPLAY_RE = re.compile(r"\\\[[\s\S]*?\\\]")
_LATEX_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*\\\]\s*$")
_LATEX_LEADING_RE = re.compile(r"^\s*\\\[[\s\S]*?\\\]\s*")

# Detects likely section headings in plain text:
# short line, no trailing period, optionally starts with a number
_HEADING_RE = re.compile(r"^(?:\d+[\.\d]*\s+)?[A-Z][^\n]{0,80}$")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def strip_latex(s: str) -> str:
    s = _LATEX_DISPLAY_RE.sub("", s)
    s = _LATEX_LEADING_RE.sub("", s)
    return normalize_ws(s)


def normalize_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def is_noise_sentence(s: str) -> bool:
    s = normalize_ws(s)
    if not s or len(s) < 35:
        return True
    if _LATEX_BLOCK_RE.match(s):
        return True
    return False


def dedup_paragraphs(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen, out = set(), []
    for p in lines:
        key = normalize_ws(p).lower()
        if not key:
            continue
        if len(key) < 40:   # keep short lines (headings)
            out.append(p)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return "\n".join(out).strip()


def detect_heading(line: str) -> Optional[str]:
    """Return the heading text if this line looks like a section heading, else None."""
    line = line.strip()
    if not line or len(line) > 100 or line.endswith("."):
        return None
    if _HEADING_RE.match(line):
        return line
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE LOADERS  (return List[Chunk], error_str)
# ══════════════════════════════════════════════════════════════════════════════

def _make_chunks_from_text(
    text: str,
    source_type: str,
    section_map: Optional[List[Tuple[int, str]]] = None,  # [(word_offset, heading)]
    page_map: Optional[List[Tuple[int, int]]] = None,     # [(word_offset, page_num)]
    chunk_size: int = 150,
    overlap: int = 40,
) -> List[Chunk]:
    """
    Core chunker. Respects paragraph boundaries, tags each chunk with the
    nearest preceding heading and page number from the provided maps.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    raw_chunks: List[str] = []
    current_words: List[str] = []
    word_offsets: List[int] = []  # starting word offset of each raw_chunk
    global_word_offset = 0

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunk_text = " ".join(current_words)
            if len(current_words) >= 10:
                raw_chunks.append(chunk_text)
                word_offsets.append(global_word_offset - len(current_words))
            current_words = current_words[-overlap:] + para_words
        else:
            current_words.extend(para_words)

        global_word_offset += len(para_words)

        while len(current_words) > chunk_size * 1.5:
            chunk_text = " ".join(current_words[:chunk_size])
            if len(chunk_text.strip()) >= 10:
                raw_chunks.append(chunk_text)
                word_offsets.append(global_word_offset - len(current_words))
            current_words = current_words[chunk_size - overlap:]

    if len(current_words) >= 10:
        raw_chunks.append(" ".join(current_words))
        word_offsets.append(global_word_offset - len(current_words))

    # Resolve section and page for each chunk via maps
    def resolve(maps, offset):
        if not maps:
            return None
        val = None
        for map_offset, map_val in maps:
            if map_offset <= offset:
                val = map_val
            else:
                break
        return val

    chunks = []
    for i, (raw, offset) in enumerate(zip(raw_chunks, word_offsets)):
        section = resolve(section_map, offset) or ""
        page = resolve(page_map, offset)
        chunks.append(Chunk(text=raw, section=section, page=page, source_type=source_type))

    return chunks


def load_pdf(file_bytes: bytes) -> Tuple[List[Chunk], str]:
    if not PDF_SUPPORT:
        return [], "pdfplumber not installed. Add it to requirements.txt."
    try:
        all_lines: List[Tuple[int, str, int]] = []  # (word_offset, line_text, page_num)
        word_offset = 0
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        all_lines.append((word_offset, line, page_num))
                        word_offset += len(line.split())

        if not all_lines:
            return [], "PDF parsed but no text found. It may be a scanned/image-only PDF."

        # Build section and page maps
        section_map: List[Tuple[int, str]] = []
        page_map: List[Tuple[int, int]] = []
        full_text_lines = []

        current_page = None
        for offset, line, page_num in all_lines:
            if page_num != current_page:
                page_map.append((offset, page_num))
                current_page = page_num
            heading = detect_heading(line)
            if heading:
                section_map.append((offset, heading))
            full_text_lines.append(line)

        full_text = "\n".join(full_text_lines)
        full_text = dedup_paragraphs(full_text)
        chunks = _make_chunks_from_text(full_text, "pdf", section_map, page_map)

        if not chunks:
            return [], "Could not extract usable chunks from this PDF."
        return chunks, ""
    except Exception as e:
        return [], f"PDF parse error: {e}"


def load_url(url: str) -> Tuple[List[Chunk], str]:
    if not URL_SUPPORT:
        return [], "requests/beautifulsoup4 not installed."
    url = normalize_url(url)
    if not url:
        return [], "Please enter a URL."
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract heading structure BEFORE stripping tags
        section_map: List[Tuple[int, str]] = []
        word_offset = 0
        text_lines = []

        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
            if tag.name in ("h1", "h2", "h3", "h4"):
                heading_text = tag.get_text(strip=True)
                if heading_text:
                    section_map.append((word_offset, heading_text[:80]))
            line = tag.get_text(separator=" ", strip=True)
            if line:
                text_lines.append(line)
                word_offset += len(line.split())

        text = "\n".join(text_lines).strip()
        if len(text) < 100:
            return [], (
                "Page fetched but no usable text extracted. "
                "The site may require JavaScript. "
                "Try copying the text and using Paste Text instead."
            )

        text = dedup_paragraphs(text)
        chunks = _make_chunks_from_text(text, "url", section_map=section_map, page_map=None)
        if not chunks:
            return [], "Could not extract usable chunks from this page."
        return chunks, ""

    except requests.exceptions.Timeout:
        return [], "Request timed out (15s). Try Paste Text instead."
    except requests.exceptions.ConnectionError:
        return [], "Could not connect. Check the URL and try again."
    except requests.exceptions.HTTPError as e:
        return [], f"HTTP {e.response.status_code}: page may require login or does not exist."
    except Exception as e:
        return [], f"Unexpected fetch error: {e}"


def load_paste(text: str) -> Tuple[List[Chunk], str]:
    if not text.strip():
        return [], "No text provided."
    # Detect headings from plain text
    section_map: List[Tuple[int, str]] = []
    word_offset = 0
    for line in text.splitlines():
        heading = detect_heading(line.strip())
        if heading:
            section_map.append((word_offset, heading))
        word_offset += len(line.split())

    cleaned = dedup_paragraphs(text)
    chunks = _make_chunks_from_text(cleaned, "paste", section_map=section_map, page_map=None)
    if not chunks:
        return [], "Not enough text to index. Try a longer document."
    return chunks, ""


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class DocRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def query(self, question: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def score_sentences(self, question: str, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        q_vec = self.vectorizer.transform([question])
        s_mat = self.vectorizer.transform(sentences)
        return list(cosine_similarity(q_vec, s_mat).flatten())


# ══════════════════════════════════════════════════════════════════════════════
# MMR + ANSWER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def match_label(score: float) -> Tuple[str, str]:
    if score >= 0.25:
        return "strong match", "match-high"
    if score >= 0.08:
        return "partial match", "match-medium"
    return "weak match", "match-low"


def select_sentences_mmr(
    retriever: DocRetriever,
    question: str,
    sentences: List[str],
    k: int = 2,
    min_rel: float = 0.08,
    lambda_div: float = 0.75,
    max_chars: int = 400,
) -> List[int]:
    if not sentences:
        return []
    q_vec = retriever.vectorizer.transform([question])
    s_mat = retriever.vectorizer.transform(sentences)
    rel = cosine_similarity(q_vec, s_mat).flatten()
    sim = cosine_similarity(s_mat, s_mat)

    candidates = [i for i, r in enumerate(rel) if r >= min_rel]
    if not candidates:
        return []

    selected: List[int] = []
    total_chars = 0

    while candidates and len(selected) < k and total_chars < max_chars:
        best_i, best_score = None, -1e9
        for i in candidates:
            red = float(np.max(sim[i, selected])) if selected else 0.0
            mmr = lambda_div * float(rel[i]) - (1.0 - lambda_div) * red
            if mmr > best_score:
                best_score, best_i = mmr, i
        if best_i is None:
            break
        if selected and (total_chars + len(sentences[best_i])) > max_chars:
            break
        selected.append(best_i)
        total_chars += len(sentences[best_i]) + 1
        candidates.remove(best_i)

    return sorted(selected)


def extract_answer_sentences(
    retriever: DocRetriever,
    question: str,
    results: List[Tuple[Chunk, float]],
) -> Tuple[str, Chunk]:
    """
    Pool sentences from top chunks, run MMR selection, return answer text
    and the source Chunk for attribution.
    """
    pool: List[str] = []
    pool_meta: List[int] = []  # which result index each sentence came from

    for ri, (chunk, score) in enumerate(results[:3]):
        for sent in split_sentences(chunk.text):
            if not is_noise_sentence(sent):
                cleaned = strip_latex(sent)
                if cleaned:
                    pool.append(cleaned)
                    pool_meta.append(ri)

    if not pool:
        # fallback: first two sentences of best chunk
        fallback_sents = split_sentences(results[0][0].text)[:2]
        return " ".join(fallback_sents), results[0][0]

    chosen_idx = select_sentences_mmr(retriever, question, pool, k=3, min_rel=0.06, max_chars=450)

    if not chosen_idx:
        return strip_latex(pool[0]), results[0][0]

    answer = " ".join(pool[i] for i in chosen_idx)

    # Attribution: use the chunk that contributed the most selected sentences
    from collections import Counter
    best_result_idx = Counter(pool_meta[i] for i in chosen_idx).most_common(1)[0][0]
    source_chunk = results[best_result_idx][0]

    return answer, source_chunk


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_base(chunks: List[Chunk], source_name: str) -> str:
    """Store retriever in session state. Returns error or ''."""
    if len(chunks) < 2:
        return "Not enough text to index. Try a longer document."
    try:
        st.session_state.retriever = DocRetriever(chunks)
        st.session_state.source_name = source_name
        st.session_state.chunk_count = len(chunks)
        st.session_state.messages = []
        return ""
    except Exception as e:
        return f"Indexing error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING ANSWER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def stream_answer(text: str, delay: float = 0.018):
    """
    Word-by-word generator for st.write_stream.
    When we add a real LLM later, this becomes a token stream from the API.
    """
    words = text.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for _key, _default in [
    ("retriever", None),
    ("source_name", None),
    ("chunk_count", 0),
    ("messages", []),
    ("source_type", "URL"),
    ("pdf_bytes", None),
    ("pdf_name", ""),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📄 Papertrail")
    st.markdown("Load a document, then ask it anything.")
    st.markdown("---")

    source_type = st.radio(
        "Source",
        ["URL", "PDF Upload", "Paste Text"],
        index=["URL", "PDF Upload", "Paste Text"].index(st.session_state.source_type),
        label_visibility="collapsed",
    )
    st.session_state.source_type = source_type

    if source_type == "PDF Upload":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded is not None:
            raw_bytes = uploaded.read()
            if raw_bytes:
                st.session_state.pdf_bytes = raw_bytes
                st.session_state.pdf_name = uploaded.name

        if st.session_state.pdf_bytes and st.button("Build Knowledge Base", key="build_pdf"):
            with st.spinner("Reading PDF..."):
                chunks, err = load_pdf(st.session_state.pdf_bytes)
            if err:
                st.error(err)
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(chunks, st.session_state.pdf_name)
                if err:
                    st.error(err)
                else:
                    st.session_state.pdf_bytes = None
                    st.rerun()

    if st.session_state.retriever:
        st.markdown("---")
        st.caption(f"Active: {str(st.session_state.source_name)[:40]}")
        st.caption(f"{st.session_state.chunk_count} chunks indexed")
        if st.button("Clear & start over"):
            for k, v in [("retriever", None), ("source_name", None),
                         ("chunk_count", 0), ("messages", []),
                         ("pdf_bytes", None), ("pdf_name", "")]:
                st.session_state[k] = v
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="doc-header">Papertrail</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Ask anything from your document</div>', unsafe_allow_html=True)

# URL and Paste Text inputs (main area = light bg, no CSS fights)
if source_type == "URL":
    col1, col2 = st.columns([5, 1])
    with col1:
        url_val = st.text_input("URL", placeholder="https://example.com/article",
                                label_visibility="collapsed", key="url_input")
    with col2:
        fetch_clicked = st.button("Fetch", use_container_width=True)

    if fetch_clicked:
        if not url_val.strip():
            st.warning("Enter a URL first.")
        else:
            with st.spinner("Fetching..."):
                chunks, err = load_url(url_val)
            if err:
                st.error(err)
                st.info("Tip: copy the page text and use Paste Text instead.")
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(chunks, url_val.strip())
                if err:
                    st.error(err)
                else:
                    st.rerun()

elif source_type == "Paste Text":
    pasted = st.text_area("Paste text", height=200,
                          placeholder="Paste any text here -- articles, docs, notes...",
                          label_visibility="collapsed", key="paste_input")
    if st.button("Build Knowledge Base", key="build_paste"):
        if not pasted.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Indexing..."):
                chunks, err = load_paste(pasted)
            if err:
                st.error(err)
            else:
                err = build_knowledge_base(chunks, "Pasted text")
                if err:
                    st.error(err)
                else:
                    st.rerun()

# Source badge
if st.session_state.retriever and st.session_state.source_name:
    src = html.escape(str(st.session_state.source_name))
    st.markdown(
        f'<div class="source-row">'
        f'<span class="source-badge" title="{src}">{src}</span>'
        f'<span class="chunk-count">{st.session_state.chunk_count} chunks</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Empty states
if not st.session_state.retriever:
    msg = "Upload a PDF from the sidebar to begin." if source_type == "PDF Upload" \
        else "Load a document above to begin."
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">Knowledge base ready — ask your first question.</div>',
        unsafe_allow_html=True,
    )

# ── Chat history (already answered turns) ─────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        safe = html.escape(msg["content"])
        st.markdown(f'<div class="msg-user">{safe}</div>', unsafe_allow_html=True)
    else:
        # Answer bubble
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="msg-label">Most relevant passage</div>'
            f'{html.escape(msg["content"])}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Attribution bar beneath the bubble
        attr = msg.get("attribution", "")
        match_cls = msg.get("match_css", "match-low")
        match_lbl = msg.get("match_label", "")
        if attr or match_lbl:
            parts = []
            if attr:
                parts.append(f'<span class="attr-section">{html.escape(attr)}</span>')
            if match_lbl:
                parts.append(f'<span class="match-pill {match_cls}">{match_lbl}</span>')
            st.markdown(
                f'<div class="attribution-bar">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )

# ── Live answer (streaming) ────────────────────────────────────────────────────
if st.session_state.retriever:
    question = st.chat_input("Ask a question about your document...")

    if question and question.strip():
        q = question.strip()

        # Show user message immediately
        safe_q = html.escape(q)
        st.markdown(f'<div class="msg-user">{safe_q}</div>', unsafe_allow_html=True)

        # Retrieve + extract
        results = st.session_state.retriever.query(q, top_k=5)

        if not results or results[0][1] < 0.01:
            answer_text = "Nothing relevant found. Try rephrasing using keywords from the document."
            source_chunk = None
            score = 0.0
        else:
            answer_text, source_chunk = extract_answer_sentences(
                st.session_state.retriever, q, results
            )
            score = results[0][1]

        label, css_class = match_label(score)
        attribution = source_chunk.attribution_label() if source_chunk else ""

        # Stream the answer into a bot bubble
        with st.container():
            st.markdown('<div class="msg-bot"><div class="msg-label">Most relevant passage</div>', unsafe_allow_html=True)
            st.write_stream(stream_answer(answer_text))
            st.markdown('</div>', unsafe_allow_html=True)

        # Attribution bar
        if attribution or label:
            parts = []
            if attribution:
                parts.append(f'<span class="attr-section">{html.escape(attribution)}</span>')
            parts.append(f'<span class="match-pill {css_class}">{label}</span>')
            st.markdown(
                f'<div class="attribution-bar">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )

        # Persist to history
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "attribution": attribution,
            "match_label": label,
            "match_css": css_class,
        })