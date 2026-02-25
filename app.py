import html
import io
import re
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Cross-encoder (sentence-transformers) ─────────────────────────────────────
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_SUPPORT = True
except ImportError:
    CROSS_ENCODER_SUPPORT = False

# ── HuggingFace Inference API for generation ───────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN = None  # set via st.secrets; works without token on free tier (rate limited)

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
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
section[data-testid="stSidebar"] .stButton button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: opacity 0.15s;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
section[data-testid="stSidebar"] .stButton button:hover { opacity: 0.85; }

/* ── Main typography ── */
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
    color: var(--paper);
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
    margin: 0.75rem clamp(20px, 10%, 80px) 0.75rem 0;
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
.match-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
}
.match-high   { background: #d4edda; color: #155724; }
.match-medium { background: #fff3cd; color: #856404; }
.match-low    { background: #f8d7da; color: #721c24; }

/* ── Attribution ── */
.attr-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin: -0.4rem clamp(20px, 10%, 80px) 0.75rem 0;
    padding: 0 1.1rem;
}
.attr-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    background: #ede8df;
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 3px;
    letter-spacing: 0.03em;
}

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


# ── Text utilities ─────────────────────────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list) or isinstance(x, tuple):
        return " ".join(str(i) for i in x if i is not None)
    return str(x)

def normalize_ws(s: str) -> str:
    s = as_text(s)
    return re.sub(r"\s+", " ", s or "").strip()


def dedup_paragraphs(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    paras = [ln for ln in lines if ln]
    seen = set()
    out = []
    for p in paras:
        key = normalize_ws(p).lower()
        if not key:
            continue
        if len(key) < 40:
            out.append(p)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return "\n".join(out).strip()


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 40) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunk = " ".join(current_words)
            if len(current_words) >= 10:
                chunks.append(chunk)
            current_words = current_words[-overlap:] + para_words
        else:
            current_words.extend(para_words)
        while len(current_words) > chunk_size * 1.5:
            chunk = " ".join(current_words[:chunk_size])
            if len(chunk.strip()) >= 10:
                chunks.append(chunk)
            current_words = current_words[chunk_size - overlap:]

    if len(current_words) >= 10:
        chunks.append(" ".join(current_words))

    return chunks


def match_label(score: float) -> Tuple[str, str]:
    if score >= 0.25:
        return "strong match", "match-high"
    if score >= 0.08:
        return "partial match", "match-medium"
    return "weak match", "match-low"


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents


# ── Source loaders ─────────────────────────────────────────────────────────────
def load_pdf(file_bytes: bytes) -> Tuple[str, dict, dict, str]:
    if not PDF_SUPPORT:
        return "", {}, {}, "pdfplumber not installed. Add it to requirements.txt."
    try:
        pages_text, section_map, page_map = [], {}, {}
        para_idx = 0
        _hre = re.compile(r"^(?:\d+[\d\.]*\s+)?[A-Z][^\n]{0,80}$")
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                page_map[para_idx] = page_num
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if _hre.match(line) and not line.endswith(".") and len(line) < 80:
                        section_map[para_idx] = line
                    para_idx += 1
                pages_text.append(raw)
        text = "\n\n".join(pages_text).strip()
        if not text:
            return "", {}, {}, "PDF parsed but no text found. It may be a scanned/image-only PDF."
        return text, section_map, page_map, ""
    except Exception as e:
        return "", {}, {}, f"PDF parse error: {e}"


def load_url(url: str) -> Tuple[str, dict, str]:
    if not URL_SUPPORT:
        return "", {}, "requests/beautifulsoup4 not installed."
    url = normalize_url(url)
    if not url:
        return "", {}, "Please enter a URL."
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

        # Step 1: capture heading positions BEFORE stripping anything
        section_map = {}
        current_heading = ""
        para_idx = 0
        for tag in soup.find_all(True):
            if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                current_heading = tag.get_text(strip=True)
            elif tag.name in ("p", "li", "td", "div", "blockquote", "pre"):
                # only leaf-ish nodes (skip containers whose children we will also visit)
                if not tag.find(["p", "li", "td", "div", "blockquote"]):
                    if current_heading:
                        section_map[para_idx] = current_heading
                    para_idx += 1

        # Step 2: strip noise, then get_text on whatever remains
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "form", "button", "iframe"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True).strip()

        if len(text) < 100:
            return "", {}, (
                "Page fetched but no usable text extracted. "
                "The site may require JavaScript. "
                "Try copying the text and using Paste Text instead."
            )
        return text, section_map, ""
    except requests.exceptions.Timeout:
        return "", {}, "Request timed out (15s). Try Paste Text instead."
    except requests.exceptions.ConnectionError:
        return "", {}, "Could not connect. Check the URL and try again."
    except requests.exceptions.HTTPError as e:
        return "", {}, f"HTTP {e.response.status_code}: page may require login or does not exist."
    except Exception as e:
        return "", {}, f"Unexpected fetch error: {e}"


# ── Retriever ──────────────────────────────────────────────────────────────────
class DocRetriever:
    def __init__(self, chunks: List[str],
                 chunk_sections: Optional[List[str]] = None,
                 chunk_pages: Optional[List[Optional[int]]] = None):
        self.chunks = chunks
        self.chunk_sections = chunk_sections or [""] * len(chunks)
        self.chunk_pages    = chunk_pages    or [None] * len(chunks)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    def query(self, question: str, top_k: int = 20) -> List[Tuple[str, float, int]]:
        """Returns (chunk_text, tfidf_score, chunk_index) — larger pool for reranker."""
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i]), int(i)) for i in top_idx]

    def score_sentences(self, question: str, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        q_vec = self.vectorizer.transform([question])
        s_mat = self.vectorizer.transform(sentences)
        return [float(x) for x in cosine_similarity(q_vec, s_mat).flatten()]

    def get_attribution(self, chunk_idx: int) -> str:
        parts = []
        section = self.chunk_sections[chunk_idx] if chunk_idx < len(self.chunk_sections) else ""
        page    = self.chunk_pages[chunk_idx]    if chunk_idx < len(self.chunk_pages)    else None
        if section:
            parts.append(f"§ {section}")
        if page is not None:
            parts.append(f"p. {page}")
        return "  ·  ".join(parts)


def build_knowledge_base(text: str, source_name: str,
                         section_map: Optional[Dict[int, str]] = None,
                         page_map: Optional[Dict[int, int]] = None) -> str:
    cleaned = dedup_paragraphs(text)
    raw_chunks = chunk_text(cleaned)
    if len(raw_chunks) < 2:
        return "Not enough text to index. Try a longer document."

    # Resolve nearest-preceding section/page for every chunk
    def resolve(m, i):
        if not m:
            return None
        val = None
        for k in sorted(m.keys()):
            if k <= i:
                val = m[k]
        return val

    chunk_sections = [resolve(section_map, i) or "" for i in range(len(raw_chunks))]
    chunk_pages    = [resolve(page_map,    i)     for i in range(len(raw_chunks))]

    try:
        st.session_state.retriever = DocRetriever(raw_chunks, chunk_sections, chunk_pages)
        st.session_state.source_name = source_name
        st.session_state.chunk_count = len(raw_chunks)
        st.session_state.messages = []
        return ""
    except Exception as e:
        return f"Indexing error: {e}"


# ── Cross-encoder reranking ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading reranker (first run only)...")
def load_cross_encoder():
    if not CROSS_ENCODER_SUPPORT:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def rerank(question: str, candidates: List[Tuple[str, float, int]], top_k: int = 5) -> List[Tuple[str, float, int]]:
    """Cross-encoder rerank. Falls back to TF-IDF order if unavailable."""
    ce = load_cross_encoder()
    if ce is None or not candidates:
        return candidates[:top_k]
    pairs = [(question, c[0]) for c in candidates]
    try:
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
    except Exception:
        return candidates[:top_k]


# ── HuggingFace flan-t5 generation ────────────────────────────────────────────

def generate_answer_hf(question: str, context: str) -> Tuple[str, bool]:
    """Returns (answer, is_generated). Falls back silently on cold start/error."""
    if not URL_SUPPORT:
        return "", False
    prompt = (
        f"Answer the question using only the context below. Be concise.\n\n"
        f"Context: {context[:1200]}\n\nQuestion: {question}\n\nAnswer:"
    )
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = requests.post(
            HF_API_URL, headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 150}},
            timeout=30,
        )
        if resp.status_code == 503:   # model warming up — silent fallback
            return "", False
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            answer = data[0]["generated_text"].strip()
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            if answer and len(answer) > 5:
                return answer, True
        return "", False
    except Exception:
        return "", False


# ── Streaming word generator ───────────────────────────────────────────────────

def stream_words(text: str, delay: float = 0.02):
    """Word-by-word generator for st.write_stream. Swap body for LLM token stream later."""
    words = text.split()
    for i, word in enumerate(words):
        yield word + ("" if i == len(words) - 1 else " ")
        time.sleep(delay)

_LATEX_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*\\\]\s*$")

def is_noise_sentence(s: str) -> bool:
    s_norm = normalize_ws(s)
    if not s_norm:
        return True
    if _LATEX_BLOCK_RE.match(s_norm):
        return True
    if len(s_norm) < 35:
        return True
    return False


_LATEX_LEADING_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*?\\\]\s*")

def strip_leading_display_latex(s: str) -> str:
    if not s:
        return ""
    return _LATEX_LEADING_BLOCK_RE.sub("", s, count=1).strip()

def select_sentences_mmr(
    retriever: DocRetriever,
    question: str,
    sentences: List[str],
    k: int = 2,
    min_rel: float = 0.08,
    lambda_div: float = 0.75,
    max_chars: int = 320,
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
        best_i = None
        best_score = -1e9

        for i in candidates:
            red = 0.0
            if selected:
                red = float(np.max(sim[i, selected]))
            mmr = lambda_div * float(rel[i]) - (1.0 - lambda_div) * red
            if mmr > best_score:
                best_score = mmr
                best_i = i

        if best_i is None:
            break

        sent_len = len(sentences[best_i])
        if selected and (total_chars + 1 + sent_len) > max_chars:
            break

        selected.append(best_i)
        total_chars += sent_len + (1 if selected else 0)
        candidates.remove(best_i)

    return sorted(set(selected))

def _extractive_answer(retriever: DocRetriever, question: str, best_chunk: str) -> List[str]:
    sents = split_sentences(best_chunk)
    if not sents:
        return []

    filtered = []
    for s in sents:
        if is_noise_sentence(s):
            continue
        s2 = strip_leading_display_latex(s)
        if not s2:
            continue
        filtered.append(s2)

    if not filtered:
        return []

    chosen_filtered_idx = select_sentences_mmr(
        retriever, question, filtered, k=2, min_rel=0.08, lambda_div=0.75, max_chars=320,
    )

    if not chosen_filtered_idx:
        return [filtered[0]]

    picked = [filtered[j] for j in chosen_filtered_idx]
    return [x for x in picked if x]


_LATEX_DISPLAY_ANYWHERE_RE = re.compile(r"\\\[[\s\S]*?\\\]")

def remove_display_latex_anywhere(s: str) -> str:
    if not s:
        return ""
    s = _LATEX_DISPLAY_ANYWHERE_RE.sub("", s)
    return normalize_ws(s)


def build_sentence_pool(results, max_chunks: int = 3):
    chunks_sents = []
    pool_sents = []
    meta = []
    for ci, (chunk, _) in enumerate(results[:max_chunks]):
        sents = split_sentences(chunk)
        chunks_sents.append(sents)
        for si, s in enumerate(sents):
            pool_sents.append(s)
            meta.append((ci, si))
    return pool_sents, meta, chunks_sents

def select_pool_sentences_mmr(
    retriever: DocRetriever,
    question: str,
    pool_sents: List[str],
    k: int = 2,
    min_rel: float = 0.08,
    lambda_div: float = 0.75,
    max_chars: int = 320,
) -> List[int]:
    cleaned = []
    idx_map = []

    for i, s in enumerate(pool_sents):
        if is_noise_sentence(s):
            continue
        s2 = strip_leading_display_latex(s)
        if not s2:
            continue
        cleaned.append(s2)
        idx_map.append(i)

    if not cleaned:
        return []

    q_vec = retriever.vectorizer.transform([question])
    s_mat = retriever.vectorizer.transform(cleaned)

    rel = cosine_similarity(q_vec, s_mat).flatten()
    sim = cosine_similarity(s_mat, s_mat)

    candidates = [i for i, r in enumerate(rel) if r >= min_rel]
    if not candidates:
        return []

    selected = []
    total_chars = 0

    while candidates and len(selected) < k and total_chars < max_chars:
        best_i = None
        best_score = -1e9

        for i in candidates:
            red = 0.0
            if selected:
                red = float(np.max(sim[i, selected]))
            mmr = lambda_div * float(rel[i]) - (1.0 - lambda_div) * red
            if mmr > best_score:
                best_score = mmr
                best_i = i

        if best_i is None:
            break

        sent_len = len(cleaned[best_i])
        if selected and (total_chars + 1 + sent_len) > max_chars:
            break

        selected.append(best_i)
        total_chars += sent_len + 1
        candidates.remove(best_i)

    return [idx_map[i] for i in sorted(set(selected))]

def extractive_answer_from_results(retriever: DocRetriever, question: str, results, k: int = 2):
    pool_sents, meta, chunks_sents = build_sentence_pool(results, max_chunks=3)
    chosen_pool_idx = select_pool_sentences_mmr(
        retriever, question, pool_sents, k=k, min_rel=0.08, lambda_div=0.75, max_chars=320
    )

    if not chosen_pool_idx:
        best_chunk = results[0][0]
        for s in split_sentences(best_chunk):
            if not is_noise_sentence(s):
                s2 = strip_leading_display_latex(s)
                return [s2] if s2 else [], (0, None)
        return [], (0, None)

    picked = [strip_leading_display_latex(pool_sents[i]) for i in chosen_pool_idx]
    picked = [x for x in picked if x]

    chunk_votes = {}
    for i in chosen_pool_idx:
        ci, si = meta[i]
        chunk_votes.setdefault(ci, []).append(si)

    best_ci = max(chunk_votes.items(), key=lambda kv: len(kv[1]))[0]
    return picked, (best_ci, chunk_votes[best_ci])


def focused_supporting_from_indices(results, chunks_sents, chosen_chunk_index: int, chosen_sent_indices: List[int]):
    sents = chunks_sents[chosen_chunk_index]
    if not sents:
        return ""

    keep = set()
    for i in chosen_sent_indices:
        for j in range(max(0, i - 1), min(len(sents), i + 2)):
            keep.add(j)

    out = " ".join(sents[i] for i in sorted(keep)).strip()
    out = strip_leading_display_latex(out)
    out = remove_display_latex_anywhere(out)
    return out


def answer_question(retriever: DocRetriever, question: str) -> tuple:
    """
    Pipeline: TF-IDF (top-20) -> cross-encoder rerank (top-5)
              -> MMR extraction -> HF generation (falls back to extractive)
              -> section attribution
    Returns: (answer_html, answer_text, score, extras, attribution_html)
    """
    # Step 1+2: retrieve and rerank
    candidates = retriever.query(question, top_k=20)
    reranked   = rerank(question, candidates, top_k=5)

    if not reranked or reranked[0][1] < 0.01:
        empty = "Nothing relevant found. Try rephrasing using keywords from the document."
        return f'<p>{empty}</p>', empty, 0.0, [], ""

    best_chunk, best_score, best_idx = reranked[0]
    label, css_class = match_label(best_score)

    # Step 3: MMR extractive answer (drop chunk_idx for downstream helpers)
    results_for_mmr = [(c, s) for c, s, _ in reranked]
    pool_sents, meta, chunks_sents = build_sentence_pool(results_for_mmr, max_chunks=3)
    answer_sents, (sup_ci, sup_si) = extractive_answer_from_results(
        retriever, question, results_for_mmr, k=2
    )
    extractive_text = " ".join(answer_sents).strip()
    if not extractive_text:
        extractive_text = " ".join(split_sentences(best_chunk)[:2]).strip()
    if isinstance(extractive_text, (list, tuple)):
        extractive_text = " ".join(map(str, extractive_text))

    # Step 4: HF generation — falls back silently to extractive
    generated, is_generated = generate_answer_hf(question, best_chunk)
    answer_text  = generated if is_generated else extractive_text
    source_label = "generated answer" if is_generated else "extracted passage"

    # Supporting passage
    focused = ""
    if sup_si is not None:
        focused = focused_supporting_from_indices(results_for_mmr, chunks_sents, sup_ci, sup_si)
    if isinstance(focused, (list, tuple)):
        focused = " ".join(map(str, focused))

    # Step 5: attribution
    attr_text = retriever.get_attribution(best_idx)
    attribution_html = ""
    if attr_text:
        attribution_html = (
            f'<div class="attr-bar">'
            f'<span class="attr-pill">{html.escape(attr_text)}</span>'
            f'</div>'
        )

    # Build display HTML
    passages = [html.escape(f"{source_label.capitalize()}: {answer_text}")]
    if focused and normalize_ws(focused) != normalize_ws(answer_text):
        passages.append(html.escape(f"Supporting passage: {focused}"))
    answer_body = "</p><p>".join(passages)
    answer_html = (
        f'<div class="msg-label">{source_label}</div>'
        f"<p>{answer_body}</p>"
        f'<span class="match-pill {css_class}">{label}</span>'
    )

    extras = []
    seen = set()
    for chunk, score, _ in reranked[1:]:
        key = normalize_ws(chunk[:200]).lower()
        if key not in seen and score >= 0.01:
            seen.add(key); extras.append((chunk, score))

    return answer_html, answer_text, best_score, extras, attribution_html


# ── Session state init ─────────────────────────────────────────────────────────
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
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
            file_bytes = uploaded.read()
            if file_bytes:
                st.session_state.pdf_bytes = file_bytes
                st.session_state.pdf_name = uploaded.name

        if st.session_state.pdf_bytes and st.button("Build Knowledge Base", key="build_pdf"):
            with st.spinner("Reading PDF..."):
                text, section_map, page_map, err = load_pdf(st.session_state.pdf_bytes)
            if err:
                st.error(err)
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(text, st.session_state.pdf_name,
                                               section_map=section_map, page_map=page_map)
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
            st.session_state.retriever = None
            st.session_state.source_name = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.session_state.pdf_bytes = None
            st.session_state.pdf_name = ""
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="doc-header">Papertrail</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Ask anything from your document</div>', unsafe_allow_html=True)

if source_type == "URL":
    col1, col2 = st.columns([5, 1])
    with col1:
        url_val = st.text_input(
            "URL", placeholder="https://example.com/article",
            label_visibility="collapsed", key="url_input",
        )
    with col2:
        fetch_clicked = st.button("Fetch", use_container_width=True)

    if fetch_clicked:
        if not url_val.strip():
            st.warning("Enter a URL first.")
        else:
            with st.spinner("Fetching..."):
                text, section_map, err = load_url(url_val)
            if err:
                st.error(err)
                st.info("Tip: copy the page text and use Paste Text instead.")
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(text, url_val.strip(), section_map=section_map)
                if err:
                    st.error(err)
                else:
                    st.rerun()

elif source_type == "Paste Text":
    pasted = st.text_area(
        "Paste text", height=200,
        placeholder="Paste any text here -- articles, docs, notes...",
        label_visibility="collapsed", key="paste_input",
    )
    if st.button("Build Knowledge Base", key="build_paste"):
        if not pasted.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Indexing..."):
                err = build_knowledge_base(pasted.strip(), "Pasted text")
            if err:
                st.error(err)
            else:
                st.rerun()

if st.session_state.retriever and st.session_state.source_name:
    src = html.escape(str(st.session_state.source_name))
    st.markdown(
        f'<div class="source-row">'
        f'<span class="source-badge" title="{src}">{src}</span>'
        f'<span class="chunk-count">{st.session_state.chunk_count} chunks</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

if not st.session_state.retriever:
    msg = "Upload a PDF from the sidebar to begin." if source_type == "PDF Upload" \
        else "Load a document above to begin."
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">Knowledge base ready -- ask your first question.</div>',
        unsafe_allow_html=True,
    )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("attribution_html"):
            st.markdown(msg["attribution_html"], unsafe_allow_html=True)
        extras = msg.get("extras", [])
        if extras:
            with st.expander("Show supporting passages"):
                for chunk, score in extras:
                    st.caption(f"score: {score:.3f}")
                    st.write(chunk)

# ── Live input + streaming ─────────────────────────────────────────────────────
if st.session_state.retriever:
    question = st.chat_input("Ask a question about your document...")

    if question and question.strip():
        q = question.strip()

        # Show user bubble immediately
        st.markdown(f'<div class="msg-user">{html.escape(q)}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            answer_html, answer_text, score, extras, attribution_html = answer_question(
                st.session_state.retriever, q
            )

        # Stream the plain-text answer word by word into a bot bubble
        st.markdown('<div class="msg-bot">', unsafe_allow_html=True)
        st.write_stream(stream_words(answer_text))
        st.markdown('</div>', unsafe_allow_html=True)

        # Attribution bar
        if attribution_html:
            st.markdown(attribution_html, unsafe_allow_html=True)

        # Supporting passages
        if extras:
            with st.expander("Show supporting passages"):
                for chunk, s in extras:
                    st.caption(f"score: {s:.3f}")
                    st.write(chunk)

        # Persist
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_html,
            "extras": extras,
            "attribution_html": attribution_html,
            "answer_text": answer_text,
        })