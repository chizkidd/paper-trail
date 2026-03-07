import html
import re
import time
import unicodedata
from typing import List, Tuple, Optional, Dict

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    import requests
    from bs4 import BeautifulSoup
    URL_SUPPORT = True
except Exception:
    URL_SUPPORT = False

HF_SUPPORT = URL_SUPPORT

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_SUPPORT = True
except Exception:
    CROSS_ENCODER_SUPPORT = False

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN = None  # set via st.secrets (optional)

st.set_page_config(
    page_title="Papertrail",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink);
    color: var(--paper);
    border-right: none;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: var(--paper);
}
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

/* Dropdown menu */
ul[role="listbox"] { background: var(--ink); }
ul[role="listbox"] li { color: var(--paper); }

/* Main typography */
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

/* Source row */
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

/* Chat bubbles */
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

/* Attribution */
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

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1.1rem;
}

/* Inputs */
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

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LATEX_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*\\\]\s*$")
_LATEX_LEADING_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*?\\\]\s*")
_LATEX_DISPLAY_ANYWHERE_RE = re.compile(r"\\\[[\s\S]*?\\\]")
_LATEX_DISPLAY_RE = re.compile(r"\\\[[\s\S]*?\\\]")
_LABEL_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z\s\-]{0,40}:\s+")
_EMBEDDED_SECTION_HEADING_RE = re.compile(
    r"""\b\d+(?:\.\d+)+\s+[^\n]{1,80}?(?=\n|\s{2,}|\s*[:\-]\s|\.(?:\s|$)|$)""",
    re.VERBOSE,
)

def as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(str(i) for i in x if i is not None)
    return str(x)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", as_text(s) or "").strip()

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def dedup_paragraphs(text: str) -> str:
    if not text:
        return ""
    paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen, out = set(), []
    for p in paras:
        key = normalize_ws(p).lower()
        if len(key) >= 40:
            if key in seen:
                continue
            seen.add(key)
        out.append(p)
    return "\n".join(out).strip()

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

def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()] if text else []

def match_label(score: float) -> Tuple[str, str]:
    if score >= 0.25:
        return "strong match", "match-high"
    if score >= 0.08:
        return "partial match", "match-medium"
    return "weak match", "match-low"

def is_noise_sentence(s: str) -> bool:
    s_norm = normalize_ws(s)
    if not s_norm:
        return True
    if _LATEX_BLOCK_RE.match(s_norm):
        return True
    if len(s_norm) < 35:
        return True
    return False

def strip_leading_display_latex(s: str) -> str:
    return _LATEX_LEADING_BLOCK_RE.sub("", s or "", count=1).strip()

def remove_display_latex_anywhere(s: str) -> str:
    return normalize_ws(_LATEX_DISPLAY_ANYWHERE_RE.sub("", s or ""))

def load_pdf(file_bytes: bytes) -> Tuple[str, Dict[int, str], Dict[int, int], str]:
    """Return (text, section_map, page_map, err). Uses PyMuPDF with block fallback."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        try:
            import pymupdf as fitz  # type: ignore
        except Exception:
            return "", {}, {}, "PyMuPDF not installed. Add 'pymupdf' to requirements.txt."

    _hre = re.compile(r"^(?:\d+[\d\.]*\s+)?[A-Z][^\n]{0,80}$")

    def normalize_preserve_newlines(t: str) -> str:
        t = unicodedata.normalize("NFKC", t or "").replace("\u00A0", " ")
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        out = []
        for line in t.splitlines():
            line = line.strip()
            if not line:
                out.append("")
                continue
            line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
            line = re.sub(r"([A-Za-z])(\d)", r"\1 \2", line)
            line = re.sub(r"(\d)([A-Za-z])", r"\1 \2", line)
            out.append(re.sub(r"\s+", " ", line).strip())
        return "\n".join(out).strip()

    def looks_scanned(t: str, min_chars: int = 40) -> bool:
        return len((t or "").strip()) < min_chars

    def extract_blocks(page) -> str:
        blocks = page.get_text("blocks") or []
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        return "\n".join((b[4] or "").strip() for b in blocks if (b[4] or "").strip())

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: List[str] = []
        section_map: Dict[int, str] = {}
        page_map: Dict[int, int] = {}
        para_idx = 0

        for page_num in range(1, doc.page_count + 1):
            page = doc.load_page(page_num - 1)
            raw = normalize_preserve_newlines(page.get_text("text") or "")

            if looks_scanned(raw):
                alt = normalize_preserve_newlines(extract_blocks(page))
                if len(alt) > len(raw):
                    raw = alt

            if looks_scanned(raw):
                raw = ""

            page_map[para_idx] = page_num
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if _hre.match(line) and not line.endswith(".") and len(line) < 80:
                    section_map[para_idx] = line
                para_idx += 1

            pages_text.append(raw)

        text = "\n\n".join([p for p in pages_text if p]).strip()
        if not text:
            return "", {}, {}, "PDF parsed but no usable text found. It may be scanned/image-only."
        return text, section_map, page_map, ""
    except Exception as e:
        return "", {}, {}, f"PDF parse error: {e}"

def load_url(url: str) -> Tuple[str, Dict[int, str], str]:
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
            )
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        section_map: Dict[int, str] = {}
        current_heading = ""
        para_idx = 0
        for tag in soup.find_all(True):
            if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                current_heading = tag.get_text(strip=True)
            elif tag.name in ("p", "li", "td", "div", "blockquote", "pre"):
                if not tag.find(["p", "li", "td", "div", "blockquote"]):
                    if current_heading:
                        section_map[para_idx] = current_heading
                    para_idx += 1

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "form", "button", "iframe"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True).strip()
        if len(text) < 100:
            return "", {}, (
                "Page fetched but no usable text extracted. "
                "The site may require JavaScript. Try Paste Text instead."
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

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

class DocRetriever:
    def __init__(
        self,
        chunks: List[str],
        chunk_sections: Optional[List[str]] = None,
        chunk_pages: Optional[List[Optional[int]]] = None,
        embed_chunks: Optional[List[str]] = None,
    ):
        self.chunks = chunks
        self.chunk_sections = chunk_sections or [""] * len(chunks)
        self.chunk_pages = chunk_pages or [None] * len(chunks)

        texts_to_embed = embed_chunks if embed_chunks is not None else chunks

        self.embedder = load_embedder()
        self.embeddings = self.embedder.encode(texts_to_embed, normalize_embeddings=True)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts_to_embed)

    def query(self, question: str, top_k: int = 60, alpha: float = 0.45) -> List[Tuple[str, float, int]]:
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        dense = np.dot(self.embeddings, q_emb[0])

        q_vec = self.vectorizer.transform([question])
        sparse = cosine_similarity(q_vec, self.matrix).flatten()

        def norm01(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            mn, mx = float(x.min()), float(x.max())
            return np.zeros_like(x) if mx - mn < 1e-12 else (x - mn) / (mx - mn)

        scores = alpha * norm01(dense) + (1.0 - alpha) * norm01(sparse)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i]), int(i)) for i in top_idx]

    def score_sentences(self, question: str, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        q_vec = self.vectorizer.transform([question])
        s_mat = self.vectorizer.transform(sentences)
        return [float(x) for x in cosine_similarity(q_vec, s_mat).flatten()]

    def get_attribution(self, chunk_idx: int) -> str:
        section = self.chunk_sections[chunk_idx] if chunk_idx < len(self.chunk_sections) else ""
        page = self.chunk_pages[chunk_idx] if chunk_idx < len(self.chunk_pages) else None
        parts = []
        if section:
            parts.append(f"§ {section}")
        if page is not None:
            parts.append(f"p. {page}")
        return "  ·  ".join(parts)

def build_knowledge_base(
    text: str,
    source_name: str,
    section_map: Optional[Dict[int, str]] = None,
    page_map: Optional[Dict[int, int]] = None,
) -> str:
    cleaned = dedup_paragraphs(text)
    raw_chunks = chunk_text(cleaned)
    if len(raw_chunks) < 2:
        return "Not enough text to index. Try a longer document."

    def resolve(m, i):
        if not m:
            return None
        val = None
        for k in sorted(m.keys()):
            if k <= i:
                val = m[k]
        return val

    chunk_sections = [resolve(section_map, i) or "" for i in range(len(raw_chunks))]
    chunk_pages = [resolve(page_map, i) for i in range(len(raw_chunks))]

    try:
        embed_chunks = [f"{sec}. {chunk}" if sec else chunk for chunk, sec in zip(raw_chunks, chunk_sections)]
        st.session_state.retriever = DocRetriever(raw_chunks, chunk_sections, chunk_pages, embed_chunks)
        st.session_state.source_name = source_name
        st.session_state.chunk_count = len(raw_chunks)
        st.session_state.messages = []
        return ""
    except Exception as e:
        return f"Indexing error: {e}"

@st.cache_resource(show_spinner="Loading reranker (first run only)...")
def load_cross_encoder():
    if not CROSS_ENCODER_SUPPORT:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None

def rerank(question: str, candidates: List[Tuple[str, float, int]], top_k: int = 5) -> List[Tuple[str, float, int]]:
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

def evidence_overlap_ratio(answer: str, evidence: str) -> float:
    a = re.findall(r"[a-zA-Z]{4,}", (answer or "").lower())
    e = set(re.findall(r"[a-zA-Z]{4,}", (evidence or "").lower()))
    if not a:
        return 0.0
    return sum(1 for w in a if w in e) / max(1, len(a))

def generate_answer_hf(question: str, context: str) -> Tuple[str, bool]:
    if not HF_SUPPORT:
        return "", False

    prompt = f"""
You are a careful assistant. Answer the QUESTION using ONLY the EVIDENCE.
If the evidence is insufficient, say: "I cannot determine this from the document."

Rules:
- Write a direct answer first (1 to 3 sentences).
- Then add "Details" as bullet points if helpful.
- Every factual claim must include a citation in parentheses using the evidence tag,
  for example: (§ Methods  ·  p. 3) or (Chunk: 12).
- Do not invent citations. Do not use outside knowledge.

EVIDENCE:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "min_new_tokens": 20,
                    "temperature": 0.2,
                    "do_sample": False,
                    "return_full_text": False,
                },
            },
            timeout=30,
        )
        if resp.status_code == 503:
            return "", False
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            answer = (data[0]["generated_text"] or "").strip()
            answer = answer.split("ANSWER:")[-1].strip()
            answer = answer.split("Answer:")[-1].strip()
            if len(answer) <= 5:
                return "", False

            words = answer.split()
            if len(words) >= 4:
                from collections import Counter
                freq = Counter(w.lower() for w in words)
                if freq.most_common(1)[0][1] / len(words) > 0.40:
                    return "", False

            answer = answer.rstrip(", ;(").strip()
            if evidence_overlap_ratio(answer, context) < 0.18:
                return "", False

            return answer, True
        return "", False
    except Exception:
        return "", False

def generate_answer_ollama(question: str, evidence: str, model: str = "llama3.1:8b") -> Tuple[str, bool]:
    if not URL_SUPPORT:
        return "", False

    prompt = f"""
You are a careful assistant. Use ONLY the evidence. Cite every claim using the bracket tags.
If insufficient, say: "I cannot determine this from the document."

EVIDENCE:
{evidence}

QUESTION:
{question}

ANSWER:
""".strip()

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if len(text) < 10:
            return "", False
        if evidence_overlap_ratio(text, evidence) < 0.12:
            return "", False
        return text, True
    except Exception:
        return "", False

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
    cleaned, idx_map = [], []
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

    q_emb = retriever.embedder.encode([question], normalize_embeddings=True)
    s_emb = retriever.embedder.encode(cleaned, normalize_embeddings=True)

    rel = np.dot(s_emb, q_emb[0])
    sim = np.dot(s_emb, s_emb.T)

    candidates = [i for i, r in enumerate(rel) if r >= min_rel]
    if not candidates:
        return []

    selected, total_chars = [], 0
    while candidates and len(selected) < k and total_chars < max_chars:
        best_i, best_score = None, -1e9
        for i in candidates:
            red = float(np.max(sim[i, selected])) if selected else 0.0
            mmr = lambda_div * float(rel[i]) - (1.0 - lambda_div) * red
            if mmr > best_score:
                best_score, best_i = mmr, i

        if best_i is None:
            break

        sent_len = len(cleaned[best_i])
        if selected and (total_chars + 1 + sent_len) > max_chars:
            break

        selected.append(best_i)
        total_chars += sent_len + 1
        candidates.remove(best_i)

    return [idx_map[i] for i in sorted(set(selected))]

def extractive_answer_from_results(retriever: DocRetriever, question: str, results):
    pool_sents, meta, chunks_sents = build_sentence_pool(results, max_chunks=3)
    chosen = select_pool_sentences_mmr(
        retriever, question, pool_sents,
        k=4, min_rel=0.25, lambda_div=0.70, max_chars=900,
    )

    if not chosen:
        best_chunk = results[0][0]
        for s in split_sentences(best_chunk):
            if not is_noise_sentence(s):
                s2 = strip_leading_display_latex(s)
                return ([s2] if s2 else []), (0, None), chunks_sents
        return [], (0, None), chunks_sents

    picked = [strip_leading_display_latex(pool_sents[i]) for i in chosen]
    picked = [x for x in picked if x]

    chunk_votes: Dict[int, List[int]] = {}
    for i in chosen:
        ci, si = meta[i]
        chunk_votes.setdefault(ci, []).append(si)

    best_ci = max(chunk_votes.items(), key=lambda kv: len(kv[1]))[0]
    return picked, (best_ci, chunk_votes[best_ci]), chunks_sents

def focused_supporting_from_indices(chunks_sents, chosen_chunk_index: int, chosen_sent_indices: List[int]):
    sents = chunks_sents[chosen_chunk_index] if chosen_chunk_index < len(chunks_sents) else []
    if not sents:
        return ""
    keep = set()
    for i in chosen_sent_indices:
        for j in range(max(0, i - 1), min(len(sents), i + 2)):
            keep.add(j)
    out = " ".join(sents[i] for i in sorted(keep)).strip()
    return remove_display_latex_anywhere(strip_leading_display_latex(out))

def build_evidence_pack(
    retriever: DocRetriever,
    reranked: List[Tuple[str, float, int]],
    question: str,
    max_chunks: int = 4,
    max_sents_per_chunk: int = 2,
    max_chars: int = 1400,
) -> str:
    lines: List[str] = []
    total = 0
    q_emb = retriever.embedder.encode([question], normalize_embeddings=True)

    for chunk, score, idx in reranked[:max_chunks]:
        if score < 0.01:
            continue
        attr = retriever.get_attribution(idx) or f"Chunk: {idx}"
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

def format_extractive_answer(answer_sents: List[str], attr_text: str) -> str:
    core = remove_display_latex_anywhere(" ".join(as_text(s) for s in (answer_sents or []) if s).strip())
    if not core:
        core = "I cannot determine this from the document."
    return f"{core}\n\nEvidence: ({attr_text})" if attr_text else core

def focused_supporting_fallback(best_chunk: str, answer_sents: List[str]) -> str:
    if answer_sents:
        return " ".join(answer_sents).strip()
    sents = []
    for s in split_sentences(best_chunk):
        if is_noise_sentence(s):
            continue
        s2 = strip_leading_display_latex(s)
        if s2:
            sents.append(s2)
        if len(sents) >= 3:
            break
    return " ".join(sents).strip()

def answer_question(retriever: DocRetriever, question: str) -> tuple:
    n_chunks = len(retriever.chunks)
    dynamic_top_k = min(40, max(20, n_chunks // 5))

    candidates = retriever.query(question, top_k=dynamic_top_k)
    reranked = rerank(question, candidates, top_k=5)

    if not reranked or reranked[0][1] < 0.01:
        empty = "Nothing relevant found. Try rephrasing using keywords from the document."
        return f"<p>{empty}</p>", empty, 0.0, [], ""

    best_chunk, best_score, best_idx = reranked[0]
    label, css_class = match_label(best_score)
    attr_text = retriever.get_attribution(best_idx)

    results_for_mmr = [(c, s) for c, s, _ in reranked]
    answer_sents, (sup_ci, sup_si), chunks_sents = extractive_answer_from_results(retriever, question, results_for_mmr)

    evidence_pack = build_evidence_pack(retriever, reranked, question, max_chunks=4, max_sents_per_chunk=2, max_chars=1400)
    mode = st.session_state.get("answer_mode", "Hugging Face (best effort)")

    generated, is_generated = "", False
    if mode == "Local (Ollama)":
        generated, is_generated = generate_answer_ollama(question, evidence_pack)
    elif mode == "Hugging Face (best effort)":
        generated, is_generated = generate_answer_hf(question, evidence_pack)

    if is_generated:
        answer_text = remove_display_latex_anywhere(as_text(generated))
        source_label = "generated answer"
    else:
        answer_text = remove_display_latex_anywhere(format_extractive_answer(answer_sents, attr_text))
        source_label = "grounded answer"

    if sup_si is not None:
        focused = focused_supporting_from_indices(chunks_sents, sup_ci, sup_si)
    else:
        focused = focused_supporting_fallback(best_chunk, answer_sents)
    focused = remove_display_latex_anywhere(as_text(focused))

    attribution_html = (
        f'<div class="attr-bar"><span class="attr-pill">{html.escape(attr_text)}</span></div>'
        if attr_text else ""
    )

    passages = [html.escape(f"{source_label.capitalize()}: {answer_text}")]
    if focused and normalize_ws(focused) != normalize_ws(answer_text):
        passages.append(html.escape(f"Supporting passage: {focused}"))

    answer_body = "</p><p>".join(passages)
    answer_html = (
        f'<div class="msg-label">{source_label}</div>'
        f"<p>{answer_body}</p>"
        f'<span class="match-pill {css_class}">{label}</span>'
    )

    extras, seen = [], set()
    for chunk, score, _ in reranked[1:]:
        if score < 0.01:
            continue
        key = normalize_ws(chunk[:200]).lower()
        if key in seen:
            continue
        seen.add(key)
        extras.append((chunk, score))

    return answer_html, answer_text, best_score, extras, attribution_html

def clean_raw_passage(text: str) -> str:
    if not text:
        return ""
    text = _LATEX_DISPLAY_RE.sub("", text)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out = []
    for p in paras:
        p = _LABEL_PREFIX_RE.sub("", p)
        p = _EMBEDDED_SECTION_HEADING_RE.sub("", p)
        p = normalize_ws(p)
        if p:
            out.append(p)
    return "\n\n".join(out).strip()

def stream_words_into_bubble(text: str, source_label: str = ""):
    ph = st.empty()
    words = (text or "").split()
    buf = ""
    label_html = f'<div class="msg-label">{html.escape(source_label)}</div>' if source_label else ""
    for w in words:
        buf += w + " "
        ph.markdown(
            f'<div class="msg-bot">{label_html}<p>{html.escape(buf)}<span style="opacity:.5">▌</span></p></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.01)
    ph.markdown(
        f'<div class="msg-bot">{label_html}<p>{html.escape(buf.strip())}</p></div>',
        unsafe_allow_html=True,
    )

for k, d in [
    ("retriever", None),
    ("source_name", None),
    ("chunk_count", 0),
    ("messages", []),
    ("source_type", "URL"),
    ("answer_mode", "Hugging Face (best effort)"),
    ("pdf_bytes", None),
    ("pdf_name", ""),
]:
    st.session_state.setdefault(k, d)

with st.sidebar:
    st.markdown("## 📄 Papertrail")
    st.markdown("Load a document, then ask it anything.")
    st.markdown("---")

    st.radio(
        "Source",
        ["URL", "PDF Upload", "Paste Text"],
        key="source_type",
        label_visibility="collapsed",
    )
    source_type = st.session_state.source_type

    if source_type == "PDF Upload":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded is not None:
            fb = uploaded.read()
            if fb:
                st.session_state.pdf_bytes = fb
                st.session_state.pdf_name = uploaded.name

        if st.session_state.pdf_bytes and st.button("Build Knowledge Base", key="build_pdf"):
            with st.spinner("Reading PDF..."):
                text, section_map, page_map, err = load_pdf(st.session_state.pdf_bytes)
            if err:
                st.error(err)
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(text, st.session_state.pdf_name, section_map=section_map, page_map=page_map)
                if err:
                    st.error(err)
                else:
                    st.session_state.pdf_bytes = None
                    st.rerun()

    if st.session_state.retriever:
        st.markdown("---")
        st.caption(f"Active: {str(st.session_state.source_name)[:40]}")
        st.caption(f"{st.session_state.chunk_count} chunks indexed")

        st.session_state.answer_mode = st.selectbox(
            "Answer mode",
            ["Structured (no LLM)", "Local (Ollama)", "Hugging Face (best effort)"],
            index=["Structured (no LLM)", "Local (Ollama)", "Hugging Face (best effort)"].index(st.session_state.answer_mode),
        )

        if st.button("Clear & start over"):
            st.session_state.retriever = None
            st.session_state.source_name = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.session_state.pdf_bytes = None
            st.session_state.pdf_name = ""
            st.rerun()

st.markdown('<div class="doc-header">Papertrail</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Ask anything from your document</div>', unsafe_allow_html=True)

if source_type == "URL":
    col1, col2 = st.columns([5, 1])
    with col1:
        url_val = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed", key="url_input")
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
        "Paste text",
        height=200,
        placeholder="Paste any text here -- articles, docs, notes...",
        label_visibility="collapsed",
        key="paste_input",
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
        f'<div class="source-row"><span class="source-badge" title="{src}">{src}</span>'
        f'<span class="chunk-count">{st.session_state.chunk_count} chunks</span></div>',
        unsafe_allow_html=True,
    )

if not st.session_state.retriever:
    msg = "Upload a PDF from the sidebar to begin." if source_type == "PDF Upload" else "Load a document above to begin."
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown('<div class="empty-state">Knowledge base ready -- ask your first question.</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg.get("role") == "user":
        st.markdown(f'<div class="msg-user">{html.escape(msg.get("content",""))}</div>', unsafe_allow_html=True)
    else:
        bot_text = msg.get("answer_text") or ""
        st.markdown(f'<div class="msg-bot">{html.escape(bot_text)}</div>', unsafe_allow_html=True)
        if msg.get("attribution_html"):
            st.markdown(msg["attribution_html"], unsafe_allow_html=True)
        extras = msg.get("extras", [])
        if extras:
            with st.expander("Show supporting passages"):
                for chunk, score in extras:
                    st.caption(f"score: {score:.3f}")
                    st.write(clean_raw_passage(chunk))

if st.session_state.retriever:
    question = st.chat_input("Ask a question about your document...")
    if question and question.strip():
        q = question.strip()

        st.markdown(f'<div class="msg-user">{html.escape(q)}</div>', unsafe_allow_html=True)

        typing_ph = st.empty()
        typing_ph.markdown(
            '<div class="msg-bot"><div class="msg-label">assistant</div><p>Typing<span style="opacity:.6">▌</span></p></div>',
            unsafe_allow_html=True,
        )

        answer_html, answer_text, score, extras, attribution_html = answer_question(st.session_state.retriever, q)
        typing_ph.empty()

        stream_words_into_bubble(answer_text, source_label="assistant")

        if attribution_html:
            st.markdown(attribution_html, unsafe_allow_html=True)

        if extras:
            with st.expander("Show supporting passages"):
                for chunk, s in extras:
                    st.caption(f"score: {s:.3f}")
                    st.write(clean_raw_passage(chunk))

        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_html,
            "extras": extras,
            "attribution_html": attribution_html,
            "answer_text": answer_text,
        })

        st.markdown(
            """
            <script>
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            </script>
            """,
            unsafe_allow_html=True,
        )
