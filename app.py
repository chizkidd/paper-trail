import html
import io

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

def normalize_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 40) -> list:
    """
    Split on paragraph/sentence boundaries first, then enforce max chunk size.
    This prevents chunks that start or end mid-sentence.
    """
    # Split into natural paragraphs first
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()
        # If adding this paragraph would exceed chunk_size, flush current buffer
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunk = " ".join(current_words)
            if len(current_words) >= 10:  # min 10 words to be useful
                chunks.append(chunk)
            # Keep overlap words for context continuity
            current_words = current_words[-overlap:] + para_words
        else:
            current_words.extend(para_words)

        # Force flush if buffer is huge
        while len(current_words) > chunk_size * 1.5:
            chunk = " ".join(current_words[:chunk_size])
            if len(chunk.strip()) >= 10:
                chunks.append(chunk)
            current_words = current_words[chunk_size - overlap:]

    # Flush remainder
    if len(current_words) >= 10:
        chunks.append(" ".join(current_words))

    return chunks


def match_label(score: float) -> tuple:
    """Return (label, css_class) for a cosine similarity score."""
    if score >= 0.25:
        return "strong match", "match-high"
    elif score >= 0.08:
        return "partial match", "match-medium"
    else:
        return "weak match", "match-low"


# ── Source loaders ─────────────────────────────────────────────────────────────

def load_pdf(file_bytes: bytes) -> tuple:
    """Returns (text, error)."""
    if not PDF_SUPPORT:
        return "", "pdfplumber not installed. Add it to requirements.txt."
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            return "", "PDF parsed but no text found. It may be a scanned/image-only PDF."
        return text, ""
    except Exception as e:
        return "", f"PDF parse error: {e}"


def load_url(url: str) -> tuple:
    """Returns (text, error)."""
    if not URL_SUPPORT:
        return "", "requests/beautifulsoup4 not installed."
    url = normalize_url(url)
    if not url:
        return "", "Please enter a URL."
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
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True).strip()
        if len(text) < 100:
            return "", (
                "Page fetched but no usable text extracted. "
                "The site may require JavaScript. "
                "Copy the page text and use Paste Text instead."
            )
        return text, ""
    except requests.exceptions.Timeout:
        return "", "Request timed out (15s). Try Paste Text instead."
    except requests.exceptions.ConnectionError:
        return "", "Could not connect. Check the URL and try again."
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP {e.response.status_code}: page may require login or does not exist."
    except Exception as e:
        return "", f"Unexpected fetch error: {e}"


# ── Retriever ──────────────────────────────────────────────────────────────────

class DocRetriever:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    def query(self, question: str, top_k: int = 3) -> list:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        # No hard threshold — always return top_k results so the user gets
        # something, and we show them the match quality label instead.
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


def build_knowledge_base(text: str, source_name: str) -> str:
    """Index text, store retriever in session state. Returns error or ''."""
    chunks = chunk_text(text)
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


def answer_question(retriever: DocRetriever, question: str) -> tuple:
    """Returns (answer_html, raw_score)."""
    results = retriever.query(question)

    if not results or results[0][1] < 0.01:
        return (
            "Nothing relevant found. Try rephrasing using specific keywords from the document.",
            0.0,
        )

    best_chunk, best_score = results[0]
    label, css_class = match_label(best_score)

    # Build a clean answer: show the best passage, then supporting passages
    passages = []
    seen = set()
    for chunk, score in results:
        if score < 0.01:
            continue
        # Deduplicate near-identical chunks (overlap can produce similar content)
        key = chunk[:60]
        if key in seen:
            continue
        seen.add(key)
        passages.append(html.escape(chunk))

    answer_body = "</p><p>".join(passages)
    answer_html = (
        f'<div class="msg-label">Most relevant passage(s)</div>'
        f"<p>{answer_body}</p>"
        f'<span class="match-pill {css_class}">{label}</span>'
    )
    return answer_html, best_score


# ── Session state init ─────────────────────────────────────────────────────────
for _key, _default in [
    ("retriever", None),
    ("source_name", None),
    ("chunk_count", 0),
    ("messages", []),
    ("source_type", "URL"),
    # Store uploaded PDF bytes so they survive reruns
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

        # Capture bytes immediately — survives subsequent reruns
        if uploaded is not None:
            file_bytes = uploaded.read()
            if file_bytes:
                st.session_state.pdf_bytes = file_bytes
                st.session_state.pdf_name = uploaded.name

        if st.session_state.pdf_bytes and st.button("Build Knowledge Base", key="build_pdf"):
            with st.spinner("Reading PDF..."):
                text, err = load_pdf(st.session_state.pdf_bytes)
            if err:
                st.error(err)
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(text, st.session_state.pdf_name)
                if err:
                    st.error(err)
                else:
                    st.session_state.pdf_bytes = None  # free memory
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

# URL and Paste Text inputs live here (light background = no CSS fights with sidebar)
if source_type == "URL":
    col1, col2 = st.columns([5, 1])
    with col1:
        url_val = st.text_input(
            "URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
            key="url_input",
        )
    with col2:
        fetch_clicked = st.button("Fetch", use_container_width=True)

    if fetch_clicked:
        if not url_val.strip():
            st.warning("Enter a URL first.")
        else:
            with st.spinner("Fetching..."):
                text, err = load_url(url_val)
            if err:
                st.error(err)
                st.info("Tip: copy the page text and use Paste Text instead.")
            else:
                with st.spinner("Indexing..."):
                    err = build_knowledge_base(text, url_val.strip())
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

# Source badge — only shown when a KB is active
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
    if source_type == "PDF Upload":
        msg = "Upload a PDF from the sidebar to begin."
    else:
        msg = "Load a document above to begin."
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">Knowledge base ready -- ask your first question.</div>',
        unsafe_allow_html=True,
    )

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        safe_content = html.escape(msg["content"])
        st.markdown(f'<div class="msg-user">{safe_content}</div>', unsafe_allow_html=True)
    else:
        # Bot messages contain pre-built HTML (already escaped where needed)
        st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# Question input — st.form fires only on explicit submit and clears after
if st.session_state.retriever:
    with st.form(key="question_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input(
                "Question",
                placeholder="What does the document say about...?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Ask", use_container_width=True)

    if submitted and question.strip():
        q = question.strip()
        answer_html, score = answer_question(st.session_state.retriever, q)
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({"role": "assistant", "content": answer_html})
        st.rerun()