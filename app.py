import streamlit as st
import numpy as np
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Optional dependencies (graceful degradation) ──────────────────────────────
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
    --ink: #1a1a2e;
    --paper: #f5f0e8;
    --accent: #c0392b;
    --muted: #7a7060;
    --border: #d4cfc4;
    --code-bg: #ede8df;
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
    border-right: none;
}
section[data-testid="stSidebar"] * { color: var(--paper) !important; }
/* URL and Paste Text inputs — match PDF upload box style */
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="textarea"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px dashed rgba(255,255,255,0.35) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="base-input"] input,
section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background: transparent !important;
    color: #f5f0e8 !important;
    -webkit-text-fill-color: #f5f0e8 !important;
    caret-color: #f5f0e8 !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    border: none !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
section[data-testid="stSidebar"] [data-baseweb="base-input"] input::placeholder,
section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder,
section[data-testid="stSidebar"] .stTextInput input::placeholder,
section[data-testid="stSidebar"] .stTextArea textarea::placeholder {
    color: rgba(245,240,232,0.4) !important;
    -webkit-text-fill-color: rgba(245,240,232,0.4) !important;
}
/* File uploader — force dark bg and light text */
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
    background: rgba(255,255,255,0.12) !important;
    color: var(--paper) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.03em;
    transition: opacity 0.15s;
}
section[data-testid="stSidebar"] .stButton button:hover { opacity: 0.85; }

/* Header */
.doc-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ink);
    margin-bottom: 0.1rem;
}
.doc-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.source-badge {
    display: inline-block;
    background: var(--ink);
    color: var(--paper);
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}
.chunk-count {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-left: 8px;
}

/* Chat messages */
.chat-wrap { max-width: 780px; }
.msg-user {
    background: var(--ink);
    color: var(--paper);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1.1rem;
    margin: 0.6rem 0 0.6rem 80px;
    font-size: 0.95rem;
    line-height: 1.5;
}
.msg-bot {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 80px 0.6rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.msg-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.4rem;
}
.confidence-bar {
    display: inline-block;
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    margin-right: 6px;
    vertical-align: middle;
}
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1.1rem;
}

/* Input area */
.stTextInput input {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    background: white !important;
    transition: border-color 0.15s;
}
.stTextInput input:focus {
    border-color: var(--ink) !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Text chunking ──────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


# ── Source loaders ─────────────────────────────────────────────────────────────
def load_pdf(file_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        st.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(pages)


def load_url(url: str) -> str:
    if not URL_SUPPORT:
        st.error("requests/beautifulsoup4 not installed.")
        return ""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        st.error(f"Could not fetch URL: {e}")
        return ""


# ── TF-IDF retriever ───────────────────────────────────────────────────────────
class DocRetriever:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    def query(self, question: str, top_k: int = 3, threshold: float = 0.10):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] >= threshold]
        return results


# ── Answer builder ─────────────────────────────────────────────────────────────
def build_answer(results: list, question: str) -> tuple[str, float, str]:
    """Return (answer_text, confidence, source_type)."""
    if not results:
        return (
            "I couldn't find relevant information in the document for that question. "
            "Try rephrasing, or check that the document covers this topic.",
            0.0,
            "no_match",
        )

    best_chunk, best_score = results[0]

    # Stitch top chunks if they add meaningful signal
    context_parts = [r[0] for r in results if r[1] > 0.05]
    combined = " [...] ".join(context_parts)

    # Trim to a readable length
    if len(combined) > 900:
        combined = combined[:900] + "..."

    return combined, best_score, "doc"


# ── Session state init ─────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar: document loading ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Papertrail")
    st.markdown("Load a document, then ask it anything.")
    st.markdown("---")

    source_type = st.radio("Source type", ["PDF Upload", "URL", "Paste Text"], label_visibility="collapsed")

    raw_text = ""
    source_label = ""

    if source_type == "PDF Upload":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            raw_text = load_pdf(uploaded.read())
            source_label = uploaded.name

    elif source_type == "URL":
        st.markdown("""
        <style>
        .custom-input input, .custom-textarea textarea {
            background: rgba(255,255,255,0.07) !important;
            border: 1.5px dashed rgba(255,255,255,0.35) !important;
            border-radius: 8px !important;
            color: #f5f0e8 !important;
            -webkit-text-fill-color: #f5f0e8 !important;
            caret-color: #f5f0e8 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.82rem !important;
            padding: 0.5rem 0.75rem !important;
        }
        .custom-input input::placeholder, .custom-textarea textarea::placeholder {
            color: rgba(245,240,232,0.4) !important;
            -webkit-text-fill-color: rgba(245,240,232,0.4) !important;
        }
        .custom-input > div, .custom-textarea > div,
        .custom-input > div > div, .custom-textarea > div > div {
            background: transparent !important;
            border: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="custom-input">', unsafe_allow_html=True)
            url_input = st.text_input("Enter URL", placeholder="https://example.com/docs", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Fetch Page") and url_input:
            with st.spinner("Fetching..."):
                raw_text = load_url(url_input)
                source_label = url_input

    else:
        st.markdown('<div class="custom-textarea">', unsafe_allow_html=True)
        pasted = st.text_area("Paste your text here", height=220, placeholder="Paste any text — articles, docs, notes...", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        source_label = "Pasted text"
        raw_text = pasted

    if raw_text and st.button("Build Knowledge Base"):
        with st.spinner("Chunking & indexing..."):
            chunks = chunk_text(raw_text)
            if len(chunks) < 2:
                st.warning("Not enough text to index. Try a longer document.")
            else:
                st.session_state.retriever = DocRetriever(chunks)
                st.session_state.source_name = source_label
                st.session_state.chunk_count = len(chunks)
                st.session_state.messages = []
                st.success(f"Ready. {len(chunks)} chunks indexed.")

    if st.session_state.retriever:
        st.markdown("---")
        if st.button("Clear & start over"):
            st.session_state.retriever = None
            st.session_state.source_name = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="doc-header">Papertrail</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Ask anything from your document</div>', unsafe_allow_html=True)

if st.session_state.source_name:
    st.markdown(
        f'<span class="source-badge">{st.session_state.source_name[:60]}</span>'
        f'<span class="chunk-count">{st.session_state.chunk_count} chunks</span>',
        unsafe_allow_html=True,
    )

# Chat history
if not st.session_state.messages and not st.session_state.retriever:
    st.markdown(
        '<div class="empty-state">Load a document from the sidebar to begin.</div>',
        unsafe_allow_html=True,
    )
elif not st.session_state.messages and st.session_state.retriever:
    st.markdown(
        '<div class="empty-state">Knowledge base ready — ask your first question.</div>',
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        confidence = msg.get("confidence", 0)
        bar_width = int(confidence * 80)
        meta = ""
        if confidence > 0:
            meta = (
                f'<div class="msg-meta">'
                f'<span class="confidence-bar" style="width:{bar_width}px"></span>'
                f'confidence {confidence:.0%}'
                f'</div>'
            )
        st.markdown(
            f'<div class="msg-bot">{msg["content"]}{meta}</div>',
            unsafe_allow_html=True,
        )

# Input
if st.session_state.retriever:
    question = st.text_input(
        "Ask a question",
        placeholder="What does the document say about...?",
        label_visibility="collapsed",
        key="question_input",
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        results = st.session_state.retriever.query(question)
        answer, confidence, source_type_result = build_answer(results, question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
        })
        st.rerun()