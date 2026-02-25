import html
import io
import re
from typing import List, Tuple



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
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def normalize_ws(s: str) -> str:
    # Collapse all whitespace to single spaces, trim.
    return re.sub(r"\s+", " ", s or "").strip()


def dedup_paragraphs(text: str) -> str:
    """
    Remove duplicated paragraphs/lines that often appear in scraped HTML or
    converted PDFs (templates, repeated headers, repeated blocks).

    De-dup is exact after whitespace normalization (keeps first occurrence).
    """
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines()]
    paras = [ln for ln in lines if ln]  # treat each non-empty line as a paragraph unit

    seen = set()
    out = []

    for p in paras:
        key = normalize_ws(p).lower()
        if not key:
            continue

        # Keep short lines (likely headings or structural labels)
        if len(key) < 40:
            out.append(p)
            continue

        if key in seen:
            continue

        seen.add(key)
        out.append(p)

    return "\n".join(out).strip()


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 40) -> List[str]:
    """
    Split on paragraph boundaries first, then enforce a max chunk size (in words),
    using overlap to preserve local context.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph would exceed chunk_size, flush current buffer
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunk = " ".join(current_words)
            if len(current_words) >= 10:  # min 10 words to be useful
                chunks.append(chunk)

            # Keep overlap words for continuity
            current_words = current_words[-overlap:] + para_words
        else:
            current_words.extend(para_words)

        # Force flush if buffer is huge
        while len(current_words) > chunk_size * 1.5:
            chunk = " ".join(current_words[:chunk_size])
            if len(chunk.strip()) >= 10:
                chunks.append(chunk)
            current_words = current_words[chunk_size - overlap :]

    # Flush remainder
    if len(current_words) >= 10:
        chunks.append(" ".join(current_words))

    return chunks


def match_label(score: float) -> Tuple[str, str]:
    """Return (label, css_class) for a cosine similarity score."""
    if score >= 0.25:
        return "strong match", "match-high"
    if score >= 0.08:
        return "partial match", "match-medium"
    return "weak match", "match-low"


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    # Basic sentence splitting; good enough for an extractive baseline.
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents


# ── Source loaders ─────────────────────────────────────────────────────────────
def load_pdf(file_bytes: bytes) -> Tuple[str, str]:
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


def load_url(url: str) -> Tuple[str, str]:
    """Returns (text, error)."""
    if not URL_SUPPORT:
        return "", "requests/beautifulsoup4 not installed."

    url = normalize_url(url)
    if not url:
        return "", "Please enter a URL."

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
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
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        # No hard threshold: always return top_k and label match quality.
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def score_sentences(self, question: str, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        q_vec = self.vectorizer.transform([question])
        s_mat = self.vectorizer.transform(sentences)
        s_scores = cosine_similarity(q_vec, s_mat).flatten()
        return [float(x) for x in s_scores]


def build_knowledge_base(text: str, source_name: str) -> str:
    """
    1) De-dup paragraphs (fix repeated blocks)
    2) Chunk
    3) Fit TF-IDF retriever and store in session state
    """
    cleaned = dedup_paragraphs(text)
    chunks = chunk_text(cleaned)

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

_LATEX_BLOCK_RE = re.compile(r"^\s*\\\[[\s\S]*\\\]\s*$")

def is_noise_sentence(s: str) -> bool:
    s_norm = normalize_ws(s)
    if not s_norm:
        return True
    # Drop pure LaTeX display blocks
    if _LATEX_BLOCK_RE.match(s_norm):
        return True
    # Drop very short structural fragments
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
    """
    Maximal Marginal Relevance selection over sentences.

    - maximizes relevance to question
    - penalizes redundancy w.r.t already selected sentences
    - returns sentence indices (in original order)
    """
    if not sentences:
        return []

    q_vec = retriever.vectorizer.transform([question])
    s_mat = retriever.vectorizer.transform(sentences)

    rel = cosine_similarity(q_vec, s_mat).flatten()  # relevance to question
    sim = cosine_similarity(s_mat, s_mat)            # sentence-to-sentence similarity

    # Candidate pool: only sentences above relevance threshold
    candidates = [i for i, r in enumerate(rel) if r >= min_rel]
    if not candidates:
        return []

    selected: List[int] = []
    total_chars = 0

    while candidates and len(selected) < k and total_chars < max_chars:
        best_i = None
        best_score = -1e9

        for i in candidates:
            # redundancy: max similarity to anything already selected
            red = 0.0
            if selected:
                red = float(np.max(sim[i, selected]))

            mmr = lambda_div * float(rel[i]) - (1.0 - lambda_div) * red

            if mmr > best_score:
                best_score = mmr
                best_i = i

        if best_i is None:
            break

        # enforce char budget
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

    # filter noise first
    filtered = []
    idx_map = []
    for i, s in enumerate(sents):
        if is_noise_sentence(s):
            continue
        s2 = strip_leading_display_latex(s)
        if not s2:
            continue
        filtered.append(s2)
        idx_map.append(i)

    if not filtered:
        return []

    # select diverse, relevant sentences
    chosen_filtered_idx = select_sentences_mmr(
        retriever,
        question,
        filtered,
        k=2,              # keep answer tight
        min_rel=0.08,     # tune globally
        lambda_div=0.75,  # more relevance than diversity
        max_chars=320,
    )

    if not chosen_filtered_idx:
        # fallback: first filtered sentence (already stripped)
        return [filtered[0]]

    picked = [filtered[j] for j in chosen_filtered_idx]
    picked = [x for x in picked if x]
    return picked


_LATEX_DISPLAY_ANYWHERE_RE = re.compile(r"\\\[[\s\S]*?\\\]")

def remove_display_latex_anywhere(s: str) -> str:
    if not s:
        return ""
    # Remove all display-math blocks like \[ ... \]
    s = _LATEX_DISPLAY_ANYWHERE_RE.sub("", s)
    # Clean up extra whitespace created by removals
    return normalize_ws(s)

def _focused_supporting_passage(
    retriever: DocRetriever, question: str, best_chunk: str, answer_sents: List[str]
) -> str:
    """
    Change #3: Answer-focused passage.
    Build a short 'supporting passage' from the best chunk that covers the answer,
    instead of dumping the entire chunk.

    Heuristic:
    - If we have answer sentences, include them plus up to 2 adjacent sentences each.
    - Else include top 4 scored sentences (in order).
    """
    sents = split_sentences(best_chunk)
    if not sents:
        return best_chunk

    if answer_sents:
        # Map answer sentences back to indices (best effort via substring match)
        idxs = set()
        for a in answer_sents:
            a_norm = normalize_ws(a)
            for i, s in enumerate(sents):
                if a_norm and a_norm in normalize_ws(s):
                    idxs.add(i)
                    break 

        # Expand window around each found index
        keep = set()
        for i in idxs:
            for j in range(max(0, i - 1), min(len(sents), i + 2)):
                keep.add(j)

        if keep:
            out = " ".join(sents[i] for i in sorted(keep)).strip()
            out = strip_leading_display_latex(out)
            out = remove_display_latex_anywhere(out)
            return out

    # fallback: top scored sentences
    scores = retriever.score_sentences(question, sents)
    ranked = np.argsort(np.array(scores))[::-1]
    pick = []
    for idx in ranked:
        if scores[idx] < 0.03:
            continue
        pick.append(int(idx))
        if len(pick) >= 4:
            break
    if not pick:
        text = " ".join(sents[:4]).strip()
        out = strip_leading_display_latex(text)
        out = remove_display_latex_anywhere(out)
        return out

    pick_sorted = sorted(set(pick))
    out = " ".join(sents[i] for i in pick_sorted).strip()
    out = strip_leading_display_latex(out)
    out = remove_display_latex_anywhere(out)
    return out


def build_sentence_pool(results, max_chunks: int = 3):
    """
    results: [(chunk, score), ...]
    returns:
      pool_sents: list[str]
      meta: list[(chunk_index, sent_index_in_chunk)]
      chunks_sents: list[list[str]]  # sentence lists per chunk
    """
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
    # Clean and filter noise first but keep mapping
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

    # Map back to original pool indices
    pool_selected = [idx_map[i] for i in sorted(set(selected))]
    return pool_selected

def extractive_answer_from_results(retriever: DocRetriever, question: str, results, k: int = 2):
    pool_sents, meta, _chunks_sents = build_sentence_pool(results, max_chunks=3)
    chosen_pool_idx = select_pool_sentences_mmr(
        retriever, question, pool_sents, k=k, min_rel=0.08, lambda_div=0.75, max_chars=320
    )

    if not chosen_pool_idx:
        # fallback: first non-noise sentence from best chunk
        best_chunk = results[0][0]
        for s in split_sentences(best_chunk):
            if not is_noise_sentence(s):
                s2 = strip_leading_display_latex(s)
                return [s2] if s2 else [], (0, None)
        return [], (0, None)

    picked = [strip_leading_display_latex(pool_sents[i]) for i in chosen_pool_idx]
    picked = [x for x in picked if x]

    # Figure out which chunk to use for supporting passage
    chunk_votes = {}
    for i in chosen_pool_idx:
        ci, si = meta[i]
        chunk_votes.setdefault(ci, []).append(si)

    # choose chunk with most selected sentences
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
    """Returns (answer_html, raw_score)."""
    results = retriever.query(question)

    pool_sents, meta, chunks_sents = build_sentence_pool(results, max_chunks=3)

    answer_sents, (support_chunk_idx, support_sent_idxs) = extractive_answer_from_results(
        retriever, question, results, k=2
    )
    answer_text = " ".join(answer_sents).strip()

    focused = ""
    if support_sent_idxs is not None:
        focused = focused_supporting_from_indices(results, chunks_sents, support_chunk_idx, support_sent_idxs)
    else:
        focused = _focused_supporting_passage(retriever, question, results[0][0], answer_sents)

    if not results or results[0][1] < 0.01:
        return (
            "Nothing relevant found. Try rephrasing using specific keywords from the document.",
            0.0,
        )

    best_chunk, best_score = results[0]
    label, css_class = match_label(best_score)

    # Change #1: Extractive answer first
    answer_sents = _extractive_answer(retriever, question, best_chunk)
    answer_text = " ".join(answer_sents) if answer_sents else split_sentences(best_chunk)[:2]

    # Change #3: Focused supporting passage
    focused = _focused_supporting_passage(retriever, question, best_chunk, answer_sents)

    passages = []
    seen = set()

    passages.append(html.escape(f"Answer: {answer_text}"))

    if focused and normalize_ws(focused) != normalize_ws(answer_text):
        passages.append(html.escape(f"Supporting passage: {focused}"))

    best_norm = normalize_ws(best_chunk)
    focused_norm = normalize_ws(focused)

    for chunk, score in results:
        if score < 0.01:
            continue

        key = normalize_ws(chunk[:200]).lower()
        if key in seen:
            continue
        seen.add(key)

        chunk_norm = normalize_ws(chunk)

        # skip the full best chunk (already represented)
        if chunk_norm == best_norm:
            continue

        # skip chunks that basically contain the focused passage
        if focused_norm and focused_norm in chunk_norm:
            continue

    extra = []
    for chunk, score in results[1:]:
        if score < 0.01:
            continue
        extra.append((chunk, score))
    st.session_state.last_extras = extra

    answer_body = "</p><p>".join(passages)
    answer_html = (
        f'<div class="msg-label">Most relevant passage(s)</div>'
        f"<p>{answer_body}</p>"
        f'<span class="match-pill {css_class}">{label}</span>'
    )
    return answer_html, best_score, extra



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


# Display math blocks like \[ ... \]
_LATEX_DISPLAY_RE = re.compile(r"\\\[[\s\S]*?\\\]")

# Label prefixes like "common structure:" or "Key idea:" at start of a line/paragraph
_LABEL_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z\s\-]{0,40}:\s+")

# Embedded numbered section headers like "8.9 Heuristic Search" anywhere in text
# Matches: 1.2 Title, 8.9 Heuristic Search, 10.3.1 Something
# Matches section numbers like 8.9 or 10.3.1, then removes the following heading phrase
# up to a boundary (newline, double space, sentence end, colon).
_EMBEDDED_SECTION_HEADING_RE = re.compile(
    r"""
    \b
    \d+(?:\.\d+)+          # section number like 8.9 or 10.3.1
    \s+                    # whitespace
    (?:[^\n]*[A-Za-z][^\n]*){1,80}?          # up to 80 chars of heading text (non-newline), non-greedy
    (?=                    # stop before boundary:
        \n                 # newline
      | \s{2,}              # double space
      | \s*[:\-–—]\s        # colon/dash separators
      | \.(?:\s|$)          # sentence end
      | $                   # end of string
    )
    """,
    re.VERBOSE,
)

def clean_raw_passage(text: str) -> str:
    if not text:
        return ""

    # Remove display math blocks
    text = _LATEX_DISPLAY_RE.sub("", text)

    # Process paragraph by paragraph to remove label prefixes reliably
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out_paras = []

    for p in paras:
        # Remove leading label prefix if present (e.g., "common structure: ")
        p = _LABEL_PREFIX_RE.sub("", p)

        # Remove embedded section heading titles anywhere
        p = _EMBEDDED_SECTION_HEADING_RE.sub("", p)

        # Normalize whitespace after removals
        p = normalize_ws(p)

        if p:
            out_paras.append(p)

    return "\n\n".join(out_paras).strip()

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        safe_content = html.escape(msg["content"])
        st.markdown(f'<div class="msg-user">{safe_content}</div>', unsafe_allow_html=True)
    else:
        # Bot messages contain pre-built HTML (already escaped where needed)
        st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        extras = msg.get("extras", [])
        if extras:
            with st.expander("Show raw passages"):
                for chunk, score in extras:
                    cleaned = clean_raw_passage(chunk)
                    st.write(cleaned)

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
        answer_html, score, extras = answer_question(st.session_state.retriever, q)
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({"role": "assistant", "content": answer_html, "extras": extras})
        st.rerun()