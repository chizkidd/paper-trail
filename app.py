import html
import logging
import re

import streamlit as st

from papertrail import config
from papertrail.ingest.pdf import load_pdf
from papertrail.ingest.web import load_url
from papertrail.qa import (
    answer_across_docs,
    answer_question_stream,
    build_knowledge_base,
    export_conversation_markdown,
    get_active_doc,
    get_active_retriever,
    get_documents,
    remove_document,
)
from papertrail.ingest.docx import load_docx
from papertrail.utils.text import clean_raw_passage
from papertrail.utils.stream import stream_words_into_bubble

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Papertrail",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Optional HF token from Streamlit secrets ───────────────────────────────────
import papertrail.llm.hf as _hf_llm  # noqa: E402
_hf_llm.HF_TOKEN = st.secrets.get("HF_TOKEN", None)

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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--paper); color: var(--ink); }
.stApp { background: var(--paper); }
.stMainBlockContainer { padding-bottom: 2rem !important; }

section[data-testid="stSidebar"] { background: var(--ink); color: var(--paper); border-right: none; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: var(--paper); }
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
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] { color: rgba(245,240,232,0.85) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif; font-weight: 500; transition: opacity 0.15s;
}
section[data-testid="stSidebar"] .stButton button:hover { opacity: 0.85; }
ul[role="listbox"] { background: var(--ink); }
ul[role="listbox"] li { color: var(--paper); }

.doc-header { font-family: 'DM Serif Display', serif; font-size: 2.6rem; letter-spacing: -0.02em; line-height: 1; color: var(--ink); margin-bottom: 0.15rem; }
.doc-sub    { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--muted); letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 1.5rem; }

.source-row   { display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; margin-top: 0.5rem; }
.source-badge { background: var(--ink); color: var(--paper); font-family: 'DM Mono', monospace; font-size: 0.7rem; padding: 3px 10px; border-radius: 3px; letter-spacing: 0.05em; max-width: 500px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chunk-count  { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--muted); }

.msg-user  { background: var(--ink); color: var(--paper); border-radius: 12px 12px 4px 12px; padding: 0.75rem 1.1rem; margin: 0.75rem 0 0.75rem clamp(20px,10%,80px); font-size: 0.95rem; line-height: 1.5; word-break: break-word; }
.msg-bot   { background: white; color: var(--ink); border: 1px solid var(--border); border-radius: 12px 12px 12px 4px; padding: 0.85rem 1.1rem; margin: 0.75rem clamp(20px,10%,80px) 0.75rem 0; font-size: 0.95rem; line-height: 1.65; box-shadow: 0 1px 4px rgba(0,0,0,0.05); word-break: break-word; }
.msg-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }

.match-pill   { display: inline-block; font-family: 'DM Mono', monospace; font-size: 0.65rem; padding: 2px 7px; border-radius: 20px; margin-top: 0.5rem; letter-spacing: 0.04em; }
.match-high   { background: #d4edda; color: #155724; }
.match-medium { background: #fff3cd; color: #856404; }
.match-low    { background: #f8d7da; color: #721c24; }

.attr-bar  { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: -0.4rem clamp(20px,10%,80px) 0.75rem 0; padding: 0 1.1rem; }
.attr-pill { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--muted); background: #ede8df; border: 1px solid var(--border); padding: 2px 8px; border-radius: 3px; letter-spacing: 0.03em; }

.empty-state { text-align: center; padding: 4rem 2rem; color: var(--muted); font-family: 'DM Serif Display', serif; font-style: italic; font-size: 1.1rem; }

.stTextInput input, .stTextArea textarea { border: 1.5px solid var(--border) !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; background: white !important; transition: border-color 0.15s; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--ink) !important; box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)

def _safe_md_to_html(text: str) -> str:
    """Escape HTML then apply minimal markdown: bold, italic, inline code, line breaks."""
    t = html.escape(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t, flags=re.DOTALL)
    t = re.sub(r"\*([^\*\n]+?)\*", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
    t = re.sub(r"\n\n+", "</p><p>", t)
    t = re.sub(r"\n", "<br>", t)
    return t


# ── Session state ──────────────────────────────────────────────────────────────
for k, d in [
    ("documents",      []),
    ("active_doc_idx", 0),
    ("source_type",    "URL"),
    ("answer_mode",    "Hugging Face (best effort)"),
    ("pdf_bytes",      None),
    ("pdf_name",       ""),
    ("search_all",     False),
]:
    st.session_state.setdefault(k, d)


def _validate_session_state() -> None:
    docs = st.session_state.get("documents", [])
    idx  = st.session_state.get("active_doc_idx", 0)
    required = {"source_name", "chunk_count", "retriever", "messages", "text_hash"}
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict) or not required.issubset(doc.keys()):
            logging.getLogger(__name__).warning(
                "Session state corruption at doc %d — resetting.", i
            )
            st.session_state.documents = []
            st.session_state.active_doc_idx = 0
            st.warning("Session was reset due to an inconsistency. Please reload your document.")
            return
    if docs and idx >= len(docs):
        st.session_state.active_doc_idx = max(0, len(docs) - 1)


_validate_session_state()

# Warm up models on first page load so the first question has no cold-start delay.
# @st.cache_resource makes these no-ops on every subsequent rerun.
from papertrail.retrieval.embedder import load_embedder as _load_embedder  # noqa: E402
from papertrail.retrieval.rerank   import load_cross_encoder as _load_ce    # noqa: E402
with st.spinner("Warming up models (first load only)…"):
    _load_embedder()
    _load_ce()

# ── Sidebar ────────────────────────────────────────────────────────────────────
active_doc = get_active_doc(st.session_state.documents, st.session_state.active_doc_idx)

with st.sidebar:
    st.markdown("## 📄 Papertrail")
    st.markdown("Load a document, then ask it anything.")
    st.markdown("---")

    # ── Document list ──────────────────────────────────────────────────────────
    docs = get_documents(st.session_state.documents)
    if docs:
        st.markdown("**Documents**")
        for i, doc in enumerate(docs):
            col_a, col_b = st.columns([5, 1])
            with col_a:
                label = doc["source_name"][:28] + "…" if len(doc["source_name"]) > 28 else doc["source_name"]
                is_active = i == st.session_state.active_doc_idx
                if st.button(
                    f"{'▶ ' if is_active else ''}{label}",
                    key=f"doc_select_{i}",
                    use_container_width=True,
                ):
                    st.session_state.active_doc_idx = i
                    st.rerun()
            with col_b:
                if st.button("✕", key=f"doc_remove_{i}"):
                    new_docs, new_idx = remove_document(
                        i, st.session_state.documents, st.session_state.active_doc_idx
                    )
                    st.session_state.documents      = new_docs
                    st.session_state.active_doc_idx = new_idx
                    st.rerun()
        # ── Conversation actions ───────────────────────────────────────────────
        if active_doc and active_doc.get("messages"):
            _col1, _col2 = st.columns(2)
            with _col1:
                if st.button("🗑 Clear chat", key="clear_chat", use_container_width=True):
                    active_doc["messages"] = []
                    st.rerun()
            with _col2:
                _md = export_conversation_markdown(active_doc)
                st.download_button(
                    label="⬇ Export",
                    data=_md,
                    file_name=f"papertrail_{active_doc['source_name'][:30]}.md",
                    mime="text/markdown",
                    key="export_md",
                    use_container_width=True,
                )
        if len(docs) > 1:
            st.checkbox("Search all documents", key="search_all")
        elif st.session_state.get("search_all"):
            st.session_state.search_all = False
        st.markdown("---")

    # ── Source selector ────────────────────────────────────────────────────────
    st.radio(
        "Source",
        ["URL", "Document Upload", "Paste Text"],
        key="source_type",
        label_visibility="collapsed",
    )
    source_type = st.session_state.source_type

    if source_type == "Document Upload":
        limit_mb = config.PDF_MAX_BYTES // (1024 * 1024)
        uploaded = st.file_uploader(
            f"Upload PDF or DOCX (max {limit_mb} MB)",
            type=["pdf", "docx"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            fb = uploaded.read()
            if fb:
                st.session_state.pdf_bytes = fb
                st.session_state.pdf_name  = uploaded.name

        if st.session_state.pdf_bytes and st.button("Build Knowledge Base", key="build_doc"):
            _ext = st.session_state.pdf_name.rsplit(".", 1)[-1].lower()
            with st.spinner(f"Reading {'DOCX' if _ext == 'docx' else 'PDF'}..."):
                if _ext == "docx":
                    text, section_map, page_map, err = load_docx(st.session_state.pdf_bytes)
                else:
                    text, section_map, page_map, err = load_pdf(st.session_state.pdf_bytes)
            if err and not text:
                st.error(err)
            else:
                if err:
                    st.info(err)  # non-fatal note (e.g. OCR used)
                with st.spinner("Indexing..."):
                    idx_err, new_docs, new_idx = build_knowledge_base(
                        text, st.session_state.pdf_name,
                        st.session_state.documents,
                        section_map=section_map, page_map=page_map,
                    )
                if idx_err:
                    st.error(idx_err)
                else:
                    st.session_state.documents      = new_docs
                    st.session_state.active_doc_idx = new_idx
                    st.session_state.pdf_bytes = None
                    st.rerun()

    # ── Active doc controls ────────────────────────────────────────────────────
    if active_doc:
        st.markdown("---")
        st.caption(f"Active: {active_doc['source_name'][:40]}")
        st.caption(f"{active_doc['chunk_count']} chunks indexed")

        st.session_state.answer_mode = st.selectbox(
            "Answer mode",
            ["Structured (no LLM)", "Local (Ollama)", "Hugging Face (best effort)"],
            index=["Structured (no LLM)", "Local (Ollama)", "Hugging Face (best effort)"].index(
                st.session_state.answer_mode
            ),
        )

    # ── System status ──────────────────────────────────────────────────────────
    with st.expander("System status", expanded=False):
        import importlib as _il
        import os as _os

        def _model_status(label: str, module: str, fn: str) -> None:
            try:
                obj = getattr(_il.import_module(module), fn)()
                st.success(f"{label}: {'ready' if obj is not None else 'loading...'}")
            except Exception as _e:
                st.error(f"{label}: {type(_e).__name__}")

        _model_status("Embedder", "papertrail.retrieval.embedder", "load_embedder")
        _model_status("Reranker", "papertrail.retrieval.rerank",   "load_cross_encoder")
        _cache = config.CACHE_DIR
        _n = len([f for f in _os.listdir(_cache) if f.endswith("_meta.json")]) if _os.path.isdir(_cache) else 0
        st.info(f"Cache: {_n} document(s) on disk")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="doc-header">Papertrail</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Ask anything from your document</div>', unsafe_allow_html=True)

# ── Source input ───────────────────────────────────────────────────────────────
if source_type == "URL":
    col1, col2 = st.columns([5, 1])
    with col1:
        url_val = st.text_input("URL", placeholder="https://example.com/article",
                                label_visibility="collapsed", key="url_input")
    with col2:
        fetch_clicked = st.button("Fetch", key="fetch_url", use_container_width=True)

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
                    err, new_docs, new_idx = build_knowledge_base(
                        text, url_val.strip(), st.session_state.documents,
                        section_map=section_map,
                    )
                if err:
                    st.error(err)
                else:
                    st.session_state.documents      = new_docs
                    st.session_state.active_doc_idx = new_idx
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
                err, new_docs, new_idx = build_knowledge_base(
                    pasted.strip(), "Pasted text", st.session_state.documents
                )
            if err:
                st.error(err)
            else:
                st.session_state.documents      = new_docs
                st.session_state.active_doc_idx = new_idx
                st.rerun()

# ── Source badge ───────────────────────────────────────────────────────────────
if active_doc:
    src = html.escape(active_doc["source_name"])
    st.markdown(
        f'<div class="source-row"><span class="source-badge" title="{src}">{src}</span>'
        f'<span class="chunk-count">{active_doc["chunk_count"]} chunks</span></div>',
        unsafe_allow_html=True,
    )

# ── Empty state ────────────────────────────────────────────────────────────────
retriever = get_active_retriever(st.session_state.documents, st.session_state.active_doc_idx)
if not retriever:
    msg = "Upload a PDF or DOCX from the sidebar to begin." if source_type == "Document Upload" \
          else "Load a document above to begin."
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)
elif not active_doc.get("messages"):
    st.markdown('<div class="empty-state">Knowledge base ready -- ask your first question.</div>',
                unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
if active_doc:
    for msg in active_doc.get("messages", []):
        if msg.get("role") == "user":
            st.markdown(f'<div class="msg-user">{html.escape(msg.get("content",""))}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="msg-bot">'
                f'<p>{_safe_md_to_html(msg.get("answer_text") or "")}</p></div>',
                unsafe_allow_html=True,
            )
            if msg.get("match_lbl"):
                st.markdown(
                    f'<span class="match-pill {html.escape(msg.get("css_class", ""))}">'
                    f'{html.escape(msg["match_lbl"])}</span>',
                    unsafe_allow_html=True,
                )
            if msg.get("attr_text"):
                attr_html = (
                    f'<div class="attr-bar">'
                    f'<span class="attr-pill">{html.escape(msg["attr_text"])}</span>'
                    f'</div>'
                )
                st.markdown(attr_html, unsafe_allow_html=True)
            if msg.get("extras"):
                with st.expander("Show supporting passages"):
                    for chunk, score in msg["extras"]:
                        st.caption(f"score: {score:.3f}")
                        st.write(clean_raw_passage(chunk))

# ── Live input ─────────────────────────────────────────────────────────────────
if retriever:
    question = st.chat_input("Ask a question about your document...")
    if question and question.strip():
        q = question.strip()
        st.markdown(f'<div class="msg-user">{html.escape(q)}</div>', unsafe_allow_html=True)

        typing_ph = st.empty()
        typing_ph.markdown(
            '<div class="msg-bot">'
            '<p>Typing<span style="opacity:.6">▌</span></p></div>',
            unsafe_allow_html=True,
        )

        try:
            if st.session_state.get("search_all") and len(st.session_state.documents) > 1:
                _res = answer_across_docs(
                    st.session_state.documents, q, mode=st.session_state.answer_mode
                )
                _text, focused, _label, css_class, match_lbl, score, extras, attr_text = _res
                typing_ph.empty()
                answer_text = stream_words_into_bubble(_text)
            else:
                metadata, token_stream = answer_question_stream(
                    retriever, q, mode=st.session_state.answer_mode
                )
                typing_ph.empty()
                answer_text = stream_words_into_bubble(token_stream)
                if not answer_text.strip():
                    answer_text = metadata["extractive_fallback"]
                    answer_text = stream_words_into_bubble(answer_text)
                css_class    = metadata["css_class"]
                match_lbl    = metadata["match_lbl"]
                attr_text    = metadata["attr_text"]
                extras       = metadata["extras"]
        except MemoryError:
            typing_ph.empty()
            st.error("Out of memory processing this question. Try a shorter document.")
            st.stop()
        except Exception as _exc:
            typing_ph.empty()
            logging.getLogger(__name__).exception("Answer pipeline failed: %r", q)
            st.error(
                "Something went wrong answering that question. "
                "Try rephrasing, or reload the document."
            )
            st.caption(f"Error detail: {type(_exc).__name__}: {_exc}")
            st.stop()

        if match_lbl:
            st.markdown(
                f'<span class="match-pill {html.escape(css_class)}">'
                f'{html.escape(match_lbl)}</span>',
                unsafe_allow_html=True,
            )

        if attr_text:
            attr_html = (
                f'<div class="attr-bar">'
                f'<span class="attr-pill">{html.escape(attr_text)}</span>'
                f'</div>'
            )
            st.markdown(attr_html, unsafe_allow_html=True)
        if extras:
            with st.expander("Show supporting passages"):
                for chunk, s in extras:
                    st.caption(f"score: {s:.3f}")
                    st.write(clean_raw_passage(chunk))

        # Store only plain text — no raw HTML — in session state.
        if active_doc is not None:
            active_doc["messages"].append({"role": "user", "content": q})
            active_doc["messages"].append({
                "role":         "assistant",
                "answer_text":  answer_text,
                "attr_text":    attr_text,
                "extras":       extras,
                "css_class":    css_class,
                "match_lbl":    match_lbl,
            })

        st.markdown(
            "<script>window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});</script>",
            unsafe_allow_html=True,
        )
