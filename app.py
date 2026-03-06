import html
import streamlit as st

from papertrail.ingest.pdf import load_pdf
from papertrail.ingest.web import load_url
from papertrail.qa import answer_question, build_knowledge_base
from papertrail.utils.text import clean_raw_passage
from papertrail.utils.stream import stream_words_into_bubble

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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--paper); color: var(--ink); }
.stApp { background: var(--paper); }

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
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div { color: rgba(245,240,232,0.85) !important; }
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
.msg-bot   { background: white; border: 1px solid var(--border); border-radius: 12px 12px 12px 4px; padding: 0.85rem 1.1rem; margin: 0.75rem clamp(20px,10%,80px) 0.75rem 0; font-size: 0.95rem; line-height: 1.65; box-shadow: 0 1px 4px rgba(0,0,0,0.05); word-break: break-word; }
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

# ── Session state ──────────────────────────────────────────────────────────────
for k, d in [
    ("retriever",   None),
    ("source_name", None),
    ("chunk_count", 0),
    ("messages",    []),
    ("source_type", "URL"),
    ("answer_mode", "Hugging Face (best effort)"),
    ("pdf_bytes",   None),
    ("pdf_name",    ""),
]:
    st.session_state.setdefault(k, d)

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
            fb = uploaded.read()
            if fb:
                st.session_state.pdf_bytes = fb
                st.session_state.pdf_name  = uploaded.name

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
        st.session_state.answer_mode = st.selectbox(
            "Answer mode",
            ["Structured (no LLM)", "Local (Ollama)", "Hugging Face (best effort)"],
            index=["Structured (no LLM)", "Local (Ollama)", "Hugging Face (best effort)"].index(
                st.session_state.answer_mode
            ),
        )
        if st.button("Clear & start over"):
            for k in ("retriever","source_name","chunk_count","messages","pdf_bytes","pdf_name"):
                st.session_state[k] = None if k not in ("messages",) else []
            st.session_state.pdf_bytes = None
            st.session_state.pdf_name  = ""
            st.rerun()

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
                if err:  st.error(err)
                else:    st.rerun()

elif source_type == "Paste Text":
    pasted = st.text_area("Paste text", height=200,
                          placeholder="Paste any text here -- articles, docs, notes...",
                          label_visibility="collapsed", key="paste_input")
    if st.button("Build Knowledge Base", key="build_paste"):
        if not pasted.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Indexing..."):
                err = build_knowledge_base(pasted.strip(), "Pasted text")
            if err:  st.error(err)
            else:    st.rerun()

# ── Source badge ───────────────────────────────────────────────────────────────
if st.session_state.retriever and st.session_state.source_name:
    src = html.escape(str(st.session_state.source_name))
    st.markdown(
        f'<div class="source-row">'
        f'<span class="source-badge" title="{src}">{src}</span>'
        f'<span class="chunk-count">{st.session_state.chunk_count} chunks</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.retriever:
    msg = "Upload a PDF from the sidebar to begin." if source_type == "PDF Upload" \
          else "Load a document above to begin."
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)
elif not st.session_state.messages:
    st.markdown('<div class="empty-state">Knowledge base ready -- ask your first question.</div>',
                unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg.get("role") == "user":
        st.markdown(f'<div class="msg-user">{html.escape(msg.get("content",""))}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot">{html.escape(msg.get("answer_text") or "")}</div>',
                    unsafe_allow_html=True)
        if msg.get("attribution_html"):
            st.markdown(msg["attribution_html"], unsafe_allow_html=True)
        if msg.get("extras"):
            with st.expander("Show supporting passages"):
                for chunk, score in msg["extras"]:
                    st.caption(f"score: {score:.3f}")
                    st.write(clean_raw_passage(chunk))

# ── Live input ─────────────────────────────────────────────────────────────────
if st.session_state.retriever:
    question = st.chat_input("Ask a question about your document...")
    if question and question.strip():
        q = question.strip()
        st.markdown(f'<div class="msg-user">{html.escape(q)}</div>', unsafe_allow_html=True)

        typing_ph = st.empty()
        typing_ph.markdown(
            '<div class="msg-bot"><div class="msg-label">assistant</div>'
            '<p>Typing<span style="opacity:.6">▌</span></p></div>',
            unsafe_allow_html=True,
        )

        answer_html, answer_text, score, extras, attribution_html = \
            answer_question(st.session_state.retriever, q)
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
            "role": "assistant", "content": answer_html,
            "extras": extras, "attribution_html": attribution_html,
            "answer_text": answer_text,
        })
        st.markdown(
            "<script>window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});</script>",
            unsafe_allow_html=True,
        )
