import html
import time
import streamlit as st


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
