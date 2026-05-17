import html
import streamlit as st

_STREAM_BATCH = 6  # render every N words — smooth UX without per-word DOM thrash


def stream_words_into_bubble(text: str, source_label: str = "") -> None:
    """Render text into a chat bubble with batched word streaming.

    Batches DOM updates every _STREAM_BATCH words — no sleep() calls —
    so the Streamlit server thread is never blocked for the full answer duration.
    """
    words = (text or "").split()
    ph = st.empty()
    label_html = (
        f'<div class="msg-label">{html.escape(source_label)}</div>'
        if source_label else ""
    )

    if not words:
        ph.markdown(
            f'<div class="msg-bot">{label_html}<p></p></div>',
            unsafe_allow_html=True,
        )
        return

    buf = ""
    for i, w in enumerate(words):
        buf += w + " "
        if (i + 1) % _STREAM_BATCH == 0:
            ph.markdown(
                f'<div class="msg-bot">{label_html}'
                f'<p>{html.escape(buf)}'
                f'<span style="opacity:.5">▌</span></p></div>',
                unsafe_allow_html=True,
            )

    ph.markdown(
        f'<div class="msg-bot">{label_html}<p>{html.escape(buf.strip())}</p></div>',
        unsafe_allow_html=True,
    )
