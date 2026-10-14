import html
from typing import Iterator, Union

import streamlit as st

_STREAM_BATCH = 6  # render every N words/tokens — smooth UX without per-word DOM thrash


def stream_words_into_bubble(
    text_or_iter: Union[str, Iterator[str]],
    source_label: str = "",
) -> str:
    """
    Render text into a chat bubble. Accepts a str (batch/replay mode) or an
    Iterator[str] of tokens (live streaming mode). Returns the full accumulated text.

    Batches DOM updates every _STREAM_BATCH words/tokens — no sleep() calls —
    so the Streamlit server thread is never blocked.
    """
    ph = st.empty()
    label_html = (
        f'<div class="msg-label">{html.escape(source_label)}</div>'
        if source_label else ""
    )

    if isinstance(text_or_iter, str):
        tokens: Iterator[str] = iter((text_or_iter or "").split())
        join_space = True   # words need spaces re-inserted
    else:
        tokens = text_or_iter
        join_space = False  # LLM tokens carry their own spacing

    buf = ""
    count = 0

    for chunk in tokens:
        buf = buf + chunk + (" " if join_space else "")
        count += 1
        if count % _STREAM_BATCH == 0:
            ph.markdown(
                f'<div class="msg-bot">{label_html}'
                f'<p>{html.escape(buf)}'
                f'<span style="opacity:.5">▌</span></p></div>',
                unsafe_allow_html=True,
            )

    result = buf.strip()
    ph.markdown(
        f'<div class="msg-bot">{label_html}<p>{html.escape(result)}</p></div>',
        unsafe_allow_html=True,
    )
    return result
