import logging
from typing import Optional

import requests

from papertrail import config
from papertrail.llm import base
from papertrail.llm.base import LLMStream
from papertrail.llm.openai_compat import stream_chat

logger = logging.getLogger(__name__)

# Optional key from Streamlit secrets (set at runtime by app.py if available).
GROQ_API_KEY: Optional[str] = None

# Tried in order when the configured model is rejected; first one Groq actually serves wins.
_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "gemma2-9b-it",
]
_NOT_CHAT = ("whisper", "guard", "tts", "playai", "embed", "orpheus")

_discovered: Optional[str] = None   # remembered for the life of the process


def discover_model(exclude: str = "") -> Optional[str]:
    """Ask Groq which models are live and pick a suitable chat model."""
    global _discovered
    if _discovered and _discovered != exclude:
        return _discovered
    try:
        resp = requests.get(
            config.GROQ_API_URL.rsplit("/chat/completions", 1)[0] + "/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        ids = [m.get("id", "") for m in resp.json().get("data", [])]
    except Exception as exc:
        logger.warning("Groq model discovery failed: %s", exc)
        return None
    ids = [i for i in ids if i and i != exclude and not any(x in i.lower() for x in _NOT_CHAT)]
    pick = next((m for m in _FALLBACK_MODELS if m in ids), None) or (ids[0] if ids else None)
    if pick:
        logger.warning("Groq model %r unavailable; using %r.", exclude or config.GROQ_MODEL, pick)
        _discovered = pick
    return pick


def generate_answer_groq_stream(question: str, context: str) -> LLMStream:
    """Groq free tier (needs GROQ_API_KEY). Self-heals if the model name has been retired."""

    def source(st: LLMStream):
        model = _discovered or config.GROQ_MODEL
        kw = dict(url=config.GROQ_API_URL, token=GROQ_API_KEY, provider="Groq")
        inner = stream_chat(question, context, model=model, **kw)
        produced = False
        for tok in inner:
            produced = True
            yield tok
        st.status = inner.status
        if produced or inner.status != base.UNAVAILABLE:
            return
        alt = discover_model(exclude=model)
        if not alt:
            return
        retry = stream_chat(question, context, model=alt, **kw)
        for tok in retry:
            yield tok
        st.status = retry.status

    return LLMStream(source)
