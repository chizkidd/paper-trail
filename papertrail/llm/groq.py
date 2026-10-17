from typing import Optional

from papertrail import config
from papertrail.llm.base import LLMStream
from papertrail.llm.openai_compat import stream_chat

# Optional key from Streamlit secrets (set at runtime by app.py if available).
GROQ_API_KEY: Optional[str] = None


def generate_answer_groq_stream(question: str, context: str) -> LLMStream:
    """Groq free tier (needs GROQ_API_KEY)."""
    return stream_chat(
        question, context,
        url=config.GROQ_API_URL, model=config.GROQ_MODEL, token=GROQ_API_KEY, provider="Groq",
    )
