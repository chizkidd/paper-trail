from typing import Optional

from papertrail import config
from papertrail.llm.base import LLMStream
from papertrail.llm.openai_compat import stream_chat

# Optional token from Streamlit secrets (set at runtime by app.py if available).
HF_TOKEN: Optional[str] = None


def generate_answer_hf_stream(question: str, context: str) -> LLMStream:
    """Hugging Face Inference Providers router (needs HF_TOKEN)."""
    return stream_chat(
        question, context,
        url=config.HF_API_URL, model=config.HF_MODEL, token=HF_TOKEN, provider="HF router",
    )
