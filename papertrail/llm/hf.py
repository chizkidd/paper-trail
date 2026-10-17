import json
import logging
from typing import Optional

from papertrail import config
from papertrail.llm import base
from papertrail.llm.base import LLMStream

logger = logging.getLogger(__name__)

try:
    import requests
    _SUPPORT = True
except ImportError:
    _SUPPORT = False

# Optional token from Streamlit secrets (set at runtime by app.py if available).
HF_TOKEN: Optional[str] = None

_SYSTEM = (
    "Answer the user's QUESTION using ONLY the EVIDENCE provided.\n"
    'If the evidence is insufficient, say: "I cannot determine this from the document."\n'
    "Rules:\n"
    "- Answer in 1 to 3 plain sentences. No bullet points, no headings.\n"
    "- Do not include citations, bracket tags, or source references; the UI shows the source.\n"
    "- Do not use outside knowledge."
)


def _messages(question: str, context: str) -> list:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"EVIDENCE:\n{context}\n\nQUESTION:\n{question}"},
    ]


def _status_for_http(code: int) -> str:
    if code in (401, 403):
        return base.AUTH
    if code in (402, 429):
        return base.RATE_LIMITED
    if code in (400, 404, 410, 422, 503):
        return base.UNAVAILABLE
    return base.ERROR


def generate_answer_hf_stream(question: str, context: str) -> LLMStream:
    """Stream tokens from the Hugging Face Inference Providers router (OpenAI-compatible)."""

    def source(st: LLMStream):
        if not _SUPPORT:
            st.status = base.ERROR
            return
        if not HF_TOKEN:
            st.status = base.NO_TOKEN
            return
        try:
            resp = requests.post(
                config.HF_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {HF_TOKEN}",
                },
                json={
                    "model":       config.HF_MODEL,
                    "messages":    _messages(question, context),
                    "max_tokens":  config.HF_MAX_NEW_TOKENS,
                    "temperature": config.HF_TEMPERATURE,
                    "stream":      True,
                },
                timeout=config.HF_TIMEOUT,
                stream=True,
            )
            if resp.status_code >= 400:
                st.status = _status_for_http(resp.status_code)
                logger.warning("HF router returned %d for model %s.", resp.status_code, config.HF_MODEL)
                return
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if delta:
                    yield delta
        except requests.exceptions.Timeout:
            logger.warning("HF request timed out after %ss.", config.HF_TIMEOUT)
            st.status = base.TIMEOUT
        except Exception as exc:
            logger.warning("HF stream error: %s", exc)
            st.status = base.ERROR

    return LLMStream(source)
