"""Streaming client for OpenAI-compatible chat-completions APIs (Groq, HF router, ...)."""
import json
import logging
from typing import Optional

import requests

from papertrail import config
from papertrail.llm import base
from papertrail.llm.base import LLMStream

logger = logging.getLogger(__name__)

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


def stream_chat(
    question: str,
    context: str,
    *,
    url: str,
    model: str,
    token: Optional[str],
    provider: str,
) -> LLMStream:
    """Stream tokens from an OpenAI-compatible /chat/completions endpoint."""

    def source(st: LLMStream):
        if not token:
            st.status = base.NO_TOKEN
            return
        try:
            resp = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json={
                    "model":       model,
                    "messages":    _messages(question, context),
                    "max_tokens":  config.LLM_MAX_NEW_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                    "stream":      True,
                },
                timeout=config.LLM_TIMEOUT,
                stream=True,
            )
            if resp.status_code >= 400:
                st.status = _status_for_http(resp.status_code)
                logger.warning("%s returned %d for model %s: %s",
                               provider, resp.status_code, model, (resp.text or "")[:300])
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
            logger.warning("%s request timed out after %ss.", provider, config.LLM_TIMEOUT)
            st.status = base.TIMEOUT
        except Exception as exc:
            logger.warning("%s stream error: %s", provider, exc)
            st.status = base.ERROR

    return LLMStream(source)
