import json
import logging

from papertrail import config
from papertrail.llm import base
from papertrail.llm.base import LLMStream

logger = logging.getLogger(__name__)

try:
    import requests
    _SUPPORT = True
except ImportError:
    _SUPPORT = False


def _build_prompt(question: str, evidence: str) -> str:
    return (
        "Answer the question using ONLY the evidence. Reply in 1 to 3 plain sentences: "
        "no bullet points, no headings, no citations or bracket tags.\n"
        'If insufficient, say: "I cannot determine this from the document."\n\n'
        f"EVIDENCE:\n{evidence}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def generate_answer_ollama_stream(question: str, evidence: str) -> LLMStream:
    """Stream tokens from a local Ollama server."""

    def source(st: LLMStream):
        if not _SUPPORT:
            st.status = base.ERROR
            return
        try:
            resp = requests.post(
                config.OLLAMA_URL,
                json={
                    "model":  config.OLLAMA_MODEL,
                    "prompt": _build_prompt(question, evidence),
                    "stream": True,
                },
                timeout=config.OLLAMA_TIMEOUT,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
        except requests.exceptions.ConnectionError:
            logger.debug("Ollama not reachable at %s.", config.OLLAMA_URL)
            st.status = base.NOT_REACHABLE
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out after %ss.", config.OLLAMA_TIMEOUT)
            st.status = base.TIMEOUT
        except Exception as exc:
            logger.warning("Ollama stream error: %s", exc)
            st.status = base.ERROR

    return LLMStream(source)
