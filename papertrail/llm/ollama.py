import logging
import re
from typing import Tuple

from papertrail import config

logger = logging.getLogger(__name__)

try:
    import requests
    _SUPPORT = True
except ImportError:
    _SUPPORT = False


def _evidence_overlap_ratio(answer: str, evidence: str) -> float:
    a = re.findall(r"[a-zA-Z]{4,}", (answer or "").lower())
    e = set(re.findall(r"[a-zA-Z]{4,}", (evidence or "").lower()))
    return sum(1 for w in a if w in e) / max(1, len(a)) if a else 0.0


def generate_answer_ollama(
    question: str,
    evidence: str,
    model: str = config.OLLAMA_MODEL,
) -> Tuple[str, bool]:
    if not _SUPPORT:
        return "", False

    prompt = (
        "You are a careful assistant. Use ONLY the evidence. Cite every claim using the bracket tags.\n"
        'If insufficient, say: "I cannot determine this from the document."\n\n'
        f"EVIDENCE:\n{evidence}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    try:
        resp = requests.post(
            config.OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if len(text) < 10:
            return "", False
        if _evidence_overlap_ratio(text, evidence) < 0.12:
            return "", False
        return text, True

    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out after %ss.", config.OLLAMA_TIMEOUT)
        return "", False
    except requests.exceptions.ConnectionError:
        logger.debug("Ollama not reachable at %s.", config.OLLAMA_URL)
        return "", False
    except requests.exceptions.RequestException as exc:
        logger.warning("Ollama request failed: %s", exc)
        return "", False
    except Exception as exc:
        logger.warning("Unexpected error during Ollama generation: %s", exc)
        return "", False
