import json
import logging
import re
from typing import Iterator, Tuple

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


def _build_prompt(question: str, evidence: str) -> str:
    return (
        "Answer the question using ONLY the evidence. Reply in 1 to 3 plain sentences: "
        "no bullet points, no headings, no citations or bracket tags.\n"
        'If insufficient, say: "I cannot determine this from the document."\n\n'
        f"EVIDENCE:\n{evidence}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def generate_answer_ollama_stream(question: str, evidence: str) -> Iterator[str]:
    """Yield response tokens from the Ollama streaming API."""
    if not _SUPPORT:
        return
    prompt = _build_prompt(question, evidence)
    try:
        resp = requests.post(
            config.OLLAMA_URL,
            json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": True},
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
    except Exception as exc:
        logger.warning("Ollama stream error: %s", exc)


def generate_answer_ollama(
    question: str,
    evidence: str,
    model: str = config.OLLAMA_MODEL,
) -> Tuple[str, bool]:
    if not _SUPPORT:
        return "", False

    prompt = _build_prompt(question, evidence)

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
