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

# Optional token from Streamlit secrets (set at runtime by app.py if available).
HF_TOKEN: str | None = None


def _evidence_overlap_ratio(answer: str, evidence: str) -> float:
    a = re.findall(r"[a-zA-Z]{4,}", (answer or "").lower())
    e = set(re.findall(r"[a-zA-Z]{4,}", (evidence or "").lower()))
    return sum(1 for w in a if w in e) / max(1, len(a)) if a else 0.0


def _build_prompt(question: str, context: str) -> str:
    # Mistral-Instruct uses [INST]...[/INST] chat format
    return (
        "[INST] Answer the QUESTION using ONLY the EVIDENCE below.\n"
        'If the evidence is insufficient, say: "I cannot determine this from the document."\n\n'
        "Rules:\n"
        "- Answer in 1 to 3 plain sentences. No bullet points, no headings.\n"
        "- Do not include citations, bracket tags, or source references; the UI shows the source.\n"
        "- Do not use outside knowledge.\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION:\n{question} [/INST]"
    )


def generate_answer_hf_stream(question: str, context: str) -> Iterator[str]:
    """Yield response tokens from the HF Inference API SSE stream."""
    if not _SUPPORT:
        return
    prompt = _build_prompt(question, context)
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = requests.post(
            config.HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens":   config.HF_MAX_NEW_TOKENS,
                    "temperature":      config.HF_TEMPERATURE,
                    "do_sample":        False,
                    "return_full_text": False,
                },
                "stream": True,
            },
            timeout=config.HF_TIMEOUT,
            stream=True,
        )
        if resp.status_code in (503, 422):
            logger.warning("HF API returned %d for streaming request.", resp.status_code)
            return
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            token_obj  = data.get("token", {})
            token_text = token_obj.get("text", "")
            if token_text and not token_obj.get("special", False):
                yield token_text
            if data.get("generated_text") is not None:
                break
    except requests.exceptions.Timeout:
        logger.warning("HF API stream timed out after %ss.", config.HF_TIMEOUT)
    except Exception as exc:
        logger.warning("HF stream error: %s", exc)


def generate_answer_hf(question: str, context: str) -> Tuple[str, bool]:
    if not _SUPPORT:
        return "", False

    prompt = _build_prompt(question, context)

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        resp = requests.post(
            config.HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": config.HF_MAX_NEW_TOKENS,
                    "temperature":    config.HF_TEMPERATURE,
                    "do_sample":      False,
                    "return_full_text": False,
                },
            },
            timeout=config.HF_TIMEOUT,
        )
        if resp.status_code == 503:
            logger.warning("HF model loading (503), skipping generation.")
            return "", False
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data and "generated_text" in data[0]:
            answer = (data[0]["generated_text"] or "").strip()

            # Strip any [INST] echo that leaked through
            answer = re.sub(r"\[INST\].*?\[/INST\]", "", answer, flags=re.DOTALL).strip()

            if len(answer) <= 5:
                return "", False

            words = answer.split()
            if len(words) >= 4:
                from collections import Counter
                freq = Counter(w.lower() for w in words)
                if freq.most_common(1)[0][1] / len(words) > 0.40:
                    return "", False

            answer = answer.rstrip(", ;(").strip()
            if _evidence_overlap_ratio(answer, context) < 0.18:
                return "", False

            return answer, True

        return "", False

    except requests.exceptions.Timeout:
        logger.warning("HF API request timed out after %ss.", config.HF_TIMEOUT)
        return "", False
    except requests.exceptions.RequestException as exc:
        logger.warning("HF API request failed: %s", exc)
        return "", False
    except Exception as exc:
        logger.warning("Unexpected error during HF generation: %s", exc)
        return "", False
