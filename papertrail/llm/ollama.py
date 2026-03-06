import re
from typing import Tuple

try:
    import requests
    _SUPPORT = True
except Exception:
    _SUPPORT = False


def _overlap(answer: str, evidence: str) -> float:
    a = re.findall(r"[a-zA-Z]{4,}", (answer or "").lower())
    e = set(re.findall(r"[a-zA-Z]{4,}", (evidence or "").lower()))
    return sum(1 for w in a if w in e) / max(1, len(a)) if a else 0.0


def generate_answer_ollama(question: str, evidence: str, model: str = "llama3.1:8b") -> Tuple[str, bool]:
    if not _SUPPORT: return "", False
    prompt = (
        "You are a careful assistant. Use ONLY the evidence. Cite every claim.\n"
        'If insufficient, say: "I cannot determine this from the document."\n\n'
        f"EVIDENCE:\n{evidence}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
        if len(text) < 10:              return "", False
        if _overlap(text, evidence) < 0.12: return "", False
        return text, True
    except Exception:
        return "", False
