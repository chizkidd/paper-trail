import re
from typing import Tuple

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN   = None  # set via st.secrets (optional)

try:
    import requests
    _SUPPORT = True
except Exception:
    _SUPPORT = False


def _overlap(answer: str, evidence: str) -> float:
    a = re.findall(r"[a-zA-Z]{4,}", (answer or "").lower())
    e = set(re.findall(r"[a-zA-Z]{4,}", (evidence or "").lower()))
    return sum(1 for w in a if w in e) / max(1, len(a)) if a else 0.0


def generate_answer_hf(question: str, context: str) -> Tuple[str, bool]:
    if not _SUPPORT: return "", False
    prompt = (
        "You are a careful assistant. Answer the QUESTION using ONLY the EVIDENCE.\n"
        'If insufficient, say: "I cannot determine this from the document."\n\n'
        "Rules:\n- Write a direct answer first (1 to 3 sentences).\n"
        '- Then add "Details" as bullet points if helpful.\n'
        "- Every factual claim must include a citation in parentheses, e.g.: (§ Methods · p. 3).\n"
        "- Do not invent citations. Do not use outside knowledge.\n\n"
        f"EVIDENCE:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN: headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = requests.post(
            HF_API_URL, headers=headers,
            json={"inputs": prompt, "parameters": {
                "max_new_tokens": 300, "min_new_tokens": 20,
                "temperature": 0.2, "do_sample": False, "return_full_text": False,
            }},
            timeout=30,
        )
        if resp.status_code == 503: return "", False
        resp.raise_for_status()
        data = resp.json()
        if not (isinstance(data, list) and data and "generated_text" in data[0]):
            return "", False
        answer = (data[0]["generated_text"] or "").strip()
        answer = answer.split("ANSWER:")[-1].strip()
        answer = answer.split("Answer:")[-1].strip()
        if len(answer) <= 5: return "", False
        words = answer.split()
        if len(words) >= 4:
            from collections import Counter
            if Counter(w.lower() for w in words).most_common(1)[0][1] / len(words) > 0.40:
                return "", False
        answer = answer.rstrip(", ;(").strip()
        if _overlap(answer, context) < 0.18: return "", False
        return answer, True
    except Exception:
        return "", False
