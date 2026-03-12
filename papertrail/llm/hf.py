import re
from typing import Tuple

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN   = None  # set via st.secrets (optional)

try:
    import requests
    _SUPPORT = True
except Exception:
    _SUPPORT = False


def _evidence_overlap_ratio(answer: str, evidence: str) -> float:
    a = re.findall(r"[a-zA-Z]{4,}", (answer or "").lower())
    e = set(re.findall(r"[a-zA-Z]{4,}", (evidence or "").lower()))
    return sum(1 for w in a if w in e) / max(1, len(a)) if a else 0.0


def generate_answer_hf(question: str, context: str) -> Tuple[str, bool]:
    if not _SUPPORT:
        return "", False

    # Mistral-Instruct uses [INST]...[/INST] chat format
    prompt = (
        "[INST] You are a careful assistant. Answer the QUESTION using ONLY the EVIDENCE below.\n"
        'If the evidence is insufficient, say: "I cannot determine this from the document."\n\n'
        "Rules:\n"
        "- Write a direct answer first (1 to 3 sentences).\n"
        '- Then add "Details" as bullet points if helpful.\n'
        "- Every factual claim must include a citation in parentheses using the evidence tag,\n"
        "  for example: (§ Methods · p. 3) or (Chunk: 12).\n"
        "- Do not invent citations. Do not use outside knowledge.\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION:\n{question} [/INST]"
    )

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.2,
                    "do_sample": False,
                    "return_full_text": False,
                },
            },
            timeout=45,
        )
        if resp.status_code == 503:
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
    except Exception:
        return "", False