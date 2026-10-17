import numpy as np

from papertrail.llm.guard import clean_generated, evidence_sentences, is_supported

EVIDENCE = (
    "[§ Safety] Aligned models are vulnerable because alignment is shallow and only affects the first few tokens.\n"
    "[§ Safety] Data augmentation with partial harmful prefixes deepens the refusal behaviour."
)


class FakeEmbedder:
    """Bag-of-words embedder: deterministic, no model download."""
    VOCAB: dict = {}

    def encode(self, sents, normalize_embeddings=True):
        import re
        rows = []
        for s in sents:
            toks = re.findall(r"[a-z]+", s.lower())
            for t in toks:
                self.VOCAB.setdefault(t, len(self.VOCAB))
            rows.append(toks)
        out = np.zeros((len(sents), 512))
        for i, toks in enumerate(rows):
            for t in toks:
                out[i, self.VOCAB[t] % 512] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


def test_evidence_sentences_strip_tags():
    assert evidence_sentences(EVIDENCE)[0].startswith("Aligned models")


def test_supported_answer_accepted():
    ans = "Aligned models are vulnerable because alignment is shallow and affects only the first few tokens."
    assert is_supported(ans, EVIDENCE, FakeEmbedder())


def test_invented_sentence_rejected():
    ans = ("Aligned models are vulnerable because alignment is shallow. "
           "Quantum entanglement guarantees perfect cryptographic security everywhere.")
    assert not is_supported(ans, EVIDENCE, FakeEmbedder())


def test_refusal_and_empty_rejected():
    assert not is_supported("I cannot determine this from the document.", EVIDENCE, FakeEmbedder())
    assert not is_supported("", EVIDENCE, FakeEmbedder())


def test_clean_generated_strips_tags():
    assert clean_generated("It is shallow (§ Safety · p. 3) [Chunk: 2] indeed.") == "It is shallow indeed."
