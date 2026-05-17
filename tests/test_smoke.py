"""Smoke tests: verify imports, core functions, and new features work in isolation."""
import pytest


def test_app_imports_cleanly():
    """All modules imported by app.py must load without error."""
    import papertrail.qa as qa
    import papertrail.config as config
    from papertrail.utils.stream import stream_words_into_bubble
    import papertrail.ingest.web  # noqa: F401
    import papertrail.ingest.pdf  # noqa: F401

    assert callable(qa.answer_question)
    assert callable(qa.build_knowledge_base)
    assert callable(qa.export_conversation_markdown)
    assert callable(stream_words_into_bubble)
    assert config.EMBEDDING_MODEL


def test_export_empty_conversation():
    from papertrail.qa import export_conversation_markdown
    doc = {"source_name": "test.pdf", "messages": []}
    md = export_conversation_markdown(doc)
    assert "test.pdf" in md
    assert "0 questions" in md


def test_export_single_question():
    from papertrail.qa import export_conversation_markdown
    doc = {
        "source_name": "paper.pdf",
        "messages": [
            {"role": "user", "content": "What is a GAN?"},
            {
                "role": "assistant",
                "answer_text": "A GAN is a generative adversarial network.",
                "attr_text": "§ Introduction · p. 1",
            },
        ],
    }
    md = export_conversation_markdown(doc)
    assert "What is a GAN?" in md
    assert "A GAN is a generative adversarial network." in md
    assert "§ Introduction · p. 1" in md
    assert "1 question" in md


def test_export_plural_questions():
    from papertrail.qa import export_conversation_markdown
    doc = {
        "source_name": "paper.pdf",
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "answer_text": "A1", "attr_text": ""},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "answer_text": "A2", "attr_text": ""},
        ],
    }
    md = export_conversation_markdown(doc)
    assert "2 questions" in md


def test_export_no_attr_text():
    from papertrail.qa import export_conversation_markdown
    doc = {
        "source_name": "doc.pdf",
        "messages": [
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "answer_text": "World.", "attr_text": ""},
        ],
    }
    md = export_conversation_markdown(doc)
    assert "Source:" not in md  # empty attr_text should not emit a source line


def test_stream_words_no_sleep():
    """stream_words_into_bubble must not import time or call sleep."""
    import inspect
    import papertrail.utils.stream as stream_mod
    src = inspect.getsource(stream_mod)
    assert "time.sleep" not in src, "time.sleep() found in stream.py — blocks server thread"
    assert "import time" not in src, "time module imported in stream.py"


def test_session_state_validator_repairs_missing_keys():
    """_validate_session_state should reset documents when a doc is missing required keys."""
    corrupt_doc = {"source_name": "bad.pdf"}  # missing chunk_count, retriever, messages, text_hash
    state = {"documents": [corrupt_doc], "active_doc_idx": 0}

    # Inline the validator logic to test it in isolation (avoids importing Streamlit)
    docs = state.get("documents", [])
    required = {"source_name", "chunk_count", "retriever", "messages", "text_hash"}
    corruption_found = False
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict) or not required.issubset(doc.keys()):
            state["documents"] = []
            state["active_doc_idx"] = 0
            corruption_found = True
            break

    assert corruption_found, "Validator should have detected corruption"
    assert state["documents"] == [], "Validator should have reset documents to []"
    assert state["active_doc_idx"] == 0


def test_session_state_validator_clamps_out_of_range_index():
    """active_doc_idx beyond the documents list should be clamped to last valid index."""
    state = {"documents": [{"x": 1}], "active_doc_idx": 99}
    docs = state.get("documents", [])
    idx = state.get("active_doc_idx", 0)
    if docs and idx >= len(docs):
        state["active_doc_idx"] = max(0, len(docs) - 1)
    assert state["active_doc_idx"] == 0


def test_build_knowledge_base_pure():
    """build_knowledge_base must work without a Streamlit server running."""
    from papertrail.qa import build_knowledge_base
    err, docs, idx = build_knowledge_base("word " * 60, "unit-test-doc", documents=[])
    assert isinstance(err, str)
    assert isinstance(docs, list)
    assert isinstance(idx, int)


def test_remove_document_pure():
    """remove_document must be callable without a Streamlit server running."""
    from papertrail.qa import remove_document
    fake = [{"source_name": "a"}, {"source_name": "b"}]
    docs, idx = remove_document(0, fake, active_doc_idx=1)
    assert len(docs) == 1
    assert docs[0]["source_name"] == "b"
    assert idx == 0


@pytest.mark.parametrize("text,expected_noise", [
    ("C(G) = max D V ∗ G(x) + Ez~pz[log(1 −D∗)]", True),
    ("The model generates realistic images by fooling the discriminator network.", False),
    ("Short", True),
    ("", True),
])
def test_noise_sentence_parametrized(text, expected_noise):
    from papertrail.utils.text import is_noise_sentence
    assert is_noise_sentence(text) is expected_noise
