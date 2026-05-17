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


@pytest.mark.parametrize("text,expected_noise", [
    ("C(G) = max D V ∗ G(x) + Ez~pz[log(1 −D∗)]", True),
    ("The model generates realistic images by fooling the discriminator network.", False),
    ("Short", True),
    ("", True),
])
def test_noise_sentence_parametrized(text, expected_noise):
    from papertrail.utils.text import is_noise_sentence
    assert is_noise_sentence(text) is expected_noise
