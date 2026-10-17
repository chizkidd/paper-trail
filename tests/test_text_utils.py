"""Unit tests for papertrail.utils.text (pure functions, no ML deps)."""

from papertrail.utils.text import (
    as_text,
    dedup_paragraphs,
    is_noise_sentence,
    match_label,
    normalize_url,
    normalize_ws,
    remove_display_latex_anywhere,
    split_sentences,
    strip_leading_display_latex,
)


class TestAsText:
    def test_none_returns_empty(self):
        assert as_text(None) == ""

    def test_string_passthrough(self):
        assert as_text("hello") == "hello"

    def test_list_joined(self):
        assert as_text(["a", "b", "c"]) == "a b c"

    def test_tuple_joined(self):
        assert as_text(("x", "y")) == "x y"

    def test_int_converted(self):
        assert as_text(42) == "42"

    def test_list_with_none_skips_none(self):
        assert as_text(["a", None, "b"]) == "a b"


class TestNormalizeWs:
    def test_collapses_spaces(self):
        assert normalize_ws("a  b   c") == "a b c"

    def test_strips_edges(self):
        assert normalize_ws("  hello  ") == "hello"

    def test_newlines_collapsed(self):
        assert normalize_ws("a\nb\tc") == "a b c"

    def test_empty_string(self):
        assert normalize_ws("") == ""


class TestNormalizeUrl:
    def test_adds_https(self):
        assert normalize_url("example.com") == "https://example.com"

    def test_preserves_http(self):
        assert normalize_url("http://example.com") == "http://example.com"

    def test_preserves_https(self):
        assert normalize_url("https://example.com") == "https://example.com"

    def test_strips_whitespace(self):
        assert normalize_url("  https://example.com  ") == "https://example.com"

    def test_empty_returns_empty(self):
        assert normalize_url("") == ""

    def test_none_like_empty(self):
        assert normalize_url("   ") == ""


class TestSplitSentences:
    def test_basic_split(self):
        sents = split_sentences("Hello world. How are you? I am fine!")
        assert sents == ["Hello world.", "How are you?", "I am fine!"]

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_single_sentence_no_punct(self):
        result = split_sentences("No punctuation here")
        assert result == ["No punctuation here"]


class TestMatchLabel:
    def test_strong_match(self):
        label, css = match_label(0.30)
        assert label == "strong match"
        assert css == "match-high"

    def test_partial_match(self):
        label, css = match_label(0.15)
        assert label == "partial match"
        assert css == "match-medium"

    def test_weak_match(self):
        label, css = match_label(0.05)
        assert label == "weak match"
        assert css == "match-low"

    def test_boundary_strong(self):
        label, _ = match_label(0.25)
        assert label == "strong match"

    def test_boundary_partial(self):
        label, _ = match_label(0.08)
        assert label == "partial match"


class TestIsNoiseSentence:
    def test_empty_is_noise(self):
        assert is_noise_sentence("") is True

    def test_short_is_noise(self):
        assert is_noise_sentence("Too short") is True

    def test_normal_sentence_not_noise(self):
        assert is_noise_sentence("This is a perfectly normal sentence with enough length.") is False

    def test_latex_block_is_noise(self):
        assert is_noise_sentence(r"\[ x = \frac{1}{2} \]") is True

    def test_unicode_math_star_is_noise(self):
        # ∗ is U+2217 (Mathematical Operator) — appears in PDF math extractions
        assert is_noise_sentence("C(G) = max D V (G, D) =Ex~pdata[log D∗ G(x)] + Ez~pz[log(1 −D∗ G(G(z)))]") is True

    def test_unicode_math_minus_is_noise(self):
        # − is U+2212, ∼ is U+223C
        assert is_noise_sentence("At that point C(G) achieves the value −log 4 and pg ∼ pdata holds.") is True

    def test_clean_prose_with_ascii_equals_not_noise(self):
        # Plain ASCII math notation (no Unicode symbols) in an otherwise readable sentence
        assert is_noise_sentence("The global minimum is achieved if and only if pg = pdata at convergence.") is False


class TestStripLeadingDisplayLatex:
    def test_strips_leading_block(self):
        result = strip_leading_display_latex(r"\[ E = mc^2 \] Some text after.")
        assert result == "Some text after."

    def test_no_latex_unchanged(self):
        assert strip_leading_display_latex("plain text") == "plain text"

    def test_empty_string(self):
        assert strip_leading_display_latex("") == ""


class TestRemoveDisplayLatexAnywhere:
    def test_removes_inline_block(self):
        result = remove_display_latex_anywhere(r"Before \[ x^2 \] after.")
        assert result == "Before after."

    def test_no_latex_unchanged(self):
        assert remove_display_latex_anywhere("plain text") == "plain text"

    def test_multiple_blocks_removed(self):
        result = remove_display_latex_anywhere(r"A \[ x \] B \[ y \] C")
        assert result == "A B C"


class TestDedupParagraphs:
    def test_removes_duplicate_long_lines(self):
        text = "This is a long paragraph that will definitely be considered a duplicate.\nThis is a long paragraph that will definitely be considered a duplicate."
        result = dedup_paragraphs(text)
        assert result.count("This is a long paragraph") == 1

    def test_keeps_short_duplicates(self):
        # Short lines (< 40 chars) are not deduped
        text = "Short line\nShort line"
        result = dedup_paragraphs(text)
        assert result.count("Short line") == 2

    def test_empty_input(self):
        assert dedup_paragraphs("") == ""

    def test_different_lines_kept(self):
        text = "This is the first long paragraph with enough characters.\nThis is the second long paragraph with enough characters."
        result = dedup_paragraphs(text)
        assert "first long paragraph" in result
        assert "second long paragraph" in result


class TestExtractiveAnswerFormat:
    def test_caps_sentences_and_drops_evidence_suffix(self):
        from papertrail.retrieval.evidence import format_extractive_answer
        out = format_extractive_answer(["One.", "Two.", "Three."], "§ Methods")
        assert out == "One. Two."
        assert "Evidence" not in out

    def test_strips_footnote_markers(self):
        from papertrail.retrieval.evidence import strip_footnote_markers
        assert strip_footnote_markers("at greater depths.65 This prevents it.") == "at greater depths. This prevents it."
        assert strip_footnote_markers("Scored 3.5 on v2 in 2024.") == "Scored 3.5 on v2 in 2024."

    def test_empty_falls_back(self):
        from papertrail.retrieval.evidence import format_extractive_answer
        assert "cannot determine" in format_extractive_answer([], "x")


def test_mmr_skips_fragments_and_caps_at_two():
    import numpy as np
    from papertrail.retrieval.mmr import select_pool_sentences_mmr

    class _E:
        def encode(self, sents, normalize_embeddings=True):
            return np.ones((len(sents), 4)) / 2.0

    class _R:
        embedder = _E()

    pool = [
        "Token-Depth Vulnerability is a heading with no terminator here",   # 9 words, kept
        "Too short fragment here.",                                          # < 6 words, skipped
        "Aligned models are shallow and only affect early output tokens.",
        "Data augmentation deepens refusal behaviour across the response.",
    ]
    chosen = select_pool_sentences_mmr(_R(), "q", pool, q_emb=np.ones((1, 4)) / 2.0)
    assert 1 not in chosen
    assert len(chosen) <= 2
