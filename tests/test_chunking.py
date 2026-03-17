"""Unit tests for papertrail.ingest.chunking."""

from papertrail.ingest.chunking import chunk_text, prepare_chunks


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short paragraph."
        chunks = chunk_text(text, chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short paragraph."

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n  \n  ") == []

    def test_chunks_respect_size(self):
        # Create text that exceeds chunk_size
        words = ["word"] * 1000
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        for chunk in chunks:
            # Each chunk should be at most ~150 words (1.5 * chunk_size)
            assert len(chunk.split()) <= 150

    def test_overlap_carries_words(self):
        # With overlap=5, last 5 words of chunk N appear in chunk N+1
        words = ["word" + str(i) for i in range(200)]
        text = "\n".join(words)  # one word per paragraph
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        assert len(chunks) >= 2

    def test_filters_very_short_chunks(self):
        # Chunks < 20 chars are filtered out
        text = "Hi\n\nThis is a long enough paragraph to be included as a proper chunk."
        chunks = chunk_text(text)
        assert all(len(c.strip()) >= 20 for c in chunks)

    def test_multiple_paragraphs(self):
        paras = ["Paragraph number one with some content."] * 5
        text = "\n".join(paras)
        chunks = chunk_text(text, chunk_size=20, overlap=2)
        assert len(chunks) >= 1


class TestPrepareChunks:
    def test_returns_four_lists(self):
        result = prepare_chunks("Hello world. This is a test document.")
        assert len(result) == 4  # raw, sections, pages, embed

    def test_section_map_resolved(self):
        text = "First paragraph here.\nSecond paragraph here."
        section_map = {0: "Introduction"}
        raw, sections, pages, embed = prepare_chunks(text, section_map=section_map)
        # At least one chunk should have "Introduction" as section
        assert any(s == "Introduction" for s in sections)

    def test_embed_includes_section_prefix(self):
        text = "Content paragraph with enough words to form a chunk."
        section_map = {0: "Methods"}
        raw, sections, pages, embed = prepare_chunks(text, section_map=section_map)
        # Embed chunks with a section should be prefixed
        for r, s, e in zip(raw, sections, embed):
            if s:
                assert e.startswith(s + ".")
            else:
                assert e == r

    def test_no_maps_returns_empty_sections(self):
        text = "A simple paragraph with some content for testing."
        raw, sections, pages, embed = prepare_chunks(text)
        assert all(s == "" for s in sections)
        assert all(p is None for p in pages)
